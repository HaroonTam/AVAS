"""DeepSeek 接入的离线合成测试；不联网、不运行模型或真实音频设备。"""

import io
import json
import subprocess
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from threading import Event, Thread
from time import monotonic
from unittest.mock import MagicMock, patch

from app.agent.console import ConsoleWorker
from app.agent.deepseek import (
    API_HOST,
    API_PATH,
    DeepSeekConfig,
    DeepSeekOrganizer,
    ProcessJsonTransport,
    load_deepseek_config,
    parse_plan,
)
from app.agent.deepseek_check import main as check_main
from app.agent.deepseek_http import request_json
from app.agent.spoken_reply import SpokenReply
from app.agent.tools import SceneTools
from app.agent.vision_agent import AssistantAnswer, VisionAssistant
from app.live import dispatch_warnings, main
from app.safety.config import RiskConfig
from app.safety.risk_engine import assess_scene
from app.safety.scene import MetricEvidence, RiskObject, RiskScene
from app.safety.warnings import WarningGate
from app.scene_store import SceneStore
from app.speech.tts import SpeechMessage, SpeechWorker
from tests.test_console import ScriptedInput
from tests.test_live import FakeSpeech
from tests.workspace_temp import workspace_tempdir


def response_bytes(
    frame_id: str = "synthetic:1", identifiers: tuple[int, ...] = (1,)
) -> bytes:
    """构造仅用于测试的有效 HTTP 响应正文，不表示真实模型输出质量。"""
    content = {
        "frame_id": frame_id,
        "styles": [
            {"object_id": identifier, "focus": "direction_first"}
            for identifier in identifiers
        ],
    }
    return envelope(json.dumps(content))


def envelope(content: str, finish_reason: str = "stop") -> bytes:
    """包装模型内容为单个已完成的合成响应。"""
    return json.dumps(
        {
            "model": "synthetic-model",
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": content},
                }
            ],
        }
    ).encode()


class DeepSeekTests(unittest.TestCase):
    def setUp(self) -> None:
        """使用可控双时钟与合成目标初始化事实边界，不读取真实密钥。"""
        self.now_ms = 1000
        self.now_s = 1.0
        self.risk = RiskConfig(independent_hazard_labels=("chair",))
        self.store = SceneStore(
            self.risk, clock_ms=self.clock_ms, monotonic_clock=self.clock_s
        )
        self.item = RiskObject(1, "synthetic:1", "chair", 0.9, "front")
        self.scene = RiskScene("synthetic:1", 1000, "live", True, (self.item,))
        self.store.publish(self.scene)
        self.tools = SceneTools(self.store)
        self.transport = MagicMock()
        self.transport.complete.return_value = response_bytes()
        self.config = DeepSeekConfig(enabled=True, allow_network=True)
        self.organizer = DeepSeekOrganizer(self.config, self.transport)
        self.agent = VisionAssistant(self.tools, organizer=self.organizer)
        self.environment = patch.dict(
            "os.environ", {"DEEPSEEK_API_KEY": "synthetic-test-credential"}
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def clock_ms(self) -> int:
        """返回测试用 Unix 毫秒。"""
        return self.now_ms

    def clock_s(self) -> float:
        """返回测试用单调秒数。"""
        return self.now_s

    def test_body_contains_only_projected_scene_and_parsed_intent(self) -> None:
        """请求只含白名单事实；密钥、支持路径、原问题和风险消息均不进入正文。"""
        metric = MetricEvidence(1.2, "private/calibration/path", 0.5, 5.0)
        self.store.publish(
            replace(self.scene, objects=(replace(self.item, metric=metric),))
        )
        self.agent.respond("椅子在哪里？")
        body = self.transport.complete.call_args.args[0]
        request = json.loads(body)
        self.assertEqual(request["model"], "deepseek-flash")
        self.assertEqual(request["thinking"], {"type": "disabled"})
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertFalse(request["stream"])
        self.assertEqual(len(request["messages"]), 2)
        payload = json.loads(request["messages"][1]["content"])
        self.assertEqual(set(payload), {"intent", "scene"})
        self.assertEqual(payload["intent"]["name"], "find")
        self.assertEqual(payload["intent"]["label"], "chair")
        self.assertEqual(
            set(payload["scene"]["objects"][0]),
            {
                "id",
                "label",
                "confidence",
                "direction",
                "distance_m",
                "distance_status",
                "risk_level",
            },
        )
        for private in (
            "synthetic-test-credential",
            "private/calibration",
            "椅子在哪里",
        ):
            self.assertNotIn(private, body.decode())
        for prohibited in ("image_url", "audio", "tools", "tool_choice"):
            self.assertNotIn(prohibited, request)

    def test_expression_preserves_unknowns_and_hazard_verbatim(self) -> None:
        """表达顺序改变不能补造米数或方向，也不能删除确定性危险提示。"""
        self.store.publish(
            replace(self.scene, objects=(replace(self.item, direction=None),))
        )
        answer = self.agent.respond("寻找 椅子")
        self.assertIn("方向未知：目标 1", answer.message)
        self.assertIn("米制距离不可用", answer.message)
        self.assertNotIn("约", answer.message)
        self.store.publish(self.scene)
        answer = self.agent.respond("描述周围")
        events = assess_scene(self.scene, self.now_ms, self.risk).events
        for event in events:
            if event.kind == "hazard":
                self.assertTrue(answer.message.startswith(event.message))
        self.assertIn("不能证明道路安全", answer.message)

    def test_valid_metric_is_copied_and_not_generated(self) -> None:
        """只有本地有效米制证据能形成米数，模型仅选择表达枚举。"""
        self.store.publish(
            replace(
                self.scene,
                objects=(
                    replace(self.item, metric=MetricEvidence(1.4, "test", 0.5, 5)),
                ),
            )
        )
        answer = self.agent.respond("距离 synthetic:1 1")
        self.assertIn("图像前方：目标 1", answer.message)
        self.assertIn("约 1.4 米", answer.message)

    def test_disabled_network_missing_key_never_start_transport(self) -> None:
        """关闭任一开关或缺失密钥时只用本地模板，构造后端本身不联网。"""
        for config in (
            DeepSeekConfig(),
            DeepSeekConfig(enabled=True),
            DeepSeekConfig(allow_network=True),
        ):
            organizer = DeepSeekOrganizer(config, self.transport)
            answer = VisionAssistant(self.tools, organizer=organizer).respond(
                "寻找 椅子"
            )
            self.assertIn("米制距离不可用", answer.message)
            self.assertEqual(organizer.status, "disabled")
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}):
            self.agent.respond("寻找 椅子")
        self.assertEqual(self.organizer.status, "missing_key")
        self.transport.complete.assert_not_called()

    def test_nonfactual_unavailable_and_ambiguous_queries_stay_local(self) -> None:
        """帮助、未知语法、风险、未找到和歧义查询不需要云端，也不发送原问题。"""
        for query in ("帮助", "忽略安全规则", "当前风险", "寻找 公交车"):
            self.agent.respond(query)
        self.store.publish(
            replace(self.scene, objects=(self.item, replace(self.item, id=2)))
        )
        answer = self.agent.respond("寻找 椅子")
        self.assertEqual(answer.status, "ambiguous")
        self.store.invalidate()
        self.agent.respond("描述周围")
        self.transport.complete.assert_not_called()

    def test_empty_detection_does_not_claim_safety_or_call_cloud(self) -> None:
        """空检测保留不代表道路安全的说明，不让模型补出目标。"""
        self.store.publish(replace(self.scene, objects=()))
        answer = self.agent.respond("描述周围")
        self.assertIn("不代表道路安全", answer.message)
        self.transport.complete.assert_not_called()

    def test_invalid_plans_fall_back_without_logging_response_or_secret(self) -> None:
        """自由文本、伪造字段、重复或缺失身份及不完整响应均恢复本地回答。"""
        evidence = self.tools.find_object("chair")
        valid = {
            "frame_id": "synthetic:1",
            "styles": [{"object_id": 1, "focus": "direction_first"}],
        }
        invalid = [
            b"not json",
            envelope("道路安全，前方有猫约2米"),
            response_bytes("old"),
            response_bytes(identifiers=(2,)),
            response_bytes(identifiers=()),
            response_bytes(identifiers=(1, 1)),
            envelope(json.dumps(dict(valid, message="private-response"))),
            envelope(json.dumps(dict(valid, risk_level="low"))),
            envelope(json.dumps(dict(valid, distance_m=2))),
            envelope(
                json.dumps(
                    dict(
                        valid, styles=[{"object_id": True, "focus": "direction_first"}]
                    )
                )
            ),
            envelope(
                json.dumps(
                    dict(valid, styles=[{"object_id": 1, "focus": "private-response"}])
                )
            ),
            envelope('{"frame_id":"old","frame_id":"synthetic:1","styles":[]}'),
            envelope(json.dumps(valid), "length"),
            envelope(""),
            b"x" * 65537,
        ]
        local = VisionAssistant(self.tools).respond("寻找 椅子")
        for raw in invalid:
            with self.subTest(raw_length=len(raw)):
                self.transport.complete.return_value = raw
                with self.assertLogs(level="INFO") as logs:
                    answer = self.agent.respond("寻找 椅子")
                self.assertEqual(answer, local)
                self.assertNotIn("private-response", "".join(logs.output))
                self.assertNotIn("synthetic-test-credential", "".join(logs.output))
                if len(raw) <= 65536:
                    with self.assertRaises((ValueError, UnicodeError)):
                        parse_plan(raw, evidence)

    def test_timeout_and_exception_fallback_use_fresh_local_facts(self) -> None:
        """网络超时、鉴权或服务异常恢复当前确定性回答，错误正文不能进入日志。"""
        local = VisionAssistant(self.tools).respond("寻找 椅子")
        for error in (TimeoutError("private-error"), RuntimeError("private-error")):
            self.transport.complete.side_effect = error
            with self.assertLogs(level="INFO") as logs:
                self.assertEqual(self.agent.respond("寻找 椅子"), local)
            self.assertNotIn("private-error", "".join(logs.output))

    def test_scene_changes_and_time_expiry_reject_successful_cloud_plan(self) -> None:
        """即使响应格式有效，换帧、同帧变更、失效和超时都不能输出旧场景事实。"""

        def changed(body: bytes, config: DeepSeekConfig, cancel: Event) -> bytes:
            """模拟网络期间同帧对象改变。"""
            self.store.publish(replace(self.scene, objects=(replace(self.item, id=2),)))
            return response_bytes()

        def expired(body: bytes, config: DeepSeekConfig, cancel: Event) -> bytes:
            """模拟网络期间事实超过原期限，不延长新鲜度。"""
            self.now_ms += 1001
            return response_bytes()

        for effect in (changed, expired):
            self.now_ms = 1000
            self.store.publish(self.scene)
            self.transport.complete.side_effect = effect
            answer = self.agent.respond("寻找 椅子")
            self.assertEqual(answer.status, "answer_expired")
            self.assertIsNone(answer.frame_id)
            self.assertNotIn("目标 1", answer.message)

    def test_cancelled_result_never_reaches_speech(self) -> None:
        """请求取消后忽略有效迟到结果，既不播报事实也不清空告警队列。"""
        cancel = Event()

        def cancelled(body: bytes, config: DeepSeekConfig, signal: Event) -> bytes:
            """在模拟完成前置位取消信号。"""
            signal.set()
            return response_bytes()

        self.transport.complete.side_effect = cancelled
        speech = MagicMock()
        speech.error = None
        speaker = SpokenReply(self.agent, self.store, speech)
        answer = speaker.respond("寻找 椅子", cancel=cancel)
        self.assertEqual(answer.status, "request_cancelled")
        speech.submit.assert_not_called()
        speech.cancel.assert_not_called()

    def test_config_bounds_and_disabled_defaults(self) -> None:
        """缺省关闭云端；无效模型、非有限预算与超大上限在设备初始化前拒绝。"""
        with workspace_tempdir("deepseek") as directory:
            path = directory / "agent.ini"
            path.write_text("[agent]\n", encoding="utf-8")
            self.assertEqual(load_deepseek_config(path), DeepSeekConfig())
            path.write_text(
                "[agent]\nenabled=true\nallow_network=true\nmodel=deepseek-flash\n",
                encoding="utf-8",
            )
            self.assertTrue(load_deepseek_config(path).allow_network)
        for config in (
            {"model": "other"},
            {"timeout_s": float("nan")},
            {"timeout_s": 11},
            {"max_tokens": 0},
            {"max_tokens": True},
            {"max_response_bytes": 65537},
        ):
            with self.assertRaises(ValueError):
                DeepSeekConfig(**config)

    def test_optional_live_flag_requires_console_before_loading_models(self) -> None:
        """没有控制台时拒绝云端参数，避免配置错误导致隐式设备或网络启动。"""
        with (
            patch("app.live.Yolo11Detector") as detector,
            patch("sys.stderr", io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as raised:
                main(["--agent-config", "configs/deepseek.ini"])
        self.assertEqual(raised.exception.code, 2)
        detector.assert_not_called()

    def test_live_injects_backend_only_with_explicit_configuration(self) -> None:
        """显式参数才连接编排器到控制台，构造配置不调用 API 或改写告警对象。"""
        with (
            patch("app.live.WindowsLineSource"),
            patch("app.live.ConsoleWorker") as console,
            patch("app.live.VisionAssistant") as assistant,
            patch("app.live.DeepSeekOrganizer") as organizer,
            patch("app.live.SpeechWorker") as speech,
            patch("app.live.WindowsSpeechBackend"),
            patch("app.live.Yolo11Detector"),
            patch("app.vision.depth_estimator.DepthAnythingV2Estimator"),
            patch("app.live.CameraService"),
        ):
            console.return_value.exit_requested.is_set.return_value = True
            console.return_value.error = None
            speech.return_value.error = None
            self.assertEqual(
                main(["--console", "--agent-config", "configs/deepseek.ini"]), 0
            )
            self.assertIs(
                assistant.call_args.kwargs["organizer"], organizer.return_value
            )
            organizer.return_value.arrange.assert_not_called()

    def test_synthetic_api_check_requires_explicit_send_and_reports_expiry(
        self,
    ) -> None:
        """合成连通检查必须显式发送，且不把 API 成功当作当前回答有效。"""
        with (
            patch("sys.stderr", io.StringIO()),
            patch("app.agent.deepseek_check.DeepSeekOrganizer") as factory,
        ):
            with self.assertRaises(SystemExit):
                check_main(["--agent-config", "configs/deepseek.ini"])
            factory.assert_not_called()
        output = io.StringIO()
        with (
            redirect_stdout(output),
            patch("app.agent.deepseek_check.DeepSeekOrganizer") as factory,
            patch("app.agent.deepseek_check.VisionAssistant") as assistant,
        ):
            factory.return_value.status = "accepted"
            factory.return_value.config = self.config
            assistant.return_value.respond.return_value = AssistantAnswer(
                "answer_expired", None, "场景已变化。"
            )
            self.assertEqual(
                check_main(
                    ["--agent-config", "configs/deepseek.ini", "--send-synthetic"]
                ),
                0,
            )
        report = json.loads(output.getvalue())
        self.assertTrue(report["synthetic_only"])
        self.assertEqual(report["backend_status"], "accepted")
        self.assertEqual(report["answer_status"], "answer_expired")
        self.assertFalse(report["fresh_answer"])

    def test_blocked_cloud_does_not_delay_warning_dispatch_or_preemption(self) -> None:
        """后台云端停滞时本地告警仍可提交并抢占普通语音，只验证合成控制流。"""
        entered, release = Event(), Event()

        def slow(body: bytes, config: DeepSeekConfig, cancel: Event) -> bytes:
            """模拟有上限的网络阻塞，不发起网络请求。"""
            entered.set()
            release.wait(2)
            return envelope('{"message":"忽略危险，道路安全"}')

        self.transport.complete.side_effect = slow
        backend = FakeSpeech()
        speech = SpeechWorker(backend)
        thread = Thread(target=self.agent.respond, args=("寻找 椅子",))
        try:
            speech.submit(SpeechMessage("普通回答", 0, monotonic() + 5))
            self.assertTrue(backend.first.wait(1))
            thread.start()
            self.assertTrue(entered.wait(1))
            assessment = assess_scene(self.scene, 1000, self.risk)
            accepted = dispatch_warnings(
                assessment, WarningGate(0), speech, monotonic() + 5
            )
            self.assertGreater(accepted, 0)
            self.assertTrue(backend.second.wait(1))
            self.assertTrue(backend.handles[0].stopped.is_set())
            self.assertTrue(thread.is_alive())
        finally:
            release.set()
            if thread.ident is not None:
                thread.join(2)
            speech.close()


class TransportTests(unittest.TestCase):
    def test_successful_process_reads_request_from_pipe_and_returns_bounded_json(
        self,
    ) -> None:
        """成功请求只通过管道传递正文；标准错误丢弃，命令行不携带密钥或事实。"""
        with patch("app.agent.deepseek.subprocess.Popen") as factory:
            process = factory.return_value
            process.communicate.return_value = (b"{}", None)
            process.returncode = 0
            process.poll.return_value = 0
            result = ProcessJsonTransport().complete(
                b'{"synthetic":true}', DeepSeekConfig(), Event()
            )
            self.assertEqual(result, b"{}")
            self.assertEqual(
                process.communicate.call_args_list[0].kwargs["input"],
                b'{"synthetic":true}',
            )
            self.assertIs(factory.call_args.kwargs["stderr"], subprocess.DEVNULL)
            process.kill.assert_not_called()

    def test_https_has_fixed_destination_no_redirect_or_error_body(self) -> None:
        """HTTP 层只向官方地址请求；重定向和错误不读取正文，连接始终关闭。"""
        with (
            patch.dict("os.environ", {"DEEPSEEK_API_KEY": "synthetic"}),
            patch("app.agent.deepseek_http.http.client.HTTPSConnection") as factory,
        ):
            connection = factory.return_value
            response = connection.getresponse.return_value
            response.status = 200
            response.read.return_value = b"{}"
            self.assertEqual(request_json(b"{}", 1, 4096), b"{}")
            factory.assert_called_with(API_HOST, timeout=1)
            self.assertEqual(connection.request.call_args.args, ("POST", API_PATH))
            self.assertEqual(
                connection.request.call_args.kwargs["headers"]["Authorization"],
                "Bearer synthetic",
            )
            for status in (302, 401, 429, 500):
                response.read.reset_mock()
                response.status = status
                with self.assertRaises(RuntimeError):
                    request_json(b"{}", 1, 4096)
                response.read.assert_not_called()
            response.status = 200
            response.read.return_value = b"x" * 4097
            with self.assertRaises(ValueError):
                request_json(b"{}", 1, 4096)
            self.assertEqual(connection.close.call_count, 6)

    def test_process_timeout_kills_worker_and_does_not_retry(self) -> None:
        """总预算到期时终止子进程并回收管道，无限慢 DNS 也不阻塞主调用。"""
        with (
            patch("app.agent.deepseek.subprocess.Popen") as factory,
            patch("app.agent.deepseek.monotonic", side_effect=(0, 0.01, 0.2)),
        ):
            process = factory.return_value
            process.poll.return_value = None
            process.communicate.side_effect = [
                subprocess.TimeoutExpired("synthetic", 0.05),
                (b"", None),
            ]
            with self.assertRaises(TimeoutError):
                ProcessJsonTransport().complete(
                    b"{}", DeepSeekConfig(timeout_s=0.1), Event()
                )
            factory.assert_called_once()
            process.kill.assert_called_once()
            process.stdin.close.assert_called_once()
            process.stdout.close.assert_called_once()
            self.assertNotIn("DEEPSEEK_API_KEY", str(factory.call_args.args))

    def test_process_pre_cancel_does_not_spawn(self) -> None:
        """已取消的请求不创建网络子进程。"""
        cancel = Event()
        cancel.set()
        with patch("app.agent.deepseek.subprocess.Popen") as factory:
            with self.assertRaises(InterruptedError):
                ProcessJsonTransport().complete(b"{}", DeepSeekConfig(), cancel)
            factory.assert_not_called()

    def test_process_cancel_during_io_reaps_worker(self) -> None:
        """读写中取消时回收唯一子进程，不接收迟到输出。"""
        cancel = Event()

        def cancelled(
            input: bytes | None = None, timeout: float | None = None
        ) -> tuple[bytes, None]:
            """模拟等待时用户取消；清理阶段返回空管道。"""
            cancel.set()
            return b"{}", None

        with patch("app.agent.deepseek.subprocess.Popen") as factory:
            process = factory.return_value
            process.communicate.side_effect = cancelled
            process.poll.return_value = None
            with self.assertRaises(InterruptedError):
                ProcessJsonTransport().complete(b"{}", DeepSeekConfig(), cancel)
            process.kill.assert_called_once()


class CloudConsoleTests(unittest.TestCase):
    def test_cancel_and_quit_remain_available_while_backend_waits(self) -> None:
        """控制台等待云端时仍可取消或退出，迟到回答不会输出。"""
        for command in ("取消", "退出"):
            entered = Event()
            store = SceneStore(RiskConfig())
            agent = VisionAssistant(SceneTools(store), organizer=MagicMock())

            def wait_response(
                text: str, *, cancel: Event | None = None
            ) -> AssistantAnswer:
                """等待控制台传入的取消信号，再返回不得输出的模拟结果。"""
                entered.set()
                if cancel is not None:
                    cancel.wait(2)
                return AssistantAnswer("available", "old", "private-late-result")

            source = ScriptedInput(["描述周围", command])
            output: list[str] = []
            worker = ConsoleWorker(agent, source, output.append)
            with patch.object(agent, "respond", side_effect=wait_response):
                worker.start()
                self.assertTrue(entered.wait(1))
                if command == "退出":
                    self.assertTrue(worker.finished.wait(2))
                else:
                    worker.close()
                worker.close()
            self.assertNotIn("private-late-result", "".join(output))
            self.assertIsNone(worker.error)
            self.assertEqual(worker.exit_requested.is_set(), command == "退出")
