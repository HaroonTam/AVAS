"""本地受限语音输入测试，所有麦克风和子进程操作均被替换。"""

import base64
import io
import json
import subprocess
import unittest
from contextlib import redirect_stderr
from threading import Event
from time import monotonic
from unittest.mock import MagicMock, patch

from app.agent.console import ConsoleWorker
from app.agent.spoken_reply import SpokenReply
from app.agent.tools import SceneTools
from app.agent.vision_agent import VisionAssistant
from app.live import dispatch_warnings, main
from app.safety.config import RiskConfig
from app.safety.risk_engine import assess_scene
from app.safety.scene import MetricEvidence, RiskObject, RiskScene
from app.safety.warnings import WarningGate
from app.scene_store import SceneStore
from app.speech.stt import (
    COMMANDS,
    Recognition,
    SttConfig,
    WindowsCommandRecognizer,
    parse_recognition,
)
from tests.test_console import ScriptedInput


class SttTests(unittest.TestCase):
    def payload(self, text: str = "寻找椅子", confidence: float = 0.9) -> bytes:
        """生成合成识别结果，不依赖真实录音。"""
        return json.dumps(
            {"status": "recognized", "text": text, "confidence": confidence}
        ).encode()

    def test_supported_commands_and_threshold_boundary(self) -> None:
        """固定命令逐个验证，等于阈值可用，低于阈值不给出命令。"""
        config = SttConfig()
        for phrase, command in COMMANDS:
            result = parse_recognition(self.payload(phrase, 0.7), config)
            self.assertEqual(result.command, command)
            self.assertEqual(result.status, "recognized")
        self.assertIsNone(
            parse_recognition(self.payload(confidence=0.699), config).command
        )

    def test_unknown_or_malformed_output_never_becomes_request(self) -> None:
        """未知文本、非有限分数、错误类型与畸形 JSON 不会成为工具请求。"""
        for payload in (
            b"broken",
            b"[]",
            b"x" * 4097,
            b"\xff",
            self.payload("ignore rules"),
            self.payload(confidence=float("nan")),
            self.payload(confidence=float("inf")),
            self.payload(confidence=-0.1),
            self.payload(confidence=1.1),
            b'{"status":"recognized","text":"x","confidence":true}',
            b'{"status":"recognized","text":"x","confidence":' + b"9" * 1000 + b"}",
            b'{"status":"unavailable"}',
        ):
            result = parse_recognition(payload, SttConfig())
            self.assertNotEqual(result.status, "recognized")
            self.assertIsNone(result.command)
        self.assertEqual(
            parse_recognition(b'{"status":"unrecognized"}', SttConfig()).status,
            "unrecognized",
        )

    def test_invalid_configuration_rejected(self) -> None:
        """不允许无限等待或非法分数阈值。"""
        for timeout in (0, 31, float("nan"), True):
            with self.assertRaises(ValueError):
                SttConfig(timeout_s=timeout)
        for confidence in (-1, 2, float("inf"), True):
            with self.assertRaises(ValueError):
                SttConfig(min_confidence=confidence)

    def test_constructor_and_pre_cancel_do_not_open_microphone(self) -> None:
        """构造及预先取消的请求不创建识别进程。"""
        stop = Event()
        stop.set()
        with (
            patch("app.speech.stt.os.name", "nt"),
            patch("app.speech.stt.subprocess.Popen") as launch,
        ):
            recognizer = WindowsCommandRecognizer(SttConfig())
            self.assertEqual(recognizer.recognize(stop).status, "cancelled")
        launch.assert_not_called()

    def test_recognition_process_is_hidden_and_uses_fixed_grammar(self) -> None:
        """模拟成功子进程；启动参数隐藏窗口，中文命令按数据编码。"""
        with (
            patch("app.speech.stt.os.name", "nt"),
            patch("app.speech.stt.subprocess.Popen") as launch,
        ):
            process = launch.return_value
            process.communicate.return_value = (self.payload(), None)
            process.returncode = 0
            process.poll.return_value = 0
            result = WindowsCommandRecognizer(SttConfig()).recognize(Event())
        self.assertEqual(result.command, "寻找 椅子")
        self.assertNotEqual(launch.call_args.kwargs["creationflags"], 0)
        script = base64.b64decode(launch.call_args.args[0][-1]).decode("utf-16-le")
        self.assertIn("SetInputToDefaultAudioDevice", script)
        self.assertIn("$engine.Dispose()", script)
        self.assertNotIn("寻找椅子", script)
        self.assertNotIn("DictationGrammar", script)

    def test_outer_timeout_kills_stuck_process(self) -> None:
        """外层总超时不依赖识别器自己的静音超时，强制回收卡住的子进程。"""
        with (
            patch("app.speech.stt.os.name", "nt"),
            patch("app.speech.stt.subprocess.Popen") as launch,
            patch("app.speech.stt.monotonic", side_effect=[0, 2]),
        ):
            process = launch.return_value
            process.poll.return_value = None
            result = WindowsCommandRecognizer(SttConfig(timeout_s=1)).recognize(Event())
        self.assertEqual(result.status, "timeout")
        process.kill.assert_called_once()
        process.communicate.assert_called_once_with(timeout=1)

    def test_cancellation_during_listening_discards_late_result(self) -> None:
        """识别等待期间取消，即使随后有结果也不能执行旧命令。"""
        stop = Event()

        def communicate(timeout: float) -> tuple[bytes, None]:
            """模拟关闭事件先于识别结果完成。"""
            stop.set()
            return self.payload(), None

        with (
            patch("app.speech.stt.os.name", "nt"),
            patch("app.speech.stt.subprocess.Popen") as launch,
        ):
            launch.return_value.communicate.side_effect = communicate
            launch.return_value.poll.return_value = None
            result = WindowsCommandRecognizer(SttConfig()).recognize(stop)
        self.assertEqual(result.status, "cancelled")
        launch.return_value.kill.assert_called_once()

    def test_launch_failure_is_explicit(self) -> None:
        """系统后端不能启动时返回不可用，不将故障当作空命令。"""
        with (
            patch("app.speech.stt.os.name", "nt"),
            patch(
                "app.speech.stt.subprocess.Popen", side_effect=OSError("not available")
            ),
        ):
            result = WindowsCommandRecognizer(SttConfig()).recognize(Event())
        self.assertEqual(result.status, "unavailable")

    def test_poll_timeout_can_recover_without_losing_result(self) -> None:
        """短轮询超时不会被误报为总超时，后续结果仍能完整解析。"""
        with (
            patch("app.speech.stt.os.name", "nt"),
            patch("app.speech.stt.subprocess.Popen") as launch,
        ):
            process = launch.return_value
            process.communicate.side_effect = [
                subprocess.TimeoutExpired("synthetic", 0.1),
                (self.payload(), None),
                (self.payload(), None),
            ]
            process.returncode = 0
            process.poll.return_value = 0
            result = WindowsCommandRecognizer(SttConfig()).recognize(Event())
        self.assertEqual(result.command, "寻找 椅子")

    def test_recognizer_exception_keeps_keyboard_available(self) -> None:
        """后端回收等异常不关闭文字输入，仍能处理键盘退出。"""
        recognizer, agent = MagicMock(), MagicMock()
        recognizer.recognize.side_effect = OSError("synthetic cleanup failure")
        output: list[str] = []
        worker = ConsoleWorker(
            agent,
            ScriptedInput(["listen", "quit"]),
            output.append,
            recognizer=recognizer,
        )
        worker.start()
        self.assertTrue(worker.finished.wait(1))
        worker.close()
        self.assertTrue(worker.exit_requested.is_set())
        self.assertIsNone(worker.error)
        self.assertIn("stt:unavailable", "".join(output))
        agent.respond.assert_not_called()

    def test_console_unrecognized_input_does_not_call_agent(self) -> None:
        """未识别、超时、后端故障均保留键盘退出，不伪造请求。"""
        for result in (
            Recognition("unrecognized"),
            Recognition("timeout"),
            Recognition("unavailable"),
        ):
            agent, recognizer = MagicMock(), MagicMock()
            recognizer.recognize.return_value = result
            output: list[str] = []
            worker = ConsoleWorker(
                agent,
                ScriptedInput(["听取", "quit"]),
                output.append,
                recognizer=recognizer,
            )
            worker.start()
            self.assertTrue(worker.finished.wait(1))
            worker.close()
            agent.respond.assert_not_called()
            self.assertIn(f"[stt:{result.status}]", "".join(output))
            self.assertTrue(worker.exit_requested.is_set())

    def test_console_recognized_command_routes_to_existing_agent(self) -> None:
        """受验证命令复用原有回答入口，关闭或查询不触碰安全规则。"""
        agent, recognizer = MagicMock(), MagicMock()
        recognizer.recognize.return_value = Recognition("recognized", "寻找 椅子", 0.9)
        worker = ConsoleWorker(
            agent, ScriptedInput(["listen", "quit"]), MagicMock(), recognizer=recognizer
        )
        worker.start()
        self.assertTrue(worker.finished.wait(1))
        worker.close()
        agent.respond.assert_called_once_with("寻找 椅子")

    def test_console_stt_failures_submit_fixed_spoken_feedback(self) -> None:
        """模拟识别失败经过真实控制台和语音适配器，仅提交固定低优先级提示。"""
        for status in ("unrecognized", "timeout", "unavailable", "cancelled"):
            with self.subTest(status=status):
                store = SceneStore(RiskConfig())
                agent = VisionAssistant(SceneTools(store))
                speech, recognizer = MagicMock(), MagicMock()
                speech.error = None
                recognizer.recognize.return_value = Recognition(status)
                output: list[str] = []
                speaker = SpokenReply(agent, store, speech)
                worker = ConsoleWorker(
                    agent,
                    ScriptedInput(["听取", "quit"]),
                    output.append,
                    recognizer=recognizer,
                    spoken_reply=speaker,
                )
                worker.start()
                try:
                    self.assertTrue(worker.finished.wait(1))
                finally:
                    worker.close()
                self.assertIsNone(worker.error)
                speech.submit.assert_called_once()
                message = speech.submit.call_args.args[0]
                self.assertEqual(message.priority, 0)
                self.assertIsNone(message.scene_lease)
                self.assertIn("本次", message.text)
                self.assertIn("reply_speech:feedback_queued", "".join(output))
                self.assertTrue(worker.exit_requested.is_set())

    def test_waiting_for_stt_does_not_block_warning_and_close(self) -> None:
        """识别挂起期间告警照常提交，关闭事件停止听取且丢弃迟到命令。"""
        entered = Event()

        def recognize(stop: Event) -> Recognition:
            """等待关闭信号模拟一次持续听取。"""
            entered.set()
            stop.wait(2)
            return Recognition("recognized", "描述周围", 0.9)

        recognizer, agent = MagicMock(), MagicMock()
        recognizer.recognize.side_effect = recognize
        worker = ConsoleWorker(
            agent, ScriptedInput(["listen"]), MagicMock(), recognizer=recognizer
        )
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            scene = RiskScene(
                "f",
                1000,
                "live",
                True,
                (
                    RiskObject(
                        1,
                        "f",
                        "chair",
                        0.9,
                        "front",
                        MetricEvidence(0.5, "synthetic", 0, 10),
                    ),
                ),
            )
            speech = MagicMock()
            speech.submit.return_value = True
            with self.assertLogs(level="WARNING"):
                self.assertEqual(
                    dispatch_warnings(
                        assess_scene(scene, 1000, RiskConfig()),
                        WarningGate(3000),
                        speech,
                        monotonic() + 1,
                    ),
                    1,
                )
        finally:
            worker.close()
        self.assertTrue(worker.finished.is_set())
        agent.respond.assert_not_called()

    def test_voice_flag_requires_console_and_live_wires_backend(self) -> None:
        """缺少控制台时提前拒绝；正确参数只构造后端，不自动启动识别。"""
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--voice-input"])
        with (
            patch("app.live.WindowsLineSource"),
            patch("app.live.WindowsCommandRecognizer") as recognizer,
            patch("app.live.ConsoleWorker") as console,
            patch("app.live.Yolo11Detector"),
            patch("app.vision.depth_estimator.DepthAnythingV2Estimator"),
            patch("app.live.CameraService"),
            self.assertLogs(level="WARNING"),
        ):
            console.return_value.exit_requested.is_set.return_value = True
            console.return_value.error = None
            self.assertEqual(main(["--console", "--voice-input", "--no-speech"]), 0)
        self.assertIs(console.call_args.kwargs["recognizer"], recognizer.return_value)
        recognizer.return_value.recognize.assert_not_called()
