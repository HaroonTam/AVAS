"""控制台合成测试，不打开真实终端、相机或语音设备。"""

import io
import unittest
from collections import deque
from contextlib import redirect_stdout
from threading import Event
from time import monotonic, time_ns
from unittest.mock import MagicMock, patch

from app.agent.console import ConsoleWorker, WindowsLineSource
from app.agent.tools import SceneTools
from app.agent.vision_agent import AssistantAnswer, VisionAssistant
from app.live import LiveResult, main
from app.safety.config import RiskConfig
from app.safety.scene import RiskObject, RiskScene
from app.scene_store import SceneStore


class ScriptedInput:
    def __init__(self, lines: list[str]) -> None:
        """保存有限输入，耗尽后保持空闲以验证可停止轮询。"""
        self.lines = deque(lines)

    def poll(self) -> str | None:
        """不阻塞地返回下一行。"""
        return self.lines.popleft() if self.lines else None


class ConsoleTests(unittest.TestCase):
    def setUp(self) -> None:
        """用当前合成帧构造可查询的本地交互器。"""
        self.store = SceneStore(RiskConfig(freshness_ms=10000))
        self.store.publish(
            RiskScene(
                "synthetic",
                time_ns() // 1_000_000,
                "live",
                True,
                (RiskObject(1, "synthetic", "chair", 0.9, "front"),),
            )
        )
        self.agent = VisionAssistant(SceneTools(self.store))

    def test_query_then_quit(self) -> None:
        """文字请求查询同一场景，退出命令只设置退出信号。"""
        output: list[str] = []
        worker = ConsoleWorker(
            self.agent, ScriptedInput(["寻找 椅子", "quit"]), output.append
        )
        worker.start()
        self.assertTrue(worker.finished.wait(1))
        worker.close()
        self.assertTrue(worker.exit_requested.is_set())
        self.assertIn("chair", "".join(output))
        self.assertIn("米制距离不可用", "".join(output))
        self.assertIsNone(worker.error)

    def test_idle_close_and_repeated_close_are_bounded(self) -> None:
        """未输入完整行时无需回车即可关闭，可重复调用。"""
        worker = ConsoleWorker(self.agent, ScriptedInput([]), MagicMock())
        worker.start()
        started = monotonic()
        worker.close()
        worker.close()
        self.assertLess(monotonic() - started, 1)
        self.assertTrue(worker.finished.is_set())

    def test_eof_does_not_request_perception_shutdown(self) -> None:
        """输入结束只停止交互，不关闭感知与告警。"""
        source = MagicMock()
        source.poll.side_effect = EOFError
        worker = ConsoleWorker(self.agent, source, MagicMock())
        worker.start()
        self.assertTrue(worker.finished.wait(1))
        worker.close()
        self.assertFalse(worker.exit_requested.is_set())
        self.assertIsNone(worker.error)

    def test_input_failure_is_reported_without_private_text(self) -> None:
        """输入异常仅记录异常类别，不泄露异常中的请求文本。"""
        source = MagicMock()
        source.poll.side_effect = OSError("private request")
        worker = ConsoleWorker(self.agent, source, MagicMock())
        worker.start()
        self.assertTrue(worker.finished.wait(1))
        worker.close()
        self.assertEqual(worker.error, "console unavailable: OSError")

    def test_late_answer_not_emitted_after_close(self) -> None:
        """异常慢查询不无限阻塞关闭，返回后不输出旧回答。"""
        entered, release = Event(), Event()
        output: list[str] = []

        def slow_response(text: str) -> AssistantAnswer:
            """模拟阻塞查询，不调用外部服务。"""
            entered.set()
            release.wait(2)
            return AssistantAnswer("available", "old", "不得输出的旧回答")

        with patch.object(self.agent, "respond", side_effect=slow_response):
            worker = ConsoleWorker(
                self.agent, ScriptedInput(["描述周围"]), output.append
            )
            worker.start()
            try:
                self.assertTrue(entered.wait(1))
                worker.close()
                self.assertEqual(worker.error, "console shutdown timed out")
            finally:
                release.set()
                self.assertTrue(worker.finished.wait(1))
                worker.close()
        self.assertNotIn("不得输出", "".join(output))

    def test_character_editing_overflow_and_special_keys(self) -> None:
        """字符输入有界，退格可编辑，超长行不执行合法后缀。"""
        keyboard = MagicMock()
        with (
            patch("app.agent.console.os.name", "nt"),
            patch("app.agent.console.sys.stdin.isatty", return_value=True),
            patch.dict("sys.modules", {"msvcrt": keyboard}),
        ):
            source = WindowsLineSource(MagicMock())
        chars = list("find chaiX\br\r") + ["\x00", "H"] + list("x" * 300 + "quit\r")
        keyboard.kbhit.return_value = True
        keyboard.getwch.side_effect = chars
        results: list[str] = []
        for _ in chars:
            line = source.poll()
            if line is not None:
                results.append(line)
        self.assertEqual(results[0], "find chair")
        self.assertEqual(len(results[1]), 257)
        self.assertEqual(self.agent.respond(results[1]).status, "unsupported_request")

    def test_redirected_stdin_rejected_before_loading_models(self) -> None:
        """不支持的重定向输入在设备启动前明确失败。"""
        with (
            patch("app.agent.console.sys.stdin.isatty", return_value=False),
            patch("app.live.Yolo11Detector") as detector,
            patch("app.live.CameraService") as camera,
            self.assertLogs(level="ERROR"),
        ):
            self.assertEqual(main(["--console", "--no-speech"]), 2)
        detector.assert_not_called()
        camera.assert_not_called()

    def test_console_quit_closes_camera_without_waiting_for_frame(self) -> None:
        """控制台发出退出信号后主循环清理设备，不再读取下一帧。"""
        with (
            patch("app.live.WindowsLineSource"),
            patch("app.live.ConsoleWorker") as console,
            patch("app.live.Yolo11Detector"),
            patch("app.vision.depth_estimator.DepthAnythingV2Estimator"),
            patch("app.live.CameraService") as camera,
            redirect_stdout(io.StringIO()),
            self.assertLogs(level="WARNING"),
        ):
            console.return_value.exit_requested.is_set.return_value = True
            console.return_value.error = None
            self.assertEqual(main(["--console", "--no-speech"]), 0)
        camera.return_value.read.assert_not_called()
        camera.return_value.close.assert_called_once()
        console.return_value.close.assert_called_once()

    def test_console_failure_keeps_frame_warnings_and_suppresses_frame_json(
        self,
    ) -> None:
        """交互故障后仍处理后续帧告警，控制台模式不刷逐帧 JSON。"""
        scene = self.store.read().scene
        self.assertIsNotNone(scene)
        if scene is None:
            self.fail("synthetic scene missing")
        output = io.StringIO()
        with (
            patch("app.live.WindowsLineSource"),
            patch("app.live.ConsoleWorker") as console,
            patch("app.live.Yolo11Detector"),
            patch("app.vision.depth_estimator.DepthAnythingV2Estimator"),
            patch("app.live.PerceptionPipeline") as pipeline,
            patch("app.live.CameraService") as camera,
            patch("app.live.dispatch_warnings", return_value=0) as warnings,
            redirect_stdout(output),
            self.assertLogs(level="WARNING"),
        ):
            console.return_value.exit_requested.is_set.return_value = False
            console.return_value.error = "synthetic console failure"
            camera.return_value.read.return_value.captured_monotonic = monotonic()
            pipeline.return_value.process.return_value = LiveResult(
                scene, (), "available", {}
            )
            self.assertEqual(main(["--console", "--no-speech", "--max-frames", "2"]), 2)
        self.assertEqual(warnings.call_count, 2)
        self.assertEqual(camera.return_value.read.call_count, 2)
        self.assertEqual(output.getvalue(), "")
        console.return_value.close.assert_called_once()
