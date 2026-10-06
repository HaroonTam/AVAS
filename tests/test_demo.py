"""无硬件演示的入口、可复现性和故障退出验收。"""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from app.demo import main, run_demo
from app.speech.stt import Recognition


class DemoTests(unittest.TestCase):
    def test_demo_uses_no_hardware_models_or_subprocesses(self) -> None:
        """禁止真实资源入口，确认六个场景及五个交互检查均走通。"""
        with (
            patch(
                "app.camera.camera_service.CameraService.__init__",
                side_effect=AssertionError("camera forbidden"),
            ),
            patch(
                "app.vision.detector.Yolo11Detector.__init__",
                side_effect=AssertionError("model forbidden"),
            ),
            patch(
                "app.vision.depth_estimator.DepthAnythingV2Estimator.__init__",
                side_effect=AssertionError("depth model forbidden"),
            ),
            patch(
                "app.speech.tts.WindowsSpeechBackend.__init__",
                side_effect=AssertionError("audio forbidden"),
            ),
            patch("subprocess.Popen", side_effect=AssertionError("process forbidden")),
            patch(
                "app.speech.stt.WindowsCommandRecognizer.__init__",
                side_effect=AssertionError("microphone forbidden"),
            ),
            self.assertLogs(level="WARNING"),
        ):
            report = run_demo()
        self.assertTrue(report.synthetic)
        self.assertTrue(report.passed)
        self.assertEqual(len(report.cases), 6)
        self.assertEqual(len(report.simulated_speech), 7)
        self.assertTrue(report.interaction.synthetic)
        self.assertTrue(report.interaction.passed)
        self.assertEqual(len(report.interaction.checks), 5)
        self.assertEqual(len(report.interaction.simulated_speech), 5)
        self.assertEqual(report.cases[-2].answer.status, "scene_stale")
        self.assertIsNone(report.cases[-1].answer.frame_id)

    def test_json_stdout_is_parseable_and_labels_synthetic_data(self) -> None:
        """JSON 独占 stdout，包含输入配置和合成标识，日志不污染解析。"""
        output, errors = io.StringIO(), io.StringIO()
        with (
            redirect_stdout(output),
            redirect_stderr(errors),
            self.assertLogs(level="WARNING"),
        ):
            self.assertEqual(main(["--json"]), 0)
        payload = json.loads(output.getvalue())
        self.assertTrue(payload["synthetic"])
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["schema_version"], 2)
        self.assertTrue(payload["interaction"]["passed"])
        self.assertTrue(payload["interaction"]["transcript"])
        self.assertEqual(payload["risk_config"]["freshness_ms"], 1000)
        self.assertEqual(payload["fusion_config"]["camera_orientation"], "forward")
        self.assertIn("SYNTHETIC DEMO ONLY", errors.getvalue())

    def test_repeated_demo_produces_same_semantic_report(self) -> None:
        """固定输入与合成时钟使重复执行结果一致，不依赖真实采集时间。"""
        with self.assertLogs(level="WARNING"):
            first, second = run_demo(), run_demo()
        self.assertEqual(first, second)

    def test_simulated_backend_failure_returns_nonzero(self) -> None:
        """模拟输出失败不能给出全通过结论，CLI 返回失败退出码。"""
        output = io.StringIO()
        with (
            patch(
                "app.demo.RecordingSpeech.start",
                side_effect=OSError("synthetic failure"),
            ),
            redirect_stdout(output),
            redirect_stderr(io.StringIO()),
            self.assertLogs(level="WARNING"),
        ):
            self.assertEqual(main(["--json"]), 1)
        self.assertFalse(json.loads(output.getvalue())["passed"])

    def test_broken_recognition_causes_interaction_failure(self) -> None:
        """脚本识别结果错误时交互验收失败，即使原有六个场景仍通过也不能报全通过。"""
        output = io.StringIO()
        with (
            patch(
                "app.demo_interaction.ScriptedRecognizer.recognize",
                return_value=Recognition("unavailable"),
            ),
            redirect_stdout(output),
            redirect_stderr(io.StringIO()),
            self.assertLogs(level="WARNING"),
        ):
            self.assertEqual(main(["--json"]), 1)
        payload = json.loads(output.getvalue())
        self.assertTrue(all(case["passed"] for case in payload["cases"]))
        self.assertFalse(payload["interaction"]["passed"])
        self.assertFalse(payload["passed"])
