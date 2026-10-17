"""相机启动诊断协议的合成测试，不打开设备。"""

import io
import json
import subprocess
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np

from app.camera_probe import main, probe_worker, run_probe
from tests.workspace_temp import workspace_tempdir


class CameraProbeTests(unittest.TestCase):
    def test_timeout_preserves_last_stage(self) -> None:
        """原生打开阶段卡住时保留阶段证据，不能报告首帧成功。"""
        output = b'{"stage":"camera_open_started","monotonic_s":1}\n'
        with patch("app.camera_probe.subprocess.run") as run:
            run.side_effect = subprocess.TimeoutExpired(
                "synthetic", 15, output=output + b'{"stage":'
            )
            report = run_probe(0, 640, 480, 15)
        self.assertTrue(report["timed_out"])
        self.assertFalse(report["first_frame_received"])
        self.assertIsNone(report["worker_exit_code"])
        self.assertEqual(report["events"], [json.loads(output)])
        self.assertEqual(report["unparsed_stdout"], ['{"stage":'])

    def test_worker_stage_order_and_release(self) -> None:
        """模拟相机验证阶段顺序、实际尺寸与资源释放，日志不含画面。"""
        output = io.StringIO()
        with patch("cv2.VideoCapture") as capture, redirect_stdout(output):
            capture.return_value.isOpened.return_value = True
            capture.return_value.getBackendName.return_value = "synthetic"
            capture.return_value.read.return_value = (
                True,
                np.zeros((10, 20, 3), np.uint8),
            )
            self.assertEqual(probe_worker(0, 640, 480), 0)
        capture.return_value.release.assert_called_once()
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(
            [event["stage"] for event in events],
            [
                "worker_entered",
                "opencv_import_started",
                "opencv_import_completed",
                "camera_open_started",
                "camera_open_returned",
                "resolution_set_started",
                "resolution_set_completed",
                "first_read_started",
                "first_frame",
                "release_started",
                "release_completed",
            ],
        )
        self.assertEqual(events[8]["width"], 20)
        self.assertEqual(events[8]["height"], 10)

    def test_existing_report_blocks_device_start(self) -> None:
        """历史报告存在时，在启动诊断进程前拒绝覆盖。"""
        with workspace_tempdir() as directory:
            path = directory / "probe.json"
            path.write_text("existing", encoding="utf-8")
            with patch("app.camera_probe.run_probe") as run:
                with self.assertRaises(FileExistsError):
                    main(["--output", str(path)])
                run.assert_not_called()
