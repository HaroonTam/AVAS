"""预采集工具的合成测试，不打开相机，不验证真实测距精度。"""

import json
import unittest
from pathlib import Path
from time import monotonic
from unittest.mock import MagicMock, patch

import numpy as np

from app.camera.camera_service import CameraConfig, CapturedFrame
from app.e2_capture import Measurement, capture_frame, main, save_sample, validate_roi
from tests.workspace_temp import workspace_tempdir


class E2CaptureTests(unittest.TestCase):
    def test_invalid_measurements(self) -> None:
        """拒绝非法真值与不确定度，允许明确未知精度。"""
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                Measurement(value, None, "g", "tape", "indoor", "i")
        with self.assertRaises(ValueError):
            Measurement(1.0, -0.1, "g", "tape", "indoor", "i")
        self.assertIsNone(Measurement(1.0, None, "g", "t", "c", "i").uncertainty_m)

    def test_invalid_roi(self) -> None:
        """取消选框、反向框和越界框不得保存。"""
        for roi in ((0, 0, 0, 0), (-1, 0, 2, 2), (0, 0, 11, 2), (3, 0, 2, 2)):
            with self.assertRaises(ValueError):
                validate_roi(roi, 10, 10)

    def test_camera_failure_closes(self) -> None:
        """相机超时仍关闭服务，错误不得伪装为有效样本。"""
        with patch("app.e2_capture.CameraService") as factory:
            camera = factory.return_value
            camera.read.side_effect = TimeoutError("synthetic timeout")
            with self.assertRaises(TimeoutError):
                capture_frame(CameraConfig())
            camera.close.assert_called_once()

    def test_fresh_frame_after_first_packet(self) -> None:
        """丢弃首包和陈旧帧，只返回近期帧并释放服务。"""
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        old = CapturedFrame("old", 1, monotonic() - 10, image)
        fresh = CapturedFrame("fresh", 2, monotonic(), image)
        with patch("app.e2_capture.CameraService") as factory:
            factory.return_value.read.side_effect = [old, old, fresh]
            self.assertIs(capture_frame(CameraConfig()), fresh)
            factory.return_value.close.assert_called_once()

    def test_saved_record_and_no_overwrite(self) -> None:
        """合成编码验证溯源、未知精度、非米制证据与已有样本保护。"""
        frame = CapturedFrame("test", 1, monotonic(), np.zeros((8, 8, 3), np.uint8))
        codec = MagicMock()
        codec.__version__ = "synthetic"
        codec.imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))
        with workspace_tempdir() as temp, patch.dict("sys.modules", {"cv2": codec}):
            directory = Path(temp) / "sample"
            arguments = (
                directory,
                frame,
                (1, 1, 7, 7),
                Measurement(1.0, None, "group", "tape", "indoor", "install"),
                CameraConfig(),
                Path("configs/system.ini"),
            )
            path = save_sample(*arguments)
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsNone(record["metric_evidence"])
            self.assertEqual(record["split"], "development_precapture")
            self.assertEqual(record["actual_width"], 8)
            self.assertEqual(len(record["image_sha256"]), 64)
            with self.assertRaises(FileExistsError):
                save_sample(*arguments)

    def test_invalid_cli_never_opens_camera(self) -> None:
        """无效真值在打开设备前失败。"""
        with patch("app.e2_capture.capture_frame") as capture:
            result = main(
                [
                    "--save-local",
                    "--distance-m",
                    "nan",
                    "--group-id",
                    "g",
                    "--instrument",
                    "tape",
                    "--conditions",
                    "indoor",
                    "--installation-id",
                    "i",
                ]
            )
            self.assertEqual(result, 2)
            capture.assert_not_called()
