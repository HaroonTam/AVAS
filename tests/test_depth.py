"""深度接口的合成测试，不依赖权重、网络或 GPU。"""

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.main import main
from app.vision.depth_estimator import (
    DepthAnythingV2Estimator,
    DepthConfig,
    DepthMap,
    load_depth_config,
    save_relative_depth,
)
from app.vision.detector import Detection, Image
from app.visualization import render_relative_depth, save_detections


class FakeDepthBackend:
    def __init__(self, values: DepthMap) -> None:
        """保存可控深度输出，记录是否复用了后端。"""
        self.values = values
        self.calls = 0

    def predict(self, image: Image) -> DepthMap:
        """返回合成原始输出以覆盖模型边界条件。"""
        self.calls += 1
        return self.values


class DepthTests(unittest.TestCase):
    def test_invalid_values_masked_without_mutating_backend(self) -> None:
        """有限非负值有效；负值和非有限值显式无效，禁止篡改后端数组。"""
        values = np.array([[0, 2, np.nan], [np.inf, -1, 4]], dtype=np.float32)
        backend = FakeDepthBackend(values)
        estimator = DepthAnythingV2Estimator(DepthConfig(Path("unused")), backend)
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        result = estimator.estimate(image, "frame-1")
        estimator.estimate(image, "frame-2")
        self.assertEqual(backend.calls, 2)
        self.assertEqual(result.frame_id, "frame-1")
        self.assertEqual(int(result.valid_mask.sum()), 3)
        self.assertTrue(np.isnan(result.relative_depth[1, 1]))
        self.assertEqual(values[1, 1], -1)
        self.assertFalse(result.relative_depth.flags.writeable)

    def test_wrong_size_and_all_invalid_rejected(self) -> None:
        """错位深度和完全失效深度不能成为有效场景信息。"""
        for values in (
            np.ones((3, 2), dtype=np.float32),
            np.full((2, 3), np.nan, dtype=np.float32),
        ):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                estimator = DepthAnythingV2Estimator(
                    DepthConfig(Path("unused")), FakeDepthBackend(values)
                )
                estimator.estimate(np.zeros((2, 3, 3), np.uint8), "frame")

    def test_invalid_image_and_frame_rejected(self) -> None:
        """在调用模型前拒绝空帧标识和非三通道图像。"""
        backend = FakeDepthBackend(np.ones((2, 3), np.float32))
        estimator = DepthAnythingV2Estimator(DepthConfig(Path("unused")), backend)
        with self.assertRaises(ValueError):
            estimator.estimate(np.zeros((2, 3, 3), np.uint8), "")
        with self.assertRaises(ValueError):
            estimator.estimate(np.zeros((2, 3), np.uint8), "frame")
        self.assertEqual(backend.calls, 0)

    def test_missing_weights_and_corrupt_weights_fail_locally(self) -> None:
        """缺少文件或校验失败时不尝试联网下载其他模型。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                DepthAnythingV2Estimator(DepthConfig(root))
            for name in (
                "config.json",
                "preprocessor_config.json",
                "model.safetensors",
            ):
                (root / name).write_bytes(b"invalid")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                DepthAnythingV2Estimator(DepthConfig(root))

    def test_depth_config_paths_and_validation(self) -> None:
        """模型路径以配置文件定位，非法输入尺寸必须拒绝。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "system.ini"
            path.write_text(
                "[depth]\nmodel_dir=model\nimage_size=518\ndevice=cpu\n",
                encoding="utf-8",
            )
            config = load_depth_config(path)
            self.assertEqual(config.model_dir, Path(directory) / "model")
        for size in (0, -14, 519):
            with self.subTest(size=size), self.assertRaises(ValueError):
                DepthConfig(Path("unused"), image_size=size)

    def test_preview_near_bright_invalid_magenta(self) -> None:
        """预览只表达图内相对顺序，不把无效深度渲染成正常远景。"""
        estimator = DepthAnythingV2Estimator(
            DepthConfig(Path("unused")),
            FakeDepthBackend(np.array([[0, 4, np.nan]], np.float32)),
        )
        result = estimator.estimate(np.zeros((1, 3, 3), np.uint8), "frame")
        preview = render_relative_depth(result)
        np.testing.assert_array_equal(preview[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(preview[0, 1], [255, 255, 255])
        np.testing.assert_array_equal(preview[0, 2], [255, 0, 255])

    def test_constant_map_and_raw_archive(self) -> None:
        """常量图不发生除零，原始数组和掩膜无损保存且不能覆盖。"""
        estimator = DepthAnythingV2Estimator(
            DepthConfig(Path("unused")), FakeDepthBackend(np.ones((2, 3), np.float32))
        )
        result = estimator.estimate(np.zeros((2, 3, 3), np.uint8), "frame")
        self.assertTrue((render_relative_depth(result) == 127).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "深度.npz"
            save_relative_depth(result, path)
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(
                    data["relative_depth"], result.relative_depth
                )
                np.testing.assert_array_equal(data["valid_mask"], result.valid_mask)
                self.assertEqual(str(data["units"]), "relative_inverse_depth")
            with self.assertRaises(FileExistsError):
                save_relative_depth(result, path)

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is not installed")
    def test_depth_failure_preserves_detections(self) -> None:
        """深度初始化失败时返回降级状态和非零退出码，但保留检测事实。"""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            save_detections(np.zeros((10, 10, 3), np.uint8), (), source)
            output = io.StringIO()
            with (
                patch("app.main.Yolo11Detector") as detector,
                patch(
                    "app.vision.depth_estimator.DepthAnythingV2Estimator",
                    side_effect=RuntimeError("depth failed"),
                ),
                redirect_stdout(output),
                self.assertLogs(level="ERROR"),
            ):
                detector.return_value.detect.return_value = (
                    Detection("person", 0.9, (1, 1, 5, 5)),
                )
                detector.return_value.supported_labels = ("person",)
                self.assertEqual(main(["--image", str(source), "--depth"]), 2)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["detections"][0]["label"], "person")
            self.assertEqual(payload["depth"]["status"], "unavailable")
            self.assertEqual(payload["risk_level"], "unknown")
            self.assertEqual(payload["distance_status"], "metric_unavailable")
