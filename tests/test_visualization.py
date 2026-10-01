"""绘图和显式保存行为的合成测试，不验证模型精度。"""

import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.main import main
from app.vision.detector import Detection
from app.visualization import draw_detections, save_detections


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is not installed")
class VisualizationTests(unittest.TestCase):
    def test_draw_preserves_input_and_marks_edge_box(self) -> None:
        """绘图不能修改感知输入，右下边界目标应正确显示。"""
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        result = draw_detections(image, (Detection("car", 0.8, (90, 70, 120, 100)),))
        self.assertFalse(image.any())
        self.assertTrue(result[99, 119].any())
        self.assertEqual(result.shape, image.shape)

    def test_empty_detections_preserve_pixels(self) -> None:
        """无检测结果时保留原图，不画安全通行之类的结论。"""
        image = np.full((80, 100, 3), 100, dtype=np.uint8)
        np.testing.assert_array_equal(draw_detections(image, ()), image)

    def test_out_of_bounds_is_rejected(self) -> None:
        """拒绝来自不同尺寸坐标空间的越界检测框。"""
        with self.assertRaises(ValueError):
            draw_detections(
                np.zeros((80, 100, 3), dtype=np.uint8),
                (Detection("car", 0.8, (90, 70, 101, 80)),),
            )

    def test_unicode_output_and_overwrite_protection(self) -> None:
        """中文路径可保存有效图片，重复保存不会覆盖既有实验结果。"""
        import cv2

        image = np.zeros((80, 100, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "检测结果" / "图像.png"
            save_detections(image, (), target)
            original_bytes = target.read_bytes()
            decoded = cv2.imdecode(np.frombuffer(original_bytes, np.uint8), 1)
            np.testing.assert_array_equal(decoded, image)
            with self.assertRaises(FileExistsError):
                save_detections(image, (), target)
            self.assertEqual(target.read_bytes(), original_bytes)

    def test_unsupported_extension_writes_nothing(self) -> None:
        """不支持的格式应明确失败，不生成误导性文件。"""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "result.txt"
            with self.assertRaises(ValueError):
                save_detections(np.zeros((20, 20, 3), dtype=np.uint8), (), target)
            self.assertFalse(target.exists())

    def test_cli_saves_only_when_requested(self) -> None:
        """默认运行不保存图像，只有显式参数才触发写入。"""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            save_detections(np.zeros((80, 100, 3), dtype=np.uint8), (), source)
            target = Path(directory) / "result.png"
            with (
                patch("app.main.Yolo11Detector") as detector,
                patch("app.visualization.save_detections") as save,
                redirect_stdout(io.StringIO()),
            ):
                detector.return_value.detect.return_value = ()
                detector.return_value.supported_labels = ()
                self.assertEqual(main(["--image", str(source)]), 0)
                save.assert_not_called()
                self.assertEqual(
                    main(
                        [
                            "--image",
                            str(source),
                            "--output-image",
                            str(target),
                        ]
                    ),
                    0,
                )
                save.assert_called_once()

    def test_cli_output_failure_returns_error(self) -> None:
        """拒绝覆盖输入图片，并将保存失败作为命令失败报告。"""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            save_detections(np.zeros((80, 100, 3), dtype=np.uint8), (), source)
            original = source.read_bytes()
            with (
                patch("app.main.Yolo11Detector") as detector,
                self.assertLogs(level="ERROR"),
            ):
                detector.return_value.detect.return_value = ()
                self.assertEqual(
                    main(
                        [
                            "--image",
                            str(source),
                            "--output-image",
                            str(source),
                        ]
                    ),
                    1,
                )
            self.assertEqual(source.read_bytes(), original)
