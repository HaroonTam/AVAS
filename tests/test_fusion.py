"""用合成网格验证融合与方向，不依赖模型权重。"""

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.fusion.config import FusionConfig, load_fusion_config
from app.fusion.depth_fusion import DetectionFrame, fuse_frame
from app.fusion.direction import estimate_direction
from app.main import main
from app.vision.depth_estimator import RelativeDepth
from app.vision.detector import Detection
from app.visualization import save_detections


class FusionTests(unittest.TestCase):
    def setUp(self) -> None:
        """构造中心深度为四、边缘为一的原图网格。"""
        self.values = np.ones((20, 30), np.float32)
        self.values[5:15, 10:20] = 4
        self.depth = RelativeDepth("frame", self.values, np.ones((20, 30), bool))
        self.frame = DetectionFrame(
            "frame", 30, 20, (Detection("chair", 0.9, (5, 0, 25, 20)),)
        )
        self.config = FusionConfig(camera_orientation="forward")

    def test_center_region_and_serialization(self) -> None:
        """默认只取内部区域，保留无量纲值与未知米制距离和风险。"""
        item = fuse_frame(self.frame, self.depth, self.config)[0]
        self.assertEqual(item.depth.roi, (10, 5, 20, 15))
        self.assertEqual(item.depth.relative_depth, 4)
        self.assertEqual(item.depth.retained_count, 100)
        self.assertEqual(item.depth.relative_depth_iqr, 0)
        self.assertEqual(item.direction, "front")
        self.assertIsNone(item.distance_m)
        self.assertEqual(item.risk_level, "unknown")
        self.assertEqual(item.depth.calibration_status, "not_calibrated")
        self.assertEqual(
            json.loads(json.dumps(asdict(item), allow_nan=False))["frame_id"], "frame"
        )

    def test_experiment_methods(self) -> None:
        """比较中心像素、全框均值与中位数，禁用裁尾以验证精确基线。"""
        for method, expected in (
            ("center_pixel", 4),
            ("box_mean", 1.75),
            ("box_median", 1),
        ):
            with self.subTest(method=method):
                item = fuse_frame(
                    self.frame,
                    self.depth,
                    replace(self.config, method=method, trim_fraction=0),
                )[0]
                self.assertEqual(item.depth.relative_depth, expected)

    def test_invalid_mask_nonfinite_negative_and_outliers(self) -> None:
        """独立过滤无效数值和掩膜，再排除孤立极端值；输入不被修改。"""
        self.values[5, 10:15] = [np.nan, np.inf, -1, 99999, 0]
        self.depth.valid_mask[6, 10] = False
        original = self.values.copy()
        item = fuse_frame(self.frame, self.depth, self.config)[0]
        self.assertEqual(item.depth.valid_count, 96)
        self.assertEqual(item.depth.retained_count, 94)
        self.assertEqual(item.depth.relative_depth, 4)
        np.testing.assert_array_equal(self.values, original)

    def test_insufficient_pixels_and_fraction(self) -> None:
        """总有效数、裁尾后数量或有效比例不足均明确未知。"""
        for config in (
            replace(self.config, min_valid_pixels=101),
            replace(self.config, min_valid_fraction=1),
        ):
            self.depth.valid_mask[5, 10] = False
            item = fuse_frame(self.frame, self.depth, config)[0]
            self.assertEqual(item.depth.status, "insufficient_valid_pixels")
            self.assertIsNone(item.depth.relative_depth)
        self.depth.valid_mask[:] = False
        self.assertEqual(
            fuse_frame(self.frame, self.depth, self.config)[0].depth.valid_count, 0
        )

    def test_post_trim_minimum(self) -> None:
        """裁尾后不足最小数量时不能输出统计值。"""
        self.values[5:15, 10:20] = np.arange(100, dtype=np.float32).reshape(10, 10)
        item = fuse_frame(
            self.frame, self.depth, replace(self.config, min_valid_pixels=95)
        )[0]
        self.assertEqual(item.depth.retained_count, 90)
        self.assertEqual(item.depth.status, "insufficient_valid_pixels")

    def test_quality_threshold_equality_and_mirrored_boundaries(self) -> None:
        """有效数量和比例恰等于阈值时可用，镜像后的分区边界仍归前方。"""
        self.depth.valid_mask[5:10, 10:20] = False
        config = replace(self.config, min_valid_pixels=50, min_valid_fraction=0.5)
        item = fuse_frame(self.frame, self.depth, config)[0]
        self.assertEqual(item.depth.status, "available")
        self.assertEqual(item.depth.valid_fraction, 0.5)
        for box in ((9, 0, 11, 2), (19, 0, 21, 2)):
            self.assertEqual(
                estimate_direction(box, 30, replace(config, mirrored=True)), "front"
            )

    def test_small_and_boundary_boxes(self) -> None:
        """边界框不越界，中心内缩后为空时返回空区域。"""
        frame = replace(
            self.frame, detections=(Detection("chair", 0.9, (29, 19, 30, 20)),)
        )
        self.assertEqual(
            fuse_frame(frame, self.depth, self.config)[0].depth.status, "empty_region"
        )
        item = fuse_frame(
            frame, self.depth, replace(self.config, method="center_pixel")
        )[0]
        self.assertEqual(item.depth.relative_depth, 1)
        with self.assertRaises(ValueError):
            replace(frame, width=29)

    def test_frame_and_shape_mismatch_rejected_even_without_objects(self) -> None:
        """错帧、转置、掩膜尺寸错误不能隐式融合，包括空检测。"""
        for depth in (
            replace(self.depth, frame_id="old-frame"),
            replace(self.depth, relative_depth=self.values.T),
            replace(self.depth, valid_mask=np.ones((1, 1), bool)),
            replace(self.depth, valid_mask=np.ones((20, 30), np.uint8)),
        ):
            with self.subTest(depth=depth.frame_id), self.assertRaises(ValueError):
                fuse_frame(replace(self.frame, detections=()), depth, self.config)

    def test_depth_missing_and_frame_local_ids(self) -> None:
        """深度缺失保留全部目标与方向，序号只属于本帧。"""
        frame = replace(self.frame, detections=self.frame.detections * 2)
        items = fuse_frame(frame, None, self.config)
        self.assertEqual([item.id for item in items], [1, 2])
        self.assertEqual(items[0].depth.status, "depth_unavailable")
        self.assertEqual(items[0].direction, "front")
        self.assertEqual(
            fuse_frame(replace(frame, detections=()), None, self.config), ()
        )

    def test_direction_boundaries_mirroring_and_unknown_mount(self) -> None:
        """边界归入前方，镜像交换左右，未知安装不猜测方向。"""
        for box, expected in (
            ((0, 0, 2, 2), "left"),
            ((9, 0, 11, 2), "front"),
            ((19, 0, 21, 2), "front"),
            ((28, 0, 30, 2), "right"),
        ):
            self.assertEqual(estimate_direction(box, 30, self.config), expected)
        self.assertEqual(
            estimate_direction((0, 0, 2, 2), 30, replace(self.config, mirrored=True)),
            "right",
        )
        self.assertIsNone(estimate_direction((0, 0, 2, 2), 30, FusionConfig()))
        with self.assertRaises(ValueError):
            estimate_direction((0, 0, 31, 2), 30, self.config)

    def test_configuration_and_old_config_defaults(self) -> None:
        """读取实验配置并兼容旧文件，非法参数必须失败。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.ini"
            path.write_text("[detection]\n", encoding="utf-8")
            self.assertEqual(load_fusion_config(path), FusionConfig())
            path.write_text(
                "[fusion]\nmethod=box_mean\nmirrored=true\ncamera_orientation=forward\n",
                encoding="utf-8",
            )
            self.assertTrue(load_fusion_config(path).mirrored)
            self.assertEqual(load_fusion_config(path).method, "box_mean")
        for field, value in (
            ("method", "mask_median"),
            ("inner_fraction", 0),
            ("min_valid_pixels", 0),
            ("min_valid_fraction", float("nan")),
            ("trim_fraction", 0.5),
            ("left_boundary", 0.9),
            ("right_boundary", float("inf")),
            ("camera_orientation", "backward"),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(self.config, **{field: value})

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is not installed")
    def test_cli_rejects_wrong_frame_and_preserves_detection(self) -> None:
        """错误帧深度经 CLI 降级，JSON 仍提供检测与未知距离。"""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.png"
            save_detections(np.zeros((20, 30, 3), np.uint8), (), source)
            output = io.StringIO()
            with (
                patch("app.main.Yolo11Detector") as detector,
                patch(
                    "app.vision.depth_estimator.DepthAnythingV2Estimator"
                ) as estimator,
                redirect_stdout(output),
                self.assertLogs(level="ERROR"),
            ):
                detector.return_value.detect.return_value = self.frame.detections
                detector.return_value.supported_labels = ("chair",)
                estimator.return_value.estimate.return_value = self.depth
                self.assertEqual(main(["--image", str(source), "--depth"]), 2)
            payload = json.loads(output.getvalue())
            self.assertFalse(payload["current_scene"])
            self.assertIsNone(payload["capture_timestamp_ms"])
            self.assertEqual(payload["fusion"]["status"], "depth_rejected")
            self.assertEqual(
                payload["safety"]["assessment"]["status"], "offline_not_current"
            )
            self.assertEqual(payload["safety"]["assessment"]["events"], [])
            self.assertEqual(
                payload["fusion"]["objects"][0]["depth"]["status"], "depth_unavailable"
            )
