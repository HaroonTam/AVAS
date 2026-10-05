"""基准计时、预热隔离与结果保护测试，不加载模型权重。"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from app.benchmark import (
    TimingSample,
    benchmark,
    main,
    measure_once,
    save_report,
    summarize,
    summarize_samples,
)
from app.config import DetectionConfig
from app.fusion.config import FusionConfig
from app.safety.config import RiskConfig
from app.vision.depth_estimator import RelativeDepth
from app.vision.detector import Image
from tests.test_live import FakeDetector


class FakeDepth:
    def __init__(self) -> None:
        """统计同一深度实例的调用次数。"""
        self.calls = 0

    def estimate(self, image: Image, frame_id: str) -> RelativeDepth:
        """返回同帧合成相对深度，不调用真实模型。"""
        self.calls += 1
        return RelativeDepth(
            frame_id,
            np.ones(image.shape[:2], np.float32),
            np.ones(image.shape[:2], bool),
        )


class BenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        """构造小型图像和合成检测后端。"""
        self.image = np.zeros((20, 20, 3), np.uint8)
        self.detector = FakeDetector()

    def test_exact_stage_timing_and_offline_assessment(self) -> None:
        """按固定时钟验证每段耗时，离线输入始终不进入实时告警路径。"""
        clock = iter((0.0, 0.001, 0.003, 0.006, 0.010))
        sample = measure_once(
            self.image,
            "synthetic",
            self.detector,
            FakeDepth(),
            FusionConfig(),
            RiskConfig(),
            clock.__next__,
        )
        self.assertAlmostEqual(sample.detection_ms, 1)
        self.assertAlmostEqual(sample.depth_ms, 2)
        self.assertAlmostEqual(sample.fusion_ms, 3)
        self.assertAlmostEqual(sample.offline_validation_ms, 4)
        self.assertAlmostEqual(sample.total_ms, 10)
        self.assertEqual(sample.object_count, 1)

    def test_warmup_excluded_and_instances_reused(self) -> None:
        """预热也运行完整处理但不进入正式样本，后端始终复用同一实例。"""
        depth = FakeDepth()
        samples = benchmark(
            self.image,
            "synthetic",
            self.detector,
            depth,
            FusionConfig(),
            RiskConfig(),
            warmup=2,
            iterations=3,
        )
        self.assertEqual(len(samples), 3)
        self.assertEqual(self.detector.calls, 5)
        self.assertEqual(depth.calls, 5)

    def test_disabled_depth_is_null_and_not_summarized(self) -> None:
        """深度禁用不产生零耗时或已完成深度推理的假象。"""
        samples = benchmark(
            self.image,
            "synthetic",
            self.detector,
            None,
            FusionConfig(),
            RiskConfig(),
            warmup=0,
            iterations=1,
        )
        self.assertIsNone(samples[0].depth_ms)
        self.assertNotIn("depth", summarize_samples(samples))

    def test_percentile_and_invalid_samples(self) -> None:
        """验证线性 p95 及非法统计样本拒绝。"""
        result = summarize((1, 2, 3, 4, 5))
        self.assertEqual(result.mean_ms, 3)
        self.assertEqual(result.median_ms, 3)
        self.assertAlmostEqual(result.p95_ms, 4.8)
        for values in ((), (-1.0,), (float("nan"),), (float("inf"),)):
            with self.assertRaises(ValueError):
                summarize(values)
        with self.assertRaises(ValueError):
            summarize_samples(
                (TimingSample(1, None, 1, 1, 3, 1), TimingSample(1, 1, 1, 1, 4, 1))
            )

    def test_failure_aborts_instead_of_mixing_degraded_samples(self) -> None:
        """深度故障直接中止基准，不把降级耗时算成完整流水线性能。"""
        depth = MagicMock()
        depth.estimate.side_effect = RuntimeError("synthetic failure")
        with self.assertRaises(RuntimeError):
            benchmark(
                self.image,
                "f",
                self.detector,
                depth,
                FusionConfig(),
                RiskConfig(),
                warmup=1,
                iterations=2,
            )
        self.assertEqual(self.detector.calls, 1)

    def test_iteration_bounds_rejected_before_processing(self) -> None:
        """拒绝空采样、布尔值和无界循环配置。"""
        for warmup, iterations in ((-1, 1), (101, 1), (0, 0), (0, 1001), (True, 1)):
            with self.assertRaises(ValueError):
                benchmark(
                    self.image,
                    "f",
                    self.detector,
                    None,
                    FusionConfig(),
                    RiskConfig(),
                    warmup=warmup,
                    iterations=iterations,
                )
        self.assertEqual(self.detector.calls, 0)

    def test_output_is_exclusive_and_bad_json_creates_no_file(self) -> None:
        """已有结果不覆盖；序列化失败不留下貌似完整的报告文件。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            save_report({"value": 1}, path)
            with self.assertRaises(FileExistsError):
                save_report({"value": 2}, path)
            self.assertEqual(json.loads(path.read_text())["value"], 1)
            invalid = Path(directory) / "invalid.json"
            with self.assertRaises(ValueError):
                save_report({"value": float("nan")}, invalid)
            self.assertFalse(invalid.exists())

    def test_existing_output_stops_before_loading_models(self) -> None:
        """CLI 预先拒绝旧路径，避免无意义模型初始化和覆盖旧实验。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text("original", encoding="utf-8")
            with (
                patch("app.benchmark.load_config") as config,
                self.assertLogs(level="ERROR"),
            ):
                self.assertEqual(
                    main(["--image", "unused.jpg", "--output", str(path)]), 1
                )
            config.assert_not_called()
            self.assertEqual(path.read_text(), "original")

    def test_cli_report_contains_protocol_hashes_and_sample_count(self) -> None:
        """模拟模型和解码器验证完整报告元数据，不读取真实权重或 GPU。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, weights, output = (
                root / "image.jpg",
                root / "synthetic.pt",
                root / "report.json",
            )
            image.write_bytes(b"synthetic image")
            weights.write_bytes(b"synthetic checkpoint")
            torch, cv2 = MagicMock(), MagicMock()
            torch.cuda.is_available.return_value = False
            torch.backends.mps.is_available.return_value = False
            torch.get_num_threads.return_value = 1
            cv2.imdecode.return_value = self.image
            with (
                patch.dict("sys.modules", {"torch": torch, "cv2": cv2}),
                patch(
                    "app.benchmark.load_config", return_value=DetectionConfig(weights)
                ),
                patch(
                    "app.benchmark.Yolo11Detector", return_value=self.detector
                ) as factory,
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    main(
                        [
                            "--image",
                            str(image),
                            "--output",
                            str(output),
                            "--warmup",
                            "1",
                            "--iterations",
                            "2",
                        ]
                    ),
                    0,
                )
            payload = json.loads(output.read_text())
            factory.assert_called_once()
            self.assertEqual(payload["protocol"], "offline_repeated_image_v1")
            self.assertFalse(payload["current_scene"])
            self.assertEqual(len(payload["samples"]), 2)
            self.assertEqual(payload["selected_devices"]["detection"], "cpu")
            self.assertEqual(len(payload["weight_sha256"]["yolo11"]), 64)
            self.assertIn("warning_latency", payload["unmeasured"])
