"""阶段计时和诊断隔离测试；不加载权重、不访问加速器或外部设备。"""

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from app.benchmark import (
    TimingSample,
    benchmark,
    main,
    sample_payload,
    summarize_depth_stages,
)
from app.benchmark_summary import load_run
from app.config import DetectionConfig
from app.fusion.config import FusionConfig
from app.safety.config import RiskConfig
from app.vision.depth_estimator import (
    DepthAnythingV2Estimator,
    DepthConfig,
    RelativeDepth,
    TransformersDepthBackend,
)
from app.vision.depth_profile import STAGES, DepthProfiler, synchronize_device
from app.vision.detector import Image
from tests.test_live import FakeDetector


class ProfiledFakeEstimator:
    def __init__(self, config: DepthConfig, *, profiler: DepthProfiler | None) -> None:
        """接收基准入口创建的诊断器，构造不依赖模型的完整阶段序列。"""
        self.profiler = profiler
        self.calls = 0

    def estimate(self, image: Image, frame_id: str) -> RelativeDepth:
        """按真实阶段顺序记录合成调用，返回同帧非米制深度。"""
        self.calls += 1
        if self.profiler is not None:
            self.profiler.begin()
            for stage in STAGES:
                self.profiler.mark(stage)
        return RelativeDepth(
            frame_id,
            np.ones(image.shape[:2], np.float32),
            np.ones(image.shape[:2], bool),
        )


class ProfileTests(unittest.TestCase):
    def test_exact_timings_and_synchronization_boundaries(self) -> None:
        """固定时钟验证七阶段与总时间；前置同步排除在阶段总计之外。"""
        ticks = iter((0.0, 0.001, 0.003, 0.006, 0.010, 0.015, 0.021, 0.028))
        sync = MagicMock()
        profiler = DepthProfiler("cuda", clock=ticks.__next__, synchronize=sync)
        profiler.begin()
        for stage in STAGES:
            profiler.mark(
                stage,
                synchronize=stage in {"to_device", "inference", "resize", "to_cpu"},
            )
        self.assertEqual(sync.call_count, 5)
        self.assertEqual(sync.call_args.args, ("cuda",))
        self.assertIsNotNone(profiler.last_sample)
        if profiler.last_sample is None:
            self.fail("missing sample")
        for stage, duration in zip(STAGES, range(1, 8)):
            self.assertAlmostEqual(profiler.last_sample[stage], duration)
        self.assertAlmostEqual(profiler.last_sample["total"], 28)

    def test_failed_or_incomplete_sample_cannot_reuse_previous(self) -> None:
        """新调用、错误顺序、时钟倒退与同步失败不得沿用上次完整结果。"""
        sync = MagicMock()
        profiler = DepthProfiler("cpu", synchronize=sync)
        profiler.begin()
        for stage in STAGES:
            profiler.mark(stage)
        self.assertIsNotNone(profiler.last_sample)
        profiler.begin()
        with self.assertRaises(ValueError):
            profiler.mark("inference")
        self.assertIsNone(profiler.last_sample)
        sync.side_effect = RuntimeError("sync failure")
        with self.assertRaises(RuntimeError):
            profiler.begin()
        self.assertIsNone(profiler.last_sample)
        for clocks in ((2.0, 1.0), (0.0, float("nan"))):
            ticks = iter(clocks)
            profiler = DepthProfiler(
                "cpu", clock=ticks.__next__, synchronize=MagicMock()
            )
            profiler.begin()
            with self.assertRaises(ValueError):
                profiler.mark("input_validation")
            self.assertIsNone(profiler.last_sample)

    def test_device_dispatch_uses_only_selected_backend(self) -> None:
        """模拟 CUDA/MPS 同步分派；CPU 不调用任一加速器。"""
        torch = MagicMock()
        with patch.dict("sys.modules", {"torch": torch}):
            synchronize_device("cpu")
            torch.cuda.synchronize.assert_not_called()
            torch.mps.synchronize.assert_not_called()
            synchronize_device("cuda")
            synchronize_device("mps")
            torch.cuda.synchronize.assert_called_once()
            torch.mps.synchronize.assert_called_once()
        with self.assertRaises(ValueError):
            DepthProfiler("auto")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
    def test_real_backend_cpu_operations_preserve_depth_and_default_has_no_sync(
        self,
    ) -> None:
        """用合成 CPU 张量执行真实后端方法，验证插值、掩膜及默认同步隔离。"""
        import torch

        backend = object.__new__(TransformersDepthBackend)
        backend._device = "cpu"
        backend._size = 518
        backend._profiler = None
        backend._processor = MagicMock()
        backend._processor.return_value.to.return_value = {
            "pixel_values": torch.ones((1, 3, 2, 3))
        }
        backend._model = MagicMock()
        backend._model.return_value.predicted_depth = torch.arange(
            6, dtype=torch.float32
        ).reshape(1, 2, 3)
        image = np.zeros((4, 6, 3), np.uint8)
        config = DepthConfig(Path("unused"), device="cpu")
        with patch(
            "app.vision.depth_profile.DepthProfiler.begin",
            side_effect=AssertionError("unexpected sync"),
        ):
            original = DepthAnythingV2Estimator(config, backend).estimate(image, "f")
        sync = MagicMock()
        profiler = DepthProfiler("cpu", synchronize=sync)
        backend._profiler = profiler
        actual = DepthAnythingV2Estimator(config, backend, profiler=profiler).estimate(
            image, "f"
        )
        np.testing.assert_array_equal(original.relative_depth, actual.relative_depth)
        np.testing.assert_array_equal(original.valid_mask, actual.valid_mask)
        self.assertEqual(actual.frame_id, original.frame_id)
        self.assertEqual(sync.call_count, 5)
        self.assertEqual(set(profiler.last_sample or {}), {*STAGES, "total"})
        backend._model.return_value.predicted_depth = torch.ones((2, 2, 3))
        with self.assertRaises(RuntimeError):
            DepthAnythingV2Estimator(config, backend, profiler=profiler).estimate(
                image, "f"
            )
        self.assertIsNone(profiler.last_sample)

    def test_benchmark_excludes_profile_warmup_and_rejects_missing_stages(self) -> None:
        """正式样本保持诊断归属；预热不进入统计，部分诊断与未启用深度拒绝。"""
        profiler = DepthProfiler("cpu", synchronize=MagicMock())
        depth = ProfiledFakeEstimator(DepthConfig(Path("unused")), profiler=profiler)
        samples = benchmark(
            np.zeros((20, 20, 3), np.uint8),
            "f",
            FakeDetector(),
            depth,
            FusionConfig(),
            RiskConfig(),
            warmup=2,
            iterations=3,
            profiler=profiler,
        )
        self.assertEqual(depth.calls, 5)
        self.assertEqual(len(samples), 3)
        self.assertEqual(set(summarize_depth_stages(samples)), {*STAGES, "total"})
        self.assertIsNot(samples[0].depth_stages_ms, samples[1].depth_stages_ms)
        with self.assertRaises(ValueError):
            summarize_depth_stages((replace(samples[0], depth_stages_ms={}),))
        with self.assertRaises(RuntimeError):
            benchmark(
                np.zeros((20, 20, 3), np.uint8),
                "f",
                FakeDetector(),
                None,
                FusionConfig(),
                RiskConfig(),
                warmup=0,
                iterations=1,
                profiler=profiler,
            )
        self.assertNotIn(
            "depth_stages_ms", sample_payload(TimingSample(1, None, 0, 0, 1, 0))
        )

    def test_cli_protocol_and_incompatible_summary(self) -> None:
        """验证单图和多图诊断协议、元数据、默认兼容性及旧汇总拒绝混用。"""
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--image", "unused", "--output", "unused.json", "--profile-depth"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "yolo.pt"
            weights.write_bytes(b"synthetic")
            for name in (
                "model.safetensors",
                "config.json",
                "preprocessor_config.json",
            ):
                (root / name).write_bytes(b"synthetic")
            images = [root / "first.jpg", root / "second.jpg"]
            for image in images:
                image.write_bytes(b"synthetic")
            torch, cv2 = MagicMock(), MagicMock()
            torch.cuda.is_available.return_value = False
            torch.backends.mps.is_available.return_value = False
            torch.get_num_threads.return_value = 1
            cv2.imdecode.return_value = np.zeros((20, 20, 3), np.uint8)
            for image_count in (1, 2):
                output = root / f"report{image_count}.json"
                args = [
                    "--depth",
                    "--profile-depth",
                    "--warmup",
                    "1",
                    "--iterations",
                    "2",
                    "--output",
                    str(output),
                ]
                for image in images[:image_count]:
                    args.extend(["--image", str(image)])
                with (
                    patch.dict("sys.modules", {"torch": torch, "cv2": cv2}),
                    patch(
                        "app.benchmark.load_config",
                        return_value=DetectionConfig(weights),
                    ),
                    patch(
                        "app.benchmark.load_depth_config",
                        return_value=DepthConfig(root),
                    ),
                    patch("app.benchmark.Yolo11Detector", return_value=FakeDetector()),
                    patch(
                        "app.benchmark.DepthAnythingV2Estimator",
                        side_effect=ProfiledFakeEstimator,
                    ),
                    redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(main(args), 0)
                report = json.loads(output.read_text())
                self.assertEqual(report["protocol"], "offline_depth_stage_profile_v1")
                self.assertEqual(report["schema_version"], 3)
                self.assertEqual(report["sample_count"], image_count * 2)
                self.assertEqual(len(report["images"]), image_count)
                self.assertTrue(report["profiling"]["perturbs_execution"])
                self.assertEqual(len(report["images"][0]["samples"]), 2)
                self.assertIn("depth_stages_ms", report["images"][0]["samples"][0])
                with self.assertRaises(ValueError):
                    load_run(output)
