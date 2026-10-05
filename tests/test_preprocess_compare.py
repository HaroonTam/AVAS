"""预处理对照的统计、掩膜与报告保护测试，不运行真实模型。"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from app.preprocess_compare import Pixels, Variant, compare_values, main, paired_timings
from app.vision.depth_estimator import DepthConfig


class ComparisonTests(unittest.TestCase):
    def test_value_difference_masks_and_unknown(self) -> None:
        """区分形状差异、掩膜差异、有效区域偏差和无共同有效值。"""
        a = np.array([1, 2, np.nan], np.float32)
        b = np.array([2, 2, np.nan], np.float32)
        result = compare_values(a, b)
        self.assertEqual(result["common_valid_count"], 2)
        self.assertEqual(result["mean_abs_difference"], 0.5)
        self.assertEqual(result["max_abs_difference"], 1)
        self.assertEqual(result["mask_mismatch_count"], 0)
        self.assertFalse(result["exact_equal"])
        self.assertTrue(compare_values(a, a)["exact_equal"])
        self.assertEqual(
            compare_values(a, np.ones(2, np.float32))["status"], "shape_mismatch"
        )
        mismatch = compare_values(a, np.array([1, np.nan, 3], np.float32))
        self.assertEqual(mismatch["mask_mismatch_count"], 2)
        self.assertEqual(mismatch["mean_abs_difference"], 0)
        self.assertFalse(mismatch["exact_equal"])
        invalid = compare_values(
            np.array([np.nan], np.float32), np.array([np.inf], np.float32)
        )
        self.assertEqual(invalid["status"], "no_common_valid_values")
        self.assertIsNone(invalid["mean_abs_difference"])
        self.assertFalse(invalid["exact_equal"])

    def test_alternating_pairs_exclude_warmup_and_never_cache(self) -> None:
        """两种实现每轮都重新执行，正式采样顺序独立于预热奇偶性。"""
        calls: list[Variant] = []

        def run(variant: Variant) -> Pixels:
            """记录调用顺序并返回合成张量。"""
            calls.append(variant)
            return np.zeros((1, 3, 2, 2), np.float32)

        clock = iter(tuple(float(i) for i in range(16)))
        samples = paired_timings(run, warmup=1, iterations=3, clock=clock.__next__)
        self.assertEqual(
            calls,
            [
                "pil",
                "torchvision",
                "pil",
                "torchvision",
                "torchvision",
                "pil",
                "pil",
                "torchvision",
            ],
        )
        self.assertEqual(len(samples), 3)
        self.assertEqual(
            [list(item) for item in samples],
            [["pil", "torchvision"], ["torchvision", "pil"], ["pil", "torchvision"]],
        )
        self.assertTrue(
            all(sample == {"pil": 1000, "torchvision": 1000} for sample in samples)
        )

    def test_bounds_clock_and_processing_failure(self) -> None:
        """非法采样数、时钟倒退或处理器故障中止实验，不计入部分成功值。"""
        run = MagicMock()
        for warmup, iterations in ((-1, 1), (101, 1), (True, 1), (0, 0), (0, 1001)):
            with self.assertRaises(ValueError):
                paired_timings(run, warmup=warmup, iterations=iterations)
        run.assert_not_called()
        for values in ((1.0, 0.0), (0.0, float("nan"))):
            clock = iter(values)
            with self.assertRaises(ValueError):
                paired_timings(run, warmup=0, iterations=1, clock=clock.__next__)
        run.side_effect = RuntimeError("processor failed")
        with self.assertRaises(RuntimeError):
            paired_timings(run, warmup=0, iterations=1)

    def test_cli_metadata_failure_and_exclusive_output(self) -> None:
        """模拟后端验证报告身份与原图不持久化，旧输出和失败结果受保护。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, output = root / "input.jpg", root / "report.json"
            image.write_bytes(b"synthetic")
            for name in (
                "config.json",
                "preprocessor_config.json",
                "model.safetensors",
            ):
                (root / name).write_bytes(b"synthetic")
            torch, cv2 = MagicMock(), MagicMock()
            torch.cuda.is_available.return_value = False
            torch.backends.mps.is_available.return_value = False
            torch.get_num_threads.return_value = 1
            cv2.imdecode.return_value = np.zeros((20, 30, 3), np.uint8)
            args = [
                "--image",
                str(image),
                "--output",
                str(output),
                "--warmup",
                "1",
                "--iterations",
                "2",
            ]
            with (
                patch.dict("sys.modules", {"torch": torch, "cv2": cv2}),
                patch(
                    "app.preprocess_compare.load_depth_config",
                    return_value=DepthConfig(root),
                ),
                patch("app.preprocess_compare.ComparisonBackend") as factory,
                patch(
                    "app.preprocess_compare.compare_image",
                    return_value={"synthetic": True},
                ) as compare,
                patch("app.preprocess_compare.version", return_value="synthetic"),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(main(args), 0)
                factory.assert_called_once()
                compare.assert_called_once()
                self.assertEqual(
                    compare.call_args.kwargs, {"warmup": 1, "iterations": 2}
                )
                payload = json.loads(output.read_text())
                self.assertEqual(
                    payload["protocol"], "offline_preprocessor_comparison_v1"
                )
                self.assertEqual(payload["preprocessing_device"], "cpu")
                self.assertEqual(payload["default_processor_unchanged"], "pil")
                self.assertFalse(payload["current_scene"])
                self.assertEqual(len(payload["images"][0]["image_file_sha256"]), 64)
                self.assertNotIn("pixels", payload["images"][0])
                original = output.read_bytes()
                with self.assertLogs(level="ERROR"):
                    self.assertEqual(main(args), 1)
                self.assertEqual(output.read_bytes(), original)
                factory.assert_called_once()
                bad_output = root / "bad.json"
                compare.side_effect = RuntimeError("synthetic failure")
                with self.assertLogs(level="ERROR"):
                    self.assertEqual(
                        main(["--image", str(image), "--output", str(bad_output)]), 1
                    )
                self.assertFalse(bad_output.exists())
