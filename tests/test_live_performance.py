"""实时性能测量的合成回归测试，不访问真实设备或权重。"""

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from time import monotonic, time_ns
from unittest.mock import MagicMock, patch

import numpy as np

from app.camera.camera_service import CapturedFrame
from app.live import main as live_main
from app.live_performance import PerformanceRecorder, read_jsonl, summarize_rows
from tests.test_benchmark import FakeDepth
from tests.test_live import FakeDetector
from tests.workspace_temp import workspace_tempdir


def row(index: int, end: float, *, current: bool = True) -> dict[str, object]:
    """构造具有已知时间边界的合成行；参数时间单位为秒。"""
    return {
        "performance": {
            "result_index": index,
            "warmup": index == 1,
            "loop_started_monotonic_s": 10.0,
            "captured_monotonic_s": end - 0.1,
            "dispatch_completed_monotonic_s": end,
            "width": 640,
            "height": 480,
        },
        "current_scene": current,
        "depth_status": "available",
        "latencies_ms": dict.fromkeys(
            ("detection", "depth", "fusion", "risk", "capture_to_dispatch"),
            float(index * 10),
        ),
    }


class LivePerformanceTests(unittest.TestCase):
    def test_cuda_peak_reset_and_cpu_unknown(self) -> None:
        """合成后端验证先重置后读峰值且不做同步；CPU 不伪造显存零值。"""
        for cuda in (True, False):
            with self.subTest(cuda=cuda), workspace_tempdir() as directory:
                torch = MagicMock()
                torch.__version__ = "synthetic"
                torch.cuda.is_available.return_value = cuda
                torch.backends.mps.is_available.return_value = False
                torch.cuda.get_device_name.return_value = "synthetic"
                torch.cuda.max_memory_allocated.return_value = 1234
                path = Path(directory) / "resources.json"
                recorder = PerformanceRecorder(path, Path("configs/system.ini"))
                with (
                    patch.dict("sys.modules", {"torch": torch}),
                    patch("app.live_performance.file_hash", return_value="synthetic"),
                ):
                    recorder.start()
                    if cuda:
                        torch.cuda.reset_peak_memory_stats.assert_called_once()
                    else:
                        torch.cuda.reset_peak_memory_stats.assert_not_called()
                    torch.cuda.max_memory_allocated.assert_not_called()
                    recorder.finish(False)
                torch.cuda.synchronize.assert_not_called()
                report = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(
                    report["cuda_peak_allocated_bytes"], 1234 if cuda else None
                )

    def test_warmup_stale_fault_and_fps_denominator(self) -> None:
        """剔除预热与陈旧耗时，但不从 FPS 分母扣除陈旧帧消耗的墙钟时间。"""
        rows = [row(1, 11), row(2, 12), row(3, 15, current=False), row(4, 16)]
        rows.append({"scene": {"valid": False}})
        report = summarize_rows(rows)
        self.assertEqual(report["valid_processing_fps"], 2 / 5)
        self.assertEqual(report["valid_post_warmup_count"], 2)
        self.assertEqual(report["stale_count_including_warmup"], 1)
        self.assertEqual(report["terminal_fault_count"], 1)
        self.assertEqual(
            report["summary_ms"],
            dict.fromkeys(
                ("detection", "depth", "fusion", "risk", "capture_to_dispatch"),
                {"count": 2, "mean": 30.0, "p50": 30.0, "p95": 39.0, "max": 40.0},
            ),
        )

    def test_depth_failure_excluded_and_empty_unknown(self) -> None:
        """深度失败不进入完整流水线样本池；没有测量时不得填充零延迟。"""
        failed = row(2, 12)
        failed["depth_status"] = "unavailable"
        report = summarize_rows([row(1, 11), failed])
        self.assertEqual(report["depth_degraded_count_including_warmup"], 1)
        self.assertEqual(report["valid_processing_fps"], 0)
        self.assertTrue(all(value is None for value in report["summary_ms"].values()))
        self.assertIsNone(summarize_rows([])["valid_processing_fps"])

    def test_corrupt_samples_are_rejected(self) -> None:
        """缺行、乱序、非有限耗时和缺失元数据均拒绝，不能静默形成好看的报告。"""
        invalid = row(2, 12)
        invalid["latencies_ms"] = {"detection": float("nan")}
        for rows in (
            [row(2, 12)],
            [row(1, 11), row(2, 10)],
            [row(1, 11), invalid],
            [{}],
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                summarize_rows(rows)

    def test_utf16_and_utf8_input_and_no_overwrite(self) -> None:
        """两种 PowerShell 编码均能读取；报告预留不得覆盖历史结果。"""
        with workspace_tempdir() as directory:
            path = Path(directory) / "frames.jsonl"
            for encoding in ("utf-16", "utf-8-sig"):
                path.write_text(json.dumps(row(1, 11)) + "\n", encoding=encoding)
                self.assertEqual(read_jsonl(path), [row(1, 11)])
            with self.assertRaises(FileExistsError):
                PerformanceRecorder(path, Path("configs/system.ini"))

    def test_resource_failure_is_unknown_not_zero(self) -> None:
        """显存采样失败不抛入告警控制流，失败资源为未知而非零。"""
        with workspace_tempdir() as directory:
            path = Path(directory) / "resources.json"
            recorder = PerformanceRecorder(path, Path("configs/system.ini"))
            with (
                patch("torch.cuda.is_available", side_effect=RuntimeError("synthetic")),
                self.assertLogs(level="ERROR"),
            ):
                recorder.start()
            recorder.finish(False)
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(report["resource_status"], "unavailable")
            self.assertIsNone(report["cuda_peak_allocated_bytes"])

    def test_measurement_follows_warning_dispatch(self) -> None:
        """启用测量后仍先派发告警；输出保留所有阶段且峰值只在初始化与收尾处理。"""
        frame = CapturedFrame(
            "synthetic:1",
            time_ns() // 1_000_000,
            monotonic(),
            np.zeros((20, 20, 3), np.uint8),
        )
        output = io.StringIO()
        order: list[str] = []

        def dispatch(*args: object, **kwargs: object) -> int:
            """记录合成告警提交顺序，不播放语音。"""
            order.append("dispatch")
            return 1

        def finish(failed: bool) -> None:
            """收尾时确认逐帧输出已经产生且此前已经提交告警。"""
            self.assertFalse(failed)
            self.assertEqual(order, ["dispatch"])
            self.assertIn("capture_to_dispatch", output.getvalue())
            order.append("finish")

        with (
            patch("app.live.Yolo11Detector", return_value=FakeDetector()),
            patch(
                "app.vision.depth_estimator.DepthAnythingV2Estimator",
                return_value=FakeDepth(),
            ),
            patch("app.live.CameraService") as camera,
            patch("app.live.PerformanceRecorder") as recorder,
            patch("app.live.dispatch_warnings", side_effect=dispatch),
            redirect_stdout(output),
            self.assertLogs(level="WARNING"),
        ):
            camera.return_value.read.return_value = frame
            camera.return_value.healthy.return_value = True
            recorder.return_value.finish.side_effect = finish
            self.assertEqual(
                live_main(
                    [
                        "--no-speech",
                        "--max-frames",
                        "1",
                        "--performance-output",
                        "unused.json",
                    ]
                ),
                0,
            )
            recorder.return_value.start.assert_called_once()
        payload = json.loads(output.getvalue())
        self.assertEqual(order, ["dispatch", "finish"])
        self.assertEqual(
            set(payload["latencies_ms"]),
            {"detection", "depth", "fusion", "risk", "capture_to_dispatch"},
        )
        self.assertEqual(payload["performance"]["result_index"], 1)
        self.assertEqual(payload["performance"]["width"], 20)
        self.assertEqual(payload["speech_accepted"], 1)
