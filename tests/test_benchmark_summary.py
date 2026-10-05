"""多轮汇总使用合成报告，验证可比性、统计口径及失败保护。"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from app.benchmark_summary import aggregate_reports, load_run, main, mapping


def fixture(run: int, depth: bool = True) -> dict[str, object]:
    """构造两图两样本的可比报告；不同轮时间不同，耗时可手算。"""
    images: list[dict[str, object]] = []
    for index in range(2):
        values = (1 + index * 2 + run * 4, 2 + index * 2 + run * 4)
        images.append(
            {
                "image_index": index,
                "image_name": f"{index}.jpg",
                "image_file_sha256": str(index) * 64,
                "frame_id": str(index + 2) * 64,
                "width": 20,
                "height": 30,
                "samples": [
                    {
                        "detection_ms": value,
                        "depth_ms": 1 if depth else None,
                        "fusion_ms": 0,
                        "offline_validation_ms": 0,
                        "total_ms": value + (1 if depth else 0),
                        "object_count": 1,
                    }
                    for value in values
                ],
                "summary_ms": {"total": {"mean_ms": 999}},
            }
        )
    return {
        "schema_version": 2,
        "protocol": "offline_multi_image_v1",
        "current_scene": False,
        "sampling_order": "image_major_argument_order",
        "warmup_scope": "per_image_before_its_samples",
        "iterations_scope": "per_image",
        "aggregation": "pooled_samples_equal_weight_per_image",
        "iterations": 2,
        "warmup": 1,
        "seed": 0,
        "image_count": 2,
        "sample_count": 4,
        "code": {
            "revision": "a" * 40,
            "code_dirty": False,
            "app_sha256": {"app/benchmark.py": "b" * 64},
        },
        "weight_sha256": {
            "yolo11": "c" * 64,
            **(
                {
                    f"depth/{name}": "d" * 64
                    for name in (
                        "model.safetensors",
                        "config.json",
                        "preprocessor_config.json",
                    )
                }
                if depth
                else {}
            ),
        },
        "configs": {
            "detection": {"size": 640},
            "fusion": {"method": "median"},
            "risk": {"freshness": 1000},
            "depth": {"size": 518} if depth else None,
        },
        "selected_devices": {"detection": "cpu", "depth": "cpu" if depth else None},
        "versions": {
            name: "3.12"
            for name in (
                "python",
                "numpy",
                "torch",
                "ultralytics",
                "transformers",
                "opencv-python",
            )
        },
        "platform": "synthetic",
        "processor": "synthetic",
        "torch_threads": 1,
        "accelerator_name": None,
        "measurement_started_at_utc": f"2026-10-05T12:00:{run:02d}+00:00",
        "images": images,
        "summary_ms": {"total": {"mean_ms": 999}},
    }


def change(
    report: dict[str, object], path: tuple[str | int, ...], value: object
) -> None:
    """只在合成夹具中修改指定嵌套字段，便于逐项测试拒绝行为。"""
    current: object = report
    for key in path[:-1]:
        if isinstance(current, dict) and isinstance(key, str):
            current = current[key]
        elif isinstance(current, list) and isinstance(key, int):
            current = current[key]
        else:
            raise ValueError("invalid fixture path")
    last = path[-1]
    if isinstance(current, dict) and isinstance(last, str):
        current[last] = value
    elif isinstance(current, list) and isinstance(last, int):
        current[last] = value
    else:
        raise ValueError("invalid fixture destination")


class SummaryTests(unittest.TestCase):
    def test_pooled_and_run_mean_statistics_are_distinct(self) -> None:
        """从原始样本重算统计，区分合并分位数、轮均值分布及轮间样本标准差。"""
        with tempfile.TemporaryDirectory() as directory:
            paths = tuple(Path(directory) / f"{i}.json" for i in range(2))
            for i, path in enumerate(paths):
                path.write_text(json.dumps(fixture(i)), encoding="utf-8")
            result = aggregate_reports(paths)
            self.assertEqual(result["sample_count"], 8)
            pooled = mapping(mapping(result["pooled_summary_ms"])["total"])
            between = mapping(mapping(result["run_mean_summary_ms"])["total"])
            self.assertEqual(pooled["mean_ms"], 5.5)
            self.assertAlmostEqual(pooled["p95_ms"], 8.65)
            self.assertAlmostEqual(between["p95_ms"], 7.3)
            self.assertAlmostEqual(between["sample_stddev_ms"], 8**0.5)
            images = result["images"]
            self.assertIsInstance(images, list)
            if isinstance(images, list):
                image = mapping(images[0])
                self.assertEqual(image["sample_count"], 4)
                self.assertEqual(
                    mapping(mapping(image["pooled_summary_ms"])["total"])["mean_ms"],
                    4.5,
                )

    def test_identity_changes_are_rejected(self) -> None:
        """输入、权重、配置、采样协议、设备、版本及源码变化不能混合汇总。"""
        edits: tuple[tuple[tuple[str | int, ...], object], ...] = (
            (("images", 0, "frame_id"), "f" * 64),
            (("images", 0, "image_file_sha256"), "f" * 64),
            (("images", 0, "width"), 21),
            (("weight_sha256", "yolo11"), "f" * 64),
            (("configs", "detection", "size"), 320),
            (("selected_devices", "detection"), "cuda"),
            (("versions", "python"), "3.13"),
            (("code", "revision"), "f" * 40),
            (("code", "app_sha256", "app/benchmark.py"), "f" * 64),
            (("warmup",), 2),
            (("seed",), 1),
            (("torch_threads",), 2),
            (("sampling_order",), "shuffled"),
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = (Path(directory) / "a.json", Path(directory) / "b.json")
            paths[0].write_text(json.dumps(fixture(0)), encoding="utf-8")
            for field, value in edits:
                with self.subTest(field=field):
                    second = fixture(1)
                    change(second, field, value)
                    paths[1].write_text(json.dumps(second), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        aggregate_reports(paths)

    def test_malformed_or_incomplete_runs_are_rejected(self) -> None:
        """拒绝未知修订、错误计数、负数、非有限值、布尔耗时及阶段不一致。"""
        edits: tuple[tuple[tuple[str | int, ...], object], ...] = (
            (("code", "code_dirty"), True),
            (("code", "revision"), None),
            (("code", "app_sha256"), {}),
            (("sample_count",), 3),
            (("image_count",), True),
            (("iterations",), 3),
            (("images", 0, "samples"), []),
            (("images", 0, "image_index"), 1),
            (("images", 0, "samples", 0, "total_ms"), -1),
            (("images", 0, "samples", 0, "total_ms"), float("nan")),
            (("images", 0, "samples", 0, "total_ms"), float("inf")),
            (("images", 0, "samples", 0, "total_ms"), True),
            (("images", 0, "samples", 0, "total_ms"), 999),
            (("images", 0, "samples", 0, "depth_ms"), None),
            (("current_scene",), True),
            (("schema_version",), 1),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for field, value in edits:
                with self.subTest(field=field):
                    report = fixture(0)
                    change(report, field, value)
                    path.write_text(json.dumps(report), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_run(path)
            for raw in ('{"schema_version":2,"schema_version":2}', "[]", "{"):
                path.write_text(raw, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_run(path)
            report = fixture(0)
            del report["versions"]
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(KeyError):
                load_run(path)

    def test_duplicates_and_round_count_are_rejected(self) -> None:
        """同一报告的路径别名、复制件及重新序列化副本不得重复计入实验。"""
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "a.json", Path(directory) / "b.json"
            report = fixture(0)
            first.write_text(json.dumps(report), encoding="utf-8")
            for paths in ((), (first,), (first,) * 21, (first, first)):
                with self.assertRaises(ValueError):
                    aggregate_reports(paths)
            for indent in (None, 2):
                second.write_text(
                    json.dumps(deepcopy(report), indent=indent), encoding="utf-8"
                )
                with self.assertRaises(ValueError):
                    aggregate_reports((first, second))

    def test_disabled_depth_and_cli_output_protection(self) -> None:
        """无深度报告不生成深度统计；CLI 保留原始输入并拒绝覆盖输出。"""
        with tempfile.TemporaryDirectory() as directory:
            paths = tuple(Path(directory) / f"{i}.json" for i in range(2))
            output = Path(directory) / "summary.json"
            for i, path in enumerate(paths):
                path.write_text(json.dumps(fixture(i, depth=False)), encoding="utf-8")
            originals = [path.read_bytes() for path in paths]
            args = [
                "--report",
                str(paths[0]),
                "--report",
                str(paths[1]),
                "--output",
                str(output),
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            result = json.loads(output.read_text())
            self.assertNotIn("depth", result["pooled_summary_ms"])
            original_output = output.read_bytes()
            with (
                patch("app.benchmark_summary.aggregate_reports") as aggregate,
                self.assertLogs(level="ERROR"),
            ):
                self.assertEqual(main(args), 1)
                aggregate.assert_not_called()
            self.assertEqual(output.read_bytes(), original_output)
            self.assertEqual([path.read_bytes() for path in paths], originals)
            invalid_output = Path(directory) / "invalid.json"
            with self.assertLogs(level="ERROR"):
                self.assertEqual(
                    main(["--report", str(paths[0]), "--output", str(invalid_output)]),
                    1,
                )
            self.assertFalse(invalid_output.exists())
