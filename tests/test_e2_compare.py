"""标定比较的合成测试；不访问模型、相机或网络。"""

import json
import unittest
from dataclasses import replace
from pathlib import Path

from app.e2_compare import (
    Calibration,
    Observation,
    check_disjoint,
    fit_calibration,
    index_rows,
    load_pair,
    metrics,
    predict,
    prediction_value,
    run_comparison,
)
from app.e2_metric import FILES, REVISION
from app.e2_metric import MODEL_ID as METRIC_ID
from app.e2_offline import digest
from app.vision.depth_estimator import MODEL_ID, MODEL_REVISION, WEIGHTS_SHA256
from tests.workspace_temp import workspace_tempdir


def write_json(path: Path, value: dict[str, object]) -> None:
    """写入合成 JSON；不触及真实数据。"""
    path.write_text(json.dumps(value), encoding="utf-8")


def observation(x: float | None, z: float, key: str = "one") -> Observation:
    """构造有完整注解的合成观测。"""
    return Observation(
        key,
        "fit_group",
        key,
        key,
        z,
        x,
        z,
        "available" if x is not None else "insufficient_valid_pixels",
        "available",
        z,
    )


def fixture(root: Path) -> tuple[dict[str, Path], Path, Path]:
    """构造完整双划分的六份报告及冻结源文件，供边界和泄漏测试。"""
    plan_path = root / "plan.json"
    plan: dict[str, object] = {
        "schema": "e2_collection_plan_v1",
        "collection_id": "synthetic",
        "installation_id": "synthetic_install",
        "groups": {
            "fit_group": "fit",
            "validation_group": "validation",
            "hidden": "test",
        },
        "distance_targets_m": [0.5, 1.0],
        "independent_placements_per_distance": 1,
    }
    write_json(plan_path, plan)
    protocol = root / "protocol.json"
    write_json(
        protocol,
        {
            "schema": "e2_comparison_protocol_v1",
            "candidates": ["A1", "A2", "B0"],
            "inverse_denominator_epsilon_per_m": 1e-8,
            "near_distance_m": 1.05,
        },
    )
    inputs: dict[str, Path] = {}
    for split in ("fit", "validation"):
        base_rows: list[dict[str, object]] = []
        rel_rows: list[dict[str, object]] = []
        metric_rows: list[dict[str, object]] = []
        for i, truth in enumerate((0.5, 1.0)):
            key = f"{split}_{i}"
            directory = root / key
            directory.mkdir()
            (directory / "frame.png").write_bytes(key.encode())
            collection: dict[str, object] = {
                "collection_id": "synthetic",
                "plan_sha256": digest(plan_path),
                "installation_id": "synthetic_install",
                "split": split,
                "group_id": f"{split}_group",
                "placement_id": f"p{i + 1:02}",
            }
            record: dict[str, object] = {
                "sample_id": key,
                "collection": collection,
                "measurement": {"distance_m": truth},
                "roi_xyxy": [0, 0, 20, 20],
                "image_sha256": digest(directory / "frame.png"),
            }
            record_path = directory / "record.json"
            write_json(record_path, record)
            base: dict[str, object] = {
                **record,
                "record_path": str(record_path),
                "record_sha256": digest(record_path),
                "analysis_roi": [0, 10, 20, 20],
                "statistics_roi": [5, 13, 15, 17],
                "metric_evidence": None,
            }
            base_rows.append(base)
            rel_rows.append(
                {
                    **base,
                    "depth_summary": {
                        "status": "available",
                        "relative_depth": 1 / truth,
                        "roi": base["statistics_roi"],
                        "units": "relative_inverse_depth",
                    },
                }
            )
            metric_rows.append(
                {
                    **base,
                    "truth_m": truth,
                    "metric_summary": {
                        "status": "available",
                        "predicted_depth_m": truth + 0.2,
                        "roi": base["statistics_roi"],
                        "units": "m",
                    },
                }
            )
        common: dict[str, object] = {
            "split": split,
            "roi_policy": "board_lower_half_v1",
            "config_sha256": "synthetic_config",
            "sample_count": 2,
            "runner_sha256": "synthetic_runner",
        }
        for kind in ("audit", "relative", "metric"):
            inputs[f"{split}_{kind}"] = root / f"{split}_{kind}.json"
        write_json(
            inputs[f"{split}_audit"],
            {
                **common,
                "schema": "e2_offline_v1",
                "mode": "audit",
                "samples": base_rows,
            },
        )
        write_json(
            inputs[f"{split}_relative"],
            {
                **common,
                "schema": "e2_offline_v1",
                "mode": "relative",
                "samples": rel_rows,
                "model": {
                    "id": MODEL_ID,
                    "revision": MODEL_REVISION,
                    "sha256": WEIGHTS_SHA256,
                },
            },
        )
        write_json(
            inputs[f"{split}_metric"],
            {
                **common,
                "schema": "e2_metric_v1",
                "samples": metric_rows,
                "model_id": METRIC_ID,
                "revision": REVISION,
                "files_sha256": FILES,
                "manifest_sha256": digest(inputs[f"{split}_audit"]),
            },
        )
    return inputs, plan_path, protocol


class E2CompareTests(unittest.TestCase):
    def test_known_equations(self) -> None:
        """恢复已知线性及逆深度方程，并检验未参与拟合的预测。"""
        for candidate, truths in [
            ("A1", [3.0, 5.0, 7.0]),
            ("A2", [1 / 3, 1 / 5, 1 / 7]),
        ]:
            rows = [observation(float(i + 1), z, str(i)) for i, z in enumerate(truths)]
            calibration = fit_calibration(rows, candidate)
            self.assertAlmostEqual(calibration.a, 2)
            self.assertAlmostEqual(calibration.b, 1)
            result = predict(observation(4, 1), calibration, 1e-8)
            self.assertAlmostEqual(
                result["prediction_m"], 9 if candidate == "A1" else 1 / 9
            )
            self.assertTrue(result["input_extrapolation"])
            self.assertIsNone(result["metric_evidence"])

    def test_degenerate_fit(self) -> None:
        """输入恒定、缺失或样本不足时不伪造拟合系数。"""
        for rows, status in [
            ([], "insufficient_fit"),
            ([observation(None, 1)], "insufficient_fit"),
            ([observation(2, 1), observation(2, 3)], "rank_deficient_fit"),
        ]:
            calibration = fit_calibration(rows, "A2")
            self.assertEqual(calibration.status, status)
            self.assertIsNone(
                predict(observation(3, 2), calibration, 1e-8)["prediction_m"]
            )

    def test_singular_nonpositive_and_nonfinite(self) -> None:
        """逆变换边界及非有限预测被拒绝，不裁剪成可用距离。"""
        calibration = Calibration("A2", 1, 0, "fitted", 3, 0, 3, 0.5, 3)
        for value in [-1, 0, 1e-8]:
            result = predict(observation(value, 1), calibration, 1e-8)
            self.assertIsNone(result["prediction_m"])
        self.assertIsNotNone(
            predict(observation(2e-8, 1), calibration, 1e-8)["prediction_m"]
        )
        linear = replace(calibration, candidate="A1")
        for value in [-1, 0, float("nan"), float("inf")]:
            self.assertIsNone(
                predict(observation(value, 1), linear, 1e-8)["prediction_m"]
            )

    def test_coverage_and_empty_metrics(self) -> None:
        """失败行保留在覆盖率分母，空有效集合的误差不可写成零。"""
        calibration = Calibration("A1", 1, 0, "fitted", 2, 0, 3, 0.5, 3)
        rows = [
            predict(observation(2, 1), calibration, 1e-8),
            predict(observation(None, 2), calibration, 1e-8),
        ]
        summary = metrics(rows)
        self.assertEqual(summary["coverage"], 0.5)
        self.assertEqual(summary["mae_m"], 1)
        self.assertEqual(summary["overestimate_count"], 1)
        self.assertIsNone(metrics(rows[1:])["mae_m"])
        self.assertIsNone(metrics([])["coverage"])

    def test_prediction_boundary(self) -> None:
        """错误、缺失统计和错误单位不被当作有效模型数值。"""
        self.assertEqual(
            prediction_value({"error": "synthetic"}, True), (None, "inference_error")
        )
        with self.assertRaises(ValueError):
            prediction_value(
                {
                    "depth_summary": {
                        "status": "available",
                        "units": "m",
                        "relative_depth": 1,
                    }
                },
                True,
            )
        with self.assertRaises(ValueError):
            prediction_value(
                {
                    "depth_summary": {
                        "status": "available",
                        "units": "relative_inverse_depth",
                        "relative_depth": float("nan"),
                    }
                },
                True,
            )

    def test_overlap_and_duplicate(self) -> None:
        """同图跨划分或样本重复会被拒绝。"""
        row = observation(1, 1)
        with self.assertRaises(ValueError):
            check_disjoint([row], [replace(row, sample_id="other", group="validation")])
        with self.assertRaises(ValueError):
            index_rows(
                {"sample_count": 2, "samples": [{"sample_id": "a"}, {"sample_id": "a"}]}
            )

    def test_complete_run_and_validation_cannot_change_fit(self) -> None:
        """改变验证模型值只影响验证结果，参数保持一致，且禁止覆盖输出。"""
        with workspace_tempdir() as root:
            inputs, plan, protocol = fixture(root)
            first = run_comparison(inputs, plan, protocol, root / "out1")
            path = inputs["validation_relative"]
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["samples"][0]["depth_summary"]["relative_depth"] = 100
            write_json(path, raw)
            second = run_comparison(inputs, plan, protocol, root / "out2")
            for candidate in ("A1", "A2"):
                self.assertEqual(
                    first["candidates"][candidate]["calibration"],
                    second["candidates"][candidate]["calibration"],
                )
            self.assertEqual(
                first["paired_common_valid"]["validation"]["common_valid_count"], 2
            )
            self.assertFalse(first["production_approved"])
            self.assertTrue((root / "out1/inputs/fit_audit.json").exists())
            with self.assertRaises(ValueError):
                run_comparison(inputs, plan, protocol, root / "out1")

    def test_source_and_pair_mismatches(self) -> None:
        """错误划分、篡改图像、真值或统计区域均导致整批拒绝。"""
        for fault in ("split", "truth", "roi", "source", "missing", "model"):
            with self.subTest(fault=fault), workspace_tempdir() as root:
                inputs, plan, _ = fixture(root)
                path = inputs["validation_metric"]
                raw = json.loads(path.read_text(encoding="utf-8"))
                if fault == "split":
                    raw["split"] = "fit"
                elif fault == "truth":
                    raw["samples"][0]["truth_m"] = 7
                elif fault == "roi":
                    raw["samples"][0]["metric_summary"]["roi"] = [0, 0, 1, 1]
                elif fault == "source":
                    (root / "validation_0/frame.png").write_bytes(b"changed")
                elif fault == "missing":
                    raw["samples"].pop()
                    raw["sample_count"] = 1
                else:
                    raw["revision"] = "different_model"
                write_json(path, raw)
                with self.assertRaises(ValueError):
                    load_pair(
                        inputs["validation_audit"],
                        inputs["validation_relative"],
                        path,
                        "validation",
                        plan,
                    )

    def test_failed_inference_remains_in_denominator(self) -> None:
        """整批报告保留失败推理，配对有效交集不能替代全量覆盖率。"""
        with workspace_tempdir() as root:
            inputs, plan, protocol = fixture(root)
            path = inputs["validation_metric"]
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["samples"][0].pop("metric_summary")
            raw["samples"][0]["error"] = "synthetic failure"
            write_json(path, raw)
            report = run_comparison(inputs, plan, protocol, root / "out")
            summary = report["candidates"]["B0"]["splits"]["validation"]["summary"]
            self.assertEqual(summary["overall"]["coverage"], 0.5)
            self.assertEqual(summary["overall"]["rejections"], {"inference_error": 1})
            self.assertEqual(
                report["paired_common_valid"]["validation"]["common_valid_count"], 1
            )

    def test_test_split_is_locked(self) -> None:
        """测试集合入口在读取任何文件之前拒绝。"""
        with self.assertRaisesRegex(ValueError, "test data locked"):
            load_pair(
                Path("unused"), Path("unused"), Path("unused"), "test", Path("unused")
            )
