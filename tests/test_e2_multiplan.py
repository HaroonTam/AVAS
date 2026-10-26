"""多计划比较和研究验收的合成测试，无模型、网络或相机。"""

import json
import unittest
from pathlib import Path

from app.e2_collection import read_object
from app.e2_compare import fit_calibration, load_pair, predict, summarize
from app.e2_multiplan import acceptance, guarded_rows, load_sources, run_multiplan
from app.e2_offline import digest
from tests.test_e2_compare import fixture, observation, write_json
from tests.workspace_temp import workspace_tempdir

ROOT = Path(__file__).resolve().parents[1]


def distinct_fixture(root: Path) -> tuple[dict[str, Path], Path]:
    """为补充集合生成不同身份和图片字节，重新计算全部依赖哈希。"""
    root.mkdir()
    inputs, plan, _ = fixture(root)
    raw_plan = read_object(plan)
    raw_plan["groups"] = {
        "extra_fit_group": "fit",
        "extra_validation_group": "validation",
    }
    write_json(plan, raw_plan)
    for split in ("fit", "validation"):
        audit = json.loads(inputs[f"{split}_audit"].read_text(encoding="utf-8"))
        relative = json.loads(inputs[f"{split}_relative"].read_text(encoding="utf-8"))
        metric = json.loads(inputs[f"{split}_metric"].read_text(encoding="utf-8"))
        for base, rel, met in zip(
            audit["samples"], relative["samples"], metric["samples"]
        ):
            path = Path(base["record_path"])
            record = json.loads(path.read_text(encoding="utf-8"))
            record["sample_id"] = "extra_" + record["sample_id"]
            record["collection"]["group_id"] = f"extra_{split}_group"
            record["collection"]["plan_sha256"] = digest(plan)
            (path.parent / "frame.png").write_bytes(record["sample_id"].encode())
            record["image_sha256"] = digest(path.parent / "frame.png")
            write_json(path, record)
            for row in (base, rel, met):
                row.update(
                    sample_id=record["sample_id"],
                    collection=record["collection"],
                    image_sha256=record["image_sha256"],
                    record_sha256=digest(path),
                )
        write_json(inputs[f"{split}_audit"], audit)
        metric["manifest_sha256"] = digest(inputs[f"{split}_audit"])
        write_json(inputs[f"{split}_relative"], relative)
        write_json(inputs[f"{split}_metric"], metric)
    return inputs, plan


def multiplan_fixture(root: Path) -> Path:
    """生成四种数据角色及完整固定哈希的多计划清单。"""
    original = root / "original"
    original.mkdir()
    inputs, plan, _ = fixture(original)
    extra_inputs, extra_plan = distinct_fixture(root / "extra")
    sources = []
    for role, split, paths, current_plan, group in [
        ("original_fit", "fit", inputs, plan, "fit_group"),
        ("old_validation", "validation", inputs, plan, "validation_group"),
        ("supplemental_fit", "fit", extra_inputs, extra_plan, "extra_fit_group"),
        (
            "new_validation",
            "validation",
            extra_inputs,
            extra_plan,
            "extra_validation_group",
        ),
    ]:
        files = {k: paths[f"{split}_{k}"] for k in ("audit", "relative", "metric")}
        files["plan"] = current_plan
        sources.append(
            {
                "role": role,
                "groups": [group],
                **{k: str(p.resolve()) for k, p in files.items()},
                "sha256": {k: digest(p) for k, p in files.items()},
            }
        )
    recipe = root / "recipe.json"
    write_json(
        recipe,
        {
            "schema": "e2_multiplan_recipe_v1",
            "inverse_denominator_epsilon_per_m": 1e-8,
            "sources": sources,
        },
    )
    return recipe


class MultiplanTests(unittest.TestCase):
    def test_full_run_and_validation_isolation(self) -> None:
        """新验证模型值变化不会改变联合拟合参数，旧验证保持单独报告。"""
        with workspace_tempdir() as root:
            recipe = multiplan_fixture(root)
            criteria = ROOT / "configs/e2_research_acceptance.json"
            first = run_multiplan(recipe, criteria, root / "first")
            data = json.loads(recipe.read_text(encoding="utf-8"))
            source = data["sources"][3]
            path = Path(source["relative"])
            prediction = json.loads(path.read_text(encoding="utf-8"))
            prediction["samples"][0]["depth_summary"]["relative_depth"] = 100
            write_json(path, prediction)
            source["sha256"]["relative"] = digest(path)
            write_json(recipe, data)
            second = run_multiplan(recipe, criteria, root / "second")
            for name in ("A1", "A2", "original_A1"):
                self.assertEqual(
                    first["candidates"][name]["calibration"],
                    second["candidates"][name]["calibration"],
                )
            self.assertEqual(
                first["candidates"]["A1"]["calibration"]["fitted_count"], 4
            )
            self.assertEqual(first["role_counts"]["new_validation"], 2)
            self.assertFalse(first["production_approved"])
            self.assertFalse(first["test_unlocked"])
            with self.assertRaises(ValueError):
                run_multiplan(recipe, criteria, root / "first")

    def test_hash_role_and_overlap_rejection(self) -> None:
        """篡改清单、测试角色或重复数据源不能进入比较。"""
        for fault in ("hash", "role", "duplicate", "missing"):
            with self.subTest(fault=fault), workspace_tempdir() as root:
                path = multiplan_fixture(root)
                recipe = json.loads(path.read_text(encoding="utf-8"))
                if fault == "hash":
                    recipe["sources"][0]["sha256"]["relative"] = "wrong"
                elif fault == "role":
                    recipe["sources"][0]["role"] = "test"
                elif fault == "duplicate":
                    recipe["sources"].append(recipe["sources"][0])
                else:
                    recipe["sources"].pop()
                with self.assertRaises(ValueError):
                    load_sources(recipe)

    def test_group_selection_is_explicit_and_complete(self) -> None:
        """允许按计划选择完整布景，但禁止跨划分选择或空选择。"""
        with workspace_tempdir() as root:
            inputs, plan, _ = fixture(root)
            for groups in ([], ["validation_group"], ["fit_group", "fit_group"]):
                with self.assertRaises(ValueError):
                    load_pair(
                        inputs["fit_audit"],
                        inputs["fit_relative"],
                        inputs["fit_metric"],
                        "fit",
                        plan,
                        selected_groups=groups,
                    )
            rows = load_pair(
                inputs["fit_audit"],
                inputs["fit_relative"],
                inputs["fit_metric"],
                "fit",
                plan,
                selected_groups=["fit_group"],
            )
            self.assertEqual(len(rows), 2)

    def test_range_rejection_does_not_depend_on_truth(self) -> None:
        """范围标记决定拒绝，改变真值不改变拒绝状态；不修改原行。"""
        calibration = fit_calibration([observation(1, 1), observation(2, 2)], "A1")
        raw = predict(observation(3, 1), calibration, 1e-8)
        modified = {**raw, "truth_m": 10}
        for row in (raw, modified):
            self.assertIsNone(guarded_rows([row])[0]["prediction_m"])
            self.assertIsNotNone(row["prediction_m"])

    def test_acceptance_boundaries_and_empty(self) -> None:
        """阈值包含端点，低覆盖率和空近距离不能由低平均误差掩盖。"""
        targets = read_object(ROOT / "configs/e2_research_acceptance.json")["targets"]
        calibration = fit_calibration([observation(1, 1), observation(2, 2)], "A1")
        row = predict(observation(1, 1), calibration, 1e-8)
        summary = summarize([row], 1.05)
        summary["overall"].update(
            mae_m=0.2, rmse_m=0.3, absrel=0.2, max_absolute_error_m=0.5
        )
        summary["near"]["max_overestimate_m"] = 0.2
        self.assertTrue(acceptance(summary, targets)["all_pass"])
        summary["overall"]["mae_m"] = 0.20000001
        self.assertFalse(acceptance(summary, targets)["all_pass"])
        summary["overall"].update(mae_m=0, coverage=0.89)
        self.assertIn("overall_coverage", acceptance(summary, targets)["failed_checks"])
        self.assertFalse(acceptance(summarize([], 1.05), targets)["all_pass"])
