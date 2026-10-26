"""多计划 E2 开发比较：合并拟合，分开旧验证与新验证，不解锁测试。"""

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from app.e2_capture import ROOT
from app.e2_collection import read_object
from app.e2_compare import (
    Calibration,
    Observation,
    check_disjoint,
    fit_calibration,
    load_pair,
    predict,
    summarize,
)
from app.e2_offline import digest, mapping, number, string


def string_list(value: object) -> list[str]:
    """收窄非空字符串列表，拒绝隐式选组。"""
    if not isinstance(value, list) or not value:
        raise ValueError("expected nonempty group list")
    return [string(item) for item in value]


def guarded_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """范围外候选返回不可用，保留全部行及原始范围标记，不使用真值决定拒绝。"""
    result: list[dict[str, object]] = []
    for original in rows:
        row = dict(original)
        reasons = []
        if row["prediction_m"] is not None:
            if row["input_extrapolation"]:
                reasons.append("outside_fit_input_range")
            if row["prediction_outside_fit_truth_range"]:
                reasons.append("outside_fit_truth_range")
            if reasons:
                row.update(prediction_m=None, error_m=None, reason="+".join(reasons))
        result.append(row)
    return result


def acceptance(
    summary: dict[str, object], targets: dict[str, object]
) -> dict[str, object]:
    """按已确认目标逐项判定，零有效或缺失指标不得通过；仅为开发验收。"""
    overall = mapping(summary["overall"])
    groups = mapping(summary["by_group"])
    distances = mapping(summary["by_nominal_distance_m"])
    near = mapping(summary["near"])
    checks: dict[str, bool] = {}
    for key, target in [
        ("mae_m", "valid_mae_m_max"),
        ("rmse_m", "valid_rmse_m_max"),
        ("absrel", "valid_absrel_max"),
        ("max_absolute_error_m", "valid_absolute_error_m_max"),
    ]:
        checks[key] = overall[key] is not None and number(overall[key]) <= number(
            targets[target]
        )
    checks["overall_coverage"] = overall["coverage"] is not None and number(
        overall["coverage"]
    ) >= number(targets["overall_coverage_min"])
    for label, items, target in [
        ("group_coverage", groups, "each_group_coverage_min"),
        ("distance_coverage", distances, "each_nominal_distance_coverage_min"),
    ]:
        checks[label] = bool(items) and all(
            mapping(item)["coverage"] is not None
            and number(mapping(item)["coverage"]) >= number(targets[target])
            for item in items.values()
        )
    checks["near_overestimate"] = near["max_overestimate_m"] is not None and number(
        near["max_overestimate_m"]
    ) <= number(targets["near_overestimate_m_max"])
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "failed_checks": [key for key, passed in checks.items() if not passed],
        "role": "development_only_not_independent_test_or_production_approval",
    }


def load_sources(
    recipe: dict[str, object],
) -> tuple[dict[str, list[Observation]], dict[str, Path]]:
    """校验多份冻结输入与全部布景，拒绝跨计划配置漂移、重复图像或角色混用。"""
    sources = recipe.get("sources")
    if recipe.get("schema") != "e2_multiplan_recipe_v1" or not isinstance(
        sources, list
    ):
        raise ValueError("invalid recipe")
    datasets: dict[str, list[Observation]] = {
        role: []
        for role in (
            "original_fit",
            "supplemental_fit",
            "old_validation",
            "new_validation",
        )
    }
    snapshots: dict[str, Path] = {}
    previous: list[list[Observation]] = []
    reference: dict[str, object] | None = None
    installation: str | None = None
    for i, item in enumerate(sources):
        source = mapping(item)
        role = string(source.get("role"))
        if role not in datasets:
            raise ValueError("unsupported role; test data locked")
        split = "fit" if role.endswith("fit") else "validation"
        paths = {
            key: ROOT / string(source.get(key))
            for key in ("audit", "relative", "metric", "plan")
        }
        hashes = mapping(source.get("sha256"))
        for key, path in paths.items():
            if digest(path) != hashes.get(key):
                raise ValueError(f"source hash mismatch: {i} {key}")
            snapshots[f"source_{i}_{key}"] = path
        groups = string_list(source.get("groups"))
        rows = load_pair(
            paths["audit"],
            paths["relative"],
            paths["metric"],
            split,
            paths["plan"],
            selected_groups=groups,
        )
        plan = read_object(paths["plan"])
        current_installation = string(plan.get("installation_id"))
        if installation is not None and installation != current_installation:
            raise ValueError("mixed installations")
        installation = current_installation
        signature: dict[str, object] = {}
        for kind in ("audit", "relative", "metric"):
            report = read_object(paths[kind])
            for key in (
                "config_sha256",
                "roi_policy",
                "runner_sha256",
                "fusion_config",
                "image_size",
                "model",
                "files_sha256",
            ):
                signature[f"{kind}_{key}"] = report.get(key)
        if reference is not None and signature != reference:
            raise ValueError("cross-source model/configuration/runner mismatch")
        reference = signature
        for prior in previous:
            check_disjoint(prior, rows)
        previous.append(rows)
        datasets[role].extend(rows)
    if any(not rows for rows in datasets.values()):
        raise ValueError("all four data roles required")
    return datasets, snapshots


def run_multiplan(
    recipe_path: Path, criteria_path: Path, output: Path
) -> dict[str, object]:
    """仅 fit 联合拟合三候选；分别评估旧/新验证及范围拒绝后的固定研究目标。"""
    if output.exists():
        raise ValueError("output already exists")
    recipe = read_object(recipe_path)
    criteria = read_object(criteria_path)
    if (
        criteria.get("schema") != "e2_research_acceptance_v1"
        or criteria.get("status") != "user_confirmed_research_targets"
        or criteria.get("all_targets_required") is not True
    ):
        raise ValueError("expected confirmed research targets")
    targets = mapping(criteria.get("targets"))
    near = number(targets.get("near_truth_m_lte"))
    epsilon = number(recipe.get("inverse_denominator_epsilon_per_m"))
    if near <= 0 or epsilon <= 0:
        raise ValueError("limits must be positive")
    for key, value in targets.items():
        threshold = number(value)
        if threshold <= 0 or ("coverage" in key and threshold > 1):
            raise ValueError("invalid research target")
    datasets, snapshots = load_sources(recipe)
    fit = datasets["original_fit"] + datasets["supplemental_fit"]
    evaluations = {
        "fit": fit,
        "old_validation": datasets["old_validation"],
        "new_validation": datasets["new_validation"],
    }
    calibrations = {name: fit_calibration(fit, name) for name in ("A1", "A2")}
    calibrations["B0"] = Calibration(
        "B0",
        None,
        None,
        "unvalidated_baseline",
        0,
        None,
        None,
        min(r.truth_m for r in fit),
        max(r.truth_m for r in fit),
    )
    calibrations["original_A1"] = fit_calibration(datasets["original_fit"], "A1")
    candidates: dict[str, object] = {}
    raw_predictions: dict[str, dict[str, list[dict[str, object]]]] = {}
    for name, calibration in calibrations.items():
        splits: dict[str, object] = {}
        raw_predictions[name] = {}
        for split, rows in evaluations.items():
            raw = [predict(row, calibration, epsilon) for row in rows]
            guarded = guarded_rows(raw)
            summary = summarize(guarded, near)
            raw_predictions[name][split] = raw
            splits[split] = {
                "raw": {"summary": summarize(raw, near), "samples": raw},
                "guarded": {"summary": summary, "samples": guarded},
                "research_acceptance": acceptance(summary, targets),
            }
        candidates[name] = {"calibration": asdict(calibration), "splits": splits}
    paired: dict[str, object] = {}
    for split in evaluations:
        paired[split] = {}
        modes: dict[str, object] = {}
        for mode in ("raw", "guarded"):
            rows_by_candidate = {
                name: (
                    raw_predictions[name][split]
                    if mode == "raw"
                    else guarded_rows(raw_predictions[name][split])
                )
                for name in ("A1", "A2", "B0")
            }
            common = set.intersection(
                *[
                    {
                        string(row["sample_id"])
                        for row in rows
                        if row["prediction_m"] is not None
                    }
                    for rows in rows_by_candidate.values()
                ]
            )
            modes[mode] = {
                "common_valid_count": len(common),
                "candidates": {
                    name: summarize(
                        [row for row in rows if row["sample_id"] in common], near
                    )
                    for name, rows in rows_by_candidate.items()
                },
            }
        paired[split] = modes
    snapshots.update(recipe=recipe_path, criteria=criteria_path)
    report: dict[str, object] = {
        "schema": "e2_multiplan_comparison_v1",
        "role": "development_only",
        "production_approved": False,
        "test_unlocked": False,
        "metric_evidence": None,
        "primary_candidate": "A1",
        "range_policy": (
            "inclusive_fit_input_and_fit_truth_prediction_ranges_no_clipping"
        ),
        "range_status": "candidate_development_rule_not_validated_operating_range",
        "baseline_note": "original_A1 uses only original fit parameters and ranges",
        "recipe": recipe,
        "numpy_version": np.__version__,
        "criteria": criteria,
        "role_counts": {role: len(rows) for role, rows in datasets.items()},
        "candidates": candidates,
        "paired_common_valid": paired,
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": digest(path)}
            for name, path in snapshots.items()
        },
        "source_sha256": {
            "multiplan": digest(Path(__file__)),
            "compare": digest(ROOT / "app/e2_compare.py"),
        },
        "code_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
    }
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "inputs"
    snapshot.mkdir()
    for name, path in snapshots.items():
        (snapshot / f"{name}.json").write_bytes(path.read_bytes())
    (output / "multiplan_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (output / "compare_snapshot.py").write_bytes(
        (ROOT / "app/e2_compare.py").read_bytes()
    )
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main(argv: list[str] | None = None) -> int:
    """运行带哈希的多计划比较清单，失败时返回非零退出码。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument(
        "--criteria", type=Path, default=ROOT / "configs/e2_research_acceptance.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run_multiplan(args.recipe, args.criteria, args.output)
    except (ValueError, OSError, KeyError) as exc:
        print(f"Multiplan comparison failed: {exc}")
        return 2
    print(args.output / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
