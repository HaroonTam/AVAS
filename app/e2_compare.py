"""E2 数值报告标定比较；只拟合 fit，不运行模型、不签发生产证据。"""

import argparse
import json
import math
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from app.e2_capture import ROOT
from app.e2_collection import read_object
from app.e2_metric import FILES, REVISION
from app.e2_metric import MODEL_ID as METRIC_ID
from app.e2_offline import digest, mapping, number, string
from app.vision.depth_estimator import MODEL_ID, MODEL_REVISION, WEIGHTS_SHA256


@dataclass(frozen=True)
class Observation:
    sample_id: str
    group: str
    placement: str
    image_hash: str
    truth_m: float
    relative: float | None
    metric_m: float | None
    relative_reason: str
    metric_reason: str
    nominal_m: float


@dataclass(frozen=True)
class Calibration:
    candidate: str
    a: float | None
    b: float | None
    status: str
    fitted_count: int
    input_min: float | None
    input_max: float | None
    truth_min_m: float | None
    truth_max_m: float | None


def index_rows(report: dict[str, object]) -> dict[str, dict[str, object]]:
    """校验样本计数及唯一标识，拒绝空集合和重复记录。"""
    rows = report.get("samples")
    if (
        not isinstance(rows, list)
        or not rows
        or report.get("sample_count") != len(rows)
    ):
        raise ValueError("invalid sample count")
    result: dict[str, dict[str, object]] = {}
    for value in rows:
        row = mapping(value)
        key = string(row.get("sample_id"))
        if key in result:
            raise ValueError("duplicate sample id")
        result[key] = row
    return result


def prediction_value(
    row: dict[str, object], relative: bool
) -> tuple[float | None, str]:
    """保留推理失败和无效统计的拒绝原因，严格检查可用值的单位。"""
    if "error" in row:
        return None, "inference_error"
    summary = mapping(row.get("depth_summary" if relative else "metric_summary"))
    status = string(summary.get("status"))
    if status != "available":
        return None, status
    units = "relative_inverse_depth" if relative else "m"
    if summary.get("units") != units:
        raise ValueError("prediction units mismatch")
    value = number(summary.get("relative_depth" if relative else "predicted_depth_m"))
    if not relative and value <= 0:
        return None, "nonpositive_metric"
    return value, "available"


def load_pair(
    audit_path: Path,
    relative_path: Path,
    metric_path: Path,
    split: str,
    plan_path: Path,
) -> list[Observation]:
    """对齐冻结清单与两份报告，验证源文件、计划、安装及完整摆放编号。"""
    if split not in ("fit", "validation"):
        raise ValueError("test data locked")
    audit, relative, metric = [
        read_object(p) for p in (audit_path, relative_path, metric_path)
    ]
    plan = read_object(plan_path)
    if plan.get("schema") != "e2_collection_plan_v1":
        raise ValueError("invalid plan")
    for report in (audit, relative, metric):
        if (
            report.get("split") != split
            or report.get("roi_policy") != "board_lower_half_v1"
        ):
            raise ValueError("split or ROI policy mismatch")
        if report.get("config_sha256") != audit.get("config_sha256"):
            raise ValueError("configuration mismatch")
    if (audit.get("schema"), audit.get("mode")) != ("e2_offline_v1", "audit"):
        raise ValueError("expected frozen audit")
    if (relative.get("schema"), relative.get("mode")) != ("e2_offline_v1", "relative"):
        raise ValueError("expected relative report")
    if relative.get("model") != {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "sha256": WEIGHTS_SHA256,
    }:
        raise ValueError("relative model mismatch")
    if (
        metric.get("schema") != "e2_metric_v1"
        or metric.get("model_id") != METRIC_ID
        or metric.get("revision") != REVISION
        or metric.get("files_sha256") != FILES
        or metric.get("manifest_sha256") != digest(audit_path)
    ):
        raise ValueError("metric model or manifest mismatch")
    if relative.get("fusion_config") != audit.get("fusion_config"):
        raise ValueError("fusion configuration mismatch")
    ar, rr, mr = [index_rows(report) for report in (audit, relative, metric)]
    if set(ar) != set(rr) or set(ar) != set(mr):
        raise ValueError("sample sets mismatch")
    groups = mapping(plan.get("groups"))
    targets = plan.get("distance_targets_m")
    repeats = plan.get("independent_placements_per_distance")
    if not isinstance(targets, list) or not targets or type(repeats) is not int:
        raise ValueError("invalid placement plan")
    if not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("invalid repeats")
    nominal = [number(t) for t in targets for _ in range(repeats)]
    expected = {
        (g, f"p{i + 1:02}")
        for g, role in groups.items()
        if role == split
        for i in range(len(nominal))
    }
    observed: set[tuple[str, str]] = set()
    result: list[Observation] = []
    for key in sorted(ar):
        base, rel, met = ar[key], rr[key], mr[key]
        for field in ("record_sha256", "image_sha256", "collection", "analysis_roi"):
            if base.get(field) is None or not base[field] == rel.get(field) == met.get(
                field
            ):
                raise ValueError(f"paired {field} mismatch")
        collection = mapping(base.get("collection"))
        group = string(collection.get("group_id"))
        placement = string(collection.get("placement_id"))
        if (
            (group, placement) not in expected
            or (group, placement) in observed
            or collection.get("split") != split
            or collection.get("plan_sha256") != digest(plan_path)
            or collection.get("installation_id") != plan.get("installation_id")
            or collection.get("collection_id") != plan.get("collection_id")
        ):
            raise ValueError("collection identity or placement mismatch")
        observed.add((group, placement))
        record_path = Path(string(base.get("record_path")))
        if digest(record_path) != base["record_sha256"]:
            raise ValueError("source record changed")
        record = read_object(record_path)
        if (
            record.get("collection") != collection
            or record.get("sample_id") != key
            or record.get("measurement") != base.get("measurement")
            or record.get("roi_xyxy") != base.get("roi_xyxy")
            or record.get("image_sha256") != base["image_sha256"]
            or digest(record_path.parent / "frame.png") != base["image_sha256"]
        ):
            raise ValueError("source metadata or image changed")
        if rel.get("measurement") != base.get("measurement"):
            raise ValueError("relative truth mismatch")
        truth = number(mapping(base.get("measurement")).get("distance_m"))
        if truth <= 0 or number(met.get("truth_m")) != truth:
            raise ValueError("metric truth mismatch")
        for row, summary_key in ((rel, "depth_summary"), (met, "metric_summary")):
            if row.get("metric_evidence") is not None:
                raise ValueError("unexpected production evidence")
            if "error" not in row:
                summary = mapping(row.get(summary_key))
                if summary.get("roi") != base.get("statistics_roi"):
                    raise ValueError("statistics ROI mismatch")
        rv, reason_r = prediction_value(rel, True)
        mv, reason_m = prediction_value(met, False)
        result.append(
            Observation(
                key,
                group,
                placement,
                string(base["image_sha256"]),
                truth,
                rv,
                mv,
                reason_r,
                reason_m,
                nominal[int(placement[1:]) - 1],
            )
        )
    if observed != expected:
        raise ValueError("incomplete registered split")
    return result


def check_disjoint(fit: list[Observation], validation: list[Observation]) -> None:
    """拒绝拟合与验证之间共享样本、图像或布景，防止直接数据泄漏。"""
    for attribute in ("sample_id", "image_hash", "group"):
        if {getattr(r, attribute) for r in fit} & {
            getattr(r, attribute) for r in validation
        }:
            raise ValueError(f"fit/validation overlap: {attribute}")


def fit_calibration(rows: list[Observation], candidate: str) -> Calibration:
    """对给定拟合行做带截距最小二乘；秩不足或非有限系数时明确失败。"""
    if candidate not in ("A1", "A2"):
        raise ValueError("unsupported candidate")
    valid = [r for r in rows if r.relative is not None]
    empty = Calibration(
        candidate, None, None, "insufficient_fit", len(valid), None, None, None, None
    )
    if len(valid) < 2:
        return empty
    x = np.asarray([r.relative for r in valid], dtype=np.float64)
    y = np.asarray([r.truth_m for r in valid], dtype=np.float64)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any(y <= 0):
        raise ValueError("invalid fitting data")
    target = y if candidate == "A1" else 1.0 / y
    coefficients, _, rank, _ = np.linalg.lstsq(
        np.column_stack((x, np.ones(len(x)))), target, rcond=None
    )
    if rank != 2 or not np.all(np.isfinite(coefficients)):
        return Calibration(
            candidate,
            None,
            None,
            "rank_deficient_fit",
            len(valid),
            float(x.min()),
            float(x.max()),
            float(y.min()),
            float(y.max()),
        )
    return Calibration(
        candidate,
        float(coefficients[0]),
        float(coefficients[1]),
        "fitted",
        len(valid),
        float(x.min()),
        float(x.max()),
        float(y.min()),
        float(y.max()),
    )


def predict(
    row: Observation, calibration: Calibration, epsilon: float
) -> dict[str, object]:
    """计算离线诊断预测；不裁剪，拒绝非正或奇异结果并标记外推，不代表生产支持。"""
    reason = "available"
    value: float | None = None
    extrapolated = False
    outside_truth_range = False
    if calibration.candidate == "B0":
        value, reason = row.metric_m, row.metric_reason
    elif calibration.status != "fitted":
        reason = calibration.status
    elif row.relative is None:
        reason = row.relative_reason
    else:
        assert calibration.a is not None and calibration.b is not None
        value = calibration.a * row.relative + calibration.b
        if calibration.candidate == "A2":
            if value <= epsilon:
                value, reason = None, "inverse_denominator_nonpositive_or_small"
            else:
                value = 1.0 / value
        assert calibration.input_min is not None and calibration.input_max is not None
        extrapolated = (
            not calibration.input_min <= row.relative <= calibration.input_max
        )
    if value is not None and (not math.isfinite(value) or value <= 0):
        value, reason = None, "nonpositive_or_nonfinite_prediction"
    if value is not None and calibration.truth_min_m is not None:
        assert calibration.truth_max_m is not None
        outside_truth_range = (
            not calibration.truth_min_m <= value <= calibration.truth_max_m
        )
    return {
        "sample_id": row.sample_id,
        "group": row.group,
        "placement": row.placement,
        "nominal_m": row.nominal_m,
        "truth_m": row.truth_m,
        "prediction_m": value,
        "reason": reason,
        "error_m": None if value is None else value - row.truth_m,
        "input_extrapolation": extrapolated,
        "prediction_outside_fit_truth_range": outside_truth_range,
        "metric_evidence": None,
    }


def metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    """计算有效预测误差及全量覆盖；空有效集合返回空指标，保留拒绝原因。"""
    valid = [r for r in rows if r["prediction_m"] is not None]
    result: dict[str, object] = {
        "total": len(rows),
        "valid": len(valid),
        "coverage": len(valid) / len(rows) if rows else None,
        "rejections": dict(
            Counter(string(r["reason"]) for r in rows if r["prediction_m"] is None)
        ),
        "input_extrapolation_count": sum(bool(r["input_extrapolation"]) for r in rows),
        "prediction_outside_fit_truth_range_count": sum(
            bool(r["prediction_outside_fit_truth_range"]) for r in rows
        ),
    }
    if not valid:
        result.update(
            dict.fromkeys(
                (
                    "mae_m",
                    "rmse_m",
                    "absrel",
                    "bias_m",
                    "p95_absolute_error_m",
                    "max_absolute_error_m",
                    "overestimate_count",
                    "max_overestimate_m",
                )
            )
        )
        return result
    errors = np.asarray([number(r["error_m"]) for r in valid])
    truth = np.asarray([number(r["truth_m"]) for r in valid])
    result.update(
        mae_m=float(np.abs(errors).mean()),
        rmse_m=float(np.sqrt(np.square(errors).mean())),
        absrel=float((np.abs(errors) / truth).mean()),
        bias_m=float(errors.mean()),
        p95_absolute_error_m=float(np.quantile(np.abs(errors), 0.95)),
        max_absolute_error_m=float(np.abs(errors).max()),
        overestimate_count=int((errors > 0).sum()),
        max_overestimate_m=float(max(0.0, errors.max())),
    )
    return result


def summarize(rows: list[dict[str, object]], near_m: float) -> dict[str, object]:
    """按布景及预注册摆放档位统计，近距离切片由真值定义且不作为风险阈值。"""
    return {
        "overall": metrics(rows),
        "by_group": {
            g: metrics([r for r in rows if r["group"] == g])
            for g in sorted({string(r["group"]) for r in rows})
        },
        "by_nominal_distance_m": {
            str(d): metrics([r for r in rows if r["nominal_m"] == d])
            for d in sorted({number(r["nominal_m"]) for r in rows})
        },
        "near_truth_m_lte": near_m,
        "near": metrics([r for r in rows if number(r["truth_m"]) <= near_m]),
    }


def run_comparison(
    inputs: dict[str, Path],
    plan_path: Path,
    protocol_path: Path,
    output: Path,
) -> dict[str, object]:
    """冻结六份输入并运行仅 fit 拟合的开发比较；拒绝覆盖已有输出和跨划分混用。"""
    if output.exists():
        raise ValueError("output already exists")
    protocol = read_object(protocol_path)
    if protocol.get("schema") != "e2_comparison_protocol_v1" or protocol.get(
        "candidates"
    ) != ["A1", "A2", "B0"]:
        raise ValueError("invalid comparison protocol")
    epsilon = number(protocol.get("inverse_denominator_epsilon_per_m"))
    near = number(protocol.get("near_distance_m"))
    if epsilon <= 0 or near <= 0:
        raise ValueError("protocol limits must be positive")
    datasets = {
        split: load_pair(
            inputs[f"{split}_audit"],
            inputs[f"{split}_relative"],
            inputs[f"{split}_metric"],
            split,
            plan_path,
        )
        for split in ("fit", "validation")
    }
    check_disjoint(datasets["fit"], datasets["validation"])
    for suffix in ("audit", "relative", "metric"):
        fit_report = read_object(inputs[f"fit_{suffix}"])
        validation_report = read_object(inputs[f"validation_{suffix}"])
        for field in (
            "config_sha256",
            "roi_policy",
            "runner_sha256",
            "fusion_config",
            "image_size",
            "model",
            "files_sha256",
        ):
            if fit_report.get(field) != validation_report.get(field):
                raise ValueError(f"cross-split {suffix} {field} mismatch")
    calibrations = [fit_calibration(datasets["fit"], name) for name in ("A1", "A2")]
    calibrations.append(
        Calibration(
            "B0",
            None,
            None,
            "unvalidated_baseline",
            0,
            None,
            None,
            min(row.truth_m for row in datasets["fit"]),
            max(row.truth_m for row in datasets["fit"]),
        )
    )
    candidates: dict[str, object] = {}
    predictions: dict[str, dict[str, list[dict[str, object]]]] = {}
    for calibration in calibrations:
        predictions[calibration.candidate] = {
            split: [predict(row, calibration, epsilon) for row in rows]
            for split, rows in datasets.items()
        }
        candidates[calibration.candidate] = {
            "calibration": asdict(calibration),
            "splits": {
                split: {"summary": summarize(rows, near), "samples": rows}
                for split, rows in predictions[calibration.candidate].items()
            },
        }
    paired: dict[str, object] = {}
    for split in datasets:
        common = set.intersection(
            *[
                {
                    string(r["sample_id"])
                    for r in predictions[c][split]
                    if r["prediction_m"] is not None
                }
                for c in predictions
            ]
        )
        paired[split] = {
            "common_valid_count": len(common),
            "candidates": {
                c: summarize(
                    [r for r in predictions[c][split] if r["sample_id"] in common], near
                )
                for c in predictions
            },
        }
    source_paths = {**inputs, "collection_plan": plan_path, "protocol": protocol_path}
    report: dict[str, object] = {
        "schema": "e2_comparison_v1",
        "role": "development_candidate_comparison",
        "production_approved": False,
        "metric_evidence": None,
        "selection": "pending_acceptance_protocol_and_independent_test",
        "protocol": protocol,
        "fit_method": (
            "OLS: A1 minimizes squared Z residuals; A2 squared inverse-Z residuals"
        ),
        "extrapolation_policy": "flag_only_no_clipping_no_production_support",
        "near_slice_role": "descriptive_only_not_a_safety_threshold",
        "numpy_version": np.__version__,
        "runner_sha256": digest(Path(__file__)),
        "code_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": digest(path)}
            for name, path in source_paths.items()
        },
        "candidates": candidates,
        "paired_common_valid": paired,
    }
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "inputs"
    snapshot.mkdir()
    (output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    for name, path in source_paths.items():
        (snapshot / f"{name}.json").write_bytes(path.read_bytes())
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main(argv: list[str] | None = None) -> int:
    """显式接收拟合及验证数值报告，不提供测试数据入口或自动部署行为。"""
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("fit", "validation"):
        for kind in ("audit", "relative", "metric"):
            parser.add_argument(f"--{split}-{kind}", type=Path, required=True)
    parser.add_argument(
        "--plan", type=Path, default=ROOT / "configs/e2_collection.json"
    )
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/e2_comparison.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    inputs = {
        f"{split}_{kind}": getattr(args, f"{split}_{kind}")
        for split in ("fit", "validation")
        for kind in ("audit", "relative", "metric")
    }
    try:
        run_comparison(inputs, args.plan, args.protocol, args.output)
    except (ValueError, OSError, KeyError) as exc:
        print(f"Comparison failed: {exc}")
        return 2
    print(args.output / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
