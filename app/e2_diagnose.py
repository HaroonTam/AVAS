"""已有 E2 数据的布景诊断；不运行模型，不生成新标定候选。"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from app.e2_capture import ROOT, validate_roi
from app.e2_collection import read_object
from app.e2_compare import fit_calibration, index_rows, predict
from app.e2_multiplan import load_sources
from app.e2_offline import digest, mapping, number, read_image, string
from app.vision.detector import Image


def rectangle(value: object, width: int, height: int) -> tuple[int, int, int, int]:
    """检查保存的像素矩形，拒绝空区域、浮点坐标及画面外坐标。"""
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("expected four ROI coordinates")
    if any(type(v) is not int for v in value):
        raise ValueError("expected integer ROI coordinates")
    roi = (int(value[0]), int(value[1]), int(value[2]), int(value[3]))
    validate_roi(roi, width, height)
    return roi


def brightness(image: Image) -> dict[str, float]:
    """计算 BGR 编码亮度代理；不是照度、曝光参数或反光判定。"""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected uint8 BGR image")
    if image.size == 0:
        raise ValueError("empty image")
    gray = image.astype(np.float64) @ np.asarray([0.114, 0.587, 0.299])
    return {
        "gray_mean": float(np.mean(gray)),
        "gray_p95": float(np.quantile(gray, 0.95)),
        "gray_ge250_fraction": float(np.mean(gray >= 250)),
        "gray_le5_fraction": float(np.mean(gray <= 5)),
    }


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    """返回描述性统计；空值集合保留缺失，不把无观测写成零。"""
    if not values:
        return {"n": 0, "mean": None, "min": None, "max": None, "span": None}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "min": min(values),
        "max": max(values),
        "span": max(values) - min(values),
    }


def describe(rows: list[dict[str, object]]) -> dict[str, object]:
    """按布景或距离档描述原始数值，不改变可用性或选择低误差样本。"""
    fields = (
        "truth_m",
        "relative",
        "relative_iqr",
        "A1_error_m",
        "B0_error_m",
        "roi_width_px",
        "roi_height_px",
        "statistics_area_px",
        "frame_gray_mean",
        "statistics_gray_mean",
        "statistics_gray_ge250_fraction",
    )
    return {
        "count": len(rows),
        "border_touch_count": sum(bool(r["roi_touches_border"]) for r in rows),
        **{
            key: numeric_summary([number(r[key]) for r in rows if r[key] is not None])
            for key in fields
        },
        "A1_mae_m": numeric_summary(
            [abs(number(r["A1_error_m"])) for r in rows if r["A1_error_m"] is not None]
        )["mean"],
    }


def run_diagnosis(recipe_path: Path, output: Path) -> dict[str, object]:
    """核验冻结来源后分析像素及现有公式；保存逐样本结果和分档统计，拒绝覆盖。"""
    if output.exists():
        raise ValueError("output already exists")
    recipe = read_object(recipe_path)
    datasets, snapshots = load_sources(recipe)
    fit = datasets["original_fit"] + datasets["supplemental_fit"]
    calibration = fit_calibration(fit, "A1")
    audits: dict[str, dict[str, object]] = {}
    relatives: dict[str, dict[str, object]] = {}
    for key, path in snapshots.items():
        if key.endswith("_audit"):
            audits.update(index_rows(read_object(path)))
        if key.endswith("_relative"):
            relatives.update(index_rows(read_object(path)))
    rows: list[dict[str, object]] = []
    for role, observations in datasets.items():
        for obs in observations:
            audit = audits[obs.sample_id]
            frame = Path(string(audit["record_path"])).parent / "frame.png"
            image = read_image(frame)
            height, width = image.shape[:2]
            roi = rectangle(audit["roi_xyxy"], width, height)
            stats = rectangle(audit["statistics_roi"], width, height)
            x1, y1, x2, y2 = stats
            frame_light = brightness(image)
            stats_light = brightness(image[y1:y2, x1:x2])
            relative_summary = relatives[obs.sample_id].get("depth_summary")
            iqr = (
                mapping(relative_summary).get("relative_depth_iqr")
                if relative_summary is not None
                else None
            )
            prediction = predict(
                obs, calibration, number(recipe["inverse_denominator_epsilon_per_m"])
            )
            rows.append(
                {
                    **asdict(obs),
                    "role": role,
                    "frame_path": str(frame),
                    "conditions": mapping(audit["measurement"]).get("conditions"),
                    "roi_xyxy": roi,
                    "statistics_roi": stats,
                    "roi_width_px": roi[2] - roi[0],
                    "roi_height_px": roi[3] - roi[1],
                    "roi_touches_border": roi[0] == 0
                    or roi[1] == 0
                    or roi[2] == width
                    or roi[3] == height,
                    "statistics_area_px": (x2 - x1) * (y2 - y1),
                    "relative_iqr": iqr,
                    "A1_prediction_m": prediction["prediction_m"],
                    "A1_error_m": prediction["error_m"],
                    "B0_error_m": None
                    if obs.metric_m is None
                    else obs.metric_m - obs.truth_m,
                    **{f"frame_{k}": v for k, v in frame_light.items()},
                    **{f"statistics_{k}": v for k, v in stats_light.items()},
                }
            )
    groups: dict[str, object] = {}
    for group in sorted({string(r["group"]) for r in rows}):
        selected = [r for r in rows if r["group"] == group]
        groups[group] = {
            "role": selected[0]["role"],
            "overall": describe(selected),
            "by_nominal_m": {
                str(distance): describe(
                    [r for r in selected if r["nominal_m"] == distance]
                )
                for distance in sorted({number(r["nominal_m"]) for r in selected})
            },
        }
    report: dict[str, object] = {
        "schema": "e2_scene_diagnosis_v1",
        "sample_count": len(rows),
        "role": "exploratory_descriptive_not_causal",
        "test_unlocked": False,
        "production_approved": False,
        "metric_evidence": None,
        "A1_calibration": asdict(calibration),
        "brightness_definition": (
            "0.114 B + 0.587 G + 0.299 R on uint8 encoded pixels; not illuminance"
        ),
        "errors_definition": (
            "raw available prediction minus measured truth; no range rejection"
        ),
        "numpy_version": np.__version__,
        "source_sha256": digest(Path(__file__)),
        "recipe_sha256": digest(recipe_path),
        "input_hashes": {name: digest(path) for name, path in snapshots.items()},
        "groups": groups,
        "samples": rows,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "diagnosis_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (output / "recipe.json").write_bytes(recipe_path.read_bytes())
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report


def main() -> int:
    """提供纯离线诊断入口，失败返回退出码2。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe", type=Path, default=ROOT / "configs/e2_boundary_comparison.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run_diagnosis(args.recipe, args.output)
    except (ValueError, OSError, KeyError) as exc:
        print(f"Diagnosis failed: {exc}")
        return 2
    print(args.output / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
