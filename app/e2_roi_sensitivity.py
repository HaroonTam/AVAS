"""固定12帧的相对深度ROI敏感性对照；准备阶段无推理，推理须显式触发。"""

import argparse
import json
import subprocess
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

from app.e2_capture import ROOT, validate_roi
from app.e2_collection import read_object
from app.e2_compare import index_rows
from app.e2_diagnose import rectangle
from app.e2_multiplan import load_sources
from app.e2_offline import digest, mapping, number, read_image, string
from app.fusion.config import FusionConfig, load_fusion_config
from app.fusion.depth_fusion import sampling_region, summarize_depth
from app.vision.depth_estimator import (
    MODEL_ID,
    MODEL_REVISION,
    WEIGHTS_SHA256,
    DepthAnythingV2Estimator,
    RelativeDepth,
    load_depth_config,
    save_relative_depth,
)
from app.vision.detector import select_device


def write_json(path: Path, value: dict[str, object]) -> None:
    """只写入新JSON，不覆盖既有实验结果。"""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def build_manifest(recipe: Path, protocol: Path, config: Path) -> dict[str, object]:
    """完整验证84条来源，再按预定布景/名义距离选全12条，不按误差选图。"""
    rule = read_object(protocol)
    if (
        rule.get("schema") != "e2_roi_sensitivity_protocol_v1"
        or rule.get("groups") != ["scene10", "scene11", "scene12"]
        or rule.get("nominal_distances_m") != [0.5, 3.0]
        or rule.get("repeats_per_bin") != 2
        or rule.get("sample_count") != 12
        or rule.get("variants")
        != ["original_statistics_v1", "center_half_of_statistics_v1"]
        or rule.get("center_fraction") != 0.5
        or any(
            rule.get(k) is not False
            for k in ("candidate_selection", "test_unlocked", "production_approved")
        )
    ):
        raise ValueError("unsupported fixed ROI protocol")
    datasets, inputs = load_sources(read_object(recipe))
    fusion = load_fusion_config(config)
    if fusion.method != "center_region_median" or fusion.inner_fraction != 0.5:
        raise ValueError("expected original center-region configuration")
    audits: dict[str, dict[str, object]] = {}
    relatives: dict[str, dict[str, object]] = {}
    for name, path in inputs.items():
        if name.endswith("_audit"):
            report = read_object(path)
            if report.get("config_sha256") != digest(config):
                raise ValueError("current configuration differs from source")
            if report.get("fusion_config") != asdict(fusion):
                raise ValueError("source fusion configuration differs")
            audits.update(index_rows(report))
        if name.endswith("_relative"):
            relatives.update(index_rows(read_object(path)))
    selected: list[dict[str, object]] = []
    counts: dict[tuple[str, float], int] = {}
    for role, observations in datasets.items():
        for obs in observations:
            if (
                obs.group not in rule["groups"]
                or obs.nominal_m not in rule["nominal_distances_m"]
            ):
                continue
            base = audits[obs.sample_id]
            record_path = Path(string(base["record_path"]))
            record = read_object(record_path)
            width, height = record["actual_width"], record["actual_height"]
            if type(width) is not int or type(height) is not int:
                raise ValueError("invalid source image dimensions")
            analysis = rectangle(base["analysis_roi"], width, height)
            original = rectangle(base["statistics_roi"], width, height)
            if sampling_region(analysis, fusion) != original:
                raise ValueError("frozen ROI differs from production sampling")
            center = sampling_region(original, fusion)
            validate_roi(center, width, height)
            key = (obs.group, obs.nominal_m)
            counts[key] = counts.get(key, 0) + 1
            selected.append(
                {
                    "sample_id": obs.sample_id,
                    "group": obs.group,
                    "placement": obs.placement,
                    "role": role,
                    "nominal_m": obs.nominal_m,
                    "truth_m": obs.truth_m,
                    "record_path": str(record_path),
                    "record_sha256": base["record_sha256"],
                    "image_sha256": base["image_sha256"],
                    "frame_id": record["frame_id"],
                    "image_width": width,
                    "image_height": height,
                    "analysis_roi": list(analysis),
                    "original_roi": list(original),
                    "center_roi": list(center),
                    "historical_summary": relatives[obs.sample_id].get("depth_summary"),
                }
            )
    expected = {(g, d): 2 for g in rule["groups"] for d in rule["nominal_distances_m"]}
    if counts != expected or len(selected) != 12:
        raise ValueError("incomplete fixed selection")
    selected.sort(key=sample_order)
    depth_config = load_depth_config(config)
    files = ("config.json", "preprocessor_config.json", "model.safetensors")
    model_hashes = {name: digest(depth_config.model_dir / name) for name in files}
    if model_hashes["model.safetensors"] != WEIGHTS_SHA256:
        raise ValueError("relative weight identity mismatch")
    return {
        "schema": "e2_roi_sensitivity_manifest_v1",
        "sample_count": 12,
        "recipe_path": str(recipe.resolve()),
        "recipe_sha256": digest(recipe),
        "protocol_path": str(protocol.resolve()),
        "protocol_sha256": digest(protocol),
        "config_path": str(config.resolve()),
        "config_sha256": digest(config),
        "protocol": rule,
        "fusion_config": asdict(fusion),
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "files_sha256": model_hashes,
        },
        "image_size": depth_config.image_size,
        "requested_device": depth_config.device,
        "test_unlocked": False,
        "production_approved": False,
        "samples": selected,
    }


def sample_order(row: dict[str, object]) -> tuple[str, float, str]:
    """按布景、名义距离和摆放号稳定排序，避免文件系统顺序影响执行。"""
    return string(row["group"]), number(row["nominal_m"]), string(row["placement"])


def compare_regions(
    depth: RelativeDepth,
    analysis: tuple[int, int, int, int],
    original: tuple[int, int, int, int],
    center: tuple[int, int, int, int],
    fusion: FusionConfig,
) -> dict[str, object]:
    """在同一原图深度网格对比两区域，缩小框不重复收缩；缺失差值保持未知。"""
    if depth.relative_depth.shape != depth.valid_mask.shape:
        raise ValueError("depth and mask grid mismatch")
    if (
        sampling_region(analysis, fusion) != original
        or sampling_region(original, fusion) != center
    ):
        raise ValueError("ROI derivation mismatch")
    height, width = depth.relative_depth.shape
    for roi in (analysis, original, center):
        validate_roi(roi, width, height)
    baseline = asdict(summarize_depth(analysis, depth, fusion))
    smaller = asdict(
        summarize_depth(center, depth, replace(fusion, method="box_median"))
    )
    x, y = baseline["relative_depth"], smaller["relative_depth"]
    return {
        "original": baseline,
        "center_half": smaller,
        "center_minus_original_relative": None
        if x is None or y is None
        else number(y) - number(x),
    }


def create_estimator(config: Path) -> tuple[DepthAnythingV2Estimator, str]:
    """显式推理时解析实际设备并只加载一次本地固定模型。"""
    import torch

    depth = load_depth_config(config)
    device = select_device(
        depth.device, torch.cuda.is_available(), torch.backends.mps.is_available()
    )
    return DepthAnythingV2Estimator(replace(depth, device=device)), device


def available_pair(row: dict[str, object]) -> bool:
    """只有同帧两区域都产生可用统计且未报错，才计作有效对照。"""
    return "error" not in row and all(
        mapping(row.get(key, {})).get("status") == "available"
        for key in ("original", "center_half")
    )


def infer(manifest: Path, output: Path) -> dict[str, object]:
    """核验冻结清单后推理全帧一次并保存浮点图，逐条保留失败；不拟合或选型。"""
    if output.exists():
        raise FileExistsError("output already exists")
    frozen = read_object(manifest)
    config = Path(string(frozen["config_path"]))
    expected = build_manifest(
        Path(string(frozen["recipe_path"])),
        Path(string(frozen["protocol_path"])),
        config,
    )
    if frozen != expected:
        raise ValueError("manifest or input identity changed; prepare a new experiment")
    estimator, device = create_estimator(config)
    fusion = load_fusion_config(config)
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_bytes(manifest.read_bytes())
    (output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    rows: list[dict[str, object]] = []
    for value in frozen["samples"]:
        source = mapping(value)
        row: dict[str, object] = {**source, "metric_evidence": None}
        try:
            image = read_image(Path(string(source["record_path"])).parent / "frame.png")
            height, width = image.shape[:2]
            if (width, height) != (source["image_width"], source["image_height"]):
                raise ValueError("source image grid mismatch")
            depth = estimator.estimate(image, string(source["frame_id"]))
            if (
                depth.frame_id != source["frame_id"]
                or depth.relative_depth.shape != image.shape[:2]
            ):
                raise ValueError("depth frame identity or grid mismatch")
            result = compare_regions(
                depth,
                rectangle(source["analysis_roi"], width, height),
                rectangle(source["original_roi"], width, height),
                rectangle(source["center_roi"], width, height),
                fusion,
            )
            row.update(result)
            history = source["historical_summary"]
            old = None if history is None else mapping(history).get("relative_depth")
            current = mapping(result["original"])["relative_depth"]
            row["rerun_minus_historical_relative"] = (
                None
                if old is None or current is None
                else number(current) - number(old)
            )
            path = output / f"depth_{len(rows) + 1:02}.npz"
            save_relative_depth(depth, path)
            row["depth_file"] = path.name
            row["depth_sha256"] = digest(path)
        except (ValueError, RuntimeError, OSError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    report: dict[str, object] = {
        "schema": "e2_roi_sensitivity_result_v1",
        "sample_count": len(rows),
        "error_count": sum("error" in row for row in rows),
        "available_pair_count": sum(available_pair(row) for row in rows),
        "unavailable_pair_count": sum(not available_pair(row) for row in rows),
        "manifest_sha256": digest(manifest),
        "runner_sha256": digest(Path(__file__)),
        "model": frozen["model"],
        "image_size": frozen["image_size"],
        "device": device,
        "fusion_config": frozen["fusion_config"],
        "package_versions": {
            name: version(name) for name in ("torch", "transformers", "numpy")
        },
        "code_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "role": "exploratory_roi_sensitivity_no_candidate_selection",
        "test_unlocked": False,
        "production_approved": False,
        "metric_evidence": None,
        "samples": rows,
    }
    write_json(output / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    """准备清单不加载模型；只有显式infer模式才运行用户请求的离线推理。"""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--infer", action="store_true")
    parser.add_argument(
        "--recipe", type=Path, default=ROOT / "configs/e2_boundary_comparison.json"
    )
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/e2_roi_sensitivity.json"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/system.ini")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.prepare:
            if args.output.exists():
                raise FileExistsError("output already exists")
            manifest = build_manifest(args.recipe, args.protocol, args.config)
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output / "manifest.json", manifest)
            print(args.output / "manifest.json")
        else:
            if args.manifest is None:
                raise ValueError("--infer requires --manifest")
            result = infer(args.manifest, args.output)
            print(args.output / "report.json")
            if result["unavailable_pair_count"]:
                return 2
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"ROI sensitivity failed: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
