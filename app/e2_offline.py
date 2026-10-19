"""E2 本地数据审查与显式相对深度诊断；不签发米制证据。"""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from app.e2_capture import ROOT, Measurement, validate_roi
from app.fusion.config import FusionConfig, load_fusion_config
from app.fusion.depth_fusion import sampling_region, summarize_depth
from app.vision.depth_estimator import (
    MODEL_ID,
    MODEL_REVISION,
    WEIGHTS_SHA256,
    DepthAnythingV2Estimator,
    load_depth_config,
)
from app.vision.detector import Image


def digest(path: Path) -> str:
    """计算本地记录或图像哈希，用于冻结输入身份。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mapping(value: object) -> dict[str, object]:
    """收窄 JSON 对象类型，拒绝非字符串键。"""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("expected JSON object")
    return dict(value)


def string(value: object) -> str:
    """读取非空文本字段，不自动把缺失值转换为字符串。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expected nonempty string")
    return value


def number(value: object) -> float:
    """只接受真实数值字段，排除布尔值与非有限值。"""
    if type(value) not in (float, int) or not isinstance(value, (float, int)):
        raise ValueError("expected numeric value")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("expected finite number")
    return result


def read_image(path: Path) -> Image:
    """以原始 BGR 网格解码本地图像；无模型、网络或设备调用。"""
    import cv2

    image: Image | None = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), 1)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return image


@dataclass(frozen=True)
class Sample:
    path: Path
    record_hash: str
    image_path: Path
    image_hash: str
    frame_id: str
    roi: tuple[int, int, int, int]
    measurement: Measurement
    split: str = "development_precapture"
    collection: dict[str, object] | None = None


def load_sample(
    path: Path,
    fusion: FusionConfig,
    config_hash: str,
    expected_split: str = "development_precapture",
) -> Sample:
    """验证预采集身份、配置快照、尺寸、真值及哈希；原记录只读。"""
    raw = mapping(json.loads(path.read_text(encoding="utf-8")))
    if expected_split not in {"development_precapture", "fit", "validation"}:
        raise ValueError("test data locked or unsupported split")
    expected_schema = (
        "e2_board_precapture_v1"
        if expected_split == "development_precapture"
        else "e2_board_collection_v1"
    )
    if (
        raw.get("schema") != expected_schema
        or raw.get("split") != expected_split
        or raw.get("roi_source") != "manual_board_diagnostic_not_yolo"
        or raw.get("sample_id") != path.parent.name
        or raw.get("image") != "frame.png"
    ):
        raise ValueError("unsupported sample identity or protocol")
    if (
        raw.get("fusion_config") != asdict(fusion)
        or raw.get("config_sha256") != config_hash
    ):
        raise ValueError("sample configuration differs from current frozen config")
    image_path = path.parent / "frame.png"
    image_hash = digest(image_path)
    if raw.get("image_sha256") != image_hash:
        raise ValueError("image hash mismatch")
    image = read_image(image_path)
    if image.shape[:2] != (raw.get("actual_height"), raw.get("actual_width")):
        raise ValueError("decoded dimensions differ from record")
    coords = raw.get("roi_xyxy")
    if not isinstance(coords, list) or len(coords) != 4:
        raise ValueError("expected four ROI coordinates")
    if any(type(item) is not int for item in coords):
        raise ValueError("ROI coordinates must be integers")
    roi = (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))
    validate_roi(roi, image.shape[1], image.shape[0])
    measured = mapping(raw.get("measurement"))
    uncertainty = measured.get("uncertainty_m")
    measurement = Measurement(
        number(measured.get("distance_m")),
        None if uncertainty is None else number(uncertainty),
        string(measured.get("group_id")),
        string(measured.get("instrument")),
        string(measured.get("conditions")),
        string(measured.get("installation_id")),
    )
    collection = None
    if expected_split != "development_precapture":
        collection = mapping(raw.get("collection"))
        if (
            collection.get("split") != expected_split
            or collection.get("group_id") != measurement.group_id
            or collection.get("installation_id") != measurement.installation_id
            or collection.get("roi_policy") != "board_lower_half_v1"
        ):
            raise ValueError("collection metadata mismatch")
    return Sample(
        path.resolve(),
        digest(path),
        image_path.resolve(),
        image_hash,
        string(raw.get("frame_id")),
        roi,
        measurement,
        expected_split,
        collection,
    )


def frozen_paths(manifest: Path, root: Path) -> list[Path]:
    """读取审查清单并拒绝路径越界、记录变更与重复条目。"""
    report = mapping(json.loads(manifest.read_text(encoding="utf-8")))
    if report.get("schema") != "e2_offline_v1" or report.get("mode") != "audit":
        raise ValueError("expected audit manifest")
    rows = report.get("samples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty manifest")
    paths: list[Path] = []
    for row in rows:
        entry = mapping(row)
        path = Path(string(entry.get("record_path"))).resolve()
        if not path.is_relative_to(root.resolve()) or path in paths:
            raise ValueError("duplicate or out-of-root record")
        if digest(path) != entry.get("record_sha256"):
            raise ValueError("record changed after audit")
        paths.append(path)
    return paths


def run(
    paths: list[Path],
    output: Path,
    config: Path,
    *,
    infer_relative: bool,
    board_lower_half: bool = False,
    expected_split: str = "development_precapture",
) -> dict[str, object]:
    """先完整验证输入，再可选单次加载模型；诊断失败逐条保留，不拟合或选型。"""
    if output.exists():
        raise FileExistsError("output directory already exists")
    fusion = load_fusion_config(config)
    samples = [
        load_sample(path, fusion, digest(config), expected_split) for path in paths
    ]
    if not samples:
        raise ValueError("no samples")
    if len({sample.image_hash for sample in samples}) != len(samples):
        raise ValueError("duplicate images; inspect before proceeding")
    if len({sample.measurement.installation_id for sample in samples}) != 1:
        raise ValueError("mixed installations")
    if expected_split != "development_precapture":
        identities = {
            (
                string(s.collection.get("collection_id")),
                string(s.collection.get("plan_sha256")),
            )
            for s in samples
            if s.collection is not None
        }
        if len(identities) != 1 or not board_lower_half:
            raise ValueError("mixed collection plans or incompatible ROI policy")
    output.mkdir(parents=True, exist_ok=False)
    estimator = (
        DepthAnythingV2Estimator(load_depth_config(config)) if infer_relative else None
    )
    rows: list[dict[str, object]] = []
    for sample in samples:
        x1, y1, x2, y2 = sample.roi
        analysis_roi = (
            (x1, (y1 + y2 + 1) // 2, x2, y2) if board_lower_half else sample.roi
        )
        image = read_image(sample.image_path)
        validate_roi(analysis_roi, image.shape[1], image.shape[0])
        inner_roi = sampling_region(analysis_roi, fusion)
        validate_roi(inner_roi, image.shape[1], image.shape[0])
        row: dict[str, object] = {
            "sample_id": sample.path.parent.name,
            "record_path": str(sample.path),
            "record_sha256": sample.record_hash,
            "image_sha256": sample.image_hash,
            "measurement": asdict(sample.measurement),
            "collection": sample.collection,
            "roi_xyxy": sample.roi,
            "analysis_roi": analysis_roi,
            "statistics_roi": inner_roi,
            "metric_evidence": None,
        }
        if estimator is not None:
            try:
                depth = estimator.estimate(image, sample.frame_id)
                row["depth_summary"] = asdict(
                    summarize_depth(analysis_roi, depth, fusion)
                )
            except (ValueError, RuntimeError, OSError) as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    report: dict[str, object] = {
        "schema": "e2_offline_v1",
        "roi_policy": "board_lower_half_v1" if board_lower_half else "original_v1",
        "code_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "runner_sha256": digest(Path(__file__)),
        "mode": "relative" if infer_relative else "audit",
        "sample_count": len(rows),
        "config_sha256": digest(config),
        "fusion_config": asdict(fusion),
        "split": expected_split,
        "dataset_role": "development_not_independent_test",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "sha256": WEIGHTS_SHA256}
        if infer_relative
        else None,
        "samples": rows,
    }
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main(argv: list[str] | None = None) -> int:
    """默认只审查本地数据；推理必须显式开启并使用冻结审查清单。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/system.ini")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-sample", action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--infer-relative", action="store_true")
    parser.add_argument("--board-lower-half", action="store_true")
    parser.add_argument(
        "--split", choices=("development_precapture", "fit", "validation")
    )
    parser.add_argument("--collection-id")
    args = parser.parse_args(argv)
    if args.infer_relative and args.manifest is None:
        parser.error("--infer-relative requires a frozen --manifest")
    if args.manifest is not None and (args.exclude_sample or args.board_lower_half):
        parser.error("cannot change exclusions after freezing manifest")
    if args.manifest and (args.split or args.collection_id):
        parser.error("split and collection must come from frozen manifest")
    try:
        if args.manifest:
            paths = frozen_paths(args.manifest, args.root)
            frozen = mapping(json.loads(args.manifest.read_text(encoding="utf-8")))
            if frozen.get("roi_policy") not in {"original_v1", "board_lower_half_v1"}:
                raise ValueError("unknown frozen ROI policy")
            lower_half = frozen.get("roi_policy") == "board_lower_half_v1"
            expected_split = string(frozen.get("split", "development_precapture"))
        else:
            expected_split = args.split or "development_precapture"
            if expected_split != "development_precapture" and not args.collection_id:
                raise ValueError("formal audit requires --collection-id")
            lower_half = args.board_lower_half
            candidates = sorted(args.root.glob("e2_*/record.json"))
            names = {path.parent.name for path in candidates}
            if set(args.exclude_sample) - names:
                raise ValueError("excluded sample does not exist")
            paths = [p for p in candidates if p.parent.name not in args.exclude_sample]
            selected: list[Path] = []
            for path in paths:
                metadata = mapping(json.loads(path.read_text(encoding="utf-8")))
                if metadata.get("split") != expected_split:
                    continue
                if args.collection_id:
                    context = mapping(metadata.get("collection"))
                    if context.get("collection_id") != args.collection_id:
                        continue
                selected.append(path)
            paths = selected
        report = run(
            paths,
            args.output,
            args.config,
            infer_relative=args.infer_relative,
            board_lower_half=lower_half,
            expected_split=expected_split,
        )
        print(args.output / "report.json")
        rows = report["samples"]
        assert isinstance(rows, list)
        return 2 if any("error" in mapping(row) for row in rows) else 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"E2 offline failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
