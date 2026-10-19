"""用户显式运行的 E2 平面板预采集；不加载模型，不生成米制证据。"""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from time import monotonic
from uuid import uuid4

from app.camera.camera_service import CameraConfig, CameraService, CapturedFrame
from app.e2_collection import Collection, Target, prepare_collection
from app.fusion.config import load_fusion_config

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Measurement:
    distance_m: float
    uncertainty_m: float | None
    group_id: str
    instrument: str
    conditions: str
    installation_id: str

    def __post_init__(self) -> None:
        """验证真值为正有限米数；未知不确定度保留为空，不伪造仪器精度。"""
        if not isfinite(self.distance_m) or self.distance_m <= 0:
            raise ValueError("distance must be positive and finite")
        if self.uncertainty_m is not None and (
            not isfinite(self.uncertainty_m) or self.uncertainty_m < 0
        ):
            raise ValueError("uncertainty must be finite and nonnegative")
        for value in (
            self.group_id,
            self.instrument,
            self.conditions,
            self.installation_id,
        ):
            if not value.strip():
                raise ValueError("measurement descriptions must not be empty")


def validate_roi(roi: tuple[int, int, int, int], width: int, height: int) -> None:
    """验证原图半开区间板面框，拒绝空框、越界和非整数坐标。"""
    x1, y1, x2, y2 = roi
    if any(type(value) is not int for value in roi) or not (
        0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
    ):
        raise ValueError("invalid board ROI")


def capture_frame(config: CameraConfig) -> CapturedFrame:
    """有界采集并丢弃首包；仅返回近期帧，退出前释放相机，无模型或语音。"""
    camera = CameraService(config)
    try:
        camera.start()
        camera.read()
        deadline = monotonic() + 5.0
        while monotonic() < deadline:
            frame = camera.read()
            if 0 <= monotonic() - frame.captured_monotonic <= 1.0:
                return frame
        raise TimeoutError("no fresh frame within capture budget")
    finally:
        camera.close()


def save_sample(
    directory: Path,
    frame: CapturedFrame,
    roi: tuple[int, int, int, int],
    measurement: Measurement,
    config: CameraConfig,
    config_path: Path,
    collection: Collection | None = None,
) -> Path:
    """独占创建样本目录，保存无损原图与完整记录；失败残留不构成有效样本。"""
    import cv2

    validate_roi(roi, frame.image.shape[1], frame.image.shape[0])
    if collection is not None and (
        collection.group_id != measurement.group_id
        or collection.installation_id != measurement.installation_id
    ):
        raise ValueError("collection differs from measurement")
    fusion = asdict(load_fusion_config(config_path))
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    ok, encoded = cv2.imencode(".png", frame.image)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    data = encoded.tobytes()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "frame.png").write_bytes(data)
    record = {
        "schema": "e2_board_collection_v1" if collection else "e2_board_precapture_v1",
        "sample_id": directory.name,
        "split": collection.split if collection else "development_precapture",
        "collection": asdict(collection) if collection else None,
        "measurement": asdict(measurement),
        "distance_definition": "approximate_optical_axis_Z_to_front_board_plane",
        "reference": "ground_projection_of_lens_housing_to_board_front",
        "reference_uncertainty": "housing_to_optical_center_offset_not_calibrated",
        "installation": {
            "computer": "OMEN by HP Gaming Laptop 16-wf0xxx",
            "camera": "built_in_index_0",
            "height_m_user_reported": 0.265,
            "orientation_user_reported": "screen_vertical_lens_horizontal_forward",
            "intrinsics": None,
            "mirroring_verified": False,
        },
        "camera_request": asdict(config),
        "actual_width": int(frame.image.shape[1]),
        "actual_height": int(frame.image.shape[0]),
        "frame_id": frame.frame_id,
        "captured_at_ms": frame.captured_at_ms,
        "image": "frame.png",
        "image_sha256": hashlib.sha256(data).hexdigest(),
        "roi_xyxy": roi,
        "roi_source": "manual_board_diagnostic_not_yolo",
        "fusion_config": fusion,
        "config_sha256": config_hash,
        "code_revision": revision,
        "working_tree_dirty": dirty,
        "opencv_version": cv2.__version__,
        "model_inference_performed": False,
        "metric_evidence": None,
    }
    path = directory / "record.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return path


def main(argv: list[str] | None = None) -> int:
    """显式保存许可与真值参数齐全后采一帧；人工选框确认后才落盘。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-local", action="store_true", required=True)
    parser.add_argument("--distance-m", type=float, required=True)
    parser.add_argument("--uncertainty-m", type=float)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--conditions", required=True)
    parser.add_argument("--installation-id", required=True)
    parser.add_argument("--collection-plan", type=Path)
    parser.add_argument("--placement-id")
    parser.add_argument("--target-id")
    parser.add_argument("--target-material")
    parser.add_argument("--target-width-m", type=float)
    parser.add_argument("--target-height-m", type=float)
    args = parser.parse_args(argv)
    target_fields = (
        args.placement_id,
        args.target_id,
        args.target_material,
        args.target_width_m,
        args.target_height_m,
    )
    if args.collection_plan is not None and any(x is None for x in target_fields):
        parser.error("collection plan requires placement id and all target fields")
    if args.collection_plan is None and any(x is not None for x in target_fields):
        parser.error("target fields require --collection-plan")
    try:
        measurement = Measurement(
            args.distance_m,
            args.uncertainty_m,
            args.group_id,
            args.instrument,
            args.conditions,
            args.installation_id,
        )
        collection = None
        if args.collection_plan is not None:
            collection = prepare_collection(
                args.collection_plan,
                args.group_id,
                args.installation_id,
                args.placement_id,
                Target(
                    args.target_id,
                    args.target_material,
                    args.target_width_m,
                    args.target_height_m,
                ),
                ROOT / "outputs",
            )
        config = CameraConfig(startup_timeout_s=15.0)
        frame = capture_frame(config)
        import cv2

        try:
            x, y, width, height = cv2.selectROI(
                "E2 board ROI: Enter accept / C cancel",
                frame.image,
                showCrosshair=True,
                fromCenter=False,
            )
        finally:
            cv2.destroyAllWindows()
        roi = (int(x), int(y), int(x + width), int(y + height))
        validate_roi(roi, frame.image.shape[1], frame.image.shape[0])
        print(f"Actual size: {frame.image.shape[1]}x{frame.image.shape[0]}")
        print("Confirm board surface, measured distance and fixed installation.")
        if input("Save locally? Type yes: ").strip().lower() != "yes":
            print("Cancelled; no image saved.")
            return 1
        path = save_sample(
            ROOT / "outputs" / f"e2_{uuid4().hex}",
            frame,
            roi,
            measurement,
            config,
            ROOT / "configs" / "system.ini",
            collection,
        )
        print(path)
        return 0
    except (ValueError, RuntimeError, TimeoutError, OSError) as exc:
        print(f"Capture failed: {exc}")
        return 2
    except (KeyboardInterrupt, EOFError):
        print("Cancelled.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
