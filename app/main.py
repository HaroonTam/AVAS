"""Offline image detection CLI; stdout contains structured JSON only."""

import argparse
import hashlib
import json
import logging
from configparser import Error as ConfigError
from dataclasses import asdict
from pathlib import Path
from time import perf_counter, time_ns

from app.config import load_config
from app.fusion.config import load_fusion_config
from app.fusion.depth_fusion import DetectionFrame, fuse_frame
from app.safety.config import load_risk_config
from app.safety.risk_engine import assess_scene
from app.safety.scene import offline_scene
from app.vision.detector import Yolo11Detector


def main(argv: list[str] | None = None) -> int:
    """运行离线图片检测；默认配置定位到项目目录，失败时返回状态码 1。"""
    parser = argparse.ArgumentParser(description="YOLO11 offline image detection")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "system.ini",
        help="Config path (default: project configs/system.ini)",
    )
    parser.add_argument("--image", type=Path, required=True, help="Local image path")
    parser.add_argument(
        "--output-image",
        type=Path,
        help="Save annotated PNG/JPEG locally (opt-in; existing files are protected)",
    )
    parser.add_argument(
        "--depth", action="store_true", help="Enable local relative depth"
    )
    parser.add_argument(
        "--output-depth", type=Path, help="Save raw relative depth .npz"
    )
    parser.add_argument(
        "--output-depth-image", type=Path, help="Save relative depth preview"
    )
    args = parser.parse_args(argv)
    if (args.output_depth or args.output_depth_image) and not args.depth:
        parser.error("--output-depth and --output-depth-image require --depth")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config = load_config(args.config)
        fusion_config = load_fusion_config(args.config)
        risk_config = load_risk_config(args.config)
        if not args.image.is_file():
            raise FileNotFoundError(f"Image missing: {args.image}")
        import cv2
        import numpy as np

        # 使用 imdecode 支持包含中文字符的 Windows 图片路径。
        image = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Image cannot be decoded")
        frame_id = hashlib.sha256(image.tobytes()).hexdigest()
        detector = Yolo11Detector(config)
        start = perf_counter()
        detections = detector.detect(image)
        elapsed_ms = (perf_counter() - start) * 1000
        depth_status: dict[str, str | int | float | None] = {"status": "disabled"}
        depth_failed = False
        depth = None
        if args.depth:
            # 可选深度分支失败时仍保留检测结果；未知距离不触发米制判断。
            try:
                from app.vision.depth_estimator import (
                    MODEL_ID,
                    MODEL_REVISION,
                    DepthAnythingV2Estimator,
                    load_depth_config,
                    save_relative_depth,
                )

                depth_config = load_depth_config(args.config)
                depth_model = DepthAnythingV2Estimator(depth_config)
                depth_start = perf_counter()
                depth = depth_model.estimate(image, frame_id)
                depth_elapsed_ms = (perf_counter() - depth_start) * 1000
                depth_status = {
                    "status": "available",
                    "units": "relative_inverse_depth",
                    "larger_values": "closer",
                    "frame_id": frame_id,
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "image_size": depth_config.image_size,
                    "inference_latency_ms": depth_elapsed_ms,
                    "valid_fraction": float(depth.valid_mask.mean()),
                }
            except Exception as exc:
                # 第三方可选模型故障不丢弃已经完成的有效目标检测。
                logging.error("Depth unavailable: %s", exc)
                depth_status = {"status": "unavailable", "reason": str(exc)}
                depth = None
                depth_failed = True
            else:
                try:
                    if args.output_depth:
                        save_relative_depth(depth, args.output_depth)
                    if args.output_depth_image:
                        from app.visualization import (
                            render_relative_depth,
                            save_detections,
                        )

                        save_detections(
                            render_relative_depth(depth), (), args.output_depth_image
                        )
                except (OSError, ValueError, RuntimeError) as exc:
                    logging.error("Depth export failed: %s", exc)
                    depth_status["export_error"] = str(exc)
                    depth_failed = True
        fusion_start = perf_counter()
        frame = DetectionFrame(frame_id, image.shape[1], image.shape[0], detections)
        try:
            observations = fuse_frame(frame, depth, fusion_config)
            fusion_status = "available"
        except ValueError as exc:
            logging.error("Fusion depth rejected: %s", exc)
            depth_status = {"status": "rejected", "reason": str(exc)}
            observations = fuse_frame(frame, None, fusion_config)
            fusion_status = "depth_rejected"
            depth_failed = True
        fusion_elapsed_ms = (perf_counter() - fusion_start) * 1000
        risk_start = perf_counter()
        assessment = assess_scene(
            offline_scene(frame_id, observations), time_ns() // 1_000_000, risk_config
        )
        risk_elapsed_ms = (perf_counter() - risk_start) * 1000
        if args.output_image is not None:
            from app.visualization import save_detections

            output_path = save_detections(image, detections, args.output_image)
            logging.info("Annotated image saved: %s", output_path)
        print(
            json.dumps(
                {
                    "source_kind": "offline_image",
                    "current_scene": False,
                    "frame_id": frame_id,
                    "capture_timestamp_ms": None,
                    "width": image.shape[1],
                    "height": image.shape[0],
                    "detection_latency_ms": elapsed_ms,
                    "supported_labels": detector.supported_labels,
                    "detections": [asdict(item) for item in detections],
                    "distance_status": "metric_unavailable",
                    "risk_level": assessment.level,
                    "safety": {
                        "assessment": asdict(assessment),
                        "latency_ms": risk_elapsed_ms,
                        "config": asdict(risk_config),
                        "thresholds_validated": False,
                        "speech_output": "not_implemented",
                    },
                    "depth": depth_status,
                    "fusion": {
                        "status": fusion_status,
                        "latency_ms": fusion_elapsed_ms,
                        "config": asdict(fusion_config),
                        "coordinate_space": "original_image",
                        "objects": [asdict(item) for item in observations],
                    },
                },
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (
        OSError,
        ValueError,
        RuntimeError,
        ImportError,
        ConfigError,
        KeyError,
    ) as exc:
        logging.error("Detection unavailable: %s", exc)
        return 1
    return 2 if depth_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
