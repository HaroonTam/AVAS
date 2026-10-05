"""本地离线图像重复处理基准；不评估精度、摄像头 FPS 或可听告警延迟。"""

import argparse
import hashlib
import json
import logging
import platform
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np

from app.config import load_config
from app.fusion.config import FusionConfig, load_fusion_config
from app.fusion.depth_fusion import DetectionFrame, fuse_frame
from app.live import DepthEstimator, Detector
from app.safety.config import RiskConfig, load_risk_config
from app.safety.risk_engine import assess_scene
from app.safety.scene import offline_scene
from app.vision.depth_estimator import DepthAnythingV2Estimator, load_depth_config
from app.vision.detector import Image, Yolo11Detector, select_device


@dataclass(frozen=True)
class TimingSample:
    detection_ms: float
    depth_ms: float | None
    fusion_ms: float
    offline_validation_ms: float
    total_ms: float
    object_count: int


@dataclass(frozen=True)
class TimingSummary:
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


def summarize(values: tuple[float, ...]) -> TimingSummary:
    """统计有限非负毫秒样本，p95 使用线性插值；空或非法样本拒绝。"""
    if not values or any(not isfinite(value) or value < 0 for value in values):
        raise ValueError("timings must be nonempty, finite and nonnegative")
    return TimingSummary(
        float(np.mean(values)),
        float(np.median(values)),
        float(np.percentile(values, 95, method="linear")),
        min(values),
        max(values),
    )


def measure_once(
    image: Image,
    frame_id: str,
    detector: Detector,
    depth_model: DepthEstimator | None,
    fusion: FusionConfig,
    risk: RiskConfig,
    clock: Callable[[], float] = perf_counter,
) -> TimingSample:
    """顺序计时到 CPU 结构化结果返回；离线风险拒绝路径不等于实时告警评估。"""
    start = clock()
    detections = detector.detect(image)
    detected = clock()
    depth = depth_model.estimate(image, frame_id) if depth_model is not None else None
    estimated = clock()
    observations = fuse_frame(
        DetectionFrame(frame_id, image.shape[1], image.shape[0], detections),
        depth,
        fusion,
    )
    fused = clock()
    assessment = assess_scene(offline_scene(frame_id, observations), 0, risk)
    ended = clock()
    if assessment.status != "offline_not_current" or assessment.events:
        raise RuntimeError("offline benchmark must not emit live warnings")
    durations = (
        detected - start,
        estimated - detected,
        fused - estimated,
        ended - fused,
        ended - start,
    )
    if any(not isfinite(value) or value < 0 for value in durations):
        raise ValueError("benchmark clock must be finite and monotonic")
    return TimingSample(
        durations[0] * 1000,
        durations[1] * 1000 if depth_model is not None else None,
        durations[2] * 1000,
        durations[3] * 1000,
        durations[4] * 1000,
        len(observations),
    )


def benchmark(
    image: Image,
    frame_id: str,
    detector: Detector,
    depth_model: DepthEstimator | None,
    fusion: FusionConfig,
    risk: RiskConfig,
    *,
    warmup: int,
    iterations: int,
) -> tuple[TimingSample, ...]:
    """复用模型执行预热与有限次正式采样；任一失败中止，禁止混入降级样本。"""
    if type(warmup) is not int or not 0 <= warmup <= 100:
        raise ValueError("warmup must be an integer within [0, 100]")
    if type(iterations) is not int or not 1 <= iterations <= 1000:
        raise ValueError("iterations must be an integer within [1, 1000]")
    samples: list[TimingSample] = []
    for index in range(warmup + iterations):
        sample = measure_once(image, frame_id, detector, depth_model, fusion, risk)
        if index >= warmup:
            samples.append(sample)
    return tuple(samples)


def summarize_samples(samples: tuple[TimingSample, ...]) -> dict[str, TimingSummary]:
    """按阶段汇总正式采样；未启用深度时不生成虚假的零耗时统计。"""
    summary = {
        "detection": summarize(tuple(item.detection_ms for item in samples)),
        "fusion": summarize(tuple(item.fusion_ms for item in samples)),
        "offline_validation": summarize(
            tuple(item.offline_validation_ms for item in samples)
        ),
        "total": summarize(tuple(item.total_ms for item in samples)),
    }
    depth = tuple(item.depth_ms for item in samples if item.depth_ms is not None)
    if depth:
        if len(depth) != len(samples):
            raise ValueError("cannot mix depth-enabled and depth-disabled samples")
        summary["depth"] = summarize(depth)
    return summary


def file_sha256(path: Path) -> str:
    """分块计算输入与权重身份，不把文件内容写入报告。"""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    """记录本地安装版本，不联网或要求可选依赖必须存在。"""
    result: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("numpy", "torch", "ultralytics", "transformers", "opencv-python"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def code_identity(root: Path) -> dict[str, object]:
    """记录 Git 修订及应用源码哈希；不读取环境文件或个人 IDE 配置。"""
    revision: str | None = None
    dirty: bool | None = None
    try:
        revision = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            .decode()
            .strip()
        )
        dirty = bool(
            subprocess.check_output(
                [
                    "git",
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                    "--",
                    "app",
                    "configs",
                    "pyproject.toml",
                    "uv.lock",
                ],
                cwd=root,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
        )
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "revision": revision,
        "code_dirty": dirty,
        "app_sha256": {
            path.relative_to(root).as_posix(): file_sha256(path)
            for path in sorted((root / "app").rglob("*.py"))
        },
    }


def json_path(value: object) -> str:
    """只转换配置中的 pathlib 路径；其他未知类型拒绝序列化。"""
    if isinstance(value, Path):
        return str(value)
    raise TypeError("unsupported report value")


def save_report(report: dict[str, object], destination: Path) -> None:
    """序列化后以独占模式保存 JSON；禁止覆盖任何已有路径。"""
    if destination.suffix.lower() != ".json":
        raise ValueError("benchmark output must use .json")
    payload = json.dumps(
        report, default=json_path, ensure_ascii=True, allow_nan=False, indent=2
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        output.write(payload + "\n")


def main(argv: list[str] | None = None) -> int:
    """显式对本地图片执行模型基准，输出不含原图、深度数组或目标事实。"""
    parser = argparse.ArgumentParser(
        description="Offline repeated-image latency benchmark"
    )
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", type=Path, default=root / "configs/system.ini")
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        required=True,
        help="Repeat for multiple images; processed in argument order",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args(argv)
    if not 0 <= args.warmup <= 100 or not 1 <= args.iterations <= 1000:
        parser.error("warmup must be 0..100 and iterations 1..1000")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError("benchmark destination already exists")
        if args.output.suffix.lower() != ".json":
            raise ValueError("benchmark output must use .json")
        detection = load_config(args.config)
        fusion, risk = load_fusion_config(args.config), load_risk_config(args.config)
        depth_config = load_depth_config(args.config) if args.depth else None
        import cv2
        import torch

        paths = tuple(path.resolve() for path in args.image)
        if not 1 <= len(paths) <= 100 or len(set(paths)) != len(paths):
            raise ValueError("provide 1..100 distinct image paths")
        inputs: list[tuple[Path, Image, str, str]] = []
        for path in paths:
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"benchmark image cannot be decoded: {path.name}")
            inputs.append(
                (
                    path,
                    image,
                    file_sha256(path),
                    hashlib.sha256(image.tobytes()).hexdigest(),
                )
            )
        cuda, mps = torch.cuda.is_available(), torch.backends.mps.is_available()
        detection_device = select_device(detection.device, cuda, mps)
        depth_device = (
            select_device(depth_config.device, cuda, mps) if depth_config else None
        )
        np.random.seed(0)
        torch.manual_seed(0)
        detector = Yolo11Detector(replace(detection, device=detection_device))
        depth_model = (
            DepthAnythingV2Estimator(replace(depth_config, device=depth_device))
            if depth_config is not None and depth_device is not None
            else None
        )
        weights = {"yolo11": file_sha256(detection.weights)}
        if depth_config is not None:
            for name in (
                "model.safetensors",
                "config.json",
                "preprocessor_config.json",
            ):
                weights[f"depth/{name}"] = file_sha256(depth_config.model_dir / name)
        measured_at = datetime.now(timezone.utc).isoformat()
        image_reports: list[dict[str, object]] = []
        all_samples: list[TimingSample] = []
        for image_index, (path, image, image_hash, frame_id) in enumerate(inputs):
            image_started = datetime.now(timezone.utc).isoformat()
            image_samples = benchmark(
                image,
                frame_id,
                detector,
                depth_model,
                fusion,
                risk,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            image_summary = summarize_samples(image_samples)
            image_reports.append(
                {
                    "image_index": image_index,
                    "image_name": path.name,
                    "image_file_sha256": image_hash,
                    "frame_id": frame_id,
                    "width": image.shape[1],
                    "height": image.shape[0],
                    "measurement_started_at_utc": image_started,
                    "samples": [asdict(sample) for sample in image_samples],
                    "summary_ms": {
                        name: asdict(value) for name, value in image_summary.items()
                    },
                }
            )
            all_samples.extend(image_samples)
        samples = tuple(all_samples)
        summary = summarize_samples(samples)
        report: dict[str, object] = {
            "schema_version": 1,
            "protocol": "offline_repeated_image_v1",
            "measurement_started_at_utc": measured_at,
            "current_scene": False,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "seed": 0,
            "configs": {
                "detection": asdict(detection),
                "depth": asdict(depth_config) if depth_config else None,
                "fusion": asdict(fusion),
                "risk": asdict(risk),
            },
            "selected_devices": {"detection": detection_device, "depth": depth_device},
            "weight_sha256": weights,
            "versions": package_versions(),
            "code": code_identity(root),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch_threads": torch.get_num_threads(),
            "accelerator_name": torch.cuda.get_device_name()
            if "cuda" in (detection_device, depth_device)
            else None,
            "samples": [asdict(sample) for sample in samples],
            "summary_ms": {name: asdict(value) for name, value in summary.items()},
            "serial_processing_rate_hz": 1000 / summary["total"].mean_ms
            if summary["total"].mean_ms > 0
            else None,
            "unmeasured": [
                "accuracy",
                "metric_distance_error",
                "camera_fps",
                "warning_latency",
                "audio_latency",
                "memory",
            ],
        }
        if len(inputs) == 1:
            for key in (
                "image_name",
                "image_file_sha256",
                "frame_id",
                "width",
                "height",
            ):
                report[key] = image_reports[0][key]
        else:
            report.update(
                {
                    "schema_version": 2,
                    "protocol": "offline_multi_image_v1",
                    "images": image_reports,
                    "image_count": len(inputs),
                    "sample_count": len(samples),
                    "sampling_order": "image_major_argument_order",
                    "warmup_scope": "per_image_before_its_samples",
                    "iterations_scope": "per_image",
                    "aggregation": "pooled_samples_equal_weight_per_image",
                }
            )
            # 多图逐次样本保存在各图片条目中，避免重复存储及丢失归属。
            del report["samples"]
        save_report(report, args.output)
        print(
            json.dumps(
                {
                    "report": str(args.output.resolve()),
                    "iterations": len(samples),
                    "mean_total_ms": summary["total"].mean_ms,
                },
                ensure_ascii=True,
            )
        )
    except Exception as exc:
        logging.error("Benchmark failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
