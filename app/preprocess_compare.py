"""PIL 与 Torchvision 的离线对照；只实验，不切换生产默认处理器。"""

import argparse
import hashlib
import json
import logging
import platform
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from app.benchmark import (
    code_identity,
    file_sha256,
    package_versions,
    save_report,
    summarize,
)
from app.vision.depth_estimator import (
    DepthAnythingV2Estimator,
    DepthConfig,
    TransformersDepthBackend,
    load_depth_config,
)
from app.vision.detector import Image, select_device

Variant = Literal["pil", "torchvision"]
Pixels = NDArray[np.float32]


def compare_values(reference: Pixels, candidate: Pixels) -> dict[str, object]:
    """比较同形浮点值和有限掩膜；差异是实现间偏差而非精度或米制误差。"""
    result: dict[str, object] = {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "exact_equal": False,
    }
    if reference.shape != candidate.shape:
        return {
            **result,
            "status": "shape_mismatch",
            "mean_abs_difference": None,
            "max_abs_difference": None,
            "mask_mismatch_count": None,
            "common_valid_count": None,
        }
    reference_mask, candidate_mask = np.isfinite(reference), np.isfinite(candidate)
    common = reference_mask & candidate_mask
    mismatch = int(np.count_nonzero(reference_mask != candidate_mask))
    count = int(np.count_nonzero(common))
    result.update({"mask_mismatch_count": mismatch, "common_valid_count": count})
    if count == 0:
        return {
            **result,
            "status": "no_common_valid_values",
            "mean_abs_difference": None,
            "max_abs_difference": None,
        }
    difference = np.abs(
        reference[common].astype(np.float64) - candidate[common].astype(np.float64)
    )
    return {
        **result,
        "status": "compared",
        "mean_abs_difference": float(difference.mean()),
        "max_abs_difference": float(difference.max()),
        "exact_equal": mismatch == 0
        and reference.dtype == candidate.dtype
        and bool(np.all(difference == 0)),
    }


def paired_timings(
    run: Callable[[Variant], Pixels],
    *,
    warmup: int,
    iterations: int,
    clock: Callable[[], float] = perf_counter,
) -> tuple[dict[str, float], ...]:
    """先交替预热，再按正式序号交替先后顺序计时；失败中止，不缓存预处理输出。"""
    if type(warmup) is not int or not 0 <= warmup <= 100:
        raise ValueError("warmup must be 0..100")
    if type(iterations) is not int or not 1 <= iterations <= 1000:
        raise ValueError("iterations must be 1..1000")
    samples: list[dict[str, float]] = []
    for phase, count in (("warmup", warmup), ("formal", iterations)):
        for index in range(count):
            order: tuple[Variant, Variant] = (
                ("pil", "torchvision") if index % 2 == 0 else ("torchvision", "pil")
            )
            values: dict[str, float] = {}
            for variant in order:
                start = clock()
                run(variant)
                duration = (clock() - start) * 1000
                if not isfinite(duration) or duration < 0:
                    raise ValueError("timing clock must be finite and monotonic")
                values[variant] = duration
            if phase == "formal":
                samples.append(values)
    return tuple(samples)


class ComparisonBackend(TransformersDepthBackend):
    def __init__(self, config: DepthConfig) -> None:
        """复用原后端的一份已校验模型；只在本实验实例中切换两个 CPU 处理器。"""
        super().__init__(config)
        from transformers import AutoImageProcessor

        self._baseline_processor = self._processor
        self._candidate_processor = AutoImageProcessor.from_pretrained(
            str(config.model_dir),
            local_files_only=True,
            trust_remote_code=False,
            backend="torchvision",
        )
        if (
            self._baseline_processor.backend != "pil"
            or self._candidate_processor.backend != "torchvision"
        ):
            raise RuntimeError("requested processor backend is unavailable")

    def select(self, variant: Variant) -> None:
        """仅切换本离线实验对象的处理器，模型、插值与有效值规则仍使用原实现。"""
        if variant not in ("pil", "torchvision"):
            raise ValueError("unknown processor variant")
        self._processor = (
            self._baseline_processor if variant == "pil" else self._candidate_processor
        )

    def pixels(self, image: Image, variant: Variant) -> Pixels:
        """从原 BGR 图重新预处理，返回 CPU float32 张量视图；不含模型或设备传输。"""
        import torch

        self.select(variant)
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        result = self._processor(
            images=rgb,
            return_tensors="pt",
            size={"height": self._size, "width": self._size},
        )
        pixels = result["pixel_values"]
        if (
            not isinstance(pixels, torch.Tensor)
            or pixels.device.type != "cpu"
            or pixels.dtype != torch.float32
            or pixels.ndim != 4
            or pixels.shape[:2] != (1, 3)
        ):
            raise RuntimeError("expected CPU float32 1x3xHxW processor output")
        values: Pixels = pixels.numpy()
        if not np.isfinite(values).all():
            raise ValueError("preprocessed tensor has nonfinite values")
        return values


def compare_image(
    image: Image,
    frame_id: str,
    backend: ComparisonBackend,
    estimator: DepthAnythingV2Estimator,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    """分别测 CPU 预处理与一次不计时的深度数值对照；不把二者混为端到端性能。"""

    def run(variant: Variant) -> Pixels:
        """绑定当前原图，每次重新执行指定处理器。"""
        return backend.pixels(image, variant)

    samples = paired_timings(run, warmup=warmup, iterations=iterations)
    pixels = compare_values(run("pil"), run("torchvision"))
    backend.select("pil")
    reference = estimator.estimate(image, frame_id)
    backend.select("torchvision")
    candidate = estimator.estimate(image, frame_id)
    return {
        "samples_ms": samples,
        "summary_ms": {
            variant: asdict(summarize(tuple(sample[variant] for sample in samples)))
            for variant in ("pil", "torchvision")
        },
        "normalized_input_comparison": pixels,
        "relative_depth_comparison": compare_values(
            reference.relative_depth, candidate.relative_depth
        ),
        "valid_mask_mismatch_count": int(
            np.count_nonzero(reference.valid_mask != candidate.valid_mask)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """显式运行处理器对照并保存新报告；异常中止，禁止覆盖旧结果或改动默认配置。"""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Offline PIL/Torchvision preprocessing comparison"
    )
    parser.add_argument("--config", type=Path, default=root / "configs/system.ini")
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args(argv)
    if not 0 <= args.warmup <= 100 or not 1 <= args.iterations <= 1000:
        parser.error("warmup must be 0..100 and iterations 1..1000")
    try:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError("comparison destination already exists")
        if args.output.suffix.lower() != ".json":
            raise ValueError("comparison output must use .json")
        paths = tuple(path.resolve() for path in args.image)
        if not 1 <= len(paths) <= 100 or len(set(paths)) != len(paths):
            raise ValueError("provide 1..100 distinct image paths")
        import cv2
        import torch

        config = load_depth_config(args.config)
        device = select_device(
            config.device, torch.cuda.is_available(), torch.backends.mps.is_available()
        )
        np.random.seed(0)
        torch.manual_seed(0)
        backend = ComparisonBackend(replace(config, device=device))
        estimator = DepthAnythingV2Estimator(config, backend)
        started = datetime.now(timezone.utc).isoformat()
        images: list[dict[str, object]] = []
        for index, path in enumerate(paths):
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"image cannot be decoded: {path.name}")
            frame_id = hashlib.sha256(image.tobytes()).hexdigest()
            images.append(
                {
                    "image_index": index,
                    "image_name": path.name,
                    "image_file_sha256": file_sha256(path),
                    "frame_id": frame_id,
                    "width": image.shape[1],
                    "height": image.shape[0],
                    **compare_image(
                        image,
                        frame_id,
                        backend,
                        estimator,
                        warmup=args.warmup,
                        iterations=args.iterations,
                    ),
                }
            )
        report: dict[str, object] = {
            "schema_version": 1,
            "protocol": "offline_preprocessor_comparison_v1",
            "current_scene": False,
            "measurement_started_at_utc": started,
            "warmup_per_variant_per_image": args.warmup,
            "iterations_per_variant_per_image": args.iterations,
            "order": "per_image_pairs_pil_first_on_even_formal_index",
            "timing_scope": (
                "CPU_BGR_conversion_processor_output_validation_and_numpy_view"
            ),
            "model_comparison": (
                "one_untimed_estimate_per_variant_per_image_shared_model"
            ),
            "seed": 0,
            "config": asdict(config),
            "selected_depth_device": device,
            "preprocessing_device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "accelerator_name": torch.cuda.get_device_name()
            if device == "cuda"
            else None,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "versions": {
                **package_versions(),
                "pillow": version("pillow"),
                "torchvision": version("torchvision"),
            },
            "weight_sha256": {
                name: file_sha256(config.model_dir / name)
                for name in (
                    "model.safetensors",
                    "config.json",
                    "preprocessor_config.json",
                )
            },
            "code": code_identity(root),
            "images": images,
            "default_processor_unchanged": "pil",
            "unmeasured": [
                "accuracy",
                "metric_distance_error",
                "camera_fps",
                "warning_latency",
                "memory",
                "end_to_end_speedup",
            ],
        }
        save_report(report, args.output)
        print(json.dumps({"report": str(args.output.resolve()), "images": len(images)}))
    except Exception as exc:
        logging.error("Preprocessor comparison failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
