"""Depth Anything V2 Small 相对逆深度；无量纲，数值越大表示相对越近。"""

import hashlib
import logging
from collections.abc import Iterator
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol

import numpy as np
from numpy.typing import NDArray

from app.vision.depth_profile import DepthProfiler
from app.vision.detector import Image, select_device

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
MODEL_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
WEIGHTS_SHA256 = "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"
DepthMap = NDArray[np.float32]


def iter_read_chunks(source: BinaryIO) -> Iterator[bytes]:
    """分块读取权重以计算校验和，避免额外载入整份权重。"""
    while block := source.read(1024 * 1024):
        yield block


@dataclass(frozen=True)
class DepthConfig:
    """独立深度模型配置，输入尺寸以 14 像素 patch 为单位。"""

    model_dir: Path
    image_size: int = 518
    device: str = "auto"

    def __post_init__(self) -> None:
        """拒绝不兼容的输入尺寸和未知设备。"""
        if self.image_size <= 0 or self.image_size % 14:
            raise ValueError("depth image_size must be a positive multiple of 14")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("depth device must be auto, cpu, cuda, or mps")


def load_depth_config(path: Path) -> DepthConfig:
    """读取深度配置；模型目录相对配置文件定位，缺少配置时明确失败。"""
    parser = ConfigParser(interpolation=None)
    with path.open(encoding="utf-8") as source:
        parser.read_file(source)
    section = parser["depth"]
    model_dir = Path(section["model_dir"])
    if not model_dir.is_absolute():
        model_dir = path.resolve().parent / model_dir
    return DepthConfig(
        model_dir.resolve(), section.getint("image_size"), section["device"]
    )


@dataclass(frozen=True)
class RelativeDepth:
    """同一离线输入的原图尺寸深度与有效掩膜；无效像素统一为 NaN。"""

    frame_id: str
    relative_depth: DepthMap
    valid_mask: NDArray[np.bool_]


class DepthBackend(Protocol):
    def predict(self, image: Image) -> DepthMap:
        """从 BGR 图像返回原图尺寸的浮点相对逆深度。"""
        ...


class DepthAnythingV2Estimator:
    def __init__(
        self,
        config: DepthConfig,
        backend: DepthBackend | None = None,
        *,
        profiler: DepthProfiler | None = None,
    ) -> None:
        """初始化一次后复用模型；仅显式传入 profiler 时采集阶段诊断。"""
        self._profiler = profiler
        self._backend = (
            backend
            if backend is not None
            else TransformersDepthBackend(config, profiler=profiler)
        )

    def estimate(self, image: Image, frame_id: str) -> RelativeDepth:
        """校验同帧深度并可选记录阶段；失败不会完成当前诊断样本。"""
        if self._profiler is not None:
            self._profiler.begin()
        if not frame_id.strip():
            raise ValueError("frame_id must not be empty")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("expected uint8 HxWx3 BGR image")
        if not image.shape[0] or not image.shape[1]:
            raise ValueError("image must not be empty")
        if self._profiler is not None:
            self._profiler.mark("input_validation")
        raw = self._backend.predict(image)
        if raw.shape != image.shape[:2] or not np.issubdtype(raw.dtype, np.floating):
            raise ValueError(
                "depth must be floating point and match original image size"
            )
        values = np.array(raw, dtype=np.float32, copy=True)
        valid = np.isfinite(values) & (values >= 0)
        if not valid.any():
            raise ValueError("depth has no valid pixels")
        values[~valid] = np.nan
        values.setflags(write=False)
        valid.setflags(write=False)
        result = RelativeDepth(frame_id, values, valid)
        if self._profiler is not None:
            self._profiler.mark("output_validation")
        return result


class TransformersDepthBackend:
    def __init__(
        self, config: DepthConfig, *, profiler: DepthProfiler | None = None
    ) -> None:
        """加载固定身份本地权重；可选诊断必须与实际执行设备一致。"""
        for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
            if not (config.model_dir / name).is_file():
                raise FileNotFoundError(
                    f"Depth model missing: {config.model_dir / name}"
                )
        digest = hashlib.sha256()
        with (config.model_dir / "model.safetensors").open("rb") as source:
            for block in iter_read_chunks(source):
                digest.update(block)
        if digest.hexdigest() != WEIGHTS_SHA256:
            raise ValueError("Depth Anything V2 Small weight SHA256 mismatch")
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._processor = AutoImageProcessor.from_pretrained(
            str(config.model_dir),
            local_files_only=True,
            trust_remote_code=False,
            backend="pil",
        )
        self._model = AutoModelForDepthEstimation.from_pretrained(
            str(config.model_dir),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
        if self._model.config.model_type != "depth_anything":
            raise ValueError("expected Depth Anything model architecture")
        if (
            getattr(self._model.config, "depth_estimation_type", "relative")
            != "relative"
        ):
            raise ValueError("expected relative depth checkpoint")
        self._device = select_device(
            config.device, torch.cuda.is_available(), torch.backends.mps.is_available()
        )
        self._model.to(self._device).eval()
        if profiler is not None and profiler.device != self._device:
            raise ValueError("profiler device differs from depth model device")
        self._profiler = profiler
        self._size = config.image_size
        logging.info("Depth Anything V2 Small device: %s", self._device)

    def predict(self, image: Image) -> DepthMap:
        """执行原有深度推理；诊断模式在设备阶段末同步并记录墙钟时间。"""
        import torch

        rgb = np.ascontiguousarray(image[:, :, ::-1])
        inputs = self._processor(
            images=rgb,
            return_tensors="pt",
            size={"height": self._size, "width": self._size},
        )
        if self._profiler is not None:
            self._profiler.mark("preprocessing")
        inputs = inputs.to(self._device)
        if self._profiler is not None:
            self._profiler.mark("to_device", synchronize=True)
        with torch.inference_mode():
            output = self._model(**inputs).predicted_depth
            if (
                not isinstance(output, torch.Tensor)
                or output.ndim != 3
                or output.shape[0] != 1
            ):
                raise RuntimeError("unexpected depth model output")
            if self._profiler is not None:
                self._profiler.mark("inference", synchronize=True)
            resized = torch.nn.functional.interpolate(
                output.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            )[0, 0]
            if self._profiler is not None:
                self._profiler.mark("resize", synchronize=True)
            values: DepthMap = resized.float().cpu().numpy()
            if self._profiler is not None:
                self._profiler.mark("to_cpu", synchronize=True)
        return values


def save_relative_depth(result: RelativeDepth, destination: Path) -> Path:
    """显式保存原始浮点深度、掩膜和帧标识；禁止覆盖已有实验文件。"""
    if destination.suffix.lower() != ".npz":
        raise ValueError("raw depth output must use .npz")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        np.savez_compressed(
            output,
            relative_depth=result.relative_depth,
            valid_mask=result.valid_mask,
            frame_id=result.frame_id,
            units="relative_inverse_depth",
            larger_values="closer",
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
        )
    return destination
