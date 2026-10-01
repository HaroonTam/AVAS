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
        self, config: DepthConfig, backend: DepthBackend | None = None
    ) -> None:
        """初始化一次后复用模型；测试可注入合成后端。"""
        self._backend = (
            backend if backend is not None else TransformersDepthBackend(config)
        )

    def estimate(self, image: Image, frame_id: str) -> RelativeDepth:
        """校验同帧深度尺寸与有效值，全无效或错位输出不能作为观测。"""
        if not frame_id.strip():
            raise ValueError("frame_id must not be empty")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("expected uint8 HxWx3 BGR image")
        if not image.shape[0] or not image.shape[1]:
            raise ValueError("image must not be empty")
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
        return RelativeDepth(frame_id, values, valid)


class TransformersDepthBackend:
    def __init__(self, config: DepthConfig) -> None:
        """只加载固定身份的本地 safetensors，不联网、不执行远程模型代码。"""
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
        self._size = config.image_size
        logging.info("Depth Anything V2 Small device: %s", self._device)

    def predict(self, image: Image) -> DepthMap:
        """BGR 转 RGB，按比例缩放输入，并用双三次插值映射回原图网格。"""
        import torch

        rgb = np.ascontiguousarray(image[:, :, ::-1])
        inputs = self._processor(
            images=rgb,
            return_tensors="pt",
            size={"height": self._size, "width": self._size},
        ).to(self._device)
        with torch.inference_mode():
            output = self._model(**inputs).predicted_depth
            if (
                not isinstance(output, torch.Tensor)
                or output.ndim != 3
                or output.shape[0] != 1
            ):
                raise RuntimeError("unexpected depth model output")
            resized = torch.nn.functional.interpolate(
                output.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            )[0, 0]
            values: DepthMap = resized.float().cpu().numpy()
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
