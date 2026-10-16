"""YOLO11 detection with validated original-image pixel coordinates."""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil, floor, isfinite
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from app.config import DetectionConfig

Image = NDArray[np.uint8]
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """xyxy half-open pixel bounds; model score is not depth/risk confidence."""

    label: str
    confidence: float
    bbox: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("label must not be empty")
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("invalid detection confidence")
        x1, y1, x2, y2 = self.bbox
        if not all(type(value) is int for value in self.bbox):
            raise ValueError("bbox must contain integer coordinates")
        if not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError("bbox must have positive area and nonnegative bounds")


def parse_detections(
    rows: Sequence[Sequence[float]],
    names: Mapping[int, str],
    width: int,
    height: int,
    confidence_threshold: float,
) -> tuple[Detection, ...]:
    """Validate Nx6 [x1,y1,x2,y2,score,class] rows; reject malformed output.

    Clip partially external boxes, drop wholly external boxes, and round outward
    for safe NumPy slicing. Coordinates must already reference the original image.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError("invalid confidence threshold")
    detections: list[Detection] = []
    for row in rows:
        if len(row) != 6 or not all(isfinite(value) for value in row):
            raise ValueError("detector returned malformed or nonfinite output")
        x1, y1, x2, y2, score, class_id = row
        if not 0 <= score <= 1 or int(class_id) != class_id:
            raise ValueError("invalid detector score or class ID")
        if int(class_id) not in names:
            raise ValueError("detector class ID is absent from model class mapping")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("detector returned inverted or empty box")
        if score < confidence_threshold:
            continue
        left, top = max(0, floor(x1)), max(0, floor(y1))
        right, bottom = min(width, ceil(x2)), min(height, ceil(y2))
        if left >= right or top >= bottom:
            continue
        detections.append(
            Detection(names[int(class_id)], score, (left, top, right, bottom))
        )
    return tuple(detections)


class DetectionBackend(Protocol):
    @property
    def names(self) -> Mapping[int, str]: ...

    def predict(self, image: Image) -> Sequence[Sequence[float]]: ...


class Yolo11Detector:
    def __init__(
        self, config: DetectionConfig, backend: DetectionBackend | None = None
    ) -> None:
        self._config = config
        self._backend = backend if backend is not None else UltralyticsBackend(config)

    @property
    def supported_labels(self) -> tuple[str, ...]:
        return tuple(self._backend.names.values())

    def detect(self, image: Image) -> tuple[Detection, ...]:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("expected uint8 HxWx3 BGR image")
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("image must not be empty")
        return parse_detections(
            self._backend.predict(image),
            self._backend.names,
            width,
            height,
            self._config.confidence,
        )


def select_device(requested: str, cuda_available: bool, mps_available: bool) -> str:
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("unsupported device")
    if requested == "auto":
        return "cuda" if cuda_available else "mps" if mps_available else "cpu"
    if (requested == "cuda" and not cuda_available) or (
        requested == "mps" and not mps_available
    ):
        LOGGER.warning("Requested %s unavailable; falling back to CPU", requested)
        return "cpu"
    return requested


class UltralyticsBackend:
    """Load a trusted local YOLO11 checkpoint once; never auto-download weights."""

    def __init__(self, config: DetectionConfig) -> None:
        if not config.weights.is_file():
            raise FileNotFoundError(
                f"YOLO11 weights missing: {config.weights}. See README.md."
            )
        if config.weights.suffix.lower() != ".pt":
            raise ValueError("this initial adapter requires YOLO11 .pt weights")
        import torch
        from ultralytics import YOLO

        self._model = YOLO(str(config.weights), task="detect")
        # Ultralytics exposes dynamic model metadata. Narrow at this boundary.
        metadata: object = self._model.model.yaml
        if not isinstance(metadata, dict):
            raise ValueError("checkpoint has no YOLO architecture metadata")
        architecture = str(metadata.get("yaml_file", ""))
        if not architecture.replace("\\", "/").split("/")[-1].startswith("yolo11"):
            raise ValueError("checkpoint architecture must identify YOLO11")
        if self._model.task != "detect":
            raise ValueError("checkpoint must be an object detection model")
        raw_names: object = self._model.names
        if not isinstance(raw_names, dict) or not raw_names:
            raise ValueError("checkpoint has no class mapping")
        self._names: dict[int, str] = {}
        for key, value in raw_names.items():
            if type(key) is not int or not isinstance(value, str) or not value.strip():
                raise ValueError("invalid model class mapping")
            self._names[key] = value
        self._device = select_device(
            config.device, torch.cuda.is_available(), torch.backends.mps.is_available()
        )
        self._config = config
        LOGGER.info("YOLO11 device: %s; classes: %s", self._device, self._names)

    @property
    def names(self) -> Mapping[int, str]:
        return dict(self._names)

    def predict(self, image: Image) -> Sequence[Sequence[float]]:
        results = self._model.predict(
            source=image,
            conf=self._config.confidence,
            imgsz=self._config.image_size,
            device=self._device,
            half=False,
            save=False,
            verbose=False,
        )
        if len(results) != 1 or results[0].boxes is None:
            raise RuntimeError("expected exactly one object detection result")
        # tensor -> plain Python values; validate structure/types before domain use.
        raw: object = results[0].boxes.data.cpu().tolist()
        if not isinstance(raw, list):
            raise ValueError("unexpected detector output")
        rows: list[list[float]] = []
        for row in raw:
            if not isinstance(row, list) or not all(
                isinstance(value, (int, float)) for value in row
            ):
                raise ValueError("unexpected detector row")
            rows.append([float(value) for value in row])
        return rows
