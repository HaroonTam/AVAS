"""Validated detection configuration. Relative weights resolve beside the INI."""

from configparser import ConfigParser
from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class DetectionConfig:
    weights: Path
    confidence: float = 0.35
    image_size: int = 640
    device: str = "auto"

    def __post_init__(self) -> None:
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be finite and within [0, 1]")
        if self.image_size <= 0 or self.image_size % 32:
            raise ValueError("image_size must be a positive multiple of 32")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")


def load_config(path: Path) -> DetectionConfig:
    parser = ConfigParser(interpolation=None)
    with path.open(encoding="utf-8") as source:
        parser.read_file(source)
    section = parser["detection"]
    weights = Path(section["weights"])
    if not weights.is_absolute():
        weights = path.resolve().parent / weights
    return DetectionConfig(
        weights=weights.resolve(),
        confidence=section.getfloat("confidence"),
        image_size=section.getint("image_size"),
        device=section["device"],
    )
