"""融合实验参数与摄像头安装约定。"""

from configparser import ConfigParser
from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class FusionConfig:
    method: str = "center_region_median"
    inner_fraction: float = 0.5
    min_valid_pixels: int = 9
    min_valid_fraction: float = 0.5
    trim_fraction: float = 0.05
    left_boundary: float = 1 / 3
    right_boundary: float = 2 / 3
    mirrored: bool = False
    camera_orientation: str = "unknown"

    def __post_init__(self) -> None:
        """验证统计参数及方向边界；未知安装朝向不得推断用户方向。"""
        if self.method not in {
            "center_pixel",
            "box_mean",
            "box_median",
            "center_region_median",
        }:
            raise ValueError("unsupported fusion method")
        if not isfinite(self.inner_fraction) or not 0 < self.inner_fraction <= 1:
            raise ValueError("inner_fraction must be in (0, 1]")
        if type(self.min_valid_pixels) is not int or self.min_valid_pixels < 1:
            raise ValueError("min_valid_pixels must be a positive integer")
        if (
            not isfinite(self.min_valid_fraction)
            or not 0 < self.min_valid_fraction <= 1
        ):
            raise ValueError("min_valid_fraction must be in (0, 1]")
        if not isfinite(self.trim_fraction) or not 0 <= self.trim_fraction < 0.5:
            raise ValueError("trim_fraction must be in [0, 0.5)")
        if not 0 < self.left_boundary < self.right_boundary < 1:
            raise ValueError("direction boundaries must satisfy 0 < left < right < 1")
        if type(self.mirrored) is not bool:
            raise ValueError("mirrored must be boolean")
        if self.camera_orientation not in {"forward", "unknown"}:
            raise ValueError("camera_orientation must be forward or unknown")


def load_fusion_config(path: Path) -> FusionConfig:
    """读取可选融合配置；旧配置使用保守默认值，朝向保持未知。"""
    parser = ConfigParser(interpolation=None)
    with path.open(encoding="utf-8") as source:
        parser.read_file(source)
    if not parser.has_section("fusion"):
        return FusionConfig()
    section = parser["fusion"]
    defaults = FusionConfig()
    return FusionConfig(
        method=section.get("method", defaults.method),
        inner_fraction=section.getfloat("inner_fraction", defaults.inner_fraction),
        min_valid_pixels=section.getint("min_valid_pixels", defaults.min_valid_pixels),
        min_valid_fraction=section.getfloat(
            "min_valid_fraction", defaults.min_valid_fraction
        ),
        trim_fraction=section.getfloat("trim_fraction", defaults.trim_fraction),
        left_boundary=section.getfloat("left_boundary", defaults.left_boundary),
        right_boundary=section.getfloat("right_boundary", defaults.right_boundary),
        mirrored=section.getboolean("mirrored", defaults.mirrored),
        camera_orientation=section.get(
            "camera_orientation", defaults.camera_orientation
        ),
    )
