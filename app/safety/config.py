"""实验性风险规则；阈值尚未经真实行走实验验证。"""

from configparser import ConfigParser
from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class RiskConfig:
    high_distance_m: float = 1.0
    medium_distance_m: float = 2.0
    min_confidence: float = 0.5
    freshness_ms: int = 1000
    cooldown_ms: int = 3000
    obstacle_labels: tuple[str, ...] = (
        "person",
        "chair",
        "bicycle",
        "car",
        "bus",
        "motorcycle",
    )
    independent_hazard_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """拒绝非有限、倒置阈值和不合法时间或类别配置。"""
        if not (
            isfinite(self.high_distance_m)
            and isfinite(self.medium_distance_m)
            and 0 < self.high_distance_m < self.medium_distance_m
        ):
            raise ValueError("risk distances must satisfy 0 < high < medium")
        if not isfinite(self.min_confidence) or not 0 <= self.min_confidence <= 1:
            raise ValueError("risk min_confidence must be within [0, 1]")
        for value in (self.freshness_ms, self.cooldown_ms):
            if type(value) is not int or value < 0:
                raise ValueError("risk times must be nonnegative integer milliseconds")
        for labels in (self.obstacle_labels, self.independent_hazard_labels):
            if any(not label.strip() or label != label.strip() for label in labels):
                raise ValueError("risk labels must be nonempty and stripped")
            if len(set(labels)) != len(labels):
                raise ValueError("duplicate risk labels")


def load_risk_config(path: Path) -> RiskConfig:
    """读取独立风险配置；旧配置文件使用明确的实验默认值。"""
    parser = ConfigParser(interpolation=None)
    with path.open(encoding="utf-8") as source:
        parser.read_file(source)
    defaults = RiskConfig()
    if not parser.has_section("risk"):
        return defaults
    section = parser["risk"]
    return RiskConfig(
        high_distance_m=section.getfloat("high_distance_m", defaults.high_distance_m),
        medium_distance_m=section.getfloat(
            "medium_distance_m", defaults.medium_distance_m
        ),
        min_confidence=section.getfloat("min_confidence", defaults.min_confidence),
        freshness_ms=section.getint("freshness_ms", defaults.freshness_ms),
        cooldown_ms=section.getint("cooldown_ms", defaults.cooldown_ms),
        obstacle_labels=tuple(
            part.strip()
            for part in section.get(
                "obstacle_labels", ",".join(defaults.obstacle_labels)
            ).split(",")
            if part.strip()
        ),
        independent_hazard_labels=tuple(
            part.strip()
            for part in section.get("independent_hazard_labels", "").split(",")
            if part.strip()
        ),
    )
