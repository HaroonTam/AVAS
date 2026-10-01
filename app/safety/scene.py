"""风险输入边界；融合结果不能自行声明风险或米制标定有效。"""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from app.fusion.depth_fusion import Observation
from app.fusion.direction import Direction


@dataclass(frozen=True)
class MetricEvidence:
    """由未来可信米制适配器提供；support_id 指向模型或标定记录，不是 LLM 文本。"""

    distance_m: float
    support_id: str
    valid_min_m: float
    valid_max_m: float

    def is_usable(self) -> bool:
        """只有有限非负值、有支持记录且处于验证范围内时可用于米制规则。"""
        return (
            bool(self.support_id.strip())
            and all(
                isfinite(value)
                for value in (self.distance_m, self.valid_min_m, self.valid_max_m)
            )
            and 0 <= self.valid_min_m <= self.distance_m <= self.valid_max_m
            and self.valid_min_m < self.valid_max_m
        )


@dataclass(frozen=True)
class RiskObject:
    id: int
    frame_id: str
    label: str
    confidence: float
    direction: Direction | None
    metric: MetricEvidence | None = None
    track_id: str | None = None

    def __post_init__(self) -> None:
        """校验对象事实；track_id 仅供未来明确保证连续性的跟踪器填写。"""
        if (
            type(self.id) is not int
            or self.id < 1
            or not self.frame_id.strip()
            or not self.label.strip()
        ):
            raise ValueError("invalid risk object identity")
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("invalid risk object confidence")
        if self.direction not in {None, "left", "front", "right"}:
            raise ValueError("invalid risk object direction")
        if self.track_id is not None and not self.track_id.strip():
            raise ValueError("empty track identity")


@dataclass(frozen=True)
class RiskScene:
    frame_id: str
    captured_at_ms: int | None
    source_kind: Literal["live", "offline_image"]
    valid: bool
    objects: tuple[RiskObject, ...]

    def __post_init__(self) -> None:
        """拒绝混帧和重复身份；失效场景可保留对象，但评估不得使用。"""
        if not self.frame_id.strip() or self.source_kind not in {
            "live",
            "offline_image",
        }:
            raise ValueError("invalid risk scene identity or source")
        if type(self.valid) is not bool:
            raise ValueError("scene valid must be boolean")
        if self.captured_at_ms is not None and (
            type(self.captured_at_ms) is not int or self.captured_at_ms < 0
        ):
            raise ValueError("capture time must be nonnegative integer milliseconds")
        if any(item.frame_id != self.frame_id for item in self.objects):
            raise ValueError("risk scene contains mismatched frames")
        ids = [item.id for item in self.objects]
        tracks = [item.track_id for item in self.objects if item.track_id is not None]
        if len(set(ids)) != len(ids) or len(set(tracks)) != len(tracks):
            raise ValueError("duplicate scene object or track identity")


def offline_scene(frame_id: str, observations: tuple[Observation, ...]) -> RiskScene:
    """转换当前离线融合结果，不复制上游风险、米制值或伪造采集时间。"""
    return RiskScene(
        frame_id,
        None,
        "offline_image",
        True,
        tuple(
            RiskObject(
                item.id,
                item.frame_id,
                item.label,
                item.confidence,
                item.direction if item.direction_status == "image_relative" else None,
            )
            for item in observations
        ),
    )
