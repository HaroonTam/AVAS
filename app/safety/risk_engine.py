"""纯本地规则风险评估；输入不接受 Agent 建议或上游风险等级。"""

from dataclasses import dataclass
from typing import Literal

from app.safety.config import RiskConfig
from app.safety.scene import RiskObject, RiskScene

RiskLevel = Literal["unknown", "low", "medium", "high"]


@dataclass(frozen=True)
class ObjectRisk:
    object_id: int
    level: RiskLevel
    reason: str


@dataclass(frozen=True)
class WarningEvent:
    key: tuple[str, ...]
    level: RiskLevel
    kind: Literal["hazard", "degraded"]
    message: str
    object_id: int | None = None


@dataclass(frozen=True)
class RiskAssessment:
    frame_id: str
    level: RiskLevel
    status: str
    objects: tuple[ObjectRisk, ...]
    events: tuple[WarningEvent, ...]


def object_risk(item: RiskObject, config: RiskConfig) -> ObjectRisk:
    """先评估独立类别规则，再评估前向米制障碍规则，缺失证据保持未知。"""
    level: RiskLevel = "unknown"
    reason = "rule_not_applicable"
    if item.confidence < config.min_confidence:
        return ObjectRisk(item.id, level, "low_detection_confidence")
    if item.direction is None:
        return ObjectRisk(item.id, level, "direction_unavailable")
    if item.direction != "front":
        return ObjectRisk(item.id, level, "outside_front_rule")
    if item.label in config.independent_hazard_labels:
        level, reason = "medium", "configured_front_category"
    if item.label not in config.obstacle_labels:
        return ObjectRisk(item.id, level, reason)
    if item.metric is None or not item.metric.is_usable():
        return ObjectRisk(
            item.id, level, reason if level != "unknown" else "metric_unavailable"
        )
    if item.metric.distance_m < config.high_distance_m:
        return ObjectRisk(item.id, "high", "front_distance_below_high")
    if item.metric.distance_m < config.medium_distance_m:
        return ObjectRisk(item.id, "medium", "front_distance_below_medium")
    if level == "medium":
        return ObjectRisk(item.id, level, reason)
    return ObjectRisk(item.id, "low", "outside_distance_thresholds")


def assess_scene(scene: RiskScene, now_ms: int, config: RiskConfig) -> RiskAssessment:
    """按采集时间拒绝离线、失效、未来或过期场景；返回独立告警和降级事件。"""
    if type(now_ms) is not int or now_ms < 0:
        raise ValueError("now_ms must be nonnegative integer milliseconds")
    status = "available"
    if scene.source_kind != "live":
        status = "offline_not_current"
    elif not scene.valid:
        status = "scene_invalid"
    elif scene.captured_at_ms is None:
        status = "capture_time_unavailable"
    elif scene.captured_at_ms > now_ms:
        status = "future_capture_time"
    elif now_ms - scene.captured_at_ms > config.freshness_ms:
        status = "scene_stale"
    if status != "available":
        events = (
            ()
            if status == "offline_not_current"
            else (
                WarningEvent(
                    ("degraded", status),
                    "unknown",
                    "degraded",
                    "当前环境信息不可用，请谨慎。",
                ),
            )
        )
        return RiskAssessment(
            scene.frame_id,
            "unknown",
            status,
            tuple(ObjectRisk(item.id, "unknown", status) for item in scene.objects),
            events,
        )
    results = tuple(object_risk(item, config) for item in scene.objects)
    levels = {result.level for result in results}
    overall: RiskLevel = "unknown"
    if "high" in levels:
        overall = "high"
    elif "medium" in levels:
        overall = "medium"
    elif results and levels == {"low"}:
        overall = "low"
    pending: list[WarningEvent] = []
    for item, result in zip(scene.objects, results):
        if result.level in {"high", "medium"}:
            identity = (
                ("track", item.track_id)
                if item.track_id is not None
                else ("frame", scene.frame_id, str(item.id))
            )
            pending.append(
                WarningEvent(
                    (*identity, item.label, "front"),
                    result.level,
                    "hazard",
                    "前方近距离障碍，请立即注意。"
                    if result.level == "high"
                    else "前方存在配置规则关注的目标，请注意。",
                    item.id,
                )
            )
    if not results or "unknown" in levels:
        reason = "no_detections" if not results else "incomplete_assessment"
        pending.append(
            WarningEvent(
                ("degraded", reason),
                "unknown",
                "degraded",
                "环境风险无法完整评估，请谨慎。",
            )
        )
    # 即使独立危险仍成立，缺失米制信息也必须保留明确的降级提示。
    if any(
        item.metric is None or not item.metric.is_usable() for item in scene.objects
    ):
        pending.append(
            WarningEvent(
                ("degraded", "metric_unavailable"),
                "unknown",
                "degraded",
                "距离信息不可用。",
            )
        )
    ordered = tuple(
        event
        for level in ("high", "medium", "unknown")
        for event in pending
        if event.level == level
    )
    return RiskAssessment(scene.frame_id, overall, status, results, ordered)
