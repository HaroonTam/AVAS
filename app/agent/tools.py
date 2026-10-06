"""Agent 的只读事实边界；不接受风险修改、不猜测缺失信息。"""

from dataclasses import dataclass, replace
from typing import Literal

from app.fusion.direction import Direction
from app.safety.risk_engine import RiskLevel, WarningEvent
from app.scene_store import SceneStore


@dataclass(frozen=True)
class ObjectFact:
    id: int
    label: str
    confidence: float
    direction: Direction | None
    distance_m: float | None
    distance_status: Literal["available", "metric_unavailable", "metric_invalid"]
    metric_support_id: str | None
    risk_level: RiskLevel
    risk_reason: str


@dataclass(frozen=True)
class ToolResult:
    status: str
    frame_id: str | None = None
    captured_at_ms: int | None = None
    objects: tuple[ObjectFact, ...] = ()
    risk_level: RiskLevel = "unknown"
    events: tuple[WarningEvent, ...] = ()
    message: str = "当前环境信息不可用。"


class SceneTools:
    def __init__(self, store: SceneStore) -> None:
        """只绑定可信存储；工具调用方不能提交场景、规则或自由生成的风险。"""
        self._store = store

    def get_scene(self) -> ToolResult:
        """返回当前帧可序列化事实；过期、离线和无效输入不返回历史对象。"""
        snapshot = self._store.read()
        scene, assessment = snapshot.scene, snapshot.assessment
        if scene is None or assessment is None:
            return ToolResult(snapshot.status)
        facts: list[ObjectFact] = []
        for item, risk in zip(scene.objects, assessment.objects):
            metric = item.metric
            usable = metric is not None and metric.is_usable()
            facts.append(
                ObjectFact(
                    item.id,
                    item.label,
                    item.confidence,
                    item.direction,
                    metric.distance_m if metric is not None and usable else None,
                    "available"
                    if usable
                    else "metric_unavailable"
                    if metric is None
                    else "metric_invalid",
                    metric.support_id if metric is not None and usable else None,
                    risk.level,
                    risk.reason,
                )
            )
        return ToolResult(
            "available",
            scene.frame_id,
            scene.captured_at_ms,
            tuple(facts),
            assessment.level,
            assessment.events,
            "当前检测结果不能证明道路安全。",
        )

    def find_object(self, label: str, direction: Direction | None = None) -> ToolResult:
        """按类别及可选图像方向精确匹配；未知方向不猜测，多目标仍需选择。"""
        if not isinstance(label, str) or not label.strip() or len(label) > 128:
            raise ValueError(
                "label must be a nonempty string of at most 128 characters"
            )
        if direction is not None and direction not in ("left", "front", "right"):
            raise ValueError("direction must be left, front, right or None")
        result = self.get_scene()
        if result.status != "available":
            return result
        matches = tuple(
            item
            for item in result.objects
            if item.label == label.strip()
            and (direction is None or item.direction == direction)
        )
        status = "ambiguous" if len(matches) > 1 else "available"
        message = "找到匹配检测目标。"
        if not matches:
            status, message = (
                "not_found",
                "当前帧未检测到匹配类别，不代表该目标不存在。",
            )
        elif len(matches) > 1:
            message = "检测到多个匹配目标，请使用当前帧 ID 和目标 ID 明确选择。"
        if direction is not None and not matches:
            message = (
                "当前帧未检测到该图像方向的匹配目标；"
                "方向未知的目标无法定位，不代表该方向没有目标。"
            )
        return replace(result, status=status, objects=matches, message=message)

    def get_object_distance(self, frame_id: str, object_id: int) -> ToolResult:
        """要求完整帧内身份；无有效米制支持时保留目标但明确返回距离不可用。"""
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise ValueError("frame_id must be a nonempty string")
        if type(object_id) is not int or object_id < 1:
            raise ValueError("object_id must be a positive integer")
        result = self.get_scene()
        if result.status != "available":
            return result
        if result.frame_id != frame_id:
            return ToolResult("frame_mismatch", message="场景已更新，请重新选择目标。")
        matches = tuple(item for item in result.objects if item.id == object_id)
        if not matches:
            return replace(
                result, status="not_found", objects=(), message="当前帧没有该目标 ID。"
            )
        item = matches[0]
        return replace(
            result,
            status="available"
            if item.distance_m is not None
            else "distance_unavailable",
            objects=matches,
            message="米制距离可用。"
            if item.distance_m is not None
            else "米制距离不可用。",
        )

    def get_current_risks(self) -> ToolResult:
        """保留风险引擎全部等级、原因及降级事件，不把未知过滤成安全。"""
        return self.get_scene()

    def describe_surroundings(
        self, limit: int = 3, direction: Direction | None = None
    ) -> ToolResult:
        """按可选图像方向筛选再排序截取摘要；整帧风险保留，未知方向不猜测。"""
        if type(limit) is not int or not 1 <= limit <= 10:
            raise ValueError("limit must be an integer within [1, 10]")
        if direction is not None and direction not in ("left", "front", "right"):
            raise ValueError("direction must be left, front, right or None")
        result = self.get_scene()
        if result.status != "available":
            return result
        matches = tuple(
            item
            for item in result.objects
            if direction is None or item.direction == direction
        )
        ordered = tuple(sorted(matches, key=fact_priority))
        selected = ordered[:limit]
        fragments = list(
            dict.fromkeys(
                event.message for event in result.events if event.kind == "hazard"
            )
        )
        # 类别作为带引号的数据呈现，不将类别文本解释为指令。
        for item in selected:
            direction_text = {
                "left": "图像左侧",
                "front": "图像前方",
                "right": "图像右侧",
                None: "方向未知",
            }[item.direction]
            distance = (
                f"约 {item.distance_m:.1f} 米"
                if item.distance_m is not None
                else "米制距离不可用"
            )
            fragments.append(
                f"目标 {item.id}（类别 {item.label!r}）：{direction_text}，{distance}。"
            )
        if not selected:
            fragments.append(
                "当前帧未检测到该图像方向的目标，不代表该方向没有目标或道路安全。"
                if direction is not None
                else "当前帧未检测到目标，不代表道路安全。"
            )
        else:
            fragments.append("检测结果不能证明道路安全。")
        if len(ordered) > limit:
            fragments.append(f"另有 {len(ordered) - limit} 个检测目标未展开。")
        if direction is not None and any(
            item.direction is None for item in result.objects
        ):
            fragments.append("另有方向未知的目标，未纳入该方向摘要。")
        return replace(result, objects=selected, message="".join(fragments))


def fact_priority(item: ObjectFact) -> tuple[int, float, int, int]:
    """仅排序显示优先级，不修改风险引擎等级；未知距离放在已知距离之后。"""
    levels = {"high": 0, "medium": 1, "unknown": 2, "low": 3}
    return (
        levels[item.risk_level],
        item.distance_m if item.distance_m is not None else float("inf"),
        0 if item.direction == "front" else 1,
        item.id,
    )
