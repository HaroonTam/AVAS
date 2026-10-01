"""告警事件冷却；输出供未来本地语音通路消费，不执行播放。"""

from app.safety.risk_engine import RiskAssessment, RiskLevel, WarningEvent


class WarningGate:
    def __init__(self, cooldown_ms: int, max_entries: int = 256) -> None:
        """建立有界去重状态；无可靠跟踪 ID 时不做跨帧对象合并。"""
        if (
            type(cooldown_ms) is not int
            or cooldown_ms < 0
            or type(max_entries) is not int
            or max_entries < 1
        ):
            raise ValueError("invalid warning gate limits")
        self._cooldown_ms = cooldown_ms
        self._max_entries = max_entries
        self._state: dict[tuple[str, ...], tuple[RiskLevel, int]] = {}
        self._last_time = -1

    def select(
        self, assessment: RiskAssessment, monotonic_ms: int
    ) -> tuple[WarningEvent, ...]:
        """筛选待输出事件；新增、重现或升级绕过冷却，消失事件立即清理。"""
        if (
            type(monotonic_ms) is not int
            or monotonic_ms < 0
            or monotonic_ms < self._last_time
        ):
            raise ValueError("warning clock must be nonnegative and monotonic")
        self._last_time = monotonic_ms
        active: dict[tuple[str, ...], tuple[RiskLevel, int]] = {}
        selected: list[WarningEvent] = []
        rank = {"unknown": 0, "low": 1, "medium": 2, "high": 3}
        for event in assessment.events:
            previous = self._state.get(event.key)
            emit = (
                previous is None
                or rank[event.level] > rank[previous[0]]
                or monotonic_ms - previous[1] >= self._cooldown_ms
            )
            if emit:
                selected.append(event)
            last_emitted = monotonic_ms if emit or previous is None else previous[1]
            if len(active) < self._max_entries:
                active[event.key] = (event.level, last_emitted)
        self._state = active
        return tuple(selected)
