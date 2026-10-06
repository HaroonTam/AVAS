"""告警事件冷却；输出供未来本地语音通路消费，不执行播放。"""

from app.safety.risk_engine import RiskAssessment, RiskLevel, WarningEvent


class WarningGate:
    def __init__(
        self,
        cooldown_ms: int,
        max_entries: int = 256,
        *,
        degraded_cooldown_ms: int | None = None,
    ) -> None:
        """建立有界去重状态；降级提示可配置独立冷却，缺省沿用普通冷却。"""
        if (
            type(cooldown_ms) is not int
            or cooldown_ms < 0
            or type(max_entries) is not int
            or max_entries < 1
            or (
                degraded_cooldown_ms is not None
                and (type(degraded_cooldown_ms) is not int or degraded_cooldown_ms < 0)
            )
        ):
            raise ValueError("invalid warning gate limits")
        self._cooldown_ms = cooldown_ms
        self._degraded_cooldown_ms = (
            cooldown_ms if degraded_cooldown_ms is None else degraded_cooldown_ms
        )
        self._max_entries = max_entries
        self._state: dict[tuple[str, ...], tuple[RiskLevel, int]] = {}
        self._last_time = -1

    def select(
        self, assessment: RiskAssessment, monotonic_ms: int
    ) -> tuple[WarningEvent, ...]:
        """筛选待输出事件；危险与降级分别冷却，新增、重现或升级立即输出。"""
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
            cooldown = (
                self._degraded_cooldown_ms
                if event.kind == "degraded"
                else self._cooldown_ms
            )
            emit = (
                previous is None
                or rank[event.level] > rank[previous[0]]
                or monotonic_ms - previous[1] >= cooldown
            )
            if emit:
                selected.append(event)
            last_emitted = monotonic_ms if emit or previous is None else previous[1]
            if len(active) < self._max_entries:
                active[event.key] = (event.level, last_emitted)
        self._state = active
        return tuple(selected)


class RepeatLimiter:
    """按提示内容限制重复播放；同一文本首次出现或等级升级时立即放行。

    无跟踪器时告警事件身份包含帧 ID，因此每帧都是新事件，仅靠 WarningGate 无法限制
    同一条危险提示的重复播放。该类只限制输出节奏，不修改事件、风险等级或证据。
    """

    def __init__(self, interval_ms: int, max_entries: int = 256) -> None:
        """建立有界内容状态；间隔为同一文本允许再次播放的最小毫秒数。"""
        if (
            type(interval_ms) is not int
            or interval_ms < 0
            or type(max_entries) is not int
            or max_entries < 1
        ):
            raise ValueError("invalid repeat limiter limits")
        self._interval_ms = interval_ms
        self._max_entries = max_entries
        self._state: dict[tuple[str, str], tuple[RiskLevel, int]] = {}

    def allow(self, event: WarningEvent, monotonic_ms: int) -> bool:
        """判断该事件能否提交播放；不修改事件本身，容量不足时不缓存但绝不丢告警。"""
        if type(monotonic_ms) is not int or monotonic_ms < 0:
            raise ValueError("repeat limiter clock must be nonnegative integer ms")
        key = (event.kind, event.message)
        previous = self._state.get(key)
        rank = {"unknown": 0, "low": 1, "medium": 2, "high": 3}
        allowed = (
            previous is None
            or rank[event.level] > rank[previous[0]]
            or monotonic_ms - previous[1] >= self._interval_ms
        )
        if allowed and (previous is not None or len(self._state) < self._max_entries):
            self._state[key] = (event.level, monotonic_ms)
        return allowed
