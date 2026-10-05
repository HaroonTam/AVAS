"""只保留最新结构化场景的进程内存储，不保存图像或音频。"""

from dataclasses import dataclass, replace
from threading import Event, Lock
from time import monotonic, time_ns
from typing import Callable

from app.safety.config import RiskConfig
from app.safety.risk_engine import RiskAssessment, assess_scene
from app.safety.scene import RiskObject, RiskScene


def wall_time_ms() -> int:
    """返回 Unix 毫秒，供当前场景时效检查使用。"""
    return time_ns() // 1_000_000


@dataclass(frozen=True)
class SceneSnapshot:
    status: str
    scene: RiskScene | None
    assessment: RiskAssessment | None


@dataclass(frozen=True)
class SceneLease:
    """仅供本地输出层使用的场景凭据，不向 Agent 暴露修改接口。"""

    frame_id: str
    captured_at_ms: int
    deadline: float
    freshness_ms: int
    _revoked: Event
    _clock_ms: Callable[[], int]
    _monotonic: Callable[[], float]

    def is_valid(self) -> bool:
        """常数时间检查撤销及双时钟期限，不查询工具、风险引擎或网络。"""
        now = self._clock_ms()
        valid = (
            not self._revoked.is_set()
            and 0 <= now - self.captured_at_ms <= self.freshness_ms
            and self._monotonic() <= self.deadline
        )
        if not valid:
            self._revoked.set()
        return valid and not self._revoked.is_set()


class SceneStore:
    def __init__(
        self,
        config: RiskConfig,
        *,
        clock_ms: Callable[[], int] = wall_time_ms,
        monotonic_clock: Callable[[], float] = monotonic,
    ) -> None:
        """使用风险配置的毫秒时效；注入时钟仅供本地测试，不暴露给 Agent。"""
        self.config = config
        self._clock_ms = clock_ms
        self._monotonic = monotonic_clock
        self._lock = Lock()
        self._scene: RiskScene | None = None
        self._deadline = 0.0
        self._status = "scene_unavailable"
        self._lease: SceneLease | None = None

    def publish(self, scene: RiskScene) -> None:
        """发布可信感知场景的不可变副本；非法输入清空旧事实后抛出异常。"""
        try:
            objects = tuple(
                RiskObject(
                    item.id,
                    item.frame_id,
                    item.label,
                    item.confidence,
                    item.direction,
                    replace(item.metric) if item.metric is not None else None,
                    item.track_id,
                )
                for item in scene.objects
            )
            candidate = replace(scene, objects=objects)
            now = self._clock_ms()
            status = assess_scene(candidate, now, self.config).status
            remaining = (
                self.config.freshness_ms - (now - candidate.captured_at_ms)
                if candidate.captured_at_ms is not None
                else 0
            )
            deadline = self._monotonic() + remaining / 1000
        except (AttributeError, TypeError, ValueError):
            self.invalidate()
            raise ValueError("invalid scene publication") from None
        with self._lock:
            if (
                status == "available"
                and self._scene is not None
                and candidate.captured_at_ms is not None
                and self._scene.captured_at_ms is not None
                and candidate.captured_at_ms < self._scene.captured_at_ms
            ):
                return
            if self._lease is not None:
                self._lease._revoked.set()
            self._scene = candidate if status == "available" else None
            self._deadline = deadline
            self._status = status
            self._lease = (
                SceneLease(
                    candidate.frame_id,
                    candidate.captured_at_ms,
                    deadline,
                    self.config.freshness_ms,
                    Event(),
                    self._clock_ms,
                    self._monotonic,
                )
                if status == "available" and candidate.captured_at_ms is not None
                else None
            )

    def acquire_lease(self) -> SceneLease | None:
        """获取当前发布版本凭据；任何后续发布都会撤销旧凭据，即使帧 ID 相同。"""
        with self._lock:
            lease = self._lease
        return lease if lease is not None and lease.is_valid() else None

    def invalidate(self) -> None:
        """相机故障或关闭时立即删除唯一场景，阻止继续查询旧事实。"""
        with self._lock:
            if self._lease is not None:
                self._lease._revoked.set()
            self._lease = None
            self._scene = None
            self._status = "scene_invalid"

    def read(self) -> SceneSnapshot:
        """读取时重新执行风险规则与双时钟时效检查；失效后不可复活旧帧。"""
        with self._lock:
            scene = self._scene
            deadline = self._deadline
            lease = self._lease
            if scene is None:
                return SceneSnapshot(self._status, None, None)
        # 风险计算在锁外完成，工具读取不能让发布端等待整次评估。
        assessment = assess_scene(scene, self._clock_ms(), self.config)
        status = assessment.status
        if status == "available" and self._monotonic() > deadline:
            status = "scene_stale"
        if status == "available" and lease is not None and not lease.is_valid():
            status = "scene_stale"
        with self._lock:
            if self._scene is not scene:
                return SceneSnapshot("scene_changed", None, None)
            if status != "available":
                if self._lease is not None:
                    self._lease._revoked.set()
                self._lease = None
                self._scene = None
                self._status = status
                return SceneSnapshot(status, None, None)
            return SceneSnapshot(status, scene, assessment)
