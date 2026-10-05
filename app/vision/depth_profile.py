"""显式启用的深度阶段墙钟诊断；设备同步会改变流水线调度。"""

from collections.abc import Callable
from math import isfinite
from time import perf_counter

STAGES = (
    "input_validation",
    "preprocessing",
    "to_device",
    "inference",
    "resize",
    "to_cpu",
    "output_validation",
)


def synchronize_device(device: str) -> None:
    """等待选定加速设备完成工作；CPU 无同步操作，未知设备拒绝。"""
    if device == "cpu":
        return
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    else:
        raise ValueError("unsupported profiling device")


class DepthProfiler:
    def __init__(
        self,
        device: str,
        *,
        clock: Callable[[], float] = perf_counter,
        synchronize: Callable[[str], None] = synchronize_device,
    ) -> None:
        """保存显式诊断设备和可注入时钟；只保留最近一次完整结果。"""
        if device not in {"cpu", "cuda", "mps"}:
            raise ValueError("profiling requires a resolved device")
        self.device = device
        self._clock = clock
        self._synchronize = synchronize
        self._previous: float | None = None
        self._start: float | None = None
        self._stages: dict[str, float] = {}
        self.last_sample: dict[str, float] | None = None

    def begin(self) -> None:
        """先清除旧结果并排空设备工作，再启动本次计时；前置等待不计入阶段。"""
        self.last_sample = None
        self._stages = {}
        self._start = self._previous = None
        self._synchronize(self.device)
        now = self._clock()
        if not isfinite(now):
            raise ValueError("profiling clock must be finite")
        self._start = self._previous = now

    def mark(self, stage: str, *, synchronize: bool = False) -> None:
        """顺序结束一个阶段；可选同步耗时归入本阶段，失败不会留下完整样本。"""
        if (
            self._previous is None
            or len(self._stages) >= len(STAGES)
            or stage != STAGES[len(self._stages)]
        ):
            self.last_sample = None
            raise ValueError("invalid profiling stage order")
        if synchronize:
            self._synchronize(self.device)
        now = self._clock()
        if not isfinite(now) or now < self._previous:
            raise ValueError("profiling clock must be finite and monotonic")
        self._stages[stage] = (now - self._previous) * 1000
        self._previous = now
        if stage == STAGES[-1] and self._start is not None:
            self.last_sample = {
                **self._stages,
                "total": (now - self._start) * 1000,
            }
