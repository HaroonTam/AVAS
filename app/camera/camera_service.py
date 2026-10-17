"""隔离摄像头原生读取；有界队列只保留近期图像。"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from queue import Empty, Full
from time import monotonic, time_ns
from uuid import uuid4

import numpy as np

from app.vision.detector import Image


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    timeout_s: float = 3.0
    startup_timeout_s: float | None = None

    def __post_init__(self) -> None:
        """验证设备与有限等待秒数；未配置启动等待时保持原读取超时行为。"""
        if any(
            type(value) is not int for value in (self.index, self.width, self.height)
        ):
            raise ValueError("camera index and dimensions must be integers")
        if self.index < 0 or self.width < 1 or self.height < 1:
            raise ValueError("invalid camera index or dimensions")
        if not 0 < self.timeout_s <= 30:
            raise ValueError("camera timeout must be in (0, 30] seconds")
        if self.startup_timeout_s is not None and not 0 < self.startup_timeout_s <= 30:
            raise ValueError("camera startup timeout must be in (0, 30] seconds")


@dataclass(frozen=True)
class CapturedFrame:
    frame_id: str
    captured_at_ms: int
    captured_monotonic: float
    image: Image

    def __post_init__(self) -> None:
        """拒绝空身份、非法时间和非 BGR 帧；时间为读取开始时的保守近似。"""
        if (
            not self.frame_id.strip()
            or type(self.captured_at_ms) is not int
            or self.captured_at_ms < 0
            or self.captured_monotonic < 0
            or not np.isfinite(self.captured_monotonic)
        ):
            raise ValueError("invalid captured frame metadata")
        if (
            self.image.dtype != np.uint8
            or self.image.ndim != 3
            or self.image.shape[2] != 3
            or min(self.image.shape[:2]) < 1
        ):
            raise ValueError("camera must return nonempty uint8 HxWx3 BGR")


def publish_latest(queue: Queue[CapturedFrame], frame: CapturedFrame) -> None:
    """队列满时丢弃旧帧；进程队列短暂竞争时允许丢帧而不阻塞。"""
    try:
        queue.put_nowait(frame)
    except Full:
        try:
            queue.get_nowait()
        except Empty:
            pass
        try:
            queue.put_nowait(frame)
        except Full:
            pass


def capture_worker(
    config: CameraConfig, queue: Queue[CapturedFrame], stop: Event, failed: Event
) -> None:
    """独占摄像头，在子进程读取并释放；异常只通过故障标志传给父进程。"""
    capture = None
    try:
        import cv2

        capture = cv2.VideoCapture(config.index)
        if not capture.isOpened():
            raise RuntimeError("camera open failed")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        session = uuid4().hex
        sequence = 0
        while not stop.is_set():
            timestamp, started = time_ns() // 1_000_000, monotonic()
            ok, image = capture.read()
            if not ok or image is None:
                raise RuntimeError("camera read failed")
            sequence += 1
            publish_latest(
                queue, CapturedFrame(f"{session}:{sequence}", timestamp, started, image)
            )
    except Exception:
        # 原生后端类型不统一，统一置故障；不记录相机画面或敏感数据。
        failed.set()
    finally:
        if capture is not None:
            capture.release()
        queue.cancel_join_thread()


class CameraService:
    def __init__(self, config: CameraConfig) -> None:
        """使用 spawn 隔离原生驱动；构造不立即打开摄像头。"""
        self.config = config
        self._closed = False
        self._shutdown_complete = False
        self._queue_closed = False
        self._startup_deadline: float | None = None
        self._received_frame = False
        self._started = False
        context = mp.get_context("spawn")
        self._queue: Queue[CapturedFrame] = context.Queue(maxsize=1)
        self._stop = context.Event()
        self._failed = context.Event()
        self._process = context.Process(
            target=capture_worker,
            args=(config, self._queue, self._stop, self._failed),
            daemon=True,
        )

    def start(self) -> None:
        """显式启动采集；可选启动预算从进程启动前起算，首帧前重试不重置预算。"""
        if self._closed:
            raise RuntimeError("camera already closed")
        if self._started:
            raise RuntimeError("camera already started")
        if self.config.startup_timeout_s is not None:
            self._startup_deadline = monotonic() + self.config.startup_timeout_s
        self._process.start()
        self._started = True

    def healthy(self) -> bool:
        """采集进程退出或故障时立即向调用方标记不可用。"""
        return (
            not self._closed and self._process.is_alive() and not self._failed.is_set()
        )

    def read(self) -> CapturedFrame:
        """首个有效包使用可选启动预算，之后恢复普通读取超时；不改帧采集时刻或新鲜度。"""
        deadline = (
            self._startup_deadline
            if not self._received_frame and self._startup_deadline is not None
            else monotonic() + self.config.timeout_s
        )
        while monotonic() < deadline:
            if not self.healthy():
                raise RuntimeError("camera unavailable")
            try:
                frame = self._queue.get(
                    timeout=min(0.1, max(0.001, deadline - monotonic()))
                )
            except Empty:
                continue
            if not self.healthy():
                raise RuntimeError("camera unavailable")
            if not isinstance(frame, CapturedFrame):
                raise RuntimeError("unexpected camera packet")
            self._received_frame = True
            return frame
        raise TimeoutError("camera read timeout")

    def close(self) -> None:
        """有界停止并确认子进程退出后才关闭句柄；仍存活则明确报错，允许再次清理。"""
        if self._shutdown_complete:
            return
        self._closed = True
        self._stop.set()
        try:
            if self._process.pid is not None:
                self._process.join(timeout=1)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=1)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(timeout=1)
                if self._process.is_alive():
                    raise RuntimeError(
                        f"camera process {self._process.pid} still running; "
                        "handle retained for cleanup retry"
                    )
                self._process.close()
            self._shutdown_complete = True
        finally:
            if not self._queue_closed:
                self._queue.close()
                self._queue.cancel_join_thread()
                self._queue_closed = True
