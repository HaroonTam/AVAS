"""有界、可抢占的本地 TTS；Windows 子进程故障不会阻塞感知。"""

from __future__ import annotations

import base64
import logging
import os
import subprocess
from dataclasses import dataclass
from threading import Condition, Thread
from time import monotonic
from typing import Protocol

from app.scene_store import SceneLease

# 本机中文 TTS 实测（2026-10-07 硬件验收，Windows SpeechSynthesizer，rate=0，
# Microsoft Huihui Desktop）：150 字符 39.47 s、8 字符 3.17 s，拟合为
# 启动约 1.13 s 加 0.256 s/字符（约 3.91 字符/秒）。下列常数取更保守的值
# （每字符约留 25% 余量），只用于播放上限与排空等待，不作为可听延迟或安全指标。
SPEECH_STARTUP_ALLOWANCE_S = 2.0
SPEECH_SECONDS_PER_CHAR = 0.32
SPEECH_MAX_PLAYBACK_S = 90.0


def speech_duration_bound(text: str) -> float:
    """按实测语速给出单条文本的播放时长上界秒数；空文本按启动余量计算。"""
    return SPEECH_STARTUP_ALLOWANCE_S + SPEECH_SECONDS_PER_CHAR * len(text)


def playback_limit(text: str, configured_timeout_s: float) -> float:
    """单条播放上限：以配置值为基础额度，再按实测语速加上文本时长并封顶。

    配置的 timeout_s 仍表示单条播放的基础等待额度；长文本按长度追加时间，
    最高不超过 SPEECH_MAX_PLAYBACK_S，避免卡死的本地播放长期占用语音队列。
    """
    return min(
        SPEECH_MAX_PLAYBACK_S,
        configured_timeout_s + SPEECH_SECONDS_PER_CHAR * len(text),
    )


@dataclass(frozen=True)
class SpeechMessage:
    text: str
    priority: int
    expires_at: float
    scene_lease: SceneLease | None = None

    def __post_init__(self) -> None:
        """只允许短文本和明确优先级；过期使用单调时钟秒数。"""
        if (
            not self.text.strip()
            or len(self.text) > 300
            or self.priority not in (0, 1, 2)
        ):
            raise ValueError("invalid speech text or priority")
        if not 0 < self.expires_at < float("inf"):
            raise ValueError("invalid speech expiry")
        if self.scene_lease is not None and self.priority != 0:
            raise ValueError("scene leases are only for ordinary answers")


class SpeechHandle(Protocol):
    def poll(self) -> int | None:
        """返回退出码，尚未结束时返回 None。"""
        ...

    def stop(self) -> None:
        """在有限时间内停止播放并释放资源。"""
        ...


class SpeechBackend(Protocol):
    def start(self, text: str) -> SpeechHandle:
        """启动本地非阻塞播放，失败时抛出异常。"""
        ...


class ProcessSpeech:
    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        """保存单次本地播放进程。"""
        self._process = process

    def poll(self) -> int | None:
        """查询播放状态，不等待声音结束。"""
        return self._process.poll()

    def stop(self) -> None:
        """终止播放并有限等待回收，供抢占和关闭使用。"""
        if self._process.poll() is None:
            self._process.kill()
        self._process.wait(timeout=1)


class WindowsSpeechBackend:
    def __init__(self, rate: int = 0) -> None:
        """选择 Windows 内置中文语音；构造不播放任何声音。"""
        if os.name != "nt":
            raise RuntimeError("local speech currently requires Windows")
        if type(rate) is not int or not -10 <= rate <= 10:
            raise ValueError("speech rate must be an integer in [-10, 10]")
        self._rate = rate

    def start(self, text: str) -> SpeechHandle:
        """文本编码为数据传递，禁止把文本插入可执行 PowerShell 语句。"""
        encoded_text = base64.b64encode(text.encode("utf-8")).decode("ascii")
        script = (
            "$ErrorActionPreference='Stop'; Add-Type -AssemblyName System.Speech; "
            "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "try { $v=@($s.GetInstalledVoices() | Where-Object { "
            "$_.Enabled -and $_.VoiceInfo.Culture.Name -like 'zh-*' }); "
            "if ($v.Count -eq 0) { throw 'No installed Chinese voice' }; "
            "$s.SelectVoice($v[0].VoiceInfo.Name); "
            f"$s.Rate={self._rate}; "
            "$t=[Text.Encoding]::UTF8.GetString("
            f"[Convert]::FromBase64String('{encoded_text}')); "
            "$s.Speak($t) } finally { $s.Dispose() }"
        )
        encoded_command = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        process = subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-EncodedCommand",
                encoded_command,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return ProcessSpeech(process)


class SpeechWorker:
    def __init__(
        self, backend: SpeechBackend, capacity: int = 4, timeout_s: float = 15
    ) -> None:
        """创建专用线程及有界队列；高优先级输入不等待普通播放。"""
        if type(capacity) is not int or capacity < 1 or not 0 < timeout_s <= 60:
            raise ValueError("invalid speech worker limits")
        self._backend = backend
        self._capacity = capacity
        self._timeout = timeout_s
        self._condition = Condition()
        self._pending: list[SpeechMessage] = []
        self._closed = False
        self._busy = False
        self._cancel = False
        self._error: str | None = None
        self._thread = Thread(target=self._run, name="local-speech", daemon=True)
        self._thread.start()

    @property
    def error(self) -> str | None:
        """向 UI 或日志暴露语音故障；无错误不等于硬件播放已验证。"""
        with self._condition:
            return self._error

    def submit(self, message: SpeechMessage) -> bool:
        """非阻塞提交；满队列优先保留更高优先级和更新的信息。"""
        with self._condition:
            if self._closed or message.expires_at <= monotonic():
                return False
            self._pending = [
                item for item in self._pending if item.expires_at > monotonic()
            ]
            if len(self._pending) >= self._capacity:
                lowest = min(item.priority for item in self._pending)
                if message.priority < lowest:
                    return False
                index = next(
                    index
                    for index, item in enumerate(self._pending)
                    if item.priority == lowest
                )
                self._pending.pop(index)
            self._pending.append(message)
            self._condition.notify()
            return True

    def cancel(self) -> None:
        """场景失效时清空旧语音并请求停止当前播放，不阻塞调用线程。"""
        with self._condition:
            self._pending.clear()
            self._cancel = True
            self._condition.notify_all()

    def _idle(self) -> bool:
        """在持锁状态下判断是否没有活动或待播消息。"""
        return not self._busy and not self._pending and not self._cancel

    def wait_idle(self, timeout_s: float) -> bool:
        """仅在停止感知后有界等待末条提示完成；返回是否已经空闲。"""
        if not 0 <= timeout_s <= 60:
            raise ValueError("invalid speech drain timeout")
        with self._condition:
            return self._condition.wait_for(self._idle, timeout=timeout_s)

    def _fail(self, error: Exception) -> None:
        """记录可访问输出故障，不让播放异常传播到感知线程。"""
        with self._condition:
            self._error = f"spoken alerts unavailable: {type(error).__name__}"
        logging.error("Spoken alerts unavailable: %s", error)

    def _clear_error(self) -> None:
        """在一次成功播放后撤销瞬时故障标记；真实后端故障会在下次播放重新置位。"""
        with self._condition:
            if self._error is None:
                return
            self._error = None
        logging.info("Spoken alerts recovered after a successful playback")

    def _run(self) -> None:
        """在后台轮询播放，紧急输入抢占较低优先级，丢弃排队过期消息并限制播放时长。"""
        handle: SpeechHandle | None = None
        active: SpeechMessage | None = None
        started = 0.0
        try:
            while True:
                with self._condition:
                    if self._closed:
                        break
                    self._pending = [
                        item for item in self._pending if item.expires_at > monotonic()
                    ]
                    highest = max((item.priority for item in self._pending), default=-1)
                    cancel = self._cancel
                    self._cancel = False
                if handle is not None and active is not None:
                    code = handle.poll()
                    timed_out = monotonic() - started >= playback_limit(
                        active.text, self._timeout
                    )
                    if (
                        code is not None
                        or timed_out
                        or cancel
                        or highest > active.priority
                    ):
                        handle.stop()
                        handle = None
                        with self._condition:
                            self._busy = False
                            self._condition.notify_all()
                        if code not in (None, 0) or timed_out:
                            self._fail(
                                RuntimeError(
                                    "local TTS failed or timed out; "
                                    "check installed Chinese voice/audio output"
                                )
                            )
                        elif code == 0:
                            # 成功播完说明后端仍可用；撤销先前一次失败留下的标记，
                            # 否则一次超时会永久关闭后续所有语音的提交与报告。
                            self._clear_error()
                if handle is None:
                    with self._condition:
                        if self._closed:
                            break
                        self._pending = [
                            item
                            for item in self._pending
                            if item.expires_at > monotonic()
                        ]
                        if self._pending:
                            highest = max(item.priority for item in self._pending)
                            index = next(
                                index
                                for index, item in enumerate(self._pending)
                                if item.priority == highest
                            )
                            active = self._pending.pop(index)
                            self._busy = True
                        else:
                            active = None
                    if active is not None:
                        try:
                            # 只进行本地常数时间检查，绝不在告警工作线程运行 Agent。
                            if (
                                active.scene_lease is not None
                                and not active.scene_lease.is_valid()
                            ):
                                with self._condition:
                                    self._busy = False
                                    self._condition.notify_all()
                                continue
                            handle = self._backend.start(active.text)
                            started = monotonic()
                        except Exception as exc:
                            self._fail(exc)
                            with self._condition:
                                self._busy = False
                                self._condition.notify_all()
                with self._condition:
                    self._condition.notify_all()
                    if not self._closed:
                        self._condition.wait(timeout=0.02)
        except Exception as exc:
            self._fail(exc)
        finally:
            with self._condition:
                self._closed = True
                self._pending.clear()
                self._cancel = False
            if handle is not None:
                try:
                    handle.stop()
                except Exception as exc:
                    self._fail(exc)

            with self._condition:
                self._busy = False
                self._condition.notify_all()

    def close(self) -> None:
        """清理待播内容、停止当前播放并有界等待线程退出。"""
        with self._condition:
            self._closed = True
            self._pending.clear()
            self._condition.notify()
        self._thread.join(timeout=2)
        if self._thread.is_alive():
            self._fail(TimeoutError("speech worker shutdown timed out"))
