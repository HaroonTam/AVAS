"""可停止的 Windows 文字控制台，独立于感知和紧急告警。"""

import os
import sys
from threading import Event, Thread
from typing import Callable, Protocol

from app.agent.spoken_reply import SpokenReply
from app.agent.vision_agent import VisionAssistant


def write_console(text: str) -> None:
    """向标准错误输出交互文字，不写入逐帧 JSON 通道。"""
    sys.stderr.write(text)
    sys.stderr.flush()


class LineSource(Protocol):
    def poll(self) -> str | None:
        """非阻塞返回完整输入行；无输入返回 None，输入结束抛出 EOFError。"""
        ...


class WindowsLineSource:
    def __init__(self, emit: Callable[[str], None] = write_console) -> None:
        """只允许 Windows 交互终端；不阻塞读取重定向文件或管道。"""
        if os.name != "nt" or not sys.stdin.isatty():
            raise RuntimeError("--console requires a Windows interactive terminal")
        import msvcrt

        self._ready = msvcrt.kbhit
        self._read = msvcrt.getwch
        self._emit = emit
        self._chars: list[str] = []
        self._overflow = False
        self._special = False

    def poll(self) -> str | None:
        """每次最多读一个字符，支持退格，超长行整体拒绝，特殊键不进入命令。"""
        if not self._ready():
            return None
        char = self._read()
        if self._special:
            self._special = False
            return None
        if char in {"\x00", "\xe0"}:
            self._special = True
            return None
        if char == "\x03":
            return "quit"
        if char == "\x1a":
            raise EOFError
        if char in {"\r", "\n"}:
            text = "".join(self._chars)
            if self._overflow:
                text = "\x00" * 257
            self._chars.clear()
            self._overflow = False
            self._emit("\n")
            return text
        if char == "\b":
            if self._chars and not self._overflow:
                self._chars.pop()
                self._emit("\b \b")
        elif char.isprintable():
            if len(self._chars) < 256 and not self._overflow:
                self._chars.append(char)
                self._emit(char)
            else:
                self._overflow = True
        return None


class ConsoleWorker:
    def __init__(
        self,
        assistant: VisionAssistant,
        source: LineSource,
        emit: Callable[[str], None] = write_console,
        *,
        spoken_reply: SpokenReply | None = None,
    ) -> None:
        """初始化单个交互线程，不建立无界请求队列；必须显式 start。"""
        self._assistant = assistant
        self._spoken_reply = spoken_reply
        self._source = source
        self._emit = emit
        self._stop = Event()
        self.exit_requested = Event()
        self.finished = Event()
        self.error: str | None = None
        self._thread = Thread(target=self._run, name="text-console", daemon=True)

    def start(self) -> None:
        """启动后台输入及查询，不让主感知循环等待键盘。"""
        self._thread.start()

    def _run(self) -> None:
        """串行处理有限命令；输入结束只关闭交互，退出命令通知主循环。"""
        try:
            self._emit(
                "文字控制台：描述周围 / 当前风险 / 寻找 椅子 / 距离 帧ID 目标ID\n"
                "输入 quit 或 退出结束运行。\n> "
            )
            while not self._stop.is_set():
                text = self._source.poll()
                if text is None:
                    self._stop.wait(0.01)
                    continue
                if self._stop.is_set():
                    break
                if text.strip() in {"quit", "exit", "退出"}:
                    self.exit_requested.set()
                    break
                if not text.strip():
                    self._emit("> ")
                    continue
                answer = (
                    self._spoken_reply.respond(text)
                    if self._spoken_reply is not None
                    else self._assistant.respond(text)
                )
                if not self._stop.is_set():
                    frame = f"，帧 {answer.frame_id}" if answer.frame_id else ""
                    self._emit(f"[{answer.status}{frame}] {answer.message}\n> ")
                    if self._spoken_reply is not None:
                        self._emit(f"[reply_speech:{self._spoken_reply.status}]\n")
        except EOFError:
            if not self._stop.is_set():
                try:
                    self._emit("\n文字输入已结束；感知与告警继续，Ctrl+C 停止程序。\n")
                except Exception as exc:
                    self.error = f"console unavailable: {type(exc).__name__}"
        except Exception as exc:
            # 不记录可能包含私人请求内容的异常正文。
            self.error = f"console unavailable: {type(exc).__name__}"
        finally:
            self.finished.set()

    def close(self) -> None:
        """有界停止交互；不关闭进程共享的标准输入输出。"""
        self._stop.set()
        if self._spoken_reply is not None:
            self._spoken_reply.close()
        if self._thread.ident is not None:
            self._thread.join(timeout=0.5)
            if self._thread.is_alive():
                self.error = "console shutdown timed out"
