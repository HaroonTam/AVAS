"""合成控制台交互验收，不创建真实终端、麦克风或音频进程。"""

import json
from collections import deque
from dataclasses import dataclass
from threading import Event

from app.agent.console import ConsoleWorker
from app.agent.spoken_reply import SpokenReply, feedback_text
from app.agent.tools import SceneTools
from app.agent.vision_agent import VisionAssistant
from app.scene_store import SceneStore
from app.speech.stt import Recognition, SttConfig, parse_recognition
from app.speech.tts import SpeechWorker


class ScriptedLines:
    def __init__(self) -> None:
        """提供有限脚本，首次听取取消，之后完成四次听取再退出。"""
        self._lines = deque(
            ["listen", "cancel", "listen", "listen", "listen", "listen", "quit"]
        )

    def poll(self) -> str | None:
        """逐行提供合成输入，脚本耗尽后保持非阻塞空闲。"""
        return self._lines.popleft() if self._lines else None


class ScriptedRecognizer:
    def __init__(self) -> None:
        """固定识别事件；不构造 Windows 识别器或访问录音设备。"""
        self.calls = 0
        self.cancelled = False

    def recognize(self, stop: Event) -> Recognition:
        """首次故意返回取消后的迟到成功；其余结果经过生产识别白名单解析。"""
        self.calls += 1
        if self.calls == 1:
            self.cancelled = stop.wait(2)
            return Recognition("recognized", "寻找 汽车", 0.9)
        events = (
            ("寻找人", 0.1),
            ("前方有什么", 0.9),
            ("寻找椅子", 0.9),
            ("左侧的椅子在哪里", 0.9),
        )
        if not 2 <= self.calls <= 5:
            return Recognition("unavailable")
        text, confidence = events[self.calls - 2]
        payload = json.dumps(
            {"status": "recognized", "text": text, "confidence": confidence}
        ).encode()
        return parse_recognition(payload, SttConfig())


@dataclass(frozen=True)
class InteractionCheck:
    name: str
    passed: bool


@dataclass(frozen=True)
class InteractionReport:
    synthetic: bool
    checks: tuple[InteractionCheck, ...]
    transcript: tuple[str, ...]
    simulated_speech: tuple[str, ...]
    passed: bool


def run_interaction_demo(
    store: SceneStore, worker: SpeechWorker, recorded: list[str]
) -> InteractionReport:
    """对调用方的合成双椅子场景执行固定交互，并验证文字、语音及取消结果。"""
    transcript: list[str] = []
    drains: list[bool] = []
    start = len(recorded)

    def emit(text: str) -> None:
        """保存有限输出，每次语音提交后等待记录后端完成，使报告顺序可复现。"""
        transcript.append(text)
        if text.startswith("[reply_speech:"):
            drains.append(worker.wait_idle(2))

    agent = VisionAssistant(SceneTools(store))
    recognizer = ScriptedRecognizer()
    console = ConsoleWorker(
        agent,
        ScriptedLines(),
        emit,
        recognizer=recognizer,
        spoken_reply=SpokenReply(agent, store, worker),
    )
    console.start()
    try:
        finished = console.finished.wait(15)
    finally:
        console.close()
    text = "".join(transcript)
    speech = tuple(recorded[start:])
    answers = [line for line in transcript if line.startswith("[available，")]
    ambiguous = [line for line in transcript if line.startswith("[ambiguous，")]
    checks = (
        InteractionCheck(
            "cancel_then_retry",
            recognizer.cancelled
            and recognizer.calls == 5
            and "[stt:cancelled]" in text
            and "寻找 汽车" not in text,
        ),
        InteractionCheck(
            "low_confidence_feedback",
            "[stt:unrecognized]" in text
            and "寻找 人" not in text
            and feedback_text("stt:unrecognized") in speech,
        ),
        InteractionCheck(
            "direction_summary_and_retry",
            len(answers) == 2
            and "目标 1（" in answers[0]
            and "目标 2（" not in answers[0]
            and "目标 2（" in answers[1]
            and "目标 1（" not in answers[1]
            and all("米制距离不可用" in answer for answer in answers),
        ),
        InteractionCheck(
            "candidate_directions",
            len(ambiguous) == 1
            and "目标 1：图像前方" in ambiguous[0]
            and "目标 2：图像左侧" in ambiguous[0]
            and "可重新按方向寻找" in ambiguous[0],
        ),
        InteractionCheck(
            "simulated_speech_and_shutdown",
            finished
            and console.exit_requested.is_set()
            and console.error is None
            and worker.error is None
            and len(drains) == 5
            and all(drains)
            and len(speech) == 5
            and feedback_text("stt:cancelled") in speech,
        ),
    )
    return InteractionReport(
        True, checks, tuple(transcript), speech, all(check.passed for check in checks)
    )
