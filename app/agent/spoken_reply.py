"""普通回答的本地语音适配；不改变紧急告警或接受任意生成文本。"""

from threading import Event
from time import monotonic

from app.agent.vision_agent import AssistantAnswer, VisionAssistant
from app.scene_store import SceneStore
from app.speech.tts import SpeechMessage, SpeechWorker


class SpokenReply:
    def __init__(
        self, assistant: VisionAssistant, store: SceneStore, speech: SpeechWorker
    ) -> None:
        """共享场景与语音队列；回答只能以最低优先级提交。"""
        self._assistant = assistant
        self._store = store
        self._speech = speech
        self.status = "not_requested"
        self._closed = Event()

    def close(self) -> None:
        """关闭普通回答提交，不取消同一队列中的紧急告警。"""
        self._closed.set()

    def respond(self, text: str) -> AssistantAnswer:
        """查询前绑定场景版本，回答复核后提交；超长或无有效依据的回答只显示文字。"""
        lease = self._store.acquire_lease()
        answer = self._assistant.respond(text)
        self.status = "not_queued"
        if self._closed.is_set():
            self.status = "closed"
            return answer
        if lease is None or answer.frame_id != lease.frame_id or not lease.is_valid():
            self.status = "scene_unavailable_or_changed"
            return answer
        if len(answer.message) > 300:
            self.status = "text_too_long"
            return answer
        if self._speech.error:
            self.status = "speech_unavailable"
            return answer
        # 队列还受现有时效上限约束；凭据负责精确的原始场景期限及撤销。
        expiry = monotonic() + self._store.config.freshness_ms / 1000
        accepted = self._speech.submit(SpeechMessage(answer.message, 0, expiry, lease))
        self.status = "queued_recheck_before_start" if accepted else "queue_rejected"
        return answer
