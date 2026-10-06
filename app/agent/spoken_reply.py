"""普通回答的本地语音适配；不改变紧急告警或接受任意生成文本。"""

from threading import Event
from time import monotonic

from app.agent.vision_agent import AssistantAnswer, VisionAssistant
from app.scene_store import SceneStore
from app.speech.tts import SpeechMessage, SpeechWorker


def feedback_text(status: str) -> str | None:
    """只将白名单状态映射为本次操作反馈，不朗读外部正文或断言当前场景。"""
    messages = {
        "unsupported_request": "本次请求不支持。请说描述周围、当前风险或寻找椅子。",
        "tools_unavailable": "本次场景查询未完成，请重试。",
        "scene_unavailable": "本次查询没有可用的环境信息，请稍后重试。",
        "scene_invalid": "本次查询没有有效的环境信息，请稍后重试。",
        "scene_stale": "本次查询的环境信息已过期，请重新查询。",
        "scene_changed": "本次查询期间场景已变化，请重新查询。",
        "answer_expired": "本次回答已失效，请重新查询。",
        "frame_mismatch": "本次选择的帧已更新，请重新选择目标。",
        "stt:unrecognized": "本次未可靠识别命令，请重新听取或使用键盘。",
        "stt:timeout": "本次听取超时，请重新听取。",
        "stt:unavailable": "本次语音输入不可用，请检查识别器和麦克风，或使用键盘。",
        "stt:cancelled": "本次听取已取消。",
    }
    return messages.get(status)


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

    def notify(self, status: str) -> None:
        """非阻塞提交固定操作反馈；无需场景凭据，沿用配置有效期和最低优先级。"""
        message = feedback_text(status)
        if self._closed.is_set():
            self.status = "closed"
        elif message is None:
            self.status = "feedback_unsupported"
        elif self._speech.error:
            self.status = "speech_unavailable"
        else:
            expiry = monotonic() + self._store.config.freshness_ms / 1000
            accepted = self._speech.submit(SpeechMessage(message, 0, expiry))
            self.status = "feedback_queued" if accepted else "queue_rejected"

    def respond(self, text: str) -> AssistantAnswer:
        """事实回答绑定场景凭据；无帧的已知失败仅朗读固定操作反馈。"""
        lease = self._store.acquire_lease()
        answer = self._assistant.respond(text)
        self.status = "not_queued"
        if self._closed.is_set():
            self.status = "closed"
            return answer
        if answer.frame_id is None and feedback_text(answer.status) is not None:
            self.notify(answer.status)
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
