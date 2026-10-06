"""普通回答的本地语音适配；不改变紧急告警或接受任意生成文本。"""

from threading import Event
from time import monotonic

from app.agent.vision_agent import AssistantAnswer, VisionAssistant, usage_help
from app.scene_store import SceneStore
from app.speech.tts import SpeechMessage, SpeechWorker, speech_duration_bound

# 普通回答与固定反馈的排队上限；防止陈旧提示长期滞留，0 表示不扩展。
MAX_QUEUE_HORIZON_S = 30.0


def feedback_text(status: str) -> str | None:
    """只将白名单状态映射为固定帮助或操作反馈，不朗读外部正文或断言当前场景。"""
    messages = {
        "help": usage_help(),
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
        self,
        assistant: VisionAssistant,
        store: SceneStore,
        speech: SpeechWorker,
        *,
        queue_horizon_s: float = 8.0,
    ) -> None:
        """共享场景与语音队列；回答只能以最低优先级提交。

        queue_horizon_s 是固定反馈与事实回答在队列中的最小保留秒数，用于覆盖
        前一条长回答按实测语速的播放时长；0 表示不扩展。事实时效仍由场景凭据
        在播放前复核决定，这里只影响消息在队列中的存活时间。
        """
        if not 0.0 <= queue_horizon_s <= MAX_QUEUE_HORIZON_S:
            raise ValueError("queue horizon must be within [0, 30] seconds")
        self._assistant = assistant
        self._store = store
        self._speech = speech
        self._queue_horizon_s = queue_horizon_s
        self._queued_bound_s = 0.0
        self._last_queued_at: float | None = None
        self.status = "not_requested"
        self._closed = Event()

    def _queue_expiry(self, text: str) -> float:
        """按实测语速给出排队有效期秒数；不比原场景新鲜度更短。

        前一条回答可能播放数十秒，固定期限会让后到的提示一入队就过期。累积量是
        本次连续提交的预估时长；距上次提交超过该时长时视为队列已排空并重新计时。
        """
        now = monotonic()
        if self._queue_horizon_s <= 0.0:
            return now + self._store.config.freshness_ms / 1000
        if (
            self._last_queued_at is None
            or now - self._last_queued_at >= self._queued_bound_s
        ):
            self._queued_bound_s = 0.0
        life_s = max(
            self._store.config.freshness_ms / 1000,
            self._queued_bound_s + self._queue_horizon_s,
            speech_duration_bound(text),
        )
        self._queued_bound_s += speech_duration_bound(text)
        self._last_queued_at = now
        return now + life_s

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
            accepted = self._speech.submit(
                SpeechMessage(message, 0, self._queue_expiry(message))
            )
            self.status = "feedback_queued" if accepted else "queue_rejected"

    def respond(self, text: str) -> AssistantAnswer:
        """事实回答绑定场景凭据；无帧的帮助及已知失败仅朗读固定白名单文本。"""
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
        # 凭据负责精确的原始场景期限及撤销；排队期限只决定消息在队列中保留多久。
        accepted = self._speech.submit(
            SpeechMessage(answer.message, 0, self._queue_expiry(answer.message), lease)
        )
        self.status = "queued_recheck_before_start" if accepted else "queue_rejected"
        return answer
