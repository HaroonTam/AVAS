"""普通回答语音边界测试，只使用模拟播放句柄。"""

import io
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from time import monotonic, time_ns
from unittest.mock import MagicMock, patch

from app.agent.spoken_reply import SpokenReply
from app.agent.tools import SceneTools
from app.agent.vision_agent import AssistantAnswer, VisionAssistant
from app.live import main
from app.safety.config import RiskConfig
from app.safety.scene import RiskObject, RiskScene
from app.scene_store import SceneStore
from app.speech.tts import SpeechMessage, SpeechWorker
from tests.test_live import FakeSpeech


class SpokenReplyTests(unittest.TestCase):
    def setUp(self) -> None:
        """构造长有效期合成场景，避免测试调度耗时造成随机过期。"""
        self.now_ms = time_ns() // 1_000_000
        self.store = SceneStore(RiskConfig(freshness_ms=10000), clock_ms=self.clock_ms)
        self.scene = RiskScene(
            "synthetic:1",
            self.now_ms,
            "live",
            True,
            (RiskObject(1, "synthetic:1", "chair", 0.9, "front"),),
        )
        self.store.publish(self.scene)
        self.agent = VisionAssistant(SceneTools(self.store))

    def clock_ms(self) -> int:
        """返回可推进的合成 Unix 毫秒。"""
        return self.now_ms

    def test_reply_is_low_priority_and_high_warning_preempts(self) -> None:
        """真实适配路径提交普通回答后，高优先级告警可以抢占其模拟播放。"""
        backend = FakeSpeech()
        worker = SpeechWorker(backend)
        try:
            speaker = SpokenReply(self.agent, self.store, worker)
            answer = speaker.respond("寻找 椅子")
            self.assertEqual(speaker.status, "queued_recheck_before_start")
            self.assertTrue(backend.first.wait(1))
            self.assertEqual(backend.messages, [answer.message])
            worker.submit(SpeechMessage("紧急告警", 2, monotonic() + 5))
            self.assertTrue(backend.second.wait(1))
            self.assertTrue(backend.handles[0].stopped.is_set())
        finally:
            worker.close()

    def test_queued_reply_dropped_after_same_frame_republication(self) -> None:
        """等待告警期间同帧重新发布也撤销回答，防止同名帧证据被替换。"""
        backend = FakeSpeech()
        worker = SpeechWorker(backend)
        try:
            worker.submit(SpeechMessage("先播告警", 2, monotonic() + 5))
            self.assertTrue(backend.first.wait(1))
            speaker = SpokenReply(self.agent, self.store, worker)
            speaker.respond("寻找 椅子")
            self.store.publish(replace(self.scene, objects=()))
            backend.handles[0].stop()
            self.assertTrue(worker.wait_idle(1))
            self.assertEqual(backend.messages, ["先播告警"])
        finally:
            worker.close()

    def test_queued_reply_dropped_after_time_expiry(self) -> None:
        """排队期限尚未到但原场景过期时，也不得播放回答。"""
        backend = FakeSpeech()
        worker = SpeechWorker(backend)
        try:
            worker.submit(SpeechMessage("先播告警", 2, monotonic() + 5))
            self.assertTrue(backend.first.wait(1))
            speaker = SpokenReply(self.agent, self.store, worker)
            speaker.respond("描述周围")
            self.now_ms += 10001
            backend.handles[0].stop()
            self.assertTrue(worker.wait_idle(1))
            self.assertEqual(backend.messages, ["先播告警"])
        finally:
            worker.close()

    def test_invalidation_revokes_lease_and_cancels_active_audio(self) -> None:
        """故障清空场景并取消语音后，旧回答播放停止。"""
        backend = FakeSpeech()
        worker = SpeechWorker(backend)
        lease = self.store.acquire_lease()
        self.assertIsNotNone(lease)
        try:
            SpokenReply(self.agent, self.store, worker).respond("寻找 椅子")
            self.assertTrue(backend.first.wait(1))
            self.store.invalidate()
            worker.cancel()
            self.assertTrue(worker.wait_idle(1))
            self.assertTrue(backend.handles[0].stopped.is_set())
            if lease is not None:
                self.assertFalse(lease.is_valid())
        finally:
            worker.close()

    def test_expired_lease_cannot_revive_when_clock_rolls_back(self) -> None:
        """凭据时效失败后，即使墙上时钟回拨也不能再次可用。"""
        lease = self.store.acquire_lease()
        self.assertIsNotNone(lease)
        if lease is None:
            self.fail("missing synthetic lease")
        self.now_ms += 10001
        self.assertFalse(lease.is_valid())
        self.now_ms -= 10001
        self.assertFalse(lease.is_valid())
        self.assertEqual(self.store.read().status, "scene_stale")

    def test_scene_change_during_response_prevents_submission(self) -> None:
        """最终回答产生期间场景变化时，旧凭据阻止回答进入语音队列。"""
        speech = MagicMock()
        speech.error = None
        answer = self.agent.respond("寻找 椅子")

        def changed_response(text: str) -> AssistantAnswer:
            """模拟查询结束后发生发布，不添加自由生成内容。"""
            self.store.publish(self.scene)
            return answer

        with patch.object(self.agent, "respond", side_effect=changed_response):
            speaker = SpokenReply(self.agent, self.store, speech)
            speaker.respond("寻找 椅子")
        speech.submit.assert_not_called()
        self.assertEqual(speaker.status, "scene_unavailable_or_changed")

    def test_unknown_request_feedback_and_long_answer_rejection(self) -> None:
        """未知请求使用固定反馈，超长事实回答不截断可能重要的风险提示。"""
        speech = MagicMock()
        speech.error = None
        speaker = SpokenReply(self.agent, self.store, speech)
        speaker.respond("未知指令")
        self.assertEqual(speaker.status, "feedback_queued")
        self.assertIn("本次请求不支持", speech.submit.call_args.args[0].text)
        speech.reset_mock()
        self.store.publish(
            replace(
                self.scene, objects=(replace(self.scene.objects[0], label="x" * 400),)
            )
        )
        speaker.respond("描述周围")
        self.assertEqual(speaker.status, "text_too_long")
        speech.submit.assert_not_called()

    def test_feedback_uses_whitelist_not_answer_body(self) -> None:
        """无帧错误状态只产生固定提示，不能借错误正文播出伪造场景或自由文本。"""
        speech = MagicMock()
        speech.error = None
        speaker = SpokenReply(self.agent, self.store, speech)
        for status in (
            "unsupported_request",
            "tools_unavailable",
            "scene_unavailable",
            "scene_invalid",
            "scene_stale",
            "scene_changed",
            "answer_expired",
            "frame_mismatch",
        ):
            with patch.object(
                self.agent,
                "respond",
                return_value=AssistantAnswer(status, None, "伪造：前方一米安全"),
            ):
                speaker.respond("合成请求")
            message = speech.submit.call_args.args[0]
            self.assertNotIn("伪造", message.text)
            self.assertNotIn("一米", message.text)
            self.assertEqual(message.priority, 0)
            self.assertIsNone(message.scene_lease)
        speech.reset_mock()
        speaker.notify("伪造：前方一米安全")
        self.assertEqual(speaker.status, "feedback_unsupported")
        with patch.object(
            self.agent,
            "respond",
            return_value=AssistantAnswer("available", None, "伪造事实"),
        ):
            speaker.respond("寻找 椅子")
        speech.submit.assert_not_called()

    def test_feedback_without_scene_is_preemptible(self) -> None:
        """无场景也能反馈本次失败；紧急告警仍抢占模拟语音。"""
        self.store.invalidate()
        backend = FakeSpeech()
        worker = SpeechWorker(backend)
        try:
            speaker = SpokenReply(self.agent, self.store, worker)
            answer = speaker.respond("描述周围")
            self.assertIsNone(answer.frame_id)
            self.assertEqual(speaker.status, "feedback_queued")
            self.assertTrue(backend.first.wait(1))
            self.assertIn("本次查询", backend.messages[0])
            worker.submit(SpeechMessage("紧急告警", 2, monotonic() + 5))
            self.assertTrue(backend.second.wait(1))
            self.assertTrue(backend.handles[0].stopped.is_set())
        finally:
            worker.close()

    def test_feedback_expires_while_waiting_for_warning(self) -> None:
        """反馈等待告警时到期就丢弃，不依赖当前场景是否恢复。"""
        backend = FakeSpeech()
        worker = SpeechWorker(backend)
        try:
            worker.submit(SpeechMessage("先播告警", 2, monotonic() + 5))
            self.assertTrue(backend.first.wait(1))
            speaker = SpokenReply(self.agent, self.store, worker)
            with patch(
                "app.agent.spoken_reply.monotonic", return_value=monotonic() - 9.5
            ):
                speaker.notify("stt:timeout")
            self.assertEqual(speaker.status, "feedback_queued")
            with patch("app.speech.tts.monotonic", return_value=monotonic() + 1):
                backend.handles[0].stop()
                self.assertTrue(worker.wait_idle(1))
            self.assertEqual(backend.messages, ["先播告警"])
        finally:
            worker.close()

    def test_feedback_backend_queue_and_closed_status(self) -> None:
        """反馈复用故障和满队列状态，关闭后不再提交，也不取消告警。"""
        speech = MagicMock()
        speaker = SpokenReply(self.agent, self.store, speech)
        speech.error = "synthetic failure"
        speaker.notify("stt:unavailable")
        self.assertEqual(speaker.status, "speech_unavailable")
        speech.submit.assert_not_called()
        speech.error = None
        speech.submit.return_value = False
        speaker.notify("stt:timeout")
        self.assertEqual(speaker.status, "queue_rejected")
        speech.reset_mock()
        speaker.close()
        speaker.notify("stt:cancelled")
        self.assertEqual(speaker.status, "closed")
        speech.submit.assert_not_called()
        speech.cancel.assert_not_called()

    def test_speech_failure_and_full_queue_are_visible(self) -> None:
        """语音后端错误和队列拒绝有明确状态，不冒充已经发声。"""
        speech = MagicMock()
        speaker = SpokenReply(self.agent, self.store, speech)
        speech.error = "backend failure"
        speaker.respond("寻找 椅子")
        self.assertEqual(speaker.status, "speech_unavailable")
        speech.submit.assert_not_called()
        speech.error = None
        speech.submit.return_value = False
        speaker.respond("寻找 椅子")
        self.assertEqual(speaker.status, "queue_rejected")
        self.assertEqual(speech.submit.call_args.args[0].priority, 0)

    def test_reply_flags_rejected_before_hardware_initialization(self) -> None:
        """缺少控制台或显式静音时拒绝朗读参数，且不启动设备。"""
        for args in (
            ["--speak-replies"],
            ["--console", "--speak-replies", "--no-speech"],
        ):
            with (
                patch("app.live.CameraService") as camera,
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                main(args)
            self.assertEqual(raised.exception.code, 2)
            camera.assert_not_called()

    def test_closed_adapter_does_not_submit_late_response(self) -> None:
        """交互关闭后即使查询返回也不提交普通语音，不调用全局取消。"""
        speech = MagicMock()
        speech.error = None
        speaker = SpokenReply(self.agent, self.store, speech)
        speaker.close()
        speaker.respond("寻找 椅子")
        self.assertEqual(speaker.status, "closed")
        speech.submit.assert_not_called()
        speech.cancel.assert_not_called()

    def test_live_connects_reply_adapter_to_shared_speech_worker(self) -> None:
        """显式朗读参数将控制台接到同一语音队列，不另起独立音频通路。"""
        with (
            patch("app.live.WindowsLineSource"),
            patch("app.live.ConsoleWorker") as console,
            patch("app.live.SpokenReply") as adapter,
            patch("app.live.SpeechWorker") as speech,
            patch("app.live.WindowsSpeechBackend"),
            patch("app.live.Yolo11Detector"),
            patch("app.vision.depth_estimator.DepthAnythingV2Estimator"),
            patch("app.live.CameraService"),
        ):
            console.return_value.exit_requested.is_set.return_value = True
            console.return_value.error = None
            speech.return_value.error = None
            self.assertEqual(main(["--console", "--speak-replies"]), 0)
        self.assertIs(adapter.call_args.args[2], speech.return_value)
        self.assertIs(console.call_args.kwargs["spoken_reply"], adapter.return_value)
