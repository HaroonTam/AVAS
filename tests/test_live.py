"""实时流水线与语音调度测试，不打开摄像头、不播放声音。"""

import base64
import io
import json
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from queue import Queue
from threading import Event
from time import monotonic, sleep, time_ns
from unittest.mock import MagicMock, patch

import numpy as np

from app.agent.vision_agent import usage_help
from app.camera.camera_service import (
    CameraConfig,
    CameraService,
    CapturedFrame,
    capture_worker,
    publish_latest,
)
from app.fusion.config import FusionConfig
from app.live import (
    SPEECH_FINAL_DRAIN_MAX_S,
    PerceptionPipeline,
    dispatch_warnings,
    final_notice_wait_seconds,
    main,
)
from app.safety.config import RiskConfig
from app.safety.risk_engine import RiskAssessment, WarningEvent, assess_scene
from app.safety.scene import MetricEvidence, RiskObject, RiskScene
from app.safety.warnings import RepeatLimiter, WarningGate
from app.scene_store import SceneStore
from app.speech.tts import (
    SpeechHandle,
    SpeechMessage,
    SpeechWorker,
    WindowsSpeechBackend,
    playback_limit,
    speech_duration_bound,
)
from app.vision.depth_estimator import RelativeDepth
from app.vision.detector import Detection, Image


class FakeDetector:
    def __init__(self) -> None:
        """记录模型调用次数以验证复用和过期帧拒绝。"""
        self.calls = 0

    def detect(self, image: Image) -> tuple[Detection, ...]:
        """返回固定框，不依赖真实模型。"""
        self.calls += 1
        return (Detection("chair", 0.9, (0, 0, 20, 20)),)


class BrokenDepth:
    def estimate(self, image: Image, frame_id: str) -> RelativeDepth:
        """模拟深度故障以验证检测仍保留。"""
        raise RuntimeError("synthetic depth failure")


class FakeHandle:
    def __init__(self) -> None:
        """模拟持续播放直到抢占或关闭。"""
        self.stopped = Event()

    def poll(self) -> int | None:
        """停止前保持播放状态。"""
        return 0 if self.stopped.is_set() else None

    def stop(self) -> None:
        """记录停止，不触发音频设备。"""
        self.stopped.set()


class FakeSpeech:
    def __init__(self) -> None:
        """保存播放顺序，通过事件同步测试。"""
        self.messages: list[str] = []
        self.handles: list[FakeHandle] = []
        self.first = Event()
        self.second = Event()

    def start(self, text: str) -> SpeechHandle:
        """快速返回模拟播放句柄，后台不等待其结束。"""
        handle = FakeHandle()
        self.messages.append(text)
        self.handles.append(handle)
        self.first.set()
        if len(self.messages) == 2:
            self.second.set()
        return handle


class ScriptedHandle:
    def __init__(self, code: int) -> None:
        """按给定退出码立即结束，用来模拟一次成功或失败的本地播放。"""
        self._code = code

    def poll(self) -> int | None:
        """始终返回预设退出码。"""
        return self._code

    def stop(self) -> None:
        """播放已经结束，无需额外动作。"""
        return None


class ScriptedSpeech:
    def __init__(self, codes: list[int]) -> None:
        """按顺序为每次播放分配退出码，并记录播放文本。"""
        self._codes = list(codes)
        self.messages: list[str] = []

    def start(self, text: str) -> SpeechHandle:
        """返回立即结束的模拟句柄，不访问音频设备。"""
        self.messages.append(text)
        code = self._codes.pop(0) if self._codes else 0
        return ScriptedHandle(code)


class LiveTests(unittest.TestCase):
    def frame(self) -> CapturedFrame:
        """创建当前合成 BGR 帧，时间与生产代码使用同一基准。"""
        return CapturedFrame(
            "test:1",
            time_ns() // 1_000_000,
            monotonic(),
            np.zeros((20, 20, 3), np.uint8),
        )

    def test_depth_failure_keeps_detection_and_models_reused(self) -> None:
        """深度异常不丢目标；米制距离继续未知，模型实例重复使用。"""
        detector = FakeDetector()
        pipeline = PerceptionPipeline(
            detector,
            BrokenDepth(),
            FusionConfig(camera_orientation="forward"),
            RiskConfig(),
        )
        with self.assertLogs(level="ERROR"):
            result = pipeline.process(self.frame())
            pipeline.process(self.frame())
        self.assertEqual(detector.calls, 2)
        self.assertEqual(result.depth_status, "unavailable")
        self.assertEqual(result.scene.objects[0].label, "chair")
        self.assertIsNone(result.scene.objects[0].metric)
        self.assertEqual(result.scene.objects[0].direction, "front")

    def test_stale_frame_skips_inference(self) -> None:
        """过期帧不消耗模型推理，也不重置时间冒充当前场景。"""
        detector = FakeDetector()
        pipeline = PerceptionPipeline(detector, None, FusionConfig(), RiskConfig())
        old = replace(self.frame(), captured_at_ms=1)
        result = pipeline.process(old)
        self.assertEqual(detector.calls, 0)
        self.assertEqual(result.depth_status, "not_processed_stale")
        self.assertEqual(
            assess_scene(result.scene, time_ns() // 1_000_000, RiskConfig()).status,
            "scene_stale",
        )

    def test_high_warning_preempts_ordinary_speech(self) -> None:
        """高风险直接打断普通语音，不等待普通播放或任何 Agent 返回。"""
        backend = FakeSpeech()
        worker = SpeechWorker(backend)
        try:
            worker.submit(SpeechMessage("普通回答", 0, monotonic() + 10))
            self.assertTrue(backend.first.wait(1))
            item = RiskObject(
                1,
                "f",
                "chair",
                0.9,
                "front",
                MetricEvidence(0.5, "synthetic-only", 0, 10),
            )
            scene = RiskScene("f", 1000, "live", True, (item,))
            assessment = assess_scene(scene, 1000, RiskConfig())
            self.assertEqual(
                dispatch_warnings(
                    assessment, WarningGate(3000), worker, monotonic() + 10
                ),
                1,
            )
            self.assertTrue(backend.second.wait(1))
            self.assertTrue(backend.handles[0].stopped.is_set())
            self.assertIn("请立即注意", backend.messages[1])
        finally:
            worker.close()
        self.assertTrue(backend.handles[-1].stopped.is_set())

    def test_degraded_notice_uses_lowest_speech_priority(self) -> None:
        """持续未知只产生降级提示，以最低优先级提交，不抢占危险或普通回答。"""
        speech = MagicMock()
        speech.submit.return_value = True
        item = RiskObject(1, "f", "chair", 0.9, "front")
        assessment = assess_scene(
            RiskScene("f", 1000, "live", True, (item,)), 1000, RiskConfig()
        )
        with self.assertLogs(level="WARNING"):
            accepted = dispatch_warnings(
                assessment, WarningGate(3000), speech, monotonic() + 10
            )
        self.assertEqual(accepted, 2)
        priorities = [call.args[0].priority for call in speech.submit.call_args_list]
        self.assertEqual(priorities, [0, 0])

    def test_frame_local_hazard_keys_are_paced_by_content(self) -> None:
        """无跟踪器时不同帧的高风险提示按内容限速，避免占满语音队列。"""
        speech = MagicMock()
        speech.submit.return_value = True
        limiter = RepeatLimiter(3000)
        accepted: list[int] = []
        for frame_id in ("f1", "f2", "f3"):
            item = RiskObject(
                1,
                frame_id,
                "chair",
                0.9,
                "front",
                MetricEvidence(0.5, "synthetic-only", 0, 10),
            )
            assessment = assess_scene(
                RiskScene(frame_id, 1000, "live", True, (item,)), 1000, RiskConfig()
            )
            with self.assertLogs(level="WARNING"):
                accepted.append(
                    dispatch_warnings(
                        assessment,
                        WarningGate(3000),
                        speech,
                        monotonic() + 10,
                        repeat_limiter=limiter,
                    )
                )
        self.assertEqual(accepted, [1, 0, 0])

    def test_expired_input_and_queue_priority(self) -> None:
        """过期内容拒绝；低优先级不能挤掉等待的紧急告警。"""
        backend = FakeSpeech()
        worker = SpeechWorker(backend, capacity=1)
        try:
            self.assertFalse(worker.submit(SpeechMessage("旧信息", 0, monotonic() - 1)))
            worker.submit(SpeechMessage("紧急一", 2, monotonic() + 10))
            self.assertTrue(backend.first.wait(1))
            self.assertTrue(worker.submit(SpeechMessage("紧急二", 2, monotonic() + 10)))
            self.assertFalse(
                worker.submit(SpeechMessage("普通信息", 0, monotonic() + 10))
            )
            backend.handles[0].stop()
            self.assertTrue(backend.second.wait(1))
            self.assertEqual(backend.messages, ["紧急一", "紧急二"])
        finally:
            worker.close()
        self.assertFalse(worker.submit(SpeechMessage("已关闭", 2, monotonic() + 10)))

    def test_speech_timeout_reports_failure(self) -> None:
        """卡住的本地播放会被终止并明确暴露故障。"""
        backend = FakeSpeech()
        worker = SpeechWorker(backend, timeout_s=0.03)
        try:
            with self.assertLogs(level="ERROR"):
                worker.submit(SpeechMessage("超时", 1, monotonic() + 10))
                self.assertTrue(backend.first.wait(1))
                self.assertTrue(backend.handles[0].stopped.wait(1))
                worker.close()
            self.assertIsNotNone(worker.error)
        finally:
            worker.close()

    def test_windows_text_is_encoded_data(self) -> None:
        """恶意形状文本不能作为 PowerShell 代码执行，使用隐藏进程启动。"""
        with (
            patch("app.speech.tts.os.name", "nt"),
            patch("app.speech.tts.subprocess.Popen") as process,
        ):
            text = "'; $(Write-Output injected) 中文"
            WindowsSpeechBackend().start(text)
            command = process.call_args.args[0]
            script = base64.b64decode(command[-1]).decode("utf-16-le")
            self.assertNotIn(text, script)
            self.assertIn(base64.b64encode(text.encode()).decode(), script)
            self.assertNotEqual(process.call_args.kwargs["creationflags"], 0)

    def test_camera_and_message_config_rejected(self) -> None:
        """拒绝非法设备配置、空图像和无限期语音。"""
        with self.assertRaises(ValueError):
            CameraConfig(index=-1)
        with self.assertRaises(ValueError):
            CameraConfig(timeout_s=float("nan"))
        with self.assertRaises(ValueError):
            replace(self.frame(), image=np.zeros((0, 20, 3), np.uint8))
        with self.assertRaises(ValueError):
            SpeechMessage("hello", 2, float("inf"))

    def test_camera_failure_invalidates_scene_and_closes_resources(self) -> None:
        """通过 CLI 模拟相机失联，确保输出失效场景并释放采集资源。"""
        import io
        import json
        from contextlib import redirect_stdout

        output = io.StringIO()
        with (
            patch("app.live.Yolo11Detector"),
            patch("app.vision.depth_estimator.DepthAnythingV2Estimator"),
            patch("app.live.CameraService") as camera,
            redirect_stdout(output),
            self.assertLogs(level="WARNING"),
        ):
            camera.return_value.read.side_effect = RuntimeError("camera lost")
            self.assertEqual(main(["--no-speech", "--max-frames", "1"]), 2)
        payload = json.loads(output.getvalue())
        self.assertFalse(payload["scene"]["valid"])
        self.assertEqual(payload["safety"]["status"], "scene_invalid")
        camera.return_value.close.assert_called_once()

    def test_latest_queue_replaces_old_frame(self) -> None:
        """满队列丢弃旧帧，只留下最新帧，不累积图像。"""
        bounded: Queue[CapturedFrame] = Queue(maxsize=1)
        queue = MagicMock(wraps=bounded)
        first = self.frame()
        second = replace(first, frame_id="test:2")
        publish_latest(queue, first)
        publish_latest(queue, second)
        self.assertIs(bounded.get_nowait(), second)
        self.assertTrue(bounded.empty())

    def test_capture_failure_sets_flag_and_releases_device(self) -> None:
        """模拟原生读取失败，验证故障信号、摄像头释放和队列清理。"""
        queue = MagicMock()
        stop, failed = MagicMock(), MagicMock()
        stop.is_set.return_value = False
        with patch("cv2.VideoCapture") as capture:
            capture.return_value.isOpened.return_value = True
            capture.return_value.read.return_value = (False, None)
            capture_worker(CameraConfig(), queue, stop, failed)
        failed.set.assert_called_once()
        capture.return_value.release.assert_called_once()
        queue.cancel_join_thread.assert_called_once()

    def test_camera_timeout_and_idempotent_shutdown(self) -> None:
        """卡住读取时有限等待，关闭强制结束隔离进程且可重复调用。"""
        with patch("app.camera.camera_service.mp.get_context") as context:
            service = CameraService(CameraConfig(timeout_s=0.01))
            process = context.return_value.Process.return_value
            queue = context.return_value.Queue.return_value
            context.return_value.Event.return_value.is_set.return_value = False
            process.is_alive.return_value = True
            from queue import Empty

            queue.get.side_effect = Empty
            service.start()
            with self.assertRaises(TimeoutError):
                service.read()
            service.close()
            service.close()
        process.terminate.assert_called_once()
        queue.close.assert_called_once()
        self.assertFalse(service.healthy())

    def test_cancel_stops_old_speech_before_failure_notice(self) -> None:
        """相机失效可撤销旧告警，最终降级提示不被原告警继续播放掩盖。"""
        backend = FakeSpeech()
        worker = SpeechWorker(backend)
        try:
            worker.submit(SpeechMessage("旧告警", 2, monotonic() + 10))
            self.assertTrue(backend.first.wait(1))
            worker.cancel()
            worker.submit(SpeechMessage("环境不可用", 1, monotonic() + 10))
            self.assertTrue(backend.second.wait(1))
            self.assertTrue(backend.handles[0].stopped.is_set())
            self.assertFalse(worker.wait_idle(0))
            backend.handles[1].stop()
            self.assertTrue(worker.wait_idle(1))
        finally:
            worker.close()

    def test_backend_start_failure_is_reported_without_blocking(self) -> None:
        """语音后端无法启动时仍能退出等待并提供明确故障。"""
        backend = MagicMock()
        backend.start.side_effect = OSError("synthetic launch failure")
        worker = SpeechWorker(backend)
        try:
            with self.assertLogs(level="ERROR"):
                worker.submit(SpeechMessage("测试", 1, monotonic() + 10))
                self.assertTrue(worker.wait_idle(1))
            self.assertIsNotNone(worker.error)
        finally:
            worker.close()

    def test_live_success_and_stale_output(self) -> None:
        """完整入口输出新鲜合成场景，过期输出则清除事实并标记无效。"""
        for stale in (False, True):
            with self.subTest(stale=stale):
                frame = self.frame()
                if stale:
                    frame = replace(frame, captured_at_ms=1)
                output = io.StringIO()
                with (
                    patch("app.live.Yolo11Detector", return_value=FakeDetector()),
                    patch(
                        "app.vision.depth_estimator.DepthAnythingV2Estimator"
                    ) as depth,
                    patch("app.live.CameraService") as camera,
                    redirect_stdout(output),
                    self.assertLogs(level="WARNING"),
                ):
                    camera.return_value.read.return_value = frame
                    camera.return_value.healthy.return_value = True
                    depth.return_value.estimate.return_value = RelativeDepth(
                        frame.frame_id,
                        np.ones((20, 20), np.float32),
                        np.ones((20, 20), bool),
                    )
                    self.assertEqual(main(["--no-speech", "--max-frames", "1"]), 0)
                payload = json.loads(output.getvalue())
                self.assertEqual(payload["current_scene"], not stale)
                self.assertEqual(payload["scene"]["valid"], not stale)
                if stale:
                    self.assertEqual(payload["scene"]["objects"], [])
                    self.assertEqual(payload["observations"], [])
                else:
                    self.assertEqual(payload["scene"]["objects"][0]["label"], "chair")
                    self.assertEqual(payload["safety"]["level"], "unknown")
                camera.return_value.close.assert_called_once()

    def test_runtime_depth_failure_returns_degraded_exit(self) -> None:
        """运行期间深度失败必须返回降级退出码，不能只在初始化失败时处理。"""
        output = io.StringIO()
        with (
            patch("app.live.Yolo11Detector", return_value=FakeDetector()),
            patch(
                "app.vision.depth_estimator.DepthAnythingV2Estimator",
                return_value=BrokenDepth(),
            ),
            patch("app.live.CameraService") as camera,
            redirect_stdout(output),
            self.assertLogs(level="WARNING"),
        ):
            camera.return_value.read.return_value = self.frame()
            self.assertEqual(main(["--no-speech", "--max-frames", "1"]), 2)
        self.assertEqual(json.loads(output.getvalue())["depth_status"], "unavailable")

    def test_scene_publication_follows_warnings_and_shutdown_invalidates(self) -> None:
        """先提交告警再发布查询场景，正常退出也清除可查询事实。"""
        store = SceneStore(RiskConfig())
        order: list[str] = []

        def dispatch(
            assessment: RiskAssessment,
            gate: WarningGate,
            speech: SpeechWorker | None,
            expires_at: float,
            *,
            repeat_limiter: RepeatLimiter | None = None,
            degraded_queue_horizon_s: float = 0.0,
        ) -> int:
            """记录告警提交位置并核对内容限速已接入，不访问音频设备。"""
            self.assertIsNotNone(repeat_limiter)
            order.append("warning")
            return 0

        def publish(scene: RiskScene) -> None:
            """验证发布场景可用并记录发布顺序。"""
            order.append("scene")
            SceneStore.publish(store, scene)
            self.assertEqual(store.read().status, "available")

        with (
            patch("app.live.Yolo11Detector", return_value=FakeDetector()),
            patch(
                "app.vision.depth_estimator.DepthAnythingV2Estimator",
                return_value=BrokenDepth(),
            ),
            patch("app.live.CameraService") as camera,
            patch("app.live.dispatch_warnings", side_effect=dispatch),
            patch.object(store, "publish", side_effect=publish),
            redirect_stdout(io.StringIO()),
            self.assertLogs(level="WARNING"),
        ):
            camera.return_value.read.return_value = self.frame()
            self.assertEqual(
                main(["--no-speech", "--max-frames", "1"], scene_store=store), 2
            )
        self.assertEqual(order, ["warning", "scene"])
        self.assertEqual(store.read().status, "scene_invalid")

    def test_publication_failure_does_not_stop_next_warning(self) -> None:
        """查询发布拒绝输入时，下一帧仍执行确定性告警。"""
        store = SceneStore(RiskConfig())
        with (
            patch("app.live.Yolo11Detector", return_value=FakeDetector()),
            patch(
                "app.vision.depth_estimator.DepthAnythingV2Estimator",
                return_value=BrokenDepth(),
            ),
            patch("app.live.CameraService") as camera,
            patch("app.live.dispatch_warnings", return_value=0) as warnings,
            patch.object(store, "publish", side_effect=ValueError("invalid scene")),
            redirect_stdout(io.StringIO()),
            self.assertLogs(level="WARNING"),
        ):
            camera.return_value.read.return_value = self.frame()
            self.assertEqual(
                main(["--no-speech", "--max-frames", "2"], scene_store=store), 2
            )
        self.assertEqual(warnings.call_count, 2)
        self.assertIsNone(store.read().scene)

    def test_playback_limit_follows_measured_speech_rate(self) -> None:
        """播放上限以配置值为基础额度、按实测语速追加并封顶。"""
        self.assertAlmostEqual(playback_limit("长" * 150, 15.0), 63.0, places=6)
        self.assertEqual(playback_limit("长" * 300, 15.0), 90.0)
        self.assertAlmostEqual(playback_limit("长" * 30, 30.0), 39.6, places=6)
        # 较小额度仍能快速触发超时终止，供卡死播放的故障路径使用。
        self.assertLess(playback_limit("超时", 0.03), 1.0)
        # 帮助全文 150 字符实测需 39.47 s，播放上限与时长上界都必须覆盖该测量值。
        self.assertGreaterEqual(playback_limit(usage_help(), 15.0), 39.47)
        self.assertGreaterEqual(speech_duration_bound(usage_help()), 39.47)

    def test_long_text_is_not_killed_by_short_configured_timeout(self) -> None:
        """长文本按长度放宽播放上限，不再被配置下限触发超时终止。"""
        backend = FakeSpeech()
        worker = SpeechWorker(backend, timeout_s=0.05)
        try:
            worker.submit(SpeechMessage("长" * 100, 0, monotonic() + 30))
            self.assertTrue(backend.first.wait(1))
            sleep(0.3)
            self.assertIsNone(worker.error)
            self.assertFalse(backend.handles[0].stopped.is_set())
        finally:
            worker.close()

    def test_successful_playback_clears_transient_failure(self) -> None:
        """一次播放失败后成功的播放必须撤销故障标记，避免永久关闭语音报告。"""
        backend = ScriptedSpeech([1, 0])
        worker = SpeechWorker(backend)
        try:
            with self.assertLogs(level="ERROR"):
                worker.submit(SpeechMessage("失败提示", 0, monotonic() + 10))
                deadline = monotonic() + 2
                while worker.error is None and monotonic() < deadline:
                    sleep(0.01)
                self.assertIsNotNone(worker.error)
                worker.submit(SpeechMessage("正常提示", 0, monotonic() + 10))
                deadline = monotonic() + 2
                while worker.error is not None and monotonic() < deadline:
                    sleep(0.01)
                self.assertIsNone(worker.error)
                self.assertEqual(backend.messages, ["失败提示", "正常提示"])
        finally:
            worker.close()

    def test_degraded_queue_horizon_spares_only_degraded_notices(self) -> None:
        """排队上限只放宽降级提示，危险告警仍使用传入的场景新鲜度。"""
        speech = MagicMock()
        speech.submit.return_value = True
        expires_at = monotonic() + 1.0
        item = RiskObject(1, "f", "chair", 0.9, "front")
        degraded = assess_scene(
            RiskScene("f", 1000, "live", True, (item,)), 1000, RiskConfig()
        )
        with self.assertLogs(level="WARNING"):
            accepted = dispatch_warnings(
                degraded,
                WarningGate(3000),
                speech,
                expires_at,
                degraded_queue_horizon_s=8.0,
            )
        self.assertEqual(accepted, 2)
        degraded_expiries = [
            call.args[0].expires_at for call in speech.submit.call_args_list
        ]
        self.assertTrue(all(value >= monotonic() + 7.0 for value in degraded_expiries))

        hazard_item = RiskObject(
            2, "f2", "chair", 0.9, "front", MetricEvidence(0.5, "synthetic-only", 0, 10)
        )
        hazard = assess_scene(
            RiskScene("f2", 1000, "live", True, (hazard_item,)), 1000, RiskConfig()
        )
        speech.reset_mock()
        with self.assertLogs(level="WARNING"):
            dispatch_warnings(
                hazard,
                WarningGate(3000),
                speech,
                expires_at,
                degraded_queue_horizon_s=8.0,
            )
        self.assertEqual(len(speech.submit.call_args_list), 1)
        self.assertEqual(speech.submit.call_args_list[0].args[0].expires_at, expires_at)

    def test_final_notice_wait_follows_measured_notice(self) -> None:
        """结束提示的排空等待按文本长度计算，而不是固定 5 秒。"""
        events = (
            WarningEvent(
                ("degraded", "scene_invalid"),
                "unknown",
                "degraded",
                "当前环境信息不可用，请谨慎。",
            ),
        )
        wait = final_notice_wait_seconds(events)
        self.assertGreater(wait, 5.0)
        self.assertLessEqual(wait, SPEECH_FINAL_DRAIN_MAX_S)
        self.assertEqual(final_notice_wait_seconds(()), 1.0)
