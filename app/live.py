"""实时本地入口：采集、感知、确定性风险与独立语音输出。"""

import argparse
import json
import logging
from configparser import ConfigParser
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import monotonic, perf_counter, time_ns
from typing import Protocol

from app.agent.console import ConsoleWorker, WindowsLineSource
from app.agent.spoken_reply import MAX_QUEUE_HORIZON_S, SpokenReply
from app.agent.tools import SceneTools
from app.agent.vision_agent import VisionAssistant
from app.camera.camera_service import CameraConfig, CameraService, CapturedFrame
from app.config import load_config
from app.fusion.config import FusionConfig, load_fusion_config
from app.fusion.depth_fusion import DetectionFrame, Observation, fuse_frame
from app.live_performance import PerformanceRecorder
from app.safety.config import RiskConfig, load_risk_config
from app.safety.risk_engine import RiskAssessment, WarningEvent, assess_scene
from app.safety.scene import RiskObject, RiskScene
from app.safety.warnings import RepeatLimiter, WarningGate
from app.scene_store import SceneStore
from app.speech.stt import SttConfig, WindowsCommandRecognizer
from app.speech.tts import (
    SpeechMessage,
    SpeechWorker,
    WindowsSpeechBackend,
    speech_duration_bound,
)
from app.vision.depth_estimator import RelativeDepth
from app.vision.detector import Detection, Image, Yolo11Detector


class Detector(Protocol):
    def detect(self, image: Image) -> tuple[Detection, ...]:
        """从同帧图像返回原图坐标检测。"""
        ...


class DepthEstimator(Protocol):
    def estimate(self, image: Image, frame_id: str) -> RelativeDepth:
        """返回同身份、同原图网格的相对深度。"""
        ...


@dataclass(frozen=True)
class LiveResult:
    scene: RiskScene
    observations: tuple[Observation, ...]
    depth_status: str
    latencies_ms: dict[str, float]


class PerceptionPipeline:
    def __init__(
        self,
        detector: Detector,
        depth: DepthEstimator | None,
        fusion: FusionConfig,
        risk: RiskConfig,
    ) -> None:
        """复用已加载模型；注入测试后端不改变确定性规则边界。"""
        self._detector = detector
        self._depth = depth
        self._fusion = fusion
        self._risk = risk

    def process(self, frame: CapturedFrame) -> LiveResult:
        """拒绝已过期输入；深度故障保留检测，检测失败由监督循环使场景失效。"""
        scene = RiskScene(frame.frame_id, frame.captured_at_ms, "live", True, ())
        if (
            assess_scene(scene, time_ns() // 1_000_000, self._risk).status
            != "available"
        ):
            return LiveResult(scene, (), "not_processed_stale", {})
        timings: dict[str, float] = {}
        started = perf_counter()
        detections = self._detector.detect(frame.image)
        timings["detection"] = (perf_counter() - started) * 1000
        depth = None
        depth_status = "unavailable"
        if self._depth is not None:
            started = perf_counter()
            try:
                depth = self._depth.estimate(frame.image, frame.frame_id)
                depth_status = "available"
            except Exception as exc:
                logging.error("Depth unavailable: %s", exc)
            timings["depth"] = (perf_counter() - started) * 1000
        started = perf_counter()
        detection_frame = DetectionFrame(
            frame.frame_id, frame.image.shape[1], frame.image.shape[0], detections
        )
        try:
            observations = fuse_frame(detection_frame, depth, self._fusion)
        except ValueError as exc:
            logging.error("Fusion depth rejected: %s", exc)
            observations = fuse_frame(detection_frame, None, self._fusion)
            depth_status = "rejected"
        timings["fusion"] = (perf_counter() - started) * 1000
        objects = tuple(
            RiskObject(
                item.id,
                item.frame_id,
                item.label,
                item.confidence,
                item.direction if item.direction_status == "image_relative" else None,
            )
            for item in observations
        )
        scene = RiskScene(frame.frame_id, frame.captured_at_ms, "live", True, objects)
        return LiveResult(scene, observations, depth_status, timings)


# 结束提示的排空等待上限秒数：超过该值即认为本地播放卡住，如实记录而不无限等待退出。
SPEECH_FINAL_DRAIN_MAX_S = 20.0


def final_notice_wait_seconds(events: tuple[WarningEvent, ...]) -> float:
    """按最长结束提示的实测时长给出有界排空等待秒数，避免最后一条提示被截断。"""
    longest = max(
        (speech_duration_bound(event.message) for event in events), default=0.0
    )
    return min(SPEECH_FINAL_DRAIN_MAX_S, longest + 1.0)


def dispatch_warnings(
    assessment: RiskAssessment,
    gate: WarningGate,
    speech: SpeechWorker | None,
    expires_at: float,
    *,
    repeat_limiter: RepeatLimiter | None = None,
    degraded_queue_horizon_s: float = 0.0,
) -> int:
    """先记录告警再提交本地播放；完全不调用或等待 Agent，返回接收事件数。

    传入 repeat_limiter 时限制同一条文本的重复播放节奏；未传入时不限制，
    供合成测试与一次性调度使用。事件仍照常记录日志。
    传入 degraded_queue_horizon_s（大于 0）时，仅降级提示的排队有效期不低于该秒数，
    并按本批已提交降级提示的预估时长累加，使后续提示在首条、第二条播放期间仍不会被
    过期清理丢弃；危险告警仍严格使用传入的场景新鲜度，不放宽陈旧危险信息的播出条件。
    """
    now_ms = int(monotonic() * 1000)
    events = gate.select(assessment, now_ms)
    accepted = 0
    # 本批已提交降级提示的预估总时长；后面的提示据此预留排队时间。
    queued_bound_s = 0.0
    for event in events:
        logging.warning("%s: %s", event.kind, event.message)
        if speech is not None:
            if repeat_limiter is not None and not repeat_limiter.allow(event, now_ms):
                continue
            # 危险告警保持可抢占优先级；降级提示降到最低，避免持续未知状态刷屏。
            if event.level == "high":
                priority = 2
            elif event.kind == "degraded":
                priority = 0
            else:
                priority = 1
            message_expiry = expires_at
            if event.kind == "degraded" and degraded_queue_horizon_s > 0:
                message_expiry = max(
                    expires_at,
                    monotonic() + queued_bound_s + degraded_queue_horizon_s,
                )
            if speech.submit(SpeechMessage(event.message, priority, message_expiry)):
                accepted += 1
                if event.kind == "degraded" and degraded_queue_horizon_s > 0:
                    queued_bound_s += speech_duration_bound(event.message)
    return accepted


def main(
    argv: list[str] | None = None, *, scene_store: SceneStore | None = None
) -> int:
    """显式启动实时摄像头；语音默认启用，Ctrl+C 清理资源并停止。"""
    parser = argparse.ArgumentParser(
        description="Local live camera and deterministic warnings"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/system.ini",
    )
    parser.add_argument(
        "--no-speech", action="store_true", help="Diagnostic mode: logs/JSON only"
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Enable Windows text queries; omit per-frame JSON output",
    )
    parser.add_argument(
        "--voice-input",
        action="store_true",
        help="Enable one-shot local command STT via console listen command",
    )
    parser.add_argument(
        "--speak-replies",
        action="store_true",
        help="Speak validated console answers at low priority (requires --console)",
    )
    parser.add_argument(
        "--performance-output",
        type=Path,
        help="Opt-in resource metadata JSON (exclusive file; requires JSONL mode)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N results; 0 runs until Ctrl+C",
    )
    args = parser.parse_args(argv)
    if args.max_frames < 0:
        parser.error("--max-frames must be nonnegative")
    if args.speak_replies and (not args.console or args.no_speech):
        parser.error("--speak-replies requires --console and enabled speech")
    if args.voice_input and not args.console:
        parser.error("--voice-input requires --console")
    if args.performance_output and args.console:
        parser.error("--performance-output requires per-frame JSONL mode")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    camera: CameraService | None = None
    speech: SpeechWorker | None = None
    console: ConsoleWorker | None = None
    performance: PerformanceRecorder | None = None
    failed = False
    try:
        settings = ConfigParser(interpolation=None)
        with args.config.open(encoding="utf-8") as source:
            settings.read_file(source)
        if args.performance_output:
            performance = PerformanceRecorder(args.performance_output, args.config)
        risk = load_risk_config(args.config)
        if scene_store is None:
            scene_store = SceneStore(risk)
        elif scene_store.config != risk:
            raise ValueError("scene store must use the live risk configuration")
        console_source = WindowsLineSource() if args.console else None
        fusion = load_fusion_config(args.config)
        camera_config = CameraConfig(
            settings.getint("camera", "index", fallback=0),
            settings.getint("camera", "width", fallback=640),
            settings.getint("camera", "height", fallback=480),
            settings.getfloat("camera", "timeout_s", fallback=3),
            startup_timeout_s=settings.getfloat(
                "camera", "startup_timeout_s", fallback=None
            ),
        )
        logging.info(
            "Camera waits: startup=%s s, read=%s s",
            camera_config.startup_timeout_s,
            camera_config.timeout_s,
        )
        # 降级提示的排队上限：0 表示沿用场景新鲜度；上限防止陈旧提示长期滞留。
        degraded_queue_horizon_s = settings.getfloat(
            "speech", "degraded_queue_horizon_s", fallback=8.0
        )
        if not 0.0 <= degraded_queue_horizon_s <= 30.0:
            raise ValueError("degraded_queue_horizon_s must be within [0, 30]")
        # 普通回答与固定反馈的排队上限：0 表示只按场景新鲜度，不扩展。
        reply_queue_horizon_s = settings.getfloat(
            "speech", "reply_queue_horizon_s", fallback=8.0
        )
        if not 0.0 <= reply_queue_horizon_s <= MAX_QUEUE_HORIZON_S:
            raise ValueError(
                f"reply_queue_horizon_s must be within [0, {MAX_QUEUE_HORIZON_S:g}]"
            )
        gate = WarningGate(
            risk.cooldown_ms, degraded_cooldown_ms=risk.degraded_cooldown_ms
        )
        # 无跟踪器时事件身份含帧 ID；用内容节奏限制同一条提示的重复播放。
        repeat_limiter = RepeatLimiter(risk.cooldown_ms)
        if not args.no_speech:
            try:
                speech = SpeechWorker(
                    WindowsSpeechBackend(settings.getint("speech", "rate", fallback=0)),
                    timeout_s=settings.getfloat("speech", "timeout_s", fallback=15),
                )
            except (OSError, ValueError, RuntimeError) as exc:
                logging.error("Spoken alerts unavailable: %s", exc)
                failed = True
        else:
            logging.warning("Spoken alerts disabled; diagnostic mode only")
        if console_source is not None:
            assistant = VisionAssistant(SceneTools(scene_store))
            console = ConsoleWorker(
                assistant,
                console_source,
                spoken_reply=SpokenReply(
                    assistant,
                    scene_store,
                    speech,
                    queue_horizon_s=reply_queue_horizon_s,
                )
                if args.speak_replies and speech is not None
                else None,
                recognizer=WindowsCommandRecognizer(
                    SttConfig(
                        timeout_s=settings.getfloat("stt", "timeout_s", fallback=10),
                        min_confidence=settings.getfloat(
                            "stt", "min_confidence", fallback=0.7
                        ),
                    )
                )
                if args.voice_input
                else None,
            )
        detector = Yolo11Detector(load_config(args.config))
        depth: DepthEstimator | None = None
        try:
            from app.vision.depth_estimator import (
                DepthAnythingV2Estimator,
                load_depth_config,
            )

            depth = DepthAnythingV2Estimator(load_depth_config(args.config))
        except Exception as exc:
            logging.error("Depth unavailable; retaining detection: %s", exc)
            failed = True
        pipeline = PerceptionPipeline(detector, depth, fusion, risk)
        if performance is not None:
            performance.start()
        camera = CameraService(camera_config)
        camera.start()
        if console is not None:
            console.start()
        count = 0
        loop_started = monotonic()
        while args.max_frames == 0 or count < args.max_frames:
            if console is not None and console.exit_requested.is_set():
                break
            try:
                frame = camera.read()
                result = pipeline.process(frame)
                if not camera.healthy():
                    raise RuntimeError("camera became unavailable during inference")
            except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
                logging.error("Current scene invalidated: %s", exc)
                invalid = RiskScene("camera-unavailable", None, "live", False, ())
                scene_store.invalidate()
                assessment = assess_scene(invalid, time_ns() // 1_000_000, risk)
                if speech is not None:
                    speech.cancel()
                dispatch_warnings(
                    assessment,
                    gate,
                    speech,
                    monotonic() + 5,
                    repeat_limiter=repeat_limiter,
                    degraded_queue_horizon_s=degraded_queue_horizon_s,
                )
                print(
                    json.dumps(
                        {
                            "scene": asdict(invalid),
                            "safety": asdict(assessment),
                            "speech_status": "unavailable"
                            if speech is None or speech.error
                            else "requested",
                        }
                    ),
                    flush=True,
                )
                failed = True
                # 此时感知已停止，只等待最终故障提示；等待按实测语速计算，超时明确记录。
                if speech is not None and not speech.wait_idle(
                    final_notice_wait_seconds(assessment.events)
                ):
                    logging.error("Final unavailable notification did not finish")
                break
            started = perf_counter()
            assessment = assess_scene(result.scene, time_ns() // 1_000_000, risk)
            result.latencies_ms["risk"] = (perf_counter() - started) * 1000
            expiry = frame.captured_monotonic + risk.freshness_ms / 1000
            if assessment.status != "available":
                expiry = monotonic() + 5
            current = assessment.status == "available"
            if not current and speech is not None:
                speech.cancel()
            display_scene = (
                result.scene
                if current
                else replace(result.scene, valid=False, objects=())
            )
            if result.depth_status in {"unavailable", "rejected"}:
                failed = True
            accepted = dispatch_warnings(
                assessment,
                gate,
                speech,
                expiry,
                repeat_limiter=repeat_limiter,
                degraded_queue_horizon_s=degraded_queue_horizon_s,
            )
            result.latencies_ms["capture_to_dispatch"] = (
                monotonic() - frame.captured_monotonic
            ) * 1000
            # 告警已独立提交，发布失败仅使查询失效，不关闭感知和告警。
            try:
                scene_store.publish(display_scene)
            except ValueError as exc:
                scene_store.invalidate()
                logging.error("Scene tools unavailable: %s", exc)
                failed = True
            if speech is not None and speech.error:
                failed = True
            measurement = {}
            if performance is not None:
                measurement = {
                    "performance": {
                        "stage_clock": "perf_counter",
                        "result_index": count + 1,
                        "warmup": count == 0,
                        "loop_started_monotonic_s": loop_started,
                        "captured_monotonic_s": frame.captured_monotonic,
                        "dispatch_completed_monotonic_s": (
                            frame.captured_monotonic
                            + result.latencies_ms["capture_to_dispatch"] / 1000
                        ),
                        "width": frame.image.shape[1],
                        "height": frame.image.shape[0],
                    }
                }
            if not args.console:
                print(
                    json.dumps(
                        {
                            "scene": asdict(display_scene),
                            "current_scene": current,
                            "observations": [
                                asdict(item) for item in result.observations
                            ]
                            if current
                            else [],
                            "safety": asdict(assessment),
                            "depth_status": result.depth_status,
                            "latencies_ms": result.latencies_ms,
                            "speech_status": "unavailable"
                            if speech is None or speech.error
                            else "worker_running",
                            "speech_accepted": accepted,
                            **measurement,
                        },
                        ensure_ascii=True,
                        allow_nan=False,
                    ),
                    flush=True,
                )
            count += 1
            if console is not None and console.error:
                logging.error("%s", console.error)
                failed = True
                console.close()
                console = None
    except KeyboardInterrupt:
        logging.info("Stopping live perception")
    except Exception as exc:
        logging.error("Live pipeline unavailable: %s", exc)
        failed = True
    finally:
        if scene_store is not None:
            scene_store.invalidate()
        if console is not None:
            console.close()
            if console.error:
                logging.error("%s", console.error)
                failed = True
        if camera is not None:
            try:
                camera.close()
            except Exception as exc:
                logging.error("Camera cleanup failed: %s", exc)
                failed = True
        if speech is not None:
            speech.close()
            failed = failed or speech.error is not None
        if performance is not None:
            try:
                performance.finish(failed)
            except Exception as exc:
                logging.error("Performance report unavailable: %s", exc)
                failed = True
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
