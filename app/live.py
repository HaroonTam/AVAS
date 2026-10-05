"""实时本地入口：采集、感知、确定性风险与独立语音输出。"""

import argparse
import json
import logging
from configparser import ConfigParser
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import monotonic, time_ns
from typing import Protocol

from app.agent.console import ConsoleWorker, WindowsLineSource
from app.agent.spoken_reply import SpokenReply
from app.agent.tools import SceneTools
from app.agent.vision_agent import VisionAssistant
from app.camera.camera_service import CameraConfig, CameraService, CapturedFrame
from app.config import load_config
from app.fusion.config import FusionConfig, load_fusion_config
from app.fusion.depth_fusion import DetectionFrame, Observation, fuse_frame
from app.safety.config import RiskConfig, load_risk_config
from app.safety.risk_engine import RiskAssessment, assess_scene
from app.safety.scene import RiskObject, RiskScene
from app.safety.warnings import WarningGate
from app.scene_store import SceneStore
from app.speech.tts import SpeechMessage, SpeechWorker, WindowsSpeechBackend
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
        started = monotonic()
        detections = self._detector.detect(frame.image)
        timings["detection"] = (monotonic() - started) * 1000
        depth = None
        depth_status = "unavailable"
        if self._depth is not None:
            started = monotonic()
            try:
                depth = self._depth.estimate(frame.image, frame.frame_id)
                depth_status = "available"
            except Exception as exc:
                logging.error("Depth unavailable: %s", exc)
            timings["depth"] = (monotonic() - started) * 1000
        started = monotonic()
        detection_frame = DetectionFrame(
            frame.frame_id, frame.image.shape[1], frame.image.shape[0], detections
        )
        try:
            observations = fuse_frame(detection_frame, depth, self._fusion)
        except ValueError as exc:
            logging.error("Fusion depth rejected: %s", exc)
            observations = fuse_frame(detection_frame, None, self._fusion)
            depth_status = "rejected"
        timings["fusion"] = (monotonic() - started) * 1000
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


def dispatch_warnings(
    assessment: RiskAssessment,
    gate: WarningGate,
    speech: SpeechWorker | None,
    expires_at: float,
) -> int:
    """先记录告警再提交本地播放；完全不调用或等待 Agent，返回接收事件数。"""
    events = gate.select(assessment, int(monotonic() * 1000))
    accepted = 0
    for event in events:
        logging.warning("%s: %s", event.kind, event.message)
        if speech is not None:
            priority = 2 if event.level == "high" else 1
            if speech.submit(SpeechMessage(event.message, priority, expires_at)):
                accepted += 1
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
        "--speak-replies",
        action="store_true",
        help="Speak validated console answers at low priority (requires --console)",
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    camera: CameraService | None = None
    speech: SpeechWorker | None = None
    console: ConsoleWorker | None = None
    failed = False
    try:
        settings = ConfigParser(interpolation=None)
        with args.config.open(encoding="utf-8") as source:
            settings.read_file(source)
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
        )
        gate = WarningGate(risk.cooldown_ms)
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
                spoken_reply=SpokenReply(assistant, scene_store, speech)
                if args.speak_replies and speech is not None
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
        camera = CameraService(camera_config)
        camera.start()
        if console is not None:
            console.start()
        count = 0
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
                dispatch_warnings(assessment, gate, speech, monotonic() + 5)
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
                # 此时感知已停止，只等待最终故障提示，超时必须明确记录。
                if speech is not None and not speech.wait_idle(5):
                    logging.error("Final unavailable notification did not finish")
                break
            started = monotonic()
            assessment = assess_scene(result.scene, time_ns() // 1_000_000, risk)
            result.latencies_ms["risk"] = (monotonic() - started) * 1000
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
            accepted = dispatch_warnings(assessment, gate, speech, expiry)
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
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
