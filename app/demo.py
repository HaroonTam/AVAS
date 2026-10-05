"""无硬件验收演示：所有输入均为合成数据，不运行真实模型或播放声音。"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from time import monotonic

import numpy as np

from app.agent.spoken_reply import SpokenReply
from app.agent.tools import SceneTools
from app.agent.vision_agent import AssistantAnswer, VisionAssistant
from app.fusion.config import FusionConfig
from app.fusion.depth_fusion import DetectionFrame, fuse_frame
from app.live import dispatch_warnings
from app.safety.config import RiskConfig
from app.safety.risk_engine import assess_scene
from app.safety.scene import MetricEvidence, RiskObject, RiskScene
from app.safety.warnings import WarningGate
from app.scene_store import SceneStore
from app.speech.tts import SpeechHandle, SpeechWorker
from app.vision.depth_estimator import RelativeDepth
from app.vision.detector import Detection


class DemoClock:
    def __init__(self) -> None:
        """使用固定合成毫秒时间，过期场景无需实际等待。"""
        self.now_ms = 1000

    def wall_ms(self) -> int:
        """返回合成墙上时间，不能解释为实际采集时间。"""
        return self.now_ms

    def monotonic_s(self) -> float:
        """返回同步推进的合成单调秒数。"""
        return self.now_ms / 1000


class SilentHandle:
    def poll(self) -> int | None:
        """立即报告模拟播放完成，不访问音频设备。"""
        return 0

    def stop(self) -> None:
        """模拟句柄没有外部资源需要释放。"""
        return None


class RecordingSpeech:
    def __init__(self) -> None:
        """只保留本次演示的有限消息列表，不持久化数据。"""
        self.messages: list[str] = []

    def start(self, text: str) -> SpeechHandle:
        """记录原本要播放的文字，不启动系统语音进程。"""
        self.messages.append(text)
        return SilentHandle()


@dataclass(frozen=True)
class DemoCase:
    name: str
    passed: bool
    answer: AssistantAnswer
    detail: str


@dataclass(frozen=True)
class DemoReport:
    schema_version: int
    synthetic: bool
    risk_config: RiskConfig
    fusion_config: FusionConfig
    cases: tuple[DemoCase, ...]
    simulated_speech: tuple[str, ...]
    passed: bool


def run_demo() -> DemoReport:
    """运行合成融合、查询、模拟语音和失效场景，返回可机读验收结果。"""
    clock = DemoClock()
    risk = RiskConfig()
    fusion = FusionConfig(camera_orientation="forward")
    store = SceneStore(risk, clock_ms=clock.wall_ms, monotonic_clock=clock.monotonic_s)
    tools = SceneTools(store)
    assistant = VisionAssistant(tools)
    frame = DetectionFrame(
        "synthetic-demo:1", 60, 40, (Detection("chair", 0.9, (20, 10, 40, 30)),)
    )
    depth = RelativeDepth(
        frame.frame_id, np.full((40, 60), 7, np.float32), np.ones((40, 60), bool)
    )
    observations = fuse_frame(frame, depth, fusion)
    scene = RiskScene(
        frame.frame_id,
        clock.now_ms,
        "live",
        True,
        tuple(
            RiskObject(
                item.id, item.frame_id, item.label, item.confidence, item.direction
            )
            for item in observations
        ),
    )
    # live 仅用于合成测试协议，不表示这里存在真实相机采集。
    store.publish(scene)
    cases: list[DemoCase] = []
    backend = RecordingSpeech()
    worker = SpeechWorker(backend)
    try:
        speaker = SpokenReply(assistant, store, worker)
        answer = speaker.respond("椅子在哪里")
        drained = worker.wait_idle(2)
        facts = tools.get_scene()
        cases.append(
            DemoCase(
                "relative_depth_is_not_meters",
                len(facts.objects) == 1
                and facts.objects[0].distance_m is None
                and facts.objects[0].direction == "front"
                and facts.risk_level == "unknown",
                answer,
                "合成相对逆深度 7 不转换成米数；方向来自真实融合代码。",
            )
        )
        cases.append(
            DemoCase(
                "ordinary_reply_simulated",
                drained
                and backend.messages == [answer.message]
                and worker.error is None,
                answer,
                "真实语音调度器调用记录后端，未播放声音。",
            )
        )
        store.publish(
            replace(scene, objects=(scene.objects[0], replace(scene.objects[0], id=2)))
        )
        answer = assistant.respond("寻找 椅子")
        cases.append(
            DemoCase(
                "multiple_matches",
                answer.status == "ambiguous",
                answer,
                "两个同类目标需要明确选择，不自动选首个。",
            )
        )
        store.publish(scene)
        metric_scene = replace(
            scene,
            objects=(
                replace(
                    scene.objects[0],
                    metric=MetricEvidence(0.5, "synthetic-demo-only", 0, 10),
                ),
            ),
        )
        # 米制证据是单独注入的合成夹具，绝不是从上面的相对深度推导。
        assessment = assess_scene(metric_scene, clock.now_ms, risk)
        accepted = dispatch_warnings(
            assessment, WarningGate(risk.cooldown_ms), worker, monotonic() + 2
        )
        drained = worker.wait_idle(2)
        store.publish(metric_scene)
        answer = assistant.respond("当前风险")
        cases.append(
            DemoCase(
                "independent_warning",
                assessment.level == "high"
                and accepted == 1
                and drained
                and assessment.events[0].message in backend.messages,
                answer,
                "告警先于 Agent 查询提交；0.5 米只是人工合成测试证据。",
            )
        )
        clock.now_ms += risk.freshness_ms + 1
        answer = assistant.respond("描述周围")
        cases.append(
            DemoCase(
                "expired_scene",
                answer.status == "scene_stale",
                answer,
                "推进测试时钟，过期事实不可查询。",
            )
        )
        store.invalidate()
        answer = assistant.respond("寻找 椅子")
        cases.append(
            DemoCase(
                "invalid_scene",
                answer.status == "scene_invalid",
                answer,
                "故障失效后不返回历史目标。",
            )
        )
    finally:
        store.invalidate()
        worker.close()
    return DemoReport(
        1,
        True,
        risk,
        fusion,
        tuple(cases),
        tuple(backend.messages),
        all(case.passed for case in cases) and worker.error is None,
    )


def main(argv: list[str] | None = None) -> int:
    """执行无硬件验收；JSON 用于自动检查，失败时返回非零退出码。"""
    parser = argparse.ArgumentParser(
        description="Synthetic demo: no camera, models or audio"
    )
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable synthetic report"
    )
    args = parser.parse_args(argv)
    print("SYNTHETIC DEMO ONLY: no camera, model inference or audio.", file=sys.stderr)
    report = run_demo()
    if args.json:
        print(json.dumps(asdict(report), ensure_ascii=True, allow_nan=False))
    else:
        print("合成演示：未使用摄像头、模型、网络或真实语音。")
        for case in report.cases:
            status = "PASS" if case.passed else "FAIL"
            print(f"{status} {case.name}: {case.answer.message}")
        print("这些结果不验证模型精度、真实性能或行走安全。")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
