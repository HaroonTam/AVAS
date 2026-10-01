"""风险规则与告警去重的合成测试，不验证真实行走安全性。"""

import ast
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from app.fusion.config import FusionConfig
from app.fusion.depth_fusion import DetectionFrame, fuse_frame
from app.safety.config import RiskConfig, load_risk_config
from app.safety.risk_engine import RiskAssessment, assess_scene
from app.safety.scene import MetricEvidence, RiskObject, RiskScene, offline_scene
from app.safety.warnings import WarningGate
from app.vision.detector import Detection


class SafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        """构造明确标为合成证据的米制输入和当前帧。"""
        self.config = RiskConfig()
        self.metric = MetricEvidence(0.5, "synthetic-test-only", 0, 10)
        self.item = RiskObject(1, "f1", "chair", 0.9, "front", self.metric, "track-1")
        self.scene = RiskScene("f1", 1000, "live", True, (self.item,))

    def evaluate(self, distance: float, now_ms: int = 1000) -> RiskAssessment:
        """按指定合成距离评估同一场景，避免测试依赖真实时钟。"""
        item = replace(self.item, metric=replace(self.metric, distance_m=distance))
        return assess_scene(replace(self.scene, objects=(item,)), now_ms, self.config)

    def test_distance_boundaries(self) -> None:
        """严格小于阈值才升级；等于一米为中，等于两米为低。"""
        for distance, level in (
            (0, "high"),
            (0.9999, "high"),
            (1, "medium"),
            (1.9999, "medium"),
            (2, "low"),
        ):
            with self.subTest(distance=distance):
                result = self.evaluate(distance)
                self.assertEqual(result.level, level)
                self.assertEqual(result.objects[0].level, level)

    def test_invalid_missing_and_out_of_range_metric(self) -> None:
        """无效或无来源的米数一律未知，不能因数值较大而当作低风险。"""
        for metric in (
            None,
            replace(self.metric, distance_m=float("nan")),
            replace(self.metric, distance_m=float("inf")),
            replace(self.metric, distance_m=-1),
            replace(self.metric, distance_m=11),
            replace(self.metric, support_id=""),
            replace(self.metric, valid_min_m=2),
        ):
            result = assess_scene(
                replace(self.scene, objects=(replace(self.item, metric=metric),)),
                1000,
                self.config,
            )
            self.assertEqual(result.level, "unknown")
            self.assertEqual(result.objects[0].reason, "metric_unavailable")
            self.assertTrue(all(event.kind == "degraded" for event in result.events))
            json.dumps(asdict(result), allow_nan=False)

    def test_independent_evidence_survives_depth_failure(self) -> None:
        """配置的独立类别危险在缺深度时保留，不被距离失败降为未知。"""
        config = replace(self.config, independent_hazard_labels=("chair",))
        scene = replace(self.scene, objects=(replace(self.item, metric=None),))
        result = assess_scene(scene, 1000, config)
        self.assertEqual(result.level, "medium")
        self.assertEqual(result.objects[0].reason, "configured_front_category")
        self.assertEqual(result.events[0].kind, "hazard")
        self.assertIn(
            ("degraded", "metric_unavailable"), [event.key for event in result.events]
        )
        self.assertEqual(assess_scene(self.scene, 1000, config).level, "high")

    def test_unknown_direction_confidence_and_category(self) -> None:
        """类别、方向和分数不满足规则时不声称低风险。"""
        for item in (
            replace(self.item, direction=None),
            replace(self.item, direction="left"),
            replace(self.item, label="unsupported"),
            replace(self.item, confidence=0.499),
        ):
            self.assertEqual(
                assess_scene(
                    replace(self.scene, objects=(item,)), 1000, self.config
                ).level,
                "unknown",
            )
        self.assertEqual(
            assess_scene(
                replace(self.scene, objects=(replace(self.item, confidence=0.5),)),
                1000,
                self.config,
            ).level,
            "high",
        )

    def test_freshness_boundary_and_unavailable_scenes(self) -> None:
        """新鲜度边界可用，未来、缺时间、失效和过期场景不能发对象告警。"""
        self.assertEqual(assess_scene(self.scene, 2000, self.config).level, "high")
        for scene, now, status in (
            (self.scene, 2001, "scene_stale"),
            (self.scene, 999, "future_capture_time"),
            (
                replace(self.scene, captured_at_ms=None),
                1000,
                "capture_time_unavailable",
            ),
            (replace(self.scene, valid=False), 1000, "scene_invalid"),
        ):
            result = assess_scene(scene, now, self.config)
            self.assertEqual(result.status, status)
            self.assertEqual(result.level, "unknown")
            self.assertTrue(all(event.kind == "degraded" for event in result.events))

    def test_offline_image_never_becomes_live_warning(self) -> None:
        """即使离线输入附带近距离数值，也不能发实时危险告警。"""
        result = assess_scene(
            replace(self.scene, source_kind="offline_image"), 1000, self.config
        )
        self.assertEqual(result.status, "offline_not_current")
        self.assertEqual(result.level, "unknown")
        self.assertEqual(result.events, ())

    def test_fusion_adapter_preserves_unknown_metric(self) -> None:
        """融合相对深度接口不提供米制证据，也不复制上游自报的风险等级。"""
        frame = DetectionFrame(
            "f1", 30, 20, (Detection("chair", 0.9, (10, 0, 20, 20)),)
        )
        observation = fuse_frame(
            frame, None, FusionConfig(camera_orientation="forward")
        )[0]
        scene = offline_scene("f1", (replace(observation, risk_level="low"),))
        self.assertIsNone(scene.objects[0].metric)
        self.assertIsNone(scene.captured_at_ms)
        self.assertEqual(assess_scene(scene, 1000, self.config).level, "unknown")

    def test_aggregation_does_not_erase_hazards_or_unknowns(self) -> None:
        """高风险优先且保留未知项；空场景和低风险加未知项不能称为安全。"""
        unknown = replace(self.item, id=2, track_id="track-2", metric=None)
        scene = replace(self.scene, objects=(self.item, unknown))
        result = assess_scene(scene, 1000, self.config)
        self.assertEqual(result.level, "high")
        self.assertEqual(result.objects[1].level, "unknown")
        far = replace(self.item, metric=replace(self.metric, distance_m=3))
        self.assertEqual(
            assess_scene(
                replace(scene, objects=(far, unknown)), 1000, self.config
            ).level,
            "unknown",
        )
        empty = assess_scene(replace(scene, objects=()), 1000, self.config)
        self.assertEqual(empty.level, "unknown")
        self.assertEqual(empty.events[0].key, ("degraded", "no_detections"))

    def test_identity_and_configuration_validation(self) -> None:
        """拒绝混帧、重复身份与非法配置，旧配置仍可读取。"""
        for objects in (
            (self.item, self.item),
            (replace(self.item, frame_id="other"),),
            (self.item, replace(self.item, id=2)),
        ):
            with self.assertRaises(ValueError):
                replace(self.scene, objects=objects)
        for field, value in (
            ("high_distance_m", 2),
            ("medium_distance_m", float("nan")),
            ("freshness_ms", -1),
            ("cooldown_ms", 1.5),
            ("min_confidence", 1.1),
        ):
            with self.assertRaises(ValueError):
                replace(self.config, **{field: value})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.ini"
            path.write_text("[detection]\n", encoding="utf-8")
            self.assertEqual(load_risk_config(path), self.config)
            path.write_text(
                "[risk]\nhigh_distance_m=0.8\nindependent_hazard_labels=car,bus\n",
                encoding="utf-8",
            )
            config = load_risk_config(path)
            self.assertEqual(config.high_distance_m, 0.8)
            self.assertEqual(config.independent_hazard_labels, ("car", "bus"))

    def test_cooldown_escalation_and_reappearance(self) -> None:
        """重复事件去重，升级和消失后重现立即输出，边界到期允许重复。"""
        gate = WarningGate(3000)
        medium = self.evaluate(1.5)
        high = self.evaluate(0.5)
        self.assertEqual(len(gate.select(medium, 0)), 1)
        self.assertEqual(gate.select(medium, 1), ())
        self.assertEqual(len(gate.select(high, 2)), 1)
        self.assertEqual(gate.select(medium, 3), ())
        self.assertEqual(len(gate.select(high, 4)), 1)
        self.assertEqual(gate.select(high, 3003), ())
        self.assertEqual(len(gate.select(high, 3004)), 1)
        gate.select(self.evaluate(3), 3005)
        self.assertEqual(len(gate.select(high, 3006)), 1)

    def test_new_objects_and_frame_local_ids_not_suppressed(self) -> None:
        """新增目标绕过冷却；帧内 ID 不能被当作跨帧身份。"""
        gate = WarningGate(3000)
        first = replace(self.scene, objects=(replace(self.item, track_id=None),))
        gate.select(assess_scene(first, 1000, self.config), 0)
        second = replace(
            first, frame_id="f2", objects=(replace(first.objects[0], frame_id="f2"),)
        )
        self.assertEqual(
            len(gate.select(assess_scene(second, 1000, self.config), 1)), 1
        )
        with self.assertRaises(ValueError):
            gate.select(self.evaluate(1), 0)

    def test_bounded_gate_and_degraded_deduplication(self) -> None:
        """状态容量耗尽不丢告警；持续降级可去重且状态变化立即输出。"""
        gate = WarningGate(3000, max_entries=1)
        result = assess_scene(
            replace(self.scene, objects=(replace(self.item, metric=None),)),
            1000,
            self.config,
        )
        self.assertEqual(len(gate.select(result, 0)), 2)
        self.assertEqual(len(gate.select(result, 1)), 1)
        stale = assess_scene(self.scene, 2001, self.config)
        self.assertEqual(len(gate.select(stale, 2)), 1)
        self.assertEqual(gate.select(stale, 3), ())

    def test_safety_modules_have_no_agent_network_or_speech_dependency(self) -> None:
        """静态检查核心仅依赖本地数据模块，不引入等待 LLM 或播放的路径。"""
        root = Path(__file__).resolve().parents[1] / "app/safety"
        allowed = {
            "configparser",
            "dataclasses",
            "math",
            "pathlib",
            "typing",
            "app.safety.config",
            "app.safety.scene",
            "app.safety.risk_engine",
            "app.fusion.depth_fusion",
            "app.fusion.direction",
        }
        for path in root.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    self.assertIn(node.module, allowed)
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertIn(alias.name, allowed)
