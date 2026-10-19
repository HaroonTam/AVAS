"""正式分组与目标元数据的合成测试，不访问设备或模型。"""

import json
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.e2_capture import main
from app.e2_collection import Target, prepare_collection
from app.e2_offline import digest, load_sample, run
from app.fusion.config import load_fusion_config
from tests.test_e2_offline import CONFIG, fixture
from tests.workspace_temp import workspace_tempdir

PLAN = Path("configs/e2_collection.json")
TARGET = Target("board02", "matte cardboard", 0.6, 0.7)


class CollectionTests(unittest.TestCase):
    def test_registered_split_and_test_lock(self) -> None:
        """预注册布景固定划分，测试组和错误安装不能采集。"""
        with workspace_tempdir() as root:
            context = prepare_collection(
                PLAN, "scene02", "omen_h0265_v1", "p1", TARGET, root
            )
            self.assertEqual(context.split, "fit")
            for group, install in (
                ("scene07", "omen_h0265_v1"),
                ("scene02", "other"),
                ("scene01", "omen_h0265_v1"),
            ):
                with self.assertRaises(ValueError):
                    prepare_collection(PLAN, group, install, "p1", TARGET, root)

    def test_duplicate_placement_and_changed_target_rejected(self) -> None:
        """防止重复摆放计数及同编号目标描述漂移。"""
        with workspace_tempdir() as root:
            context = prepare_collection(
                PLAN, "scene02", "omen_h0265_v1", "p1", TARGET, root
            )
            path = fixture(root)
            path.write_text(
                json.dumps({"collection": asdict(context)}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "placement"):
                prepare_collection(PLAN, "scene02", "omen_h0265_v1", "p1", TARGET, root)
            with self.assertRaisesRegex(ValueError, "target"):
                prepare_collection(
                    PLAN,
                    "scene02",
                    "omen_h0265_v1",
                    "p2",
                    Target("board02", "changed", 0.6, 0.7),
                    root,
                )

    def test_plan_change_rejected_after_first_capture(self) -> None:
        """首次保存后计划哈希固定，禁止悄悄改组导致泄漏。"""
        with workspace_tempdir() as root:
            context = prepare_collection(
                PLAN, "scene02", "omen_h0265_v1", "p1", TARGET, root
            )
            path = fixture(root)
            path.write_text(
                json.dumps({"collection": asdict(context)}), encoding="utf-8"
            )
            changed = root / "plan.json"
            changed.write_bytes(PLAN.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "plan changed"):
                prepare_collection(
                    changed, "scene02", "omen_h0265_v1", "p2", TARGET, root
                )

    def test_formal_sample_requires_explicit_split(self) -> None:
        """正式数据不能混入旧预采集；显式审查保存目标与划分信息。"""
        with workspace_tempdir() as root:
            context = prepare_collection(
                PLAN, "scene02", "omen_h0265_v1", "p1", TARGET, root
            )
            path = fixture(root)
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw.update(
                schema="e2_board_collection_v1", split="fit", collection=asdict(context)
            )
            raw["measurement"].update(
                group_id="scene02", installation_id="omen_h0265_v1"
            )
            path.write_text(json.dumps(raw), encoding="utf-8")
            with patch(
                "app.e2_offline.read_image",
                return_value=np.zeros((20, 20, 3), np.uint8),
            ):
                with self.assertRaises(ValueError):
                    load_sample(path, load_fusion_config(CONFIG), digest(CONFIG))
                run(
                    [path],
                    root / "audit",
                    CONFIG,
                    infer_relative=False,
                    board_lower_half=True,
                    expected_split="fit",
                )
            report = json.loads(
                (root / "audit/report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["split"], "fit")
            self.assertEqual(
                report["samples"][0]["collection"]["target"]["target_id"], "board02"
            )

    def test_test_group_fails_before_camera(self) -> None:
        """错误的测试采集请求在打开相机前返回失败。"""
        with patch("app.e2_capture.capture_frame") as camera:
            code = main(
                [
                    "--save-local",
                    "--distance-m",
                    "1",
                    "--instrument",
                    "tape",
                    "--conditions",
                    "synthetic",
                    "--installation-id",
                    "omen_h0265_v1",
                    "--group-id",
                    "scene07",
                    "--collection-plan",
                    str(PLAN),
                    "--placement-id",
                    "p1",
                    "--target-id",
                    "board02",
                    "--target-material",
                    "cardboard",
                    "--target-width-m",
                    ".6",
                    "--target-height-m",
                    ".7",
                ]
            )
            self.assertEqual(code, 2)
            camera.assert_not_called()

    def test_invalid_target_dimensions(self) -> None:
        """目标尺寸拒绝非正与非有限值，不接受猜测的空描述。"""
        for size in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                Target("board02", "cardboard", size, 0.6)
