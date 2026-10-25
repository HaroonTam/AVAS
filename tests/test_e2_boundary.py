"""边界补采计划与无硬件预览测试。"""

import json
import shutil
import subprocess
import unittest
from pathlib import Path

from app.e2_collection import Target, prepare_collection
from app.e2_offline import digest
from tests.workspace_temp import workspace_tempdir

ROOT = Path(__file__).resolve().parents[1]


class BoundaryCollectionTests(unittest.TestCase):
    def test_confirmed_targets_and_original_plan_unchanged(self) -> None:
        """确认的目标精确沿用草案，原划分哈希与测试锁定不变。"""
        draft = json.loads(
            (ROOT / "configs/e2_test_protocol.draft.json").read_text(encoding="utf-8")
        )
        confirmed = json.loads(
            (ROOT / "configs/e2_research_acceptance.json").read_text(encoding="utf-8")
        )
        self.assertEqual(confirmed["targets"], draft["proposed_research_targets"])
        self.assertFalse(confirmed["test_unlocked"])
        self.assertFalse(confirmed["production_approved"])
        self.assertEqual(
            digest(ROOT / "configs/e2_collection.json"), draft["collection_plan_sha256"]
        )

    def test_supplementary_split_and_installation(self) -> None:
        """新计划可由既有入口读取，但拒绝错误安装和测试组。"""
        target = Target("A02", "纸箱,黄色", 0.243, 0.271)
        with workspace_tempdir() as root:
            for group, filename, split in [
                ("scene09", "e2_boundary_fit.json", "fit"),
                ("scene10", "e2_boundary_fit.json", "fit"),
                ("scene11", "e2_boundary_validation.json", "validation"),
                ("scene12", "e2_boundary_fit_replacement.json", "fit"),
            ]:
                plan = ROOT / "configs" / filename
                context = prepare_collection(
                    plan, group, "omen_h0265_v1", "p01", target, root
                )
                self.assertEqual(context.split, split)
                for invalid_group, installation in [
                    (group, "changed"),
                    ("scene07", "omen_h0265_v1"),
                ]:
                    with self.assertRaises(ValueError):
                        prepare_collection(
                            plan, invalid_group, installation, "p01", target, root
                        )

    def test_preview_and_invalid_range_without_python(self) -> None:
        """预览与编号拒绝无需 Python，可在不访问相机时验证采集序列。"""
        shell = shutil.which("powershell") or shutil.which("pwsh")
        if shell is None:
            self.skipTest("PowerShell unavailable")
        script = str(ROOT / "scripts/collect_e2_boundary.ps1")
        for group, end, expected_code in [
            ("scene09", "12", 0),
            ("scene12", "12", 0),
            ("scene11", "10", 0),
            ("scene11", "12", 1),
        ]:
            result = subprocess.run(
                [
                    shell,
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    script,
                    "-Group",
                    group,
                    "-StartPlacement",
                    "1",
                    "-EndPlacement",
                    end,
                    "-Python",
                    "nonexistent-python",
                    "-Preview",
                ],
                cwd=ROOT,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, expected_code, result.stderr)
            if expected_code == 0:
                self.assertEqual(len(result.stdout.splitlines()), int(end))
                if group == "scene12":
                    self.assertIn(b"e2_boundary_fit_replacement.json", result.stdout)
