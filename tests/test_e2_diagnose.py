"""离线诊断测试，只使用合成像素与合成报告。"""

import unittest
from unittest.mock import patch

import numpy as np

from app.e2_collection import read_object
from app.e2_diagnose import brightness, numeric_summary, rectangle, run_diagnosis
from tests.test_e2_compare import write_json
from tests.test_e2_multiplan import multiplan_fixture
from tests.workspace_temp import workspace_tempdir


class DiagnoseTests(unittest.TestCase):
    def test_encoded_brightness_and_empty(self) -> None:
        """确认 BGR 顺序、亮暗像素比例和空区域拒绝，不把亮度当照度。"""
        pixels = np.array(
            [[[255, 0, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]], dtype=np.uint8
        )
        result = brightness(pixels)
        self.assertAlmostEqual(
            result["gray_mean"], (255 * 0.114 + 255 * 0.299 + 255) / 4
        )
        self.assertEqual(result["gray_ge250_fraction"], 0.25)
        self.assertEqual(result["gray_le5_fraction"], 0.25)
        with self.assertRaises(ValueError):
            brightness(pixels[:, :0])
        with self.assertRaises(ValueError):
            brightness(pixels.astype(np.float32))

    def test_coordinates_and_missing_summary(self) -> None:
        """拒绝越界、空框和隐式坐标转换；缺失统计不填零。"""
        self.assertEqual(rectangle([0, 0, 20, 20], 20, 20), (0, 0, 20, 20))
        for roi in (
            [0, 0, 21, 20],
            [0, 0, 0, 20],
            [0.0, 0, 20, 20],
            [False, 0, 20, 20],
        ):
            with self.assertRaises(ValueError):
                rectangle(roi, 20, 20)
        self.assertIsNone(numeric_summary([])["mean"])
        self.assertEqual(numeric_summary([2.0, 5.0])["span"], 3.0)

    def test_complete_run_and_source_tamper(self) -> None:
        """完整诊断保留所有角色；来源变动及覆盖输出均被拒绝。"""
        with workspace_tempdir() as root:
            recipe = multiplan_fixture(root)
            output = root / "diagnosis"
            with patch(
                "app.e2_diagnose.read_image",
                return_value=np.full((20, 20, 3), 100, dtype=np.uint8),
            ):
                report = run_diagnosis(recipe, output)
            self.assertEqual(report["sample_count"], 8)
            self.assertFalse(report["test_unlocked"])
            self.assertTrue((output / "diagnosis_snapshot.py").is_file())
            with self.assertRaisesRegex(ValueError, "already exists"):
                run_diagnosis(recipe, output)
            changed = read_object(root / "original" / "fit_relative.json")
            changed["sample_count"] = 999
            write_json(root / "original" / "fit_relative.json", changed)
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                run_diagnosis(recipe, root / "bad")
            self.assertFalse((root / "bad").exists())
