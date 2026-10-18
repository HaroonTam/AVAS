"""E2 离线流程合成测试，不启动相机、模型或网络。"""

import hashlib
import json
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.e2_offline import digest, frozen_paths, load_sample, run
from app.fusion.config import load_fusion_config
from app.vision.depth_estimator import RelativeDepth
from tests.workspace_temp import workspace_tempdir

CONFIG = Path("configs/system.ini")


def fixture(root: Path) -> Path:
    """建立具有合成图像哈希与当前融合快照的独立预采集记录。"""
    directory = root / "e2_synthetic"
    directory.mkdir()
    (directory / "frame.png").write_bytes(b"synthetic")
    record = {
        "schema": "e2_board_precapture_v1",
        "split": "development_precapture",
        "sample_id": directory.name,
        "roi_source": "manual_board_diagnostic_not_yolo",
        "image": "frame.png",
        "image_sha256": hashlib.sha256(b"synthetic").hexdigest(),
        "actual_width": 20,
        "actual_height": 20,
        "roi_xyxy": [0, 0, 20, 20],
        "frame_id": "synthetic",
        "config_sha256": digest(CONFIG),
        "fusion_config": asdict(load_fusion_config(CONFIG)),
        "measurement": {
            "distance_m": 1.0,
            "uncertainty_m": None,
            "group_id": "scene",
            "instrument": "synthetic",
            "conditions": "synthetic",
            "installation_id": "synthetic",
        },
    }
    path = directory / "record.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


class E2OfflineTests(unittest.TestCase):
    def test_lower_half_preserves_original_and_avoids_upper_hole(self) -> None:
        """派生下半板框不改原记录，生产中心统计避开上半部提手孔。"""
        with workspace_tempdir() as root:
            path = fixture(root)
            original_hash = digest(path)
            with patch(
                "app.e2_offline.read_image",
                return_value=np.zeros((20, 20, 3), np.uint8),
            ):
                run(
                    [path],
                    root / "review",
                    CONFIG,
                    infer_relative=False,
                    board_lower_half=True,
                )
            report = json.loads(
                (root / "review/report.json").read_text(encoding="utf-8")
            )
            row = report["samples"][0]
            self.assertEqual(row["roi_xyxy"], [0, 0, 20, 20])
            self.assertEqual(row["analysis_roi"], [0, 10, 20, 20])
            self.assertEqual(row["statistics_roi"], [5, 13, 15, 17])
            self.assertEqual(digest(path), original_hash)

    def test_audit_never_loads_model_and_freezes_records(self) -> None:
        """纯审查不构造模型，清单拒绝事后更改的记录。"""
        with workspace_tempdir() as root:
            path = fixture(root)
            output = root / "review"
            with (
                patch(
                    "app.e2_offline.read_image",
                    return_value=np.zeros((20, 20, 3), np.uint8),
                ),
                patch("app.e2_offline.DepthAnythingV2Estimator") as model,
            ):
                report = run([path], output, CONFIG, infer_relative=False)
                model.assert_not_called()
            self.assertEqual(report["sample_count"], 1)
            self.assertEqual(
                frozen_paths(output / "report.json", root), [path.resolve()]
            )
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                frozen_paths(output / "report.json", root)

    def test_hash_and_configuration_mismatch_rejected(self) -> None:
        """拒绝图像篡改以及与采集快照不同的配置。"""
        with workspace_tempdir() as root:
            path = fixture(root)
            with self.assertRaisesRegex(ValueError, "configuration"):
                load_sample(path, load_fusion_config(CONFIG), "wrong")
            (path.parent / "frame.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash"):
                load_sample(path, load_fusion_config(CONFIG), digest(CONFIG))

    def test_relative_uses_production_summary_without_metric(self) -> None:
        """模拟深度值验证中心统计复用，且不产生米制证据。"""
        with workspace_tempdir() as root:
            path = fixture(root)
            with (
                patch(
                    "app.e2_offline.read_image",
                    return_value=np.zeros((20, 20, 3), np.uint8),
                ),
                patch("app.e2_offline.DepthAnythingV2Estimator") as model,
            ):
                model.return_value.estimate.return_value = RelativeDepth(
                    "synthetic",
                    np.full((20, 20), 7.0, np.float32),
                    np.ones((20, 20), dtype=np.bool_),
                )
                run([path], root / "relative", CONFIG, infer_relative=True)
                model.assert_called_once()
            report = json.loads(
                (root / "relative/report.json").read_text(encoding="utf-8")
            )
            row = report["samples"][0]
            self.assertEqual(row["depth_summary"]["relative_depth"], 7.0)
            self.assertEqual(row["depth_summary"]["roi"], [5, 5, 15, 15])
            self.assertIsNone(row["metric_evidence"])

    def test_failure_is_retained_and_output_not_overwritten(self) -> None:
        """失败样本仍计入结果，重复输出路径不能覆盖旧报告。"""
        with workspace_tempdir() as root:
            path = fixture(root)
            with (
                patch(
                    "app.e2_offline.read_image",
                    return_value=np.zeros((20, 20, 3), np.uint8),
                ),
                patch("app.e2_offline.DepthAnythingV2Estimator") as model,
            ):
                model.return_value.estimate.side_effect = RuntimeError(
                    "synthetic failure"
                )
                run([path], root / "relative", CONFIG, infer_relative=True)
                with self.assertRaises(FileExistsError):
                    run([path], root / "relative", CONFIG, infer_relative=True)
            report = json.loads(
                (root / "relative/report.json").read_text(encoding="utf-8")
            )
            self.assertIn("synthetic failure", report["samples"][0]["error"])
