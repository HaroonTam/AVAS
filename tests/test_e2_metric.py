"""Metric 候选的合成边界测试，无网络、模型和硬件。"""

import hashlib
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.e2_metric import download_model, metric_summary, run_metric, verify_model
from app.e2_offline import run
from app.fusion.config import FusionConfig
from app.fusion.depth_fusion import summarize_depth
from app.vision.depth_estimator import RelativeDepth
from tests.test_e2_offline import CONFIG, fixture
from tests.workspace_temp import workspace_tempdir


class E2MetricTests(unittest.TestCase):
    def test_download_identity_and_no_overwrite(self) -> None:
        """模拟下载只在哈希匹配时发布文件，已验证文件不重复下载。"""
        content = b"synthetic"
        files = {"config.json": hashlib.sha256(content).hexdigest()}
        with workspace_tempdir() as root, patch("app.e2_metric.FILES", files):
            with patch(
                "urllib.request.urlopen", return_value=io.BytesIO(content)
            ) as request:
                download_model(root)
                download_model(root)
                request.assert_called_once()
            (root / "config.json").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_model(root)

    def test_bad_download_keeps_partial_without_publishing(self) -> None:
        """损坏下载保留 part 供排查，不生成看似可用的模型文件。"""
        with (
            workspace_tempdir() as root,
            patch("app.e2_metric.FILES", {"config.json": "wrong"}),
        ):
            with patch("urllib.request.urlopen", return_value=io.BytesIO(b"bad")):
                with self.assertRaises(ValueError):
                    download_model(root)
            self.assertTrue((root / "config.json.part").exists())
            self.assertFalse((root / "config.json").exists())

    def test_shared_statistics_preserve_production_values(self) -> None:
        """相同有效数组下两种单位路径统计一致，防止重构改变融合结果。"""
        values = np.linspace(1, 10, 400, dtype=np.float32).reshape(20, 20)
        config = FusionConfig()
        relative = summarize_depth(
            (0, 0, 20, 20),
            RelativeDepth("test", values, np.ones((20, 20), np.bool_)),
            config,
        )
        metric = metric_summary(values, (0, 0, 20, 20), config)
        self.assertEqual(metric["predicted_depth_m"], relative.relative_depth)
        self.assertEqual(metric["iqr_m"], relative.relative_depth_iqr)
        self.assertEqual(metric["retained_count"], relative.retained_count)
        self.assertEqual(metric["units"], "m")

    def test_metric_invalid_values_remain_unavailable(self) -> None:
        """非正、无穷、NaN 和模型上限外输入不能产生有效米制统计。"""
        for value in (0, -1, float("nan"), float("inf"), 21):
            summary = metric_summary(
                np.full((20, 20), value, np.float32), (0, 0, 20, 20), FusionConfig()
            )
            self.assertIsNone(summary["predicted_depth_m"])
            self.assertEqual(summary["valid_fraction"], 0)

    def test_frozen_metric_diagnostic_has_no_evidence(self) -> None:
        """模拟候选推理保持冻结样本与下半板面 ROI，不签发生产证据。"""
        with workspace_tempdir() as root:
            path = fixture(root)
            image = np.zeros((20, 20, 3), np.uint8)
            with patch("app.e2_offline.read_image", return_value=image):
                run(
                    [path],
                    root / "audit",
                    CONFIG,
                    infer_relative=False,
                    board_lower_half=True,
                )
                with (
                    patch("app.e2_metric.read_image", return_value=image),
                    patch("app.e2_metric.MetricBackend") as model,
                ):
                    model.return_value.device = "synthetic"
                    model.return_value.torch_version = "synthetic"
                    model.return_value.predict.return_value = np.full(
                        (20, 20), 2.0, np.float32
                    )
                    run_metric(
                        root / "audit/report.json",
                        root / "metric",
                        Path("unused"),
                        CONFIG,
                    )
            result = json.loads(
                (root / "metric/report.json").read_text(encoding="utf-8")
            )
            row = result["samples"][0]
            self.assertIsNone(row["metric_evidence"])
            self.assertEqual(row["metric_summary"]["predicted_depth_m"], 2)
            self.assertEqual(row["metric_summary"]["roi"], [5, 13, 15, 17])
