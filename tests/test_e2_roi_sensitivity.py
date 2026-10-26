"""固定ROI对照的合成测试；禁止加载真实模型或相机。"""

import json
import unittest
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.e2_collection import read_object
from app.e2_compare import Observation
from app.e2_offline import digest
from app.e2_roi_sensitivity import (
    build_manifest,
    compare_regions,
    infer,
    main,
    write_json,
)
from app.fusion.config import FusionConfig
from app.fusion.depth_fusion import summarize_depth
from app.vision.depth_estimator import (
    DepthAnythingV2Estimator,
    DepthConfig,
    DepthMap,
    RelativeDepth,
)
from app.vision.detector import Image
from tests.test_e2_compare import observation
from tests.workspace_temp import workspace_tempdir


@dataclass(frozen=True)
class Inputs:
    recipe: Path
    protocol: Path
    config: Path
    datasets: dict[str, list[Observation]]
    reports: dict[str, Path]
    weight_hash: str


def fixture(root: Path) -> Inputs:
    """生成三布景两距离两重复的元数据与假模型文件，图像解码在测试中替换。"""
    config = root / "system.ini"
    config.write_text(
        "[depth]\nmodel_dir = model\nimage_size = 518\ndevice = cpu\n", encoding="utf-8"
    )
    models = root / "model"
    models.mkdir()
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        (models / name).write_bytes(b"synthetic")
    recipe = root / "recipe.json"
    write_json(recipe, {})
    protocol = root / "protocol.json"
    write_json(protocol, read_object(Path("configs/e2_roi_sensitivity.json")))
    datasets: dict[str, list[Observation]] = {
        "supplemental_fit": [],
        "new_validation": [],
    }
    rows: list[dict[str, object]] = []
    for group in ("scene10", "scene11", "scene12"):
        role = "new_validation" if group == "scene11" else "supplemental_fit"
        for distance in (0.5, 3.0):
            for repeat in (1, 2):
                key = f"{group}_{distance}_{repeat}"
                record = root / key / "record.json"
                record.parent.mkdir()
                write_json(
                    record, {"actual_width": 64, "actual_height": 48, "frame_id": key}
                )
                obs = replace(
                    observation(2.0, distance, key),
                    group=group,
                    placement=f"p{len(rows) + 1:02}",
                )
                datasets[role].append(obs)
                rows.append(
                    {
                        "sample_id": key,
                        "record_path": str(record),
                        "record_sha256": digest(record),
                        "image_sha256": key,
                        "analysis_roi": [0, 24, 64, 48],
                        "statistics_roi": [16, 30, 48, 42],
                        "depth_summary": {"relative_depth": 2.0, "status": "available"},
                    }
                )
    reports: dict[str, Path] = {}
    for kind in ("audit", "relative"):
        report = root / f"{kind}.json"
        write_json(
            report,
            {
                "sample_count": 12,
                "config_sha256": digest(config),
                "fusion_config": asdict(FusionConfig()),
                "samples": rows,
            },
        )
        reports[f"source_0_{kind}"] = report
    return Inputs(
        recipe,
        protocol,
        config,
        datasets,
        reports,
        digest(models / "model.safetensors"),
    )


class SyntheticBackend:
    def __init__(self, fail_second: bool = False) -> None:
        """只保存合成失败开关，不打开设备或加载模型。"""
        self.calls = 0
        self.fail_second = fail_second

    def predict(self, image: Image) -> DepthMap:
        """生成恒定原图深度，必要时在第二张模拟失败以验证分母保留。"""
        self.calls += 1
        if self.fail_second and self.calls == 2:
            raise RuntimeError("synthetic failure")
        return np.full(image.shape[:2], 2.0, np.float32)


class RoiSensitivityTests(unittest.TestCase):
    def test_same_map_and_no_double_shrink(self) -> None:
        """原统计严格复用生产核；中心框只缩小一次，已知区域变化能被识别。"""
        values = np.ones((40, 40), np.float32)
        values[10:30, 10:30] = 4
        values[15:25, 15:25] = 8
        depth = RelativeDepth("one", values, np.ones((40, 40), np.bool_))
        fusion = FusionConfig()
        result = compare_regions(
            depth, (0, 0, 40, 40), (10, 10, 30, 30), (15, 15, 25, 25), fusion
        )
        self.assertEqual(
            result["original"], asdict(summarize_depth((0, 0, 40, 40), depth, fusion))
        )
        self.assertEqual(result["center_half"]["sample_count"], 100)
        self.assertEqual(result["center_minus_original_relative"], 4.0)
        depth = RelativeDepth("one", values, np.zeros((40, 40), np.bool_))
        self.assertIsNone(
            compare_regions(
                depth, (0, 0, 40, 40), (10, 10, 30, 30), (15, 15, 25, 25), fusion
            )["center_minus_original_relative"]
        )
        with self.assertRaisesRegex(ValueError, "derivation"):
            compare_regions(
                depth, (0, 0, 40, 40), (10, 10, 30, 30), (16, 15, 25, 25), fusion
            )

    def test_prepare_selection_and_missing_repeat(self) -> None:
        """准备阶段不加载模型，清单保留全部角色；缺少任一重复便拒绝。"""
        with workspace_tempdir() as root:
            data = fixture(root)
            with (
                patch(
                    "app.e2_roi_sensitivity.load_sources",
                    return_value=(data.datasets, data.reports),
                ),
                patch("app.e2_roi_sensitivity.WEIGHTS_SHA256", data.weight_hash),
                patch("app.e2_roi_sensitivity.create_estimator") as model,
            ):
                result = build_manifest(data.recipe, data.protocol, data.config)
                self.assertEqual(result["sample_count"], 12)
                self.assertEqual(
                    sum(s["role"] == "new_validation" for s in result["samples"]), 4
                )
                self.assertEqual(result["samples"][0]["center_roi"], [24, 33, 40, 39])
                data.datasets["new_validation"].pop()
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    build_manifest(data.recipe, data.protocol, data.config)
                model.assert_not_called()

    def test_infer_one_model_and_retained_failure(self) -> None:
        """单模型处理12张，失败帧仍计入分母，成功帧保存可重采样的无pickle浮点图。"""
        with workspace_tempdir() as root:
            data = fixture(root)
            backend = SyntheticBackend(fail_second=True)
            estimator = DepthAnythingV2Estimator(DepthConfig(root), backend=backend)
            with (
                patch(
                    "app.e2_roi_sensitivity.load_sources",
                    return_value=(data.datasets, data.reports),
                ),
                patch("app.e2_roi_sensitivity.WEIGHTS_SHA256", data.weight_hash),
                patch(
                    "app.e2_roi_sensitivity.create_estimator",
                    return_value=(estimator, "cpu"),
                ) as model,
                patch(
                    "app.e2_roi_sensitivity.read_image",
                    return_value=np.zeros((48, 64, 3), np.uint8),
                ),
            ):
                manifest = root / "manifest.json"
                write_json(
                    manifest, build_manifest(data.recipe, data.protocol, data.config)
                )
                output = root / "inferred"
                report = infer(manifest, output)
                model.assert_called_once()
                self.assertEqual(backend.calls, 12)
                self.assertEqual(report["sample_count"], 12)
                self.assertEqual(report["error_count"], 1)
                self.assertEqual(report["available_pair_count"], 11)
                self.assertEqual(report["unavailable_pair_count"], 1)
                with np.load(output / "depth_01.npz", allow_pickle=False) as saved:
                    self.assertEqual(saved["relative_depth"].shape, (48, 64))
                    self.assertEqual(saved["units"].item(), "relative_inverse_depth")
                self.assertFalse(report["test_unlocked"])
                with self.assertRaises(FileExistsError):
                    infer(manifest, output)

    def test_manifest_and_protocol_tamper_precedes_model(self) -> None:
        """改动冻结ROI或启用测试标志会在模型构造前失败。"""
        with workspace_tempdir() as root:
            data = fixture(root)
            with (
                patch(
                    "app.e2_roi_sensitivity.load_sources",
                    return_value=(data.datasets, data.reports),
                ),
                patch("app.e2_roi_sensitivity.WEIGHTS_SHA256", data.weight_hash),
                patch("app.e2_roi_sensitivity.create_estimator") as model,
            ):
                frozen = build_manifest(data.recipe, data.protocol, data.config)
                frozen["samples"][0]["center_roi"] = [0, 0, 2, 2]
                manifest = root / "manifest.json"
                write_json(manifest, frozen)
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    infer(manifest, root / "bad")
                protocol = read_object(data.protocol)
                protocol["test_unlocked"] = True
                data.protocol.write_text(json.dumps(protocol), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "fixed ROI protocol"):
                    build_manifest(data.recipe, data.protocol, data.config)
                model.assert_not_called()
                self.assertEqual(main(["--infer", "--output", str(root / "bad")]), 2)
