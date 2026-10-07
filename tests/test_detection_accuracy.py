"""检测精度入口的合成用例：标注解析、标签映射、确定性子集与端到端报告。"""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from app.config import DetectionConfig
from app.detection_accuracy import (
    load_coco_dataset,
    main,
    map_labels_to_categories,
    select_images,
    to_predictions,
)
from app.vision.detector import Detection, Image
from tests.workspace_temp import workspace_tempdir


def write_annotations(
    path: Path, file_name: str = "a.png", crowd: bool = False
) -> None:
    """写入最小可用的 COCO 检测标注，供解析与入口测试复用。"""
    annotations: list[dict[str, object]] = [
        {
            "image_id": 7,
            "category_id": 1,
            "bbox": [0.0, 0.0, 32.0, 32.0],
            "iscrowd": 0,
        }
    ]
    if crowd:
        annotations.append(
            {
                "image_id": 7,
                "category_id": 3,
                "bbox": [0.0, 0.0, 8.0, 8.0],
                "iscrowd": 1,
            }
        )
    document = {
        "images": [{"id": 7, "file_name": file_name, "width": 64, "height": 64}],
        "categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "chair"}],
        "annotations": annotations,
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def write_image(path: Path) -> None:
    """写出可解码的小图；内容不重要，只用于走通读取路径。"""
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    path.write_bytes(encoded.tobytes())


class FakeDetector:
    """固定输出的假检测器：一张图、一个 person 框，用于走通入口。"""

    def __init__(self, config: DetectionConfig) -> None:
        """记录配置，便于核对评估入口使用的阈值。"""
        self.config = config

    @property
    def supported_labels(self) -> tuple[str, ...]:
        """返回与合成标注一致的类别名。"""
        return ("person", "chair")

    def detect(self, image: Image) -> tuple[Detection, ...]:
        """返回与真值完全重合的一个 person 框。"""
        return (Detection(label="person", confidence=0.9, bbox=(0, 0, 32, 32)),)


class CocoLoaderTests(unittest.TestCase):
    def test_load_keeps_detection_fields(self) -> None:
        """解析后保留图像尺寸、类别名与真值框，并统计 crowd 数量。"""
        with workspace_tempdir() as directory:
            path = Path(directory) / "instances.json"
            write_annotations(path, crowd=True)
            dataset = load_coco_dataset(path)
        self.assertEqual(dataset.image_sizes, {"a.png": (64, 64)})
        self.assertEqual(dataset.category_names, {1: "person", 3: "chair"})
        self.assertEqual(dataset.crowd_boxes, 1)
        boxes = dataset.ground_truth["a.png"]
        self.assertEqual(len(boxes), 2)
        self.assertFalse(boxes[0].is_crowd)
        self.assertTrue(boxes[1].is_crowd)

    def test_load_rejects_incomplete_documents(self) -> None:
        """缺少 images／categories／annotations 的文档必须报错。"""
        with workspace_tempdir() as directory:
            path = Path(directory) / "broken.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_coco_dataset(path)

    def test_label_mapping_requires_every_name(self) -> None:
        """权重类别名必须全部出现在标注类别里，缺失时列出并报错。"""
        names = {1: "person", 3: "chair"}
        self.assertEqual(
            map_labels_to_categories(("person", "chair"), names),
            {"person": 1, "chair": 3},
        )
        with self.assertRaises(ValueError):
            map_labels_to_categories(("person", "door"), names)

    def test_select_images_is_deterministic(self) -> None:
        """子集只按文件名排序截断，不做随机挑选。"""
        available = {"c.png": (1, 1), "a.png": (1, 1), "b.png": (1, 1)}
        self.assertEqual(select_images(available, None), ("a.png", "b.png", "c.png"))
        self.assertEqual(select_images(available, 2), ("a.png", "b.png"))
        with self.assertRaises(ValueError):
            select_images(available, 0)

    def test_half_open_boxes_convert_to_xywh(self) -> None:
        """xyxy 半开区间相减得到宽高；未映射类别名必须报错。"""
        detections = (Detection(label="person", confidence=0.9, bbox=(1, 2, 11, 22)),)
        converted = to_predictions(detections, {"person": 1})
        self.assertEqual(converted[0].box, (1.0, 2.0, 10.0, 20.0))
        self.assertEqual(converted[0].category_id, 1)
        with self.assertRaises(ValueError):
            to_predictions(detections, {"chair": 3})


class AccuracyEntryTests(unittest.TestCase):
    def build_case(self, directory: Path, crowd: bool = False) -> tuple[Path, ...]:
        """准备图片、标注、假权重与临时配置，供入口测试使用。"""
        images = directory / "images"
        images.mkdir()
        write_image(images / "a.png")
        annotations = directory / "instances.json"
        write_annotations(annotations, crowd=crowd)
        weights = directory / "dummy.pt"
        weights.write_bytes(b"not-a-real-checkpoint")
        config = directory / "system.ini"
        config.write_text(
            "[detection]\n"
            f"weights = {weights.as_posix()}\n"
            "confidence = 0.35\n"
            "image_size = 640\n"
            "device = cpu\n",
            encoding="utf-8",
        )
        return annotations, images, config

    def test_entry_writes_metrics_and_refuses_overwrite(self) -> None:
        """端到端：AP 为 1、未测量项保持 null、报告含局限，且拒绝覆盖输出。"""
        with workspace_tempdir() as directory:
            directory = Path(directory)
            annotations, images, config = self.build_case(directory)
            output = directory / "report.json"
            arguments = [
                "--annotations",
                str(annotations),
                "--images-dir",
                str(images),
                "--config",
                str(config),
                "--output",
                str(output),
            ]
            with patch("app.detection_accuracy.Yolo11Detector", FakeDetector):
                self.assertEqual(main(arguments), 0)
                with self.assertRaises(FileExistsError):
                    main(arguments)
            report = json.loads(output.read_text(encoding="utf-8"))
            metrics = report["metrics"]
            self.assertEqual(report["schema"], "coco_val2017_detection_accuracy_v1")
            self.assertEqual(metrics["ap"], 1.0)
            self.assertEqual(metrics["ap50"], 1.0)
            self.assertEqual(metrics["ap_medium"], 1.0)
            self.assertIsNone(metrics["ap_small"])
            self.assertEqual(metrics["operating_point"]["recall"], 1.0)
            self.assertEqual(metrics["categories_without_truth"], [3])
            self.assertEqual(report["dataset"]["images_evaluated"], 1)
            self.assertEqual(report["detector"]["confidence_for_dump"], 0.001)
            self.assertEqual(report["detector"]["operating_confidence"], 0.35)
            self.assertTrue(report["limits"])
            self.assertEqual(report["per_image"][0]["true_positives"], 1)

    def test_entry_rejects_eval_above_operating_confidence(self) -> None:
        """评估阈值高于工作点阈值时直接拒绝，避免混淆两种口径。"""
        with workspace_tempdir() as directory:
            directory = Path(directory)
            annotations, images, config = self.build_case(directory)
            arguments = [
                "--annotations",
                str(annotations),
                "--images-dir",
                str(images),
                "--config",
                str(config),
                "--output",
                str(directory / "report.json"),
                "--eval-confidence",
                "0.9",
            ]
            with patch("app.detection_accuracy.Yolo11Detector", FakeDetector):
                with self.assertRaises(SystemExit):
                    main(arguments)


if __name__ == "__main__":
    unittest.main()
