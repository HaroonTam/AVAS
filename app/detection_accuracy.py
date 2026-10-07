"""COCO val2017 检测精度评估入口：独立于 app.main 与 app.live 的生产路径。

本入口只读本地权重与本地标注，做单次前向评估；不打开摄像头、麦克风或语音输出，
不调用 Agent、不联网、不下载权重，也不把预测或图像写入报告。
精度结论只适用于被评估的数据分布；COCO 类别不含「门」「台阶」。
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from app.accuracy_metrics import (
    AREA_RANGES,
    IOU_THRESHOLDS,
    MAX_DETECTIONS,
    DetectionMetrics,
    Prediction,
    TruthBox,
    evaluate_detections,
    match_image,
)
from app.benchmark import code_identity, file_sha256, package_versions, save_report
from app.config import DetectionConfig, load_config
from app.vision.detector import Detection, Image, Yolo11Detector

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CocoDataset:
    """COCO 检测标注的评估视图：图像尺寸、类别名与逐图真值框。"""

    image_sizes: dict[str, tuple[int, int]]
    category_names: dict[int, str]
    ground_truth: dict[str, tuple[TruthBox, ...]]
    crowd_boxes: int


@dataclass(frozen=True)
class ImageOutcome:
    """单张图的评估记录：真值数、预测数与工作点命中情况。"""

    file_name: str
    truth: int
    predictions: int
    true_positives: int
    false_positives: int
    false_negatives: int
    detection_ms: float


def load_coco_dataset(path: Path) -> CocoDataset:
    """读取 COCO instances 标注，只保留检测评估需要的字段并校验结构。"""
    document: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("annotations must be a JSON object")
    raw_images = document.get("images")
    raw_categories = document.get("categories")
    raw_annotations = document.get("annotations")
    if not isinstance(raw_images, list) or not raw_images:
        raise ValueError("annotations must contain a nonempty images list")
    if not isinstance(raw_categories, list) or not raw_categories:
        raise ValueError("annotations must contain a nonempty categories list")
    if not isinstance(raw_annotations, list):
        raise ValueError("annotations must contain an annotations list")
    image_sizes: dict[str, tuple[int, int]] = {}
    image_ids: dict[int, str] = {}
    for entry in raw_images:
        if not isinstance(entry, dict):
            raise ValueError("invalid image entry")
        image_id = entry.get("id")
        file_name = entry.get("file_name")
        width, height = entry.get("width"), entry.get("height")
        if (
            not isinstance(image_id, int)
            or not isinstance(file_name, str)
            or not file_name
            or not isinstance(width, int)
            or not isinstance(height, int)
            or width <= 0
            or height <= 0
        ):
            raise ValueError("invalid image entry fields")
        image_ids[image_id] = file_name
        image_sizes[file_name] = (width, height)
    category_names: dict[int, str] = {}
    for entry in raw_categories:
        if not isinstance(entry, dict):
            raise ValueError("invalid category entry")
        category_id = entry.get("id")
        name = entry.get("name")
        if (
            not isinstance(category_id, int)
            or category_id < 1
            or not isinstance(name, str)
            or not name.strip()
        ):
            raise ValueError("invalid category entry fields")
        category_names[category_id] = name
    boxes: dict[str, list[TruthBox]] = {name: [] for name in image_sizes}
    crowd_count = 0
    for entry in raw_annotations:
        if not isinstance(entry, dict):
            raise ValueError("invalid annotation entry")
        image_id = entry.get("image_id")
        category_id = entry.get("category_id")
        bbox = entry.get("bbox")
        is_crowd = entry.get("iscrowd", 0)
        if (
            not isinstance(image_id, int)
            or image_id not in image_ids
            or not isinstance(category_id, int)
            or category_id not in category_names
            or not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(value, (int, float)) for value in bbox)
            or is_crowd not in (0, 1)
        ):
            raise ValueError("invalid annotation entry fields")
        width, height = float(bbox[2]), float(bbox[3])
        if width <= 0.0 or height <= 0.0:
            raise ValueError("annotation boxes must have positive size")
        boxes[image_ids[image_id]].append(
            TruthBox(
                category_id=category_id,
                box=(float(bbox[0]), float(bbox[1]), width, height),
                is_crowd=bool(is_crowd),
            )
        )
        crowd_count += int(bool(is_crowd))
    return CocoDataset(
        image_sizes=image_sizes,
        category_names=category_names,
        ground_truth={name: tuple(items) for name, items in boxes.items() if items},
        crowd_boxes=crowd_count,
    )


def map_labels_to_categories(
    labels: tuple[str, ...], category_names: dict[int, str]
) -> dict[str, int]:
    """把权重类别名映射到 COCO category_id；名字缺失时列出并报错。"""
    name_to_id = {name: category_id for category_id, name in category_names.items()}
    missing = sorted(label for label in labels if label not in name_to_id)
    if missing:
        raise ValueError(f"detector labels absent from annotations: {missing}")
    return {label: name_to_id[label] for label in labels}


def select_images(
    available: dict[str, tuple[int, int]], limit: int | None
) -> tuple[str, ...]:
    """按文件名排序取确定性子集；上限只截断顺序，不做随机挑选。"""
    names = tuple(sorted(available))
    if limit is None:
        return names
    if limit < 1:
        raise ValueError("max images must be positive")
    return names[:limit]


def decode_image(path: Path) -> Image:
    """按生产入口的同一约定读取 BGR 图像；失败时明确报错。"""
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"cannot decode image: {path.name}")
    return image


def to_predictions(
    detections: tuple[Detection, ...],
    labels_to_categories: dict[str, int],
) -> tuple[Prediction, ...]:
    """把检测结果转成 xywh 预测；半开区间 xyxy 直接相减得到宽高。"""
    converted: list[Prediction] = []
    for item in detections:
        category_id = labels_to_categories.get(item.label)
        if category_id is None:
            raise ValueError(f"unmapped detector label: {item.label}")
        x1, y1, x2, y2 = item.bbox
        converted.append(
            Prediction(
                category_id=category_id,
                score=item.confidence,
                box=(float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
            )
        )
    return tuple(converted)


def evaluate_dataset(
    dataset: CocoDataset,
    detector: Yolo11Detector,
    images_dir: Path,
    selected: tuple[str, ...],
    labels_to_categories: dict[str, int],
    operating_confidence: float,
) -> tuple[
    dict[str, tuple[Prediction, ...]], tuple[ImageOutcome, ...], dict[str, float]
]:
    """逐图推理并统计；返回预测、逐图结果与检测耗时统计。

    逐图只做一次匹配来得到工作点的 TP/FP/FN，不对每张图重复完整评估。
    """
    predictions: dict[str, tuple[Prediction, ...]] = {}
    outcomes: list[ImageOutcome] = []
    timings: list[float] = []
    for index, file_name in enumerate(selected, 1):
        image = decode_image(images_dir / file_name)
        started = time.perf_counter()
        detections = detector.detect(image)
        latency_ms = (time.perf_counter() - started) * 1000.0
        image_predictions = to_predictions(detections, labels_to_categories)
        predictions[file_name] = image_predictions
        truth = dataset.ground_truth.get(file_name, ())
        selected_predictions = tuple(
            item for item in image_predictions if item.score >= operating_confidence
        )
        matches, truth_count = match_image(truth, selected_predictions, 0.5)
        hits = sum(1 for _, is_hit in matches if is_hit)
        outcomes.append(
            ImageOutcome(
                file_name=file_name,
                truth=truth_count,
                predictions=len(selected_predictions),
                true_positives=hits,
                false_positives=len(matches) - hits,
                false_negatives=truth_count - hits,
                detection_ms=latency_ms,
            )
        )
        timings.append(latency_ms)
        if index % 500 == 0:
            LOGGER.info("evaluated %d/%d images", index, len(selected))
    array = np.array(timings, dtype=np.float64)
    timing_summary = {
        "images": len(selected),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50, method="linear")),
        "p95_ms": float(np.percentile(array, 95, method="linear")),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }
    return predictions, tuple(outcomes), timing_summary


def _outcome_sort_key(outcome: ImageOutcome) -> tuple[int, int, str]:
    """排序键：漏检优先、误检其次、文件名最后，保证清单顺序确定。"""
    return (-outcome.false_negatives, -outcome.false_positives, outcome.file_name)


def worst_examples(
    outcomes: tuple[ImageOutcome, ...], limit: int
) -> tuple[dict[str, object], ...]:
    """按漏检优先、误检其次给出最差图像清单，供报告举漏检/误检示例。"""
    ordered = sorted(outcomes, key=_outcome_sort_key)
    return tuple(
        {
            "file_name": item.file_name,
            "truth": item.truth,
            "predictions": item.predictions,
            "true_positives": item.true_positives,
            "false_positives": item.false_positives,
            "false_negatives": item.false_negatives,
        }
        for item in ordered[:limit]
    )


def metrics_payload(metrics: DetectionMetrics) -> dict[str, object]:
    """把指标结构转成可 JSON 序列化的字典；未测量的项保持 null。"""
    return {
        "ap": metrics.ap,
        "ap50": metrics.ap50,
        "ap75": metrics.ap75,
        "ap_small": metrics.ap_small,
        "ap_medium": metrics.ap_medium,
        "ap_large": metrics.ap_large,
        "ar_1": metrics.ar_1,
        "ar_10": metrics.ar_10,
        "ar_100": metrics.ar_100,
        "categories_with_truth": metrics.categories_with_truth,
        "categories_without_truth": list(metrics.categories_without_truth),
        "images_evaluated": metrics.images_evaluated,
        "truth_instances": metrics.truth_instances,
        "predictions_total": metrics.predictions_total,
        "operating_point": {
            "confidence": metrics.operating_point.confidence,
            "iou_threshold": metrics.operating_point.iou_threshold,
            "true_positives": metrics.operating_point.true_positives,
            "false_positives": metrics.operating_point.false_positives,
            "false_negatives": metrics.operating_point.false_negatives,
            "precision": metrics.operating_point.precision,
            "recall": metrics.operating_point.recall,
            "f1": metrics.operating_point.f1,
        },
        "per_category": [
            {
                "category_id": item.category_id,
                "name": item.name,
                "truth_instances": item.truth_instances,
                "predictions": item.predictions,
                "ap": item.ap,
                "ap50": item.ap50,
            }
            for item in metrics.per_category
        ],
    }


def runtime_identity(config: DetectionConfig) -> dict[str, object]:
    """记录设备、线程与 CUDA 名称；不联网、不探测未安装的可选依赖。"""
    import torch

    identity: dict[str, object] = {
        "device_requested": config.device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": None,
        "torch_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    if torch.cuda.is_available():
        identity["cuda_device"] = torch.cuda.get_device_name(0)
    return identity


def report_payload(
    *,
    dataset: CocoDataset,
    annotations_path: Path,
    images_dir: Path,
    selected: tuple[str, ...],
    config: DetectionConfig,
    config_path: Path,
    eval_confidence: float,
    operating_confidence: float,
    labels: tuple[str, ...],
    metrics: DetectionMetrics,
    outcomes: tuple[ImageOutcome, ...],
    timing: dict[str, float],
    examples: int,
    root: Path,
) -> dict[str, object]:
    """组装完整报告：身份、数据来源、指标、逐图计数与明确的局限清单。"""
    return {
        "schema": "coco_val2017_detection_accuracy_v1",
        "dataset": {
            "name": "COCO val2017",
            "annotations": annotations_path.name,
            "annotations_sha256": file_sha256(annotations_path),
            "images_dir": str(images_dir),
            "images_available": len(dataset.image_sizes),
            "images_evaluated": len(selected),
            "subset_rule": (
                "all images"
                if len(selected) == len(dataset.image_sizes)
                else "sorted file_name prefix"
            ),
            "categories_in_annotations": len(dataset.category_names),
            "crowd_boxes": dataset.crowd_boxes,
            "license_note": (
                "COCO annotations CC BY 4.0; images keep their Flickr licenses; "
                "data stays outside version control"
            ),
        },
        "detector": {
            "weights": str(config.weights),
            "weights_sha256": file_sha256(config.weights),
            "config": config_path.name,
            "image_size": config.image_size,
            "device": config.device,
            "confidence_for_dump": eval_confidence,
            "operating_confidence": operating_confidence,
            "labels": list(labels),
        },
        "protocol": {
            "iou_thresholds": list(IOU_THRESHOLDS),
            "max_detections_per_image": MAX_DETECTIONS,
            "area_ranges": [
                {"name": name, "low": low, "high": high}
                for name, low, high in AREA_RANGES
            ],
            "ap_interpolation": "COCO 101-point",
            "out_of_range_truth": "ignored like crowd boxes",
        },
        "metrics": metrics_payload(metrics),
        "per_image": [
            {
                "file_name": item.file_name,
                "truth": item.truth,
                "predictions": item.predictions,
                "true_positives": item.true_positives,
                "false_positives": item.false_positives,
                "false_negatives": item.false_negatives,
                "detection_ms": item.detection_ms,
            }
            for item in outcomes
        ],
        "worst_examples": list(worst_examples(outcomes, examples)),
        "timing": timing,
        "timing_note": (
            "single pass around detect(); image decode and hashing excluded; "
            "app.benchmark is the dedicated repeated-latency protocol"
        ),
        "packages": package_versions(),
        "runtime": runtime_identity(config),
        "code": code_identity(root),
        "limits": [
            "仅评估 COCO 预训练权重，未在本项目数据上微调",
            "结论只适用于 COCO val2017 分布与 640 输入，不能外推到室内视障场景",
            "COCO 类别不含 door/stairs，不得声称支持门与台阶识别",
            "未测量米制距离误差、告警延迟与摄像头帧率",
            "单模型单次运行，未做跨设备或重复运行对照",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """执行 COCO val2017 精度评估；输出独占且不覆盖任何既有结果。"""
    parser = argparse.ArgumentParser(description="COCO val2017 detection accuracy")
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=root / "configs/system.ini")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-confidence", type=float, default=0.001)
    parser.add_argument("--operating-confidence", type=float)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--examples", type=int, default=10)
    args = parser.parse_args(argv)
    if not 0.0 <= args.eval_confidence <= 1.0:
        parser.error("--eval-confidence must be within 0..1")
    if args.examples < 0:
        parser.error("--examples must not be negative")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("accuracy destination already exists")
    if not args.annotations.is_file():
        raise FileNotFoundError(str(args.annotations))
    if not args.images_dir.is_dir():
        raise NotADirectoryError(str(args.images_dir))
    config = load_config(args.config)
    operating = (
        config.confidence
        if args.operating_confidence is None
        else args.operating_confidence
    )
    if not 0.0 <= operating <= 1.0:
        parser.error("--operating-confidence must be within 0..1")
    if args.eval_confidence > operating:
        parser.error("--eval-confidence must not exceed the operating confidence")
    dataset = load_coco_dataset(args.annotations)
    selected = select_images(dataset.image_sizes, args.max_images)
    detector = Yolo11Detector(replace(config, confidence=args.eval_confidence))
    labels = detector.supported_labels
    labels_to_categories = map_labels_to_categories(labels, dataset.category_names)
    predictions, outcomes, timing = evaluate_dataset(
        dataset, detector, args.images_dir, selected, labels_to_categories, operating
    )
    ground_truth = {name: dataset.ground_truth.get(name, ()) for name in selected}
    metrics = evaluate_detections(
        ground_truth, predictions, dataset.category_names, operating
    )
    report = report_payload(
        dataset=dataset,
        annotations_path=args.annotations,
        images_dir=args.images_dir,
        selected=selected,
        config=config,
        config_path=args.config,
        eval_confidence=args.eval_confidence,
        operating_confidence=operating,
        labels=labels,
        metrics=metrics,
        outcomes=outcomes,
        timing=timing,
        examples=args.examples,
        root=root,
    )
    save_report(report, args.output)
    print(
        json.dumps(
            {
                "images": len(selected),
                "ap": metrics.ap,
                "ap50": metrics.ap50,
                "ap75": metrics.ap75,
                "precision_at_operating_point": metrics.operating_point.precision,
                "recall_at_operating_point": metrics.operating_point.recall,
                "mean_detection_ms": timing["mean_ms"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
