"""COCO 风格的检测精度指标：IoU、贪心匹配、AP/AR 与工作点 P/R。

本模块只做数学与匹配，不读文件、不加载模型、不打印结果，便于用合成数据单元测试。
指标定义对齐 COCO 检测协议：101 点插值 AP、IoU 0.5:0.05:0.95、面积区间 s/m/l；
``MAX_DETECTIONS`` 按 COCO 的语义在**每张图每个类别**内取前 N 个预测（不是每类全局取
前 N，后者会把多实例类别的召回整段截掉）；面积区间以外的真值按 COCO 的做法**忽略**
（与它们重叠的预测既不记 TP 也不记 FP），crowd 真值同理。
AP 与召回都在「有真值的类别」上平均，无真值的类别返回 None 而不是 0，
以免把「不适用」当成「全错」。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray

IOU_THRESHOLDS: tuple[float, ...] = tuple(
    round(0.5 + 0.05 * index, 2) for index in range(10)
)
MAX_DETECTIONS = 100
CROWD_IGNORE_IOU = 0.5
RECALL_POINTS: NDArray[np.float64] = np.linspace(0.0, 1.0, 101)
AREA_RANGES: tuple[tuple[str, float, float], ...] = (
    ("all", 0.0, 1e10),
    ("small", 0.0, 32.0**2),
    ("medium", 32.0**2, 96.0**2),
    ("large", 96.0**2, 1e10),
)

Box = tuple[float, float, float, float]
MatchList = list[tuple[float, bool]]


@dataclass(frozen=True)
class TruthBox:
    """一个真值框：类别、xywh 像素框与 crowd 标记。"""

    category_id: int
    box: Box
    is_crowd: bool = False

    def __post_init__(self) -> None:
        """校验类别与非负宽高；crowd 真值不参与召回分母。"""
        if self.category_id < 1:
            raise ValueError("truth category id must be positive")
        _validate_box(self.box)


@dataclass(frozen=True)
class Prediction:
    """一条预测：类别、分数与 xywh 像素框。"""

    category_id: int
    score: float
    box: Box

    def __post_init__(self) -> None:
        """校验类别、有限且落在 0..1 的分数，以及非负宽高。"""
        if self.category_id < 1:
            raise ValueError("prediction category id must be positive")
        if not isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("prediction score must be finite and within 0..1")
        _validate_box(self.box)


CategoryEntry = tuple[tuple[Box, ...], tuple[Prediction, ...], tuple[bool, ...]]
CategoryIndex = dict[int, tuple[CategoryEntry, ...]]


@dataclass(frozen=True)
class PointMetrics:
    """工作点指标：固定分数阈值下的 TP/FP/FN 与精确率、召回率、F1。"""

    confidence: float
    iou_threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass(frozen=True)
class CategoryMetrics:
    """单个类别的 AP、AP50 与该类别的真值、预测计数。"""

    category_id: int
    name: str
    truth_instances: int
    predictions: int
    ap: float | None
    ap50: float | None


@dataclass(frozen=True)
class DetectionMetrics:
    """一次评估的完整结果：汇总指标、分类别结果与工作点指标。"""

    ap: float | None
    ap50: float | None
    ap75: float | None
    ap_small: float | None
    ap_medium: float | None
    ap_large: float | None
    ar_1: float | None
    ar_10: float | None
    ar_100: float | None
    categories_with_truth: int
    categories_without_truth: tuple[int, ...]
    images_evaluated: int
    truth_instances: int
    predictions_total: int
    per_category: tuple[CategoryMetrics, ...]
    operating_point: PointMetrics


def _validate_box(box: Box) -> None:
    """校验 xywh 框的有限性与正面积；不检查是否落在图像范围内。"""
    if len(box) != 4 or not all(isfinite(value) for value in box):
        raise ValueError("box must contain four finite values")
    if box[2] <= 0.0 or box[3] <= 0.0:
        raise ValueError("box width and height must be positive")


def box_area(box: Box) -> float:
    """返回 xywh 框的面积（像素平方）。"""
    return box[2] * box[3]


def box_iou(first: Box, second: Box) -> float:
    """返回两个 xywh 框的交并比；不相交时为 0。"""
    first_x2, first_y2 = first[0] + first[2], first[1] + first[3]
    second_x2, second_y2 = second[0] + second[2], second[1] + second[3]
    overlap_width = min(first_x2, second_x2) - max(first[0], second[0])
    overlap_height = min(first_y2, second_y2) - max(first[1], second[1])
    if overlap_width <= 0.0 or overlap_height <= 0.0:
        return 0.0
    intersection = overlap_width * overlap_height
    union = box_area(first) + box_area(second) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _prediction_score(prediction: Prediction) -> float:
    """排序键：预测分数；用于按分数降序排列同一张图的预测。"""
    return prediction.score


def _match_score(match: tuple[float, bool]) -> float:
    """排序键：匹配项的分数；用于全局按分数降序累计命中。"""
    return match[0]


def match_image(
    truth: Sequence[TruthBox],
    predictions: Sequence[Prediction],
    iou_threshold: float,
    ignored: Sequence[bool] | None = None,
) -> tuple[MatchList, int]:
    """在单张图上按类别贪心匹配，返回 (分数, 是否命中) 列表与召回分母。

    ``ignored`` 是逐真值的忽略标记，默认只忽略 crowd 真值；面积区间以外的真值也按
    同样方式忽略。被忽略的真值不进入召回分母，与它们重叠达到 ``CROWD_IGNORE_IOU``
    的预测既不记 TP 也不记 FP——这与 COCO 的做法一致，避免把其它尺寸的预测误记为误检。
    """
    flags = list(ignored) if ignored is not None else [item.is_crowd for item in truth]
    if len(flags) != len(truth):
        raise ValueError("ignored flags must align with truth boxes")
    ordered = sorted(predictions, key=_prediction_score, reverse=True)
    claimed: set[int] = set()
    matches: MatchList = []
    for prediction in ordered:
        best_index = -1
        best_iou = 0.0
        for index, truth_box in enumerate(truth):
            if (
                flags[index]
                or truth_box.category_id != prediction.category_id
                or index in claimed
            ):
                continue
            value = box_iou(prediction.box, truth_box.box)
            if value > best_iou:
                best_iou, best_index = value, index
        if best_index >= 0 and best_iou >= iou_threshold:
            claimed.add(best_index)
            matches.append((prediction.score, True))
            continue
        if _overlaps_ignored(truth, flags, prediction):
            continue
        matches.append((prediction.score, False))
    truth_count = sum(
        1 for index, item in enumerate(truth) if not flags[index] and not item.is_crowd
    )
    return matches, truth_count


def _overlaps_ignored(
    truth: Sequence[TruthBox],
    flags: Sequence[bool],
    prediction: Prediction,
) -> bool:
    """判断预测是否落在同类别的被忽略真值上，从而既不记命中也不记误检。"""
    return any(
        flags[index]
        and item.category_id == prediction.category_id
        and box_iou(prediction.box, item.box) >= CROWD_IGNORE_IOU
        for index, item in enumerate(truth)
    )


def average_precision(
    matches: Sequence[tuple[float, bool]],
    truth_count: int,
) -> float | None:
    """按 COCO 的 101 点插值计算 AP；该类无真值时返回 None。

    每图每类的预测上限已在收集阶段应用，因此这里不再截断。
    """
    if truth_count <= 0:
        return None
    if not matches:
        return 0.0
    ordered = sorted(matches, key=_match_score, reverse=True)
    hits = np.array([1.0 if is_hit else 0.0 for _, is_hit in ordered], dtype=np.float64)
    cumulative = np.cumsum(hits)
    precision = cumulative / np.arange(1, hits.size + 1, dtype=np.float64)
    recall = cumulative / float(truth_count)
    envelope = np.zeros(RECALL_POINTS.shape, dtype=np.float64)
    for position, point in enumerate(RECALL_POINTS):
        eligible = precision[recall >= point]
        envelope[position] = float(eligible.max()) if eligible.size else 0.0
    return float(envelope.mean())


def recall_at(
    matches: Sequence[tuple[float, bool]],
    truth_count: int,
) -> float | None:
    """返回该类别已经应用每图上限后的最大召回；无真值时返回 None。"""
    if truth_count <= 0:
        return None
    hits = sum(1 for _, is_hit in matches if is_hit)
    return hits / float(truth_count)


def _index_by_category(
    ground_truth: Mapping[str, Sequence[TruthBox]],
    predictions: Mapping[str, Sequence[Prediction]],
) -> CategoryIndex:
    """把逐图输入按类别预索引，只保留每张图上该类别的框，避免上游重复扫描。"""
    pending: dict[int, list[CategoryEntry]] = {}
    for key in sorted(set(ground_truth) | set(predictions)):
        truth_boxes = tuple(ground_truth.get(key, ()))
        prediction_boxes = tuple(predictions.get(key, ()))
        categories = {item.category_id for item in truth_boxes}
        categories.update(item.category_id for item in prediction_boxes)
        for category_id in categories:
            category_truth = tuple(
                item.box for item in truth_boxes if item.category_id == category_id
            )
            category_crowd = tuple(
                item.is_crowd for item in truth_boxes if item.category_id == category_id
            )
            category_predictions = tuple(
                item for item in prediction_boxes if item.category_id == category_id
            )
            pending.setdefault(category_id, []).append(
                (category_truth, category_predictions, category_crowd)
            )
    return {key: tuple(value) for key, value in pending.items()}


def _class_matches(
    index: CategoryIndex,
    category_id: int,
    iou_threshold: float,
    area_bounds: tuple[float, float],
    max_detections: int,
) -> tuple[MatchList, int]:
    """收集某个类别在全部非空图像上的匹配结果。

    ``max_detections`` 按 COCO 语义在每张图每个类别内生效：先按分数取该图该类的前
    ``max_detections`` 个预测，再与区间内的真值匹配；面积区间以外的真值按忽略处理。
    """
    low, high = area_bounds
    matches: MatchList = []
    truth_count = 0
    for boxes, predictions, crowd in index.get(category_id, ()):
        truth = tuple(
            TruthBox(category_id=category_id, box=box, is_crowd=is_crowd)
            for box, is_crowd in zip(boxes, crowd, strict=True)
        )
        ignored = [
            item.is_crowd or not low <= box_area(item.box) < high for item in truth
        ]
        ordered = sorted(predictions, key=_prediction_score, reverse=True)
        image_matches, image_truth = match_image(
            truth, ordered[:max_detections], iou_threshold, ignored
        )
        matches.extend(image_matches)
        truth_count += image_truth
    return matches, truth_count


def _mean(values: Sequence[float | None]) -> float | None:
    """对非 None 的值求平均；全为 None 时返回 None。"""
    present = [value for value in values if value is not None]
    if not present:
        return None
    return float(sum(present) / len(present))


def _average_precision_over_iou(
    index: CategoryIndex,
    category_id: int,
    area_bounds: tuple[float, float],
    iou_thresholds: Sequence[float],
    max_detections: int,
) -> float | None:
    """在给定 IoU 阈值集合上平均某个类别、某个面积区间的 AP。"""
    values: list[float | None] = []
    for threshold in iou_thresholds:
        matches, truth_count = _class_matches(
            index, category_id, threshold, area_bounds, max_detections
        )
        values.append(average_precision(matches, truth_count))
    return _mean(values)


def _recall_over_iou(
    index: CategoryIndex,
    category_id: int,
    area_bounds: tuple[float, float],
    max_detections: int,
    iou_thresholds: Sequence[float] = IOU_THRESHOLDS,
) -> float | None:
    """在给定 IoU 阈值集合上平均某个类别、某个面积区间的召回。"""
    values: list[float | None] = []
    for threshold in iou_thresholds:
        matches, truth_count = _class_matches(
            index, category_id, threshold, area_bounds, max_detections
        )
        values.append(recall_at(matches, truth_count))
    return _mean(values)


def evaluate_detections(
    ground_truth: Mapping[str, Sequence[TruthBox]],
    predictions: Mapping[str, Sequence[Prediction]],
    category_names: Mapping[int, str],
    operating_confidence: float,
    operating_iou: float = IOU_THRESHOLDS[0],
) -> DetectionMetrics:
    """计算 COCO 风格汇总、分类别 AP 与给定分数下的工作点指标。

    ``ground_truth`` 与 ``predictions`` 以图像键对齐；缺键按空序列处理。
    汇总指标只在实际有真值的类别上平均，并把无真值的类别列在结果里。
    """
    if not 0.0 <= operating_confidence <= 1.0:
        raise ValueError("operating confidence must be within 0..1")
    index = _index_by_category(ground_truth, predictions)
    categories = sorted(category_names)
    ap_values: list[float | None] = []
    ap50_values: list[float | None] = []
    ap75_values: list[float | None] = []
    area_values: dict[str, list[float | None]] = {key: [] for key, _, _ in AREA_RANGES}
    area_recall: dict[str, list[float | None]] = {key: [] for key, _, _ in AREA_RANGES}
    ar_values: dict[int, list[float | None]] = {1: [], 10: [], 100: []}
    per_category: list[CategoryMetrics] = []
    without_truth: list[int] = []
    truth_total = 0
    predictions_total = 0
    all_bounds = AREA_RANGES[0][1:]
    for category_id in categories:
        entries = index.get(category_id, ())
        truth_count = sum(
            1 for _, _, crowd in entries for is_crowd in crowd if not is_crowd
        )
        prediction_count = sum(
            len(category_predictions) for _, category_predictions, _ in entries
        )
        truth_total += truth_count
        predictions_total += prediction_count
        if truth_count == 0:
            without_truth.append(category_id)
        ap = _average_precision_over_iou(
            index, category_id, all_bounds, IOU_THRESHOLDS, MAX_DETECTIONS
        )
        ap50 = _average_precision_over_iou(
            index, category_id, all_bounds, (IOU_THRESHOLDS[0],), MAX_DETECTIONS
        )
        ap75 = _average_precision_over_iou(
            index, category_id, all_bounds, (IOU_THRESHOLDS[5],), MAX_DETECTIONS
        )
        ap_values.append(ap)
        ap50_values.append(ap50)
        ap75_values.append(ap75)
        for key, low, high in AREA_RANGES[1:]:
            area_values[key].append(
                _average_precision_over_iou(
                    index, category_id, (low, high), IOU_THRESHOLDS, MAX_DETECTIONS
                )
            )
            area_recall[key].append(
                _recall_over_iou(index, category_id, (low, high), MAX_DETECTIONS)
            )
        for limit in ar_values:
            ar_values[limit].append(
                _recall_over_iou(index, category_id, all_bounds, limit)
            )
        per_category.append(
            CategoryMetrics(
                category_id=category_id,
                name=category_names[category_id],
                truth_instances=truth_count,
                predictions=prediction_count,
                ap=ap,
                ap50=ap50,
            )
        )
    return DetectionMetrics(
        ap=_mean(ap_values),
        ap50=_mean(ap50_values),
        ap75=_mean(ap75_values),
        ap_small=_mean(area_values["small"]),
        ap_medium=_mean(area_values["medium"]),
        ap_large=_mean(area_values["large"]),
        ar_1=_mean(ar_values[1]),
        ar_10=_mean(ar_values[10]),
        ar_100=_mean(ar_values[100]),
        categories_with_truth=len(categories) - len(without_truth),
        categories_without_truth=tuple(without_truth),
        images_evaluated=len(set(ground_truth) | set(predictions)),
        truth_instances=truth_total,
        predictions_total=predictions_total,
        per_category=tuple(per_category),
        operating_point=operating_point(
            ground_truth, predictions, operating_confidence, operating_iou
        ),
    )


def operating_point(
    ground_truth: Mapping[str, Sequence[TruthBox]],
    predictions: Mapping[str, Sequence[Prediction]],
    confidence: float,
    iou_threshold: float = IOU_THRESHOLDS[0],
) -> PointMetrics:
    """计算固定分数阈值下的微平均 TP/FP/FN 与精确率、召回率、F1。

    工作点是生产口径（按分数过滤，不设每图预测上限），与 AP 的 maxDets 协议无关。
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be within 0..1")
    true_positives = 0
    false_positives = 0
    positives = 0
    for image_key, truth_boxes in ground_truth.items():
        selected = [
            item for item in predictions.get(image_key, ()) if item.score >= confidence
        ]
        matches, truth_count = match_image(truth_boxes, selected, iou_threshold)
        hits = sum(1 for _, is_hit in matches if is_hit)
        true_positives += hits
        false_positives += len(matches) - hits
        positives += truth_count
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else None
    )
    recall = true_positives / positives if positives else None
    if precision is not None and recall is not None and precision + recall > 0.0:
        f1: float | None = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    return PointMetrics(
        confidence=confidence,
        iou_threshold=iou_threshold,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=positives - true_positives,
        precision=precision,
        recall=recall,
        f1=f1,
    )
