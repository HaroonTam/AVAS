"""检测精度指标的合成用例：AP/AR 定义、IoU 边界、面积区间与 crowd 忽略规则。"""

import unittest

from app.accuracy_metrics import (
    Prediction,
    TruthBox,
    box_iou,
    evaluate_detections,
    operating_point,
)

CATEGORY_NAMES = {1: "person", 2: "chair"}


def perfect_truth(box: tuple[float, float, float, float]) -> dict:
    """构造单图单真值的输入，便于逐项断言。"""
    return {"img1": (TruthBox(category_id=1, box=box),)}


class AccuracyMetricTests(unittest.TestCase):
    def test_box_iou_bounds(self) -> None:
        """IoU 在不相交时为 0，在完全重合时为 1。"""
        self.assertEqual(box_iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 5.0, 5.0)), 0.0)
        self.assertEqual(box_iou((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0)), 1.0)

    def test_perfect_detection_gives_unit_ap(self) -> None:
        """完全重合的单条预测在全部 IoU 阈值上都算命中。"""
        truth = perfect_truth((0.0, 0.0, 100.0, 100.0))
        predictions = {
            "img1": (
                Prediction(category_id=1, score=0.9, box=(0.0, 0.0, 100.0, 100.0)),
            )
        }
        result = evaluate_detections(truth, predictions, CATEGORY_NAMES, 0.5)
        self.assertEqual(result.ap, 1.0)
        self.assertEqual(result.ap50, 1.0)
        self.assertEqual(result.ap75, 1.0)
        self.assertEqual(result.ar_100, 1.0)
        self.assertEqual(result.operating_point.precision, 1.0)
        self.assertEqual(result.operating_point.recall, 1.0)
        self.assertEqual(result.operating_point.f1, 1.0)
        self.assertEqual(result.categories_with_truth, 1)
        self.assertEqual(result.categories_without_truth, (2,))

    def test_missing_predictions_give_zero_not_none(self) -> None:
        """有真值但无预测时是「全错」，应当为 0 而不是未测量。"""
        result = evaluate_detections(
            perfect_truth((0.0, 0.0, 40.0, 40.0)), {}, CATEGORY_NAMES, 0.5
        )
        self.assertEqual(result.ap, 0.0)
        self.assertEqual(result.ap50, 0.0)
        by_category = {item.category_id: item for item in result.per_category}
        self.assertEqual(by_category[1].ap, 0.0)
        self.assertIsNone(by_category[2].ap)

    def test_iou_threshold_boundary_is_inclusive(self) -> None:
        """IoU 恰好等于阈值算命中，低于阈值则只在 AP50 上失败。"""
        truth = perfect_truth((0.0, 0.0, 10.0, 10.0))
        boundary = {
            "img1": (Prediction(category_id=1, score=0.9, box=(0.0, 0.0, 10.0, 5.0)),)
        }
        self.assertEqual(box_iou((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 5.0)), 0.5)
        result = evaluate_detections(truth, boundary, CATEGORY_NAMES, 0.5)
        self.assertEqual(result.ap50, 1.0)
        self.assertAlmostEqual(result.ap, 0.1, places=12)
        below = {
            "img1": (Prediction(category_id=1, score=0.9, box=(0.0, 0.0, 10.0, 4.9)),)
        }
        result = evaluate_detections(truth, below, CATEGORY_NAMES, 0.5)
        self.assertEqual(result.ap50, 0.0)

    def test_area_ranges_split_by_truth_area(self) -> None:
        """面积区间按真值框面积划分：小 100、中 2000、大 20000。"""
        truth = {
            "img1": (
                TruthBox(category_id=1, box=(0.0, 0.0, 10.0, 10.0)),
                TruthBox(category_id=1, box=(100.0, 0.0, 40.0, 50.0)),
                TruthBox(category_id=1, box=(300.0, 0.0, 100.0, 200.0)),
            )
        }
        predictions = {
            "img1": (
                Prediction(category_id=1, score=0.9, box=(0.0, 0.0, 10.0, 10.0)),
                Prediction(category_id=1, score=0.8, box=(100.0, 0.0, 40.0, 50.0)),
                Prediction(category_id=1, score=0.7, box=(300.0, 0.0, 100.0, 200.0)),
            )
        }
        result = evaluate_detections(truth, predictions, CATEGORY_NAMES, 0.5)
        self.assertEqual(result.ap_small, 1.0)
        self.assertEqual(result.ap_medium, 1.0)
        self.assertEqual(result.ap_large, 1.0)
        only_small = {"img1": (predictions["img1"][0],)}
        result = evaluate_detections(truth, only_small, CATEGORY_NAMES, 0.5)
        self.assertEqual(result.ap_small, 1.0)
        self.assertEqual(result.ap_medium, 0.0)
        self.assertEqual(result.ap_large, 0.0)

    def test_max_detections_limits_recall(self) -> None:
        """AR@1 只看最高分的一条预测，AR@10 可以看到后面的命中。"""
        truth = {
            "img1": (
                TruthBox(category_id=1, box=(0.0, 0.0, 10.0, 10.0)),
                TruthBox(category_id=1, box=(50.0, 50.0, 10.0, 10.0)),
            )
        }
        predictions = {
            "img1": (
                Prediction(category_id=1, score=0.9, box=(0.0, 0.0, 10.0, 10.0)),
                Prediction(category_id=1, score=0.8, box=(200.0, 200.0, 10.0, 10.0)),
                Prediction(category_id=1, score=0.7, box=(50.0, 50.0, 10.0, 10.0)),
            )
        }
        result = evaluate_detections(truth, predictions, CATEGORY_NAMES, 0.5)
        self.assertEqual(result.ar_1, 0.5)
        self.assertEqual(result.ar_10, 1.0)
        self.assertEqual(result.operating_point.true_positives, 2)
        self.assertEqual(result.operating_point.false_positives, 1)

    def test_crowd_truth_is_ignored_not_counted(self) -> None:
        """crowd 真值只用于忽略重叠预测，不进入召回分母、也不算误检。"""
        truth = {
            "img1": (
                TruthBox(category_id=1, box=(0.0, 0.0, 30.0, 30.0), is_crowd=True),
                TruthBox(category_id=1, box=(100.0, 100.0, 20.0, 20.0)),
            )
        }
        predictions = {
            "img1": (
                Prediction(category_id=1, score=0.9, box=(0.0, 0.0, 30.0, 30.0)),
                Prediction(category_id=1, score=0.8, box=(100.0, 100.0, 20.0, 20.0)),
            )
        }
        result = evaluate_detections(truth, predictions, CATEGORY_NAMES, 0.5)
        self.assertEqual(result.truth_instances, 1)
        self.assertEqual(result.ap, 1.0)
        self.assertEqual(result.operating_point.true_positives, 1)
        self.assertEqual(result.operating_point.false_positives, 0)

    def test_class_mismatch_is_false_positive(self) -> None:
        """框重合但类别不同时既不算命中，也不占用真值。"""
        truth = perfect_truth((0.0, 0.0, 10.0, 10.0))
        predictions = {
            "img1": (Prediction(category_id=2, score=0.9, box=(0.0, 0.0, 10.0, 10.0)),)
        }
        result = evaluate_detections(truth, predictions, CATEGORY_NAMES, 0.5)
        self.assertEqual(result.per_category[0].ap, 0.0)
        self.assertEqual(result.operating_point.false_positives, 1)
        self.assertEqual(result.operating_point.true_positives, 0)

    def test_duplicate_prediction_counts_one_hit(self) -> None:
        """同一真值只被一个预测占用，重复的高分预测算误检。"""
        truth = perfect_truth((0.0, 0.0, 10.0, 10.0))
        predictions = {
            "img1": (
                Prediction(category_id=1, score=0.9, box=(0.0, 0.0, 10.0, 10.0)),
                Prediction(category_id=1, score=0.8, box=(0.0, 0.0, 10.0, 10.0)),
            )
        }
        result = evaluate_detections(truth, predictions, CATEGORY_NAMES, 0.5)
        self.assertEqual(result.operating_point.true_positives, 1)
        self.assertEqual(result.operating_point.false_positives, 1)
        self.assertEqual(result.ar_1, 1.0)
        self.assertEqual(result.ar_100, 1.0)

    def test_operating_confidence_filters_predictions(self) -> None:
        """低于工作点分数的预测被忽略，召回为 0 且精确率未测量。"""
        truth = perfect_truth((0.0, 0.0, 10.0, 10.0))
        predictions = {
            "img1": (Prediction(category_id=1, score=0.4, box=(0.0, 0.0, 10.0, 10.0)),)
        }
        point = operating_point(truth, predictions, 0.5)
        self.assertEqual(point.true_positives, 0)
        self.assertEqual(point.false_negatives, 1)
        self.assertIsNone(point.precision)
        self.assertEqual(point.recall, 0.0)
        self.assertIsNone(point.f1)

    def test_invalid_inputs_are_rejected(self) -> None:
        """零面积框、越界分数与非法工作点分数都必须报错。"""
        with self.assertRaises(ValueError):
            TruthBox(category_id=1, box=(0.0, 0.0, 0.0, 10.0))
        with self.assertRaises(ValueError):
            Prediction(category_id=1, score=1.5, box=(0.0, 0.0, 10.0, 10.0))
        with self.assertRaises(ValueError):
            evaluate_detections({}, {}, CATEGORY_NAMES, 1.5)


if __name__ == "__main__":
    unittest.main()
