"""在原图网格融合相对逆深度，不生成米制距离或风险决策。"""

from dataclasses import dataclass
from math import ceil, floor

import numpy as np

from app.fusion.config import FusionConfig
from app.fusion.direction import Direction, estimate_direction
from app.vision.depth_estimator import RelativeDepth
from app.vision.detector import Detection


@dataclass(frozen=True)
class DetectionFrame:
    """检测所属原图身份；调用方须保证无未逆转的裁剪、缩放或填充。"""

    frame_id: str
    width: int
    height: int
    detections: tuple[Detection, ...]

    def __post_init__(self) -> None:
        """拒绝空帧身份、无效尺寸和超出原图的目标框。"""
        if not self.frame_id.strip() or self.width <= 0 or self.height <= 0:
            raise ValueError("invalid detection frame identity or dimensions")
        for detection in self.detections:
            if detection.bbox[2] > self.width or detection.bbox[3] > self.height:
                raise ValueError("detection box exceeds original-image bounds")


@dataclass(frozen=True)
class DepthSummary:
    """有效比例与离散度仅描述采样质量，不代表测距准确率。"""

    status: str
    method: str
    roi: tuple[int, int, int, int]
    sample_count: int
    valid_count: int
    retained_count: int
    valid_fraction: float
    relative_depth: float | None
    relative_depth_iqr: float | None
    units: str = "relative_inverse_depth"
    calibration_status: str = "not_calibrated"


@dataclass(frozen=True)
class Observation:
    id: int
    frame_id: str
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    direction: Direction | None
    direction_status: str
    depth: DepthSummary
    distance_m: None = None
    distance_status: str = "metric_unavailable"
    risk_level: str = "unknown"


def sampling_region(
    bbox: tuple[int, int, int, int], config: FusionConfig
) -> tuple[int, int, int, int]:
    """返回半开区间采样框；中心区域向内取整，过小区域可为空。"""
    x1, y1, x2, y2 = bbox
    if config.method == "center_pixel":
        x, y = (x1 + x2) // 2, (y1 + y2) // 2
        return x, y, x + 1, y + 1
    if config.method != "center_region_median":
        return bbox
    margin_x = (x2 - x1) * (1 - config.inner_fraction) / 2
    margin_y = (y2 - y1) * (1 - config.inner_fraction) / 2
    return (
        ceil(x1 + margin_x),
        ceil(y1 + margin_y),
        floor(x2 - margin_x),
        floor(y2 - margin_y),
    )


def summarize_depth(
    bbox: tuple[int, int, int, int], depth: RelativeDepth | None, config: FusionConfig
) -> DepthSummary:
    """过滤掩膜、非有限值、负值与分位数极端值，像素不足时返回未知。"""
    roi = sampling_region(bbox, config)
    x1, y1, x2, y2 = roi
    count = max(0, x2 - x1) * max(0, y2 - y1)
    status = "depth_unavailable" if depth is None else "empty_region"
    valid_count = retained_count = 0
    fraction = 0.0
    value = spread = None
    if depth is not None and count:
        region = depth.relative_depth[y1:y2, x1:x2]
        mask = depth.valid_mask[y1:y2, x1:x2]
        values = region[mask & np.isfinite(region) & (region >= 0)].astype(np.float64)
        valid_count = int(values.size)
        fraction = valid_count / count
        status = "insufficient_valid_pixels"
        if valid_count:
            low, high = np.quantile(
                values, [config.trim_fraction, 1 - config.trim_fraction]
            )
            retained = values[(values >= low) & (values <= high)]
            retained_count = int(retained.size)
            minimum = 1 if config.method == "center_pixel" else config.min_valid_pixels
            if retained_count >= minimum and fraction >= config.min_valid_fraction:
                status = "available"
                value = float(
                    np.mean(retained)
                    if config.method == "box_mean"
                    else np.median(retained)
                )
                q1, q3 = np.quantile(retained, [0.25, 0.75])
                spread = float(q3 - q1)
    return DepthSummary(
        status,
        config.method,
        roi,
        count,
        valid_count,
        retained_count,
        fraction,
        value,
        spread,
    )


def fuse_frame(
    frame: DetectionFrame, depth: RelativeDepth | None, config: FusionConfig
) -> tuple[Observation, ...]:
    """拒绝错帧、错尺寸或掩膜格式错误；深度缺失时仍保留目标与方向。"""
    if depth is not None:
        if depth.frame_id != frame.frame_id:
            raise ValueError("depth and detections belong to different frames")
        shape = (frame.height, frame.width)
        if depth.relative_depth.shape != shape or depth.valid_mask.shape != shape:
            raise ValueError("depth and mask must match original-image dimensions")
        if (
            not np.issubdtype(depth.relative_depth.dtype, np.floating)
            or depth.valid_mask.dtype != np.bool_
        ):
            raise ValueError("depth must be floating point with a boolean mask")
    observations: list[Observation] = []
    for identifier, detection in enumerate(frame.detections, start=1):
        direction = estimate_direction(detection.bbox, frame.width, config)
        observations.append(
            Observation(
                identifier,
                frame.frame_id,
                detection.label,
                detection.confidence,
                detection.bbox,
                direction,
                "image_relative"
                if direction is not None
                else "camera_orientation_unknown",
                summarize_depth(detection.bbox, depth, config),
            )
        )
    return tuple(observations)
