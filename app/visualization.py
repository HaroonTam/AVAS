"""离线检测结果绘制；不参与检测、距离估计或风险判断。"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from app.vision.depth_estimator import RelativeDepth
from app.vision.detector import Detection, Image


def render_relative_depth(result: RelativeDepth) -> Image:
    """生成单图归一化灰度预览，亮表示相对近，紫色表示无效像素。"""
    values = result.relative_depth
    valid = result.valid_mask
    if values.ndim != 2 or valid.shape != values.shape or not valid.any():
        raise ValueError("invalid relative depth or mask")
    minimum = float(values[valid].min())
    maximum = float(values[valid].max())
    gray = np.zeros(values.shape, dtype=np.uint8)
    if maximum > minimum:
        gray[valid] = np.clip(
            (values[valid] - minimum) / (maximum - minimum) * 255, 0, 255
        ).astype(np.uint8)
    else:
        gray[valid] = 127
    preview = np.repeat(gray[:, :, None], 3, axis=2)
    preview[~valid] = (255, 0, 255)
    return preview


def draw_detections(image: Image, detections: Sequence[Detection]) -> Image:
    """在原图副本上绘制半开坐标框和检测分数，不修改输入图像。"""
    import cv2

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected uint8 HxWx3 BGR image")
    height, width = image.shape[:2]
    if not height or not width:
        raise ValueError("image must not be empty")
    result = image.copy()
    occupied: list[tuple[int, int, int, int]] = []
    for item in detections:
        x1, y1, x2, y2 = item.bbox
        if x2 > width or y2 > height:
            raise ValueError("detection box exceeds image bounds")
        # 蓝色仅用于标记检测框，不代表任何风险等级。
        color = (255, 180, 0)
        cv2.rectangle(result, (x1, y1), (x2 - 1, y2 - 1), color, 2)
        label = f"{item.label} {item.confidence:.2f}"
        scale = 0.5
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
        )
        left = max(0, min(x1, width - text_width - 6))
        top = max(0, y1 - text_height - baseline - 6)
        label_height = text_height + baseline + 6
        # 优先向上错开标签；位置用尽时仍保留检测框，不改变任何检测数据。
        while top > 0 and any(
            left < right
            and left + text_width + 6 > other_left
            and top < bottom
            and top + label_height > other_top
            for other_left, other_top, right, bottom in occupied
        ):
            top = max(0, top - label_height - 2)
        occupied.append((left, top, left + text_width + 6, top + label_height))
        cv2.rectangle(
            result,
            (left, top),
            (
                min(width - 1, left + text_width + 6),
                min(height - 1, top + text_height + baseline + 6),
            ),
            color,
            -1,
        )
        cv2.putText(
            result,
            label,
            (left + 3, top + text_height + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return result


def save_detections(
    image: Image, detections: Sequence[Detection], destination: Path
) -> Path:
    """显式保存 PNG/JPEG 检测图；支持中文路径并拒绝覆盖已有文件。"""
    import cv2

    destination = destination.resolve()
    suffix = destination.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("output image must use .png, .jpg, or .jpeg")
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    rendered = draw_detections(image, detections)
    try:
        success, encoded = cv2.imencode(suffix, rendered)
    except cv2.error as exc:
        raise RuntimeError("Failed to encode visualization") from exc
    if not success:
        raise RuntimeError("Failed to encode visualization")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        output.write(encoded.tobytes())
    return destination
