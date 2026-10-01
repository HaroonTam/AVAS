"""图像水平分区，不表示经过标定的世界坐标。"""

from typing import Literal

from app.fusion.config import FusionConfig

Direction = Literal["left", "front", "right"]


def estimate_direction(
    bbox: tuple[int, int, int, int], width: int, config: FusionConfig
) -> Direction | None:
    """按框中心分区；镜像先反转横坐标，非前向安装返回未知。"""
    x1, y1, x2, y2 = bbox
    if width <= 0 or not 0 <= x1 < x2 <= width or not 0 <= y1 < y2:
        raise ValueError("invalid original-image box or width")
    if config.camera_orientation != "forward":
        return None
    center_sum = 2 * width - x1 - x2 if config.mirrored else x1 + x2
    position = center_sum / (2 * width)
    if position < config.left_boundary:
        return "left"
    if position > config.right_boundary:
        return "right"
    return "front"
