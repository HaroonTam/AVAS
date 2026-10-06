"""受限中文类别名称；仅用于查询别名和回答展示，不修改模型类别。"""

LABEL_NAMES: tuple[tuple[str, str], ...] = (
    ("chair", "椅子"),
    ("person", "人"),
    ("bicycle", "自行车"),
    ("car", "汽车"),
    ("bus", "公交车"),
    ("motorcycle", "摩托车"),
)


def format_label(label: str) -> str:
    """精确映射已配置类别为中文；未知名称按原文转义并加引号，不猜测翻译。"""
    return repr(dict(LABEL_NAMES).get(label, label))
