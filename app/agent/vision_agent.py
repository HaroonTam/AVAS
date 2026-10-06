"""本地受限文字交互：结构化请求、只读查询及输出前事实复核。"""

from dataclasses import dataclass
from typing import Literal

from app.agent.tools import ObjectFact, SceneTools, ToolResult
from app.fusion.direction import Direction


@dataclass(frozen=True)
class SceneRequest:
    intent: Literal["describe", "risks", "find", "distance"]
    label: str | None = None
    frame_id: str | None = None
    object_id: int | None = None
    direction: Direction | None = None

    def __post_init__(self) -> None:
        """拒绝未知意图和多余参数，避免含糊请求被悄然解释。"""
        if self.intent not in {"describe", "risks", "find", "distance"}:
            raise ValueError("unsupported intent")
        if self.direction is not None and (
            self.intent != "find" or self.direction not in ("left", "front", "right")
        ):
            raise ValueError("only find accepts direction: left, front or right")
        if self.intent == "find":
            if (
                not isinstance(self.label, str)
                or not self.label.strip()
                or len(self.label) > 128
                or self.frame_id is not None
                or self.object_id is not None
            ):
                raise ValueError(
                    "find requires a nonempty label and optional direction"
                )
        elif self.intent == "distance":
            if (
                self.label is not None
                or not isinstance(self.frame_id, str)
                or not self.frame_id.strip()
                or type(self.object_id) is not int
                or self.object_id < 1
            ):
                raise ValueError("distance requires frame_id and positive object_id")
        elif any(
            value is not None for value in (self.label, self.frame_id, self.object_id)
        ):
            raise ValueError("this intent takes no arguments")


@dataclass(frozen=True)
class AnswerDraft:
    request: SceneRequest
    evidence: ToolResult


@dataclass(frozen=True)
class AssistantAnswer:
    status: str
    frame_id: str | None
    message: str


def parse_request(text: str) -> SceneRequest | None:
    """解析有限命令及明确的中文问法；不执行文本中的指令或猜测任意意图。"""
    if not isinstance(text, str) or not text.strip() or len(text) > 256:
        return None
    text = text.strip().rstrip("？?")
    if text in {"describe", "描述周围", "周围有什么"}:
        return SceneRequest("describe")
    if text in {"risks", "当前风险", "有什么危险"}:
        return SceneRequest("risks")
    aliases = {
        "椅子": "chair",
        "人": "person",
        "自行车": "bicycle",
        "汽车": "car",
        "公交车": "bus",
        "摩托车": "motorcycle",
    }
    directions: dict[str, Direction] = {
        "左侧的": "left",
        "前方的": "front",
        "右侧的": "right",
    }
    for prefix, direction in directions.items():
        for suffix in ("在哪里", "有多远"):
            if text.startswith(prefix) and text.endswith(suffix):
                label = text[len(prefix) : -len(suffix)]
                if label in aliases:
                    return SceneRequest(
                        "find", label=aliases[label], direction=direction
                    )
    for suffix in ("在哪里", "有多远"):
        if text.endswith(suffix) and text[: -len(suffix)] in aliases:
            return SceneRequest("find", label=aliases[text[: -len(suffix)]])
    if text.startswith("find ") or text.startswith("寻找 "):
        label = text.split(" ", 1)[1].strip()
        if label and len(label) <= 128:
            return SceneRequest("find", label=aliases.get(label, label))
    parts = text.split()
    if len(parts) == 3 and parts[0] in {"distance", "距离"}:
        try:
            identifier = int(parts[2])
        except ValueError:
            return None
        if identifier > 0:
            return SceneRequest("distance", frame_id=parts[1], object_id=identifier)
    return None


def describe_fact(item: ObjectFact) -> str:
    """仅按已验证事实格式化单个目标，方向与米制距离缺失时分别说明。"""
    direction = {
        "left": "图像左侧",
        "front": "图像前方",
        "right": "图像右侧",
        None: "方向未知",
    }[item.direction]
    distance = (
        f"约 {item.distance_m:.1f} 米"
        if item.distance_m is not None
        else "米制距离不可用"
    )
    return f"目标 {item.id}（类别 {item.label!r}）：{direction}，{distance}。"


class VisionAssistant:
    def __init__(self, tools: SceneTools) -> None:
        """只依赖场景查询工具，不持有告警、语音、相机或规则修改接口。"""
        self._tools = tools

    def _query(self, request: SceneRequest) -> ToolResult:
        """按已验证的结构化请求分派工具；不接受自由工具名称。"""
        if request.intent == "describe":
            return self._tools.describe_surroundings()
        if request.intent == "risks":
            return self._tools.get_current_risks()
        if request.intent == "find" and request.label is not None:
            return self._tools.find_object(request.label, request.direction)
        if request.frame_id is not None and request.object_id is not None:
            return self._tools.get_object_distance(request.frame_id, request.object_id)
        raise ValueError("invalid request")

    def prepare(self, request: SceneRequest) -> AnswerDraft:
        """保存结构化依据，不接收或生成未经验证的自由回答文本。"""
        return AnswerDraft(request, self._query(request))

    def finalize(self, draft: AnswerDraft) -> AssistantAnswer:
        """输出前重新查询；帧、对象或风险改变时拒绝旧回答，重新提问后再选择。"""
        current = self._query(draft.request)
        if current != draft.evidence:
            return AssistantAnswer(
                "answer_expired", None, "场景已变化或信息已失效，请重新查询。"
            )
        if current.frame_id is None:
            return AssistantAnswer(current.status, None, current.message)
        if draft.request.intent == "describe":
            message = current.message
        else:
            warnings = list(dict.fromkeys(event.message for event in current.events))
            if draft.request.intent == "risks":
                if not warnings:
                    warnings.append("当前配置规则未产生告警，不代表道路安全。")
                message = "".join(warnings)
            elif current.status in {"not_found", "ambiguous"}:
                message = "".join(warnings) + current.message
                if current.status == "ambiguous":
                    message += f"帧 ID：{current.frame_id}；目标 ID："
                    message += (
                        "、".join(str(item.id) for item in current.objects) + "。"
                    )
            else:
                message = "".join(warnings)
                message += "".join(describe_fact(item) for item in current.objects)
        return AssistantAnswer(current.status, current.frame_id, message)

    def respond(self, text: str) -> AssistantAnswer:
        """同步本地文字入口；不理解时提示有限语法，工具失败时不编造事实。"""
        request = parse_request(text)
        if request is None:
            return AssistantAnswer(
                "unsupported_request",
                None,
                "请使用“描述周围”“当前风险”“寻找 椅子”"
                "“左侧的椅子在哪里”或“距离 帧ID 目标ID”。",
            )
        try:
            return self.finalize(self.prepare(request))
        except (ValueError, RuntimeError, TypeError, AttributeError):
            return AssistantAnswer("tools_unavailable", None, "场景查询暂时不可用。")
