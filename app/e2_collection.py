"""正式板面采集协议与分组校验；不访问设备或修改已有样本。"""

import hashlib
import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class Target:
    target_id: str
    material: str
    width_m: float
    height_m: float

    def __post_init__(self) -> None:
        """要求真实目标描述及正有限米制尺寸；不凭编号推断材质。"""
        if not self.target_id.strip() or not self.material.strip():
            raise ValueError("target id and material are required")
        if any(not isfinite(x) or x <= 0 for x in (self.width_m, self.height_m)):
            raise ValueError("target dimensions must be positive finite meters")


@dataclass(frozen=True)
class Collection:
    collection_id: str
    plan_sha256: str
    split: str
    group_id: str
    installation_id: str
    placement_id: str
    target: Target
    roi_policy: str = "board_lower_half_v1"


def read_object(path: Path) -> dict[str, object]:
    """收窄 JSON 顶层对象；拒绝不符合协议的输入。"""
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise ValueError("expected JSON object")
    return dict(raw)


def prepare_collection(
    plan_path: Path,
    group_id: str,
    installation_id: str,
    placement_id: str,
    target: Target,
    records_root: Path,
) -> Collection:
    """按预注册布景取得划分，冻结计划与目标信息；冲突或重复摆放在开相机前拒绝。"""
    plan = read_object(plan_path)
    groups = plan.get("groups")
    if (
        plan.get("schema") != "e2_collection_plan_v1"
        or plan.get("roi_policy") != "board_lower_half_v1"
        or not isinstance(groups, dict)
        or not groups
        or any(
            not isinstance(k, str) or v not in ("fit", "validation", "test")
            for k, v in groups.items()
        )
    ):
        raise ValueError("invalid collection plan")
    identifier = plan.get("collection_id")
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("collection id required")
    if plan.get("installation_id") != installation_id:
        raise ValueError("installation does not match collection plan")
    if group_id not in groups or not placement_id.strip():
        raise ValueError("unknown group or empty placement id")
    split = groups[group_id]
    if split == "test":
        raise ValueError(
            "test collection locked until model selection protocol is frozen"
        )
    if not isinstance(split, str):
        raise ValueError("invalid split")
    context = Collection(
        identifier,
        hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        split,
        group_id,
        installation_id,
        placement_id,
        target,
    )
    for path in records_root.glob("e2_*/record.json"):
        record = read_object(path)
        prior = record.get("collection")
        if not isinstance(prior, dict):
            measurement = record.get("measurement")
            if (
                isinstance(measurement, dict)
                and measurement.get("group_id") == group_id
            ):
                raise ValueError("group already used by development precapture")
            continue
        if prior.get("collection_id") != identifier:
            if prior.get("group_id") == group_id:
                raise ValueError("group already used by another collection")
            continue
        if prior.get("plan_sha256") != context.plan_sha256:
            raise ValueError("plan changed after collection started")
        if (
            prior.get("group_id") == group_id
            and prior.get("placement_id") == placement_id
        ):
            raise ValueError(
                "placement already recorded; use a new independent placement"
            )
        old_target = prior.get("target")
        if (
            isinstance(old_target, dict)
            and old_target.get("target_id") == target.target_id
        ):
            if old_target != asdict(target):
                raise ValueError("target metadata changed; use a new target id")
    return context
