"""严格合并同协议、同身份的离线多图报告；不加载模型或启动硬件。"""

import argparse
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from heapq import nsmallest
from math import isclose, isfinite
from pathlib import Path
from statistics import stdev

from app.benchmark import (
    TimingSample,
    file_sha256,
    save_report,
    summarize,
    summarize_samples,
)


@dataclass(frozen=True)
class ValidatedRun:
    identity: dict[str, object]
    images: tuple[dict[str, object], ...]
    samples: tuple[tuple[TimingSample, ...], ...]
    source: dict[str, object]


def mapping(value: object) -> dict[str, object]:
    """将 JSON 对象边界收窄为字符串键字典；非对象拒绝。"""
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError("expected JSON object")
    return dict(value)


def integer(value: object, minimum: int, maximum: int) -> int:
    """验证有界整数并拒绝布尔值，以避免错误的样本数或尺寸。"""
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("invalid integer metadata")
    return value


def milliseconds(value: object) -> float:
    """解析有限非负毫秒数；拒绝字符串、布尔值及非有限数。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("invalid timing value")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("timing value too large") from exc
    if not isfinite(result) or result < 0:
        raise ValueError("invalid timing value")
    return result


def text(value: object) -> str:
    """验证非空文本元数据；不自动将缺失值转成字符串。"""
    if not isinstance(value, str) or not value:
        raise ValueError("missing text metadata")
    return value


def sha256(value: object) -> str:
    """验证报告中的 SHA256 格式，而非证明其内容真实性。"""
    result = text(value)
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise ValueError("invalid SHA256 metadata")
    return result


def reject_constant(value: str) -> None:
    """拒绝 JSON 扩展的 NaN 与 Infinity，避免非法元数据参与比较。"""
    raise ValueError(f"nonfinite JSON constant: {value}")


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """拒绝重复 JSON 键，防止同一字段存在歧义。"""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def parse_sample(value: object, depth_enabled: bool) -> TimingSample:
    """验证阶段耗时、目标计数与总耗时关系；不接受混合深度模式。"""
    item = mapping(value)
    depth = item["depth_ms"]
    if (depth is not None) != depth_enabled:
        raise ValueError("depth mode differs from config")
    sample = TimingSample(
        milliseconds(item["detection_ms"]),
        milliseconds(depth) if depth_enabled else None,
        milliseconds(item["fusion_ms"]),
        milliseconds(item["offline_validation_ms"]),
        milliseconds(item["total_ms"]),
        integer(item["object_count"], 0, 1_000_000),
    )
    stages = (
        sample.detection_ms
        + (sample.depth_ms or 0)
        + sample.fusion_ms
        + sample.offline_validation_ms
    )
    if sample.total_ms + 1e-6 < stages or (
        depth_enabled
        and not isclose(sample.total_ms, stages, rel_tol=1e-8, abs_tol=1e-6)
    ):
        raise ValueError("total timing is inconsistent with stages")
    return sample


def load_run(path: Path) -> ValidatedRun:
    """读取并校验完整多图报告；只接受已知干净修订，不信任预计算汇总。"""
    payload = path.read_bytes()
    raw: object = json.loads(
        payload, parse_constant=reject_constant, object_pairs_hook=unique_object
    )
    report = mapping(raw)
    expected: dict[str, object] = {
        "schema_version": 2,
        "protocol": "offline_multi_image_v1",
        "current_scene": False,
        "sampling_order": "image_major_argument_order",
        "warmup_scope": "per_image_before_its_samples",
        "iterations_scope": "per_image",
        "aggregation": "pooled_samples_equal_weight_per_image",
    }
    for key, value in expected.items():
        if type(report[key]) is not type(value) or report[key] != value:
            raise ValueError(f"unsupported protocol field: {key}")
    iterations = integer(report["iterations"], 1, 1000)
    count = integer(report["image_count"], 2, 100)
    integer(report["warmup"], 0, 100)
    if integer(report["sample_count"], 1, 100_000) != count * iterations:
        raise ValueError("sample count mismatch")
    code = mapping(report["code"])
    if (
        code["code_dirty"] is not False
        or re.fullmatch(r"[0-9a-f]{40}", text(code["revision"])) is None
    ):
        raise ValueError("requires a known clean code revision")
    for hashes in (code["app_sha256"], report["weight_sha256"]):
        entries = mapping(hashes)
        if not entries:
            raise ValueError("missing identity hashes")
        for digest in entries.values():
            sha256(digest)
    configs = mapping(report["configs"])
    for key in ("detection", "fusion", "risk"):
        if not mapping(configs[key]):
            raise ValueError("missing configuration")
    depth_enabled = configs["depth"] is not None
    if depth_enabled and not mapping(configs["depth"]):
        raise ValueError("missing depth configuration")
    sha256(mapping(code["app_sha256"])["app/benchmark.py"])
    weights = mapping(report["weight_sha256"])
    sha256(weights["yolo11"])
    if depth_enabled:
        for name in ("model.safetensors", "config.json", "preprocessor_config.json"):
            sha256(weights[f"depth/{name}"])
    devices = mapping(report["selected_devices"])
    for key in ("detection", "depth"):
        if key == "depth" and not depth_enabled:
            if devices[key] is not None:
                raise ValueError("disabled depth must not have a device")
        elif text(devices[key]) not in ("cpu", "cuda", "mps"):
            raise ValueError("invalid device metadata")
    if "cuda" in devices.values():
        text(report["accelerator_name"])
    versions = mapping(report["versions"])
    for name in ("python", "numpy", "torch", "ultralytics", "opencv-python"):
        text(versions[name])
    if depth_enabled:
        text(versions["transformers"])
    for key in ("platform", "processor"):
        text(report[key])
    integer(report["torch_threads"], 1, 100_000)
    integer(report["seed"], 0, 2**63 - 1)
    started = text(report["measurement_started_at_utc"])
    timestamp = datetime.fromisoformat(started)
    if timestamp.utcoffset() is None:
        raise ValueError("measurement timestamp requires timezone")
    started = timestamp.astimezone(timezone.utc).isoformat()
    identity = {
        key: report[key]
        for key in (
            *expected,
            "iterations",
            "warmup",
            "seed",
            "configs",
            "selected_devices",
            "versions",
            "code",
            "platform",
            "processor",
            "torch_threads",
            "accelerator_name",
            "weight_sha256",
            "image_count",
            "sample_count",
        )
    }
    images = report["images"]
    if not isinstance(images, list) or len(images) != count:
        raise ValueError("image count mismatch")
    image_ids: list[dict[str, object]] = []
    groups: list[tuple[TimingSample, ...]] = []
    for index, value in enumerate(images):
        item = mapping(value)
        if integer(item["image_index"], 0, 99) != index:
            raise ValueError("image order mismatch")
        image_ids.append(
            {
                "image_index": index,
                "image_name": text(item["image_name"]),
                "image_file_sha256": sha256(item["image_file_sha256"]),
                "frame_id": sha256(item["frame_id"]),
                "width": integer(item["width"], 1, 1_000_000),
                "height": integer(item["height"], 1, 1_000_000),
            }
        )
        samples = item["samples"]
        if not isinstance(samples, list) or len(samples) != iterations:
            raise ValueError("per-image sample count mismatch")
        groups.append(tuple(parse_sample(sample, depth_enabled) for sample in samples))
    return ValidatedRun(
        identity,
        tuple(image_ids),
        tuple(groups),
        {
            "name": path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "measurement_started_at_utc": started,
        },
    )


def statistics_for_runs(
    runs: tuple[tuple[TimingSample, ...], ...],
) -> dict[str, object]:
    """分别汇总合并样本及每轮均值；样本标准差使用轮均值与 n-1 分母。"""
    summaries = tuple(summarize_samples(samples) for samples in runs)
    pooled = summarize_samples(tuple(sample for samples in runs for sample in samples))
    between: dict[str, object] = {}
    for stage in pooled:
        means = tuple(summary[stage].mean_ms for summary in summaries)
        between[stage] = {**asdict(summarize(means)), "sample_stddev_ms": stdev(means)}
    return {
        "sample_count": sum(len(samples) for samples in runs),
        "pooled_summary_ms": {key: asdict(value) for key, value in pooled.items()},
        "per_run_summary_ms": [
            {key: asdict(value) for key, value in summary.items()}
            for summary in summaries
        ],
        "run_mean_summary_ms": between,
    }


def sample_timings(sample: TimingSample) -> dict[str, float | None]:
    """提取毫秒阶段值；深度禁用保留 None，不把目标数量混入耗时。"""
    return {
        "detection": sample.detection_ms,
        "depth": sample.depth_ms,
        "fusion": sample.fusion_ms,
        "offline_validation": sample.offline_validation_ms,
        "total": sample.total_ms,
    }


def slowest_samples(runs: tuple[ValidatedRun, ...]) -> dict[str, object]:
    """定位总耗时最高的十个正式样本；用同轮同图阶段中位数作描述性参照。"""
    locations = nsmallest(
        10,
        (
            (-sample.total_ms, run_index, image_index, sample_index)
            for run_index, run in enumerate(runs)
            for image_index, group in enumerate(run.samples)
            for sample_index, sample in enumerate(group)
        ),
    )
    records: list[dict[str, object]] = []
    for _, run_index, image_index, sample_index in locations:
        run = runs[run_index]
        group = run.samples[image_index]
        actual = sample_timings(group[sample_index])
        summary = summarize_samples(group)
        baseline = {
            stage: summary[stage].median_ms if stage in summary else None
            for stage in actual
        }
        deltas: dict[str, float | None] = {}
        for stage, value in actual.items():
            reference = baseline[stage]
            deltas[stage] = (
                value - reference
                if value is not None and reference is not None
                else None
            )
        records.append(
            {
                "run_index": run_index,
                "image_index": image_index,
                "sample_index": sample_index,
                "image_name": run.images[image_index]["image_name"],
                "timings_ms": actual,
                "same_run_image_median_ms": baseline,
                "delta_from_median_ms": deltas,
            }
        )
    return {
        "limit": 10,
        "order": "total_descending_then_run_image_sample_index_ascending",
        "indices": "zero_based_sources_images_and_formal_samples",
        "baseline": "same_run_same_image_all_formal_samples_including_selected",
        "interpretation": "stage_medians_are_not_additive_no_causal_attribution",
        "samples": records,
    }


def aggregate_reports(paths: tuple[Path, ...]) -> dict[str, object]:
    """合并可比报告并定位最慢样本；保留来源哈希及逐图、逐轮归属。"""
    if not 2 <= len(paths) <= 20:
        raise ValueError("provide 2..20 reports")
    runs = tuple(load_run(path) for path in paths)
    first = runs[0]
    for run in runs[1:]:
        if run.identity != first.identity or run.images != first.images:
            raise ValueError(
                "reports differ in protocol, code, runtime or image identity"
            )
    for key in ("sha256", "measurement_started_at_utc"):
        if len({text(run.source[key]) for run in runs}) != len(runs):
            raise ValueError("duplicate report or measurement start time")
    return {
        "schema_version": 1,
        "protocol": "offline_multi_run_summary_v1",
        "current_scene": False,
        "run_count": len(runs),
        "comparison_identity": first.identity,
        "sources": [run.source for run in runs],
        "summarizer_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in ("benchmark_summary.py", "benchmark.py")
        },
        "aggregation": "equal_samples_per_image_and_run",
        "slowest_samples": slowest_samples(runs),
        **statistics_for_runs(
            tuple(
                tuple(sample for group in run.samples for sample in group)
                for run in runs
            )
        ),
        "images": [
            {**image, **statistics_for_runs(tuple(run.samples[i] for run in runs))}
            for i, image in enumerate(first.images)
        ],
        "limitations": [
            "metadata_equality_does_not_prove_controlled_hardware_conditions",
            "separate_reports_do_not_prove_statistical_independence",
            "descriptive_statistics_only_no_confidence_interval",
            "no_accuracy_metric_distance_camera_fps_warning_audio_or_memory_measurement",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """显式读取本地报告并独占保存汇总；任何校验失败不生成结果。"""
    parser = argparse.ArgumentParser(description="Summarize comparable offline runs")
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError("summary destination already exists")
        if args.output.suffix.lower() != ".json":
            raise ValueError("summary output must use .json")
        report = aggregate_reports(tuple(args.report))
        save_report(report, args.output)
        print(
            json.dumps(
                {"report": str(args.output.resolve()), "run_count": report["run_count"]}
            )
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        logging.error("Summary failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
