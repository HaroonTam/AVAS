"""实时测量元数据与离线汇总；不打开设备，不改变风险规则。"""

import argparse
import hashlib
import json
import logging
from configparser import ConfigParser
from importlib.metadata import PackageNotFoundError, version
from math import isfinite
from pathlib import Path
from time import get_clock_info
from typing import TextIO

import numpy as np

from app.vision.detector import select_device


def file_hash(path: Path) -> str:
    """分块计算模型或配置身份，不保存文件内容或一次载入整个权重。"""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class PerformanceRecorder:
    def __init__(self, destination: Path, config: Path) -> None:
        """在设备启动前独占预留报告；已有文件拒绝覆盖，配置只记录哈希。"""
        self._config = config
        self._cuda = False
        self._report: dict[str, object] = {
            "protocol": "live_performance_v1",
            "cuda_peak_allocated_bytes": None,
            "resource_status": "not_started",
            "clocks": {
                name: {
                    "implementation": get_clock_info(name).implementation,
                    "resolution_s": get_clock_info(name).resolution,
                }
                for name in ("perf_counter", "monotonic")
            },
        }
        self._output: TextIO = destination.open("x", encoding="utf-8")

    def start(self) -> None:
        """模型加载后、相机启动前重置分配峰值；采样故障仅标记未知，不关闭告警。"""
        try:
            import torch

            settings = ConfigParser(interpolation=None)
            settings.read(self._config, encoding="utf-8")
            cuda, mps = torch.cuda.is_available(), torch.backends.mps.is_available()
            devices = {
                stage: select_device(settings.get(stage, "device"), cuda, mps)
                for stage in ("detection", "depth")
            }
            self._cuda = "cuda" in devices.values()
            self._report.update(
                {
                    "config_sha256": hashlib.sha256(
                        self._config.read_bytes()
                    ).hexdigest(),
                    "resolved_config_devices": devices,
                    "camera_waits_s": {
                        "read": settings.getfloat("camera", "timeout_s", fallback=3),
                        "startup": settings.getfloat(
                            "camera", "startup_timeout_s", fallback=None
                        ),
                    },
                    "torch_version": str(torch.__version__),
                    "cuda_peak_scope": "after_model_load_through_shutdown",
                    "resource_status": "available" if self._cuda else "not_cuda",
                }
            )
            versions: dict[str, str | None] = {}
            for package in (
                "numpy",
                "torch",
                "ultralytics",
                "transformers",
                "opencv-python",
            ):
                try:
                    versions[package] = version(package)
                except PackageNotFoundError:
                    versions[package] = None
            self._report["package_versions"] = versions
            root = self._config.resolve().parent
            weights = root / settings.get("detection", "weights")
            depth_dir = root / settings.get("depth", "model_dir")
            hashes: dict[str, str | None] = {}
            for name, path in (
                ("yolo11", weights),
                ("depth_anything_v2", depth_dir / "model.safetensors"),
            ):
                try:
                    hashes[name] = file_hash(path)
                except OSError as exc:
                    hashes[name] = None
                    logging.error("Performance weight identity unavailable: %s", exc)
            self._report["weight_sha256"] = hashes
            if self._cuda:
                self._report["cuda_device_name"] = torch.cuda.get_device_name()
                torch.cuda.reset_peak_memory_stats()
        except Exception as exc:
            self._cuda = False
            self._report["resource_status"] = "unavailable"
            logging.error("Performance resource sampling unavailable: %s", exc)

    def finish(self, failed: bool) -> None:
        """退出时写报告并关闭句柄；显存为本进程 PyTorch 分配峰值，不是整卡占用。"""
        try:
            if self._cuda:
                try:
                    import torch

                    self._report["cuda_peak_allocated_bytes"] = int(
                        torch.cuda.max_memory_allocated()
                    )
                except Exception as exc:
                    self._report["resource_status"] = "unavailable"
                    logging.error("CUDA peak unavailable: %s", exc)
            self._report["live_failed"] = failed
            json.dump(self._report, self._output, allow_nan=False, indent=2)
            self._output.write("\n")
        finally:
            self._output.close()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """兼容 PowerShell UTF-16 与 UTF-8；非 JSON 或非对象行明确拒绝，避免静默删样本。"""
    raw = path.read_bytes()
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    rows: list[dict[str, object]] = []
    for line in raw.decode(encoding).splitlines():
        if not line.strip():
            continue
        value: object = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("JSONL row must be an object")
        rows.append(value)
    return rows


def number(value: object) -> float:
    """验证有限非负测量值；缺失、布尔值和非有限值不得作为零值参与统计。"""
    if type(value) not in (int, float):
        raise ValueError("measurement must be numeric")
    result = float(value)
    if not isfinite(result) or result < 0:
        raise ValueError("measurement must be finite and nonnegative")
    return result


def statistics(values: list[float]) -> dict[str, float | int] | None:
    """对已验证的毫秒样本计算线性插值分位数；空样本返回未知。"""
    if not values:
        return None
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50, method="linear")),
        "p95": float(np.percentile(values, 95, method="linear")),
        "max": max(values),
    }


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    """首结果单列预热；陈旧、深度故障及终止故障不入正常延迟池，仍保留分类计数。

    稳态 FPS 为首结果派发后至最后结果派发的有效完整结果数除以该墙钟区间，
    区间内故障时间不扣除；这不是原生摄像头采集帧率，也不是可听延迟。
    """
    stages = ("detection", "depth", "fusion", "risk", "capture_to_dispatch")
    samples: dict[str, list[float]] = {stage: [] for stage in stages}
    warmup: dict[str, object] | None = None
    first_end: float | None = None
    last_end: float | None = None
    loop_start: float | None = None
    processed = valid = stale = degraded = fault = 0
    dimensions: set[tuple[int, int]] = set()
    stage_clocks: set[str] = set()
    for row in rows:
        metadata = row.get("performance")
        if not isinstance(metadata, dict):
            scene = row.get("scene")
            if isinstance(scene, dict) and scene.get("valid") is False:
                fault += 1
                continue
            raise ValueError(
                "missing performance metadata; rerun with measurement enabled"
            )
        processed += 1
        stage_clock = metadata.get("stage_clock", "legacy_monotonic")
        if not isinstance(stage_clock, str) or stage_clock not in (
            "legacy_monotonic",
            "perf_counter",
        ):
            raise ValueError("unknown stage clock")
        stage_clocks.add(stage_clock)
        if len(stage_clocks) > 1:
            raise ValueError("cannot pool different stage clocks")
        if metadata.get("result_index") != processed:
            raise ValueError("missing, duplicate or reordered results")
        if metadata.get("warmup") is not (processed == 1):
            raise ValueError("invalid warmup marker")
        start = number(metadata.get("loop_started_monotonic_s"))
        end = number(metadata.get("dispatch_completed_monotonic_s"))
        captured = number(metadata.get("captured_monotonic_s"))
        if end < start or end < captured or (last_end is not None and end < last_end):
            raise ValueError("invalid monotonic boundaries")
        if loop_start is not None and start != loop_start:
            raise ValueError("mixed measurement sessions")
        loop_start, last_end = start, end
        width, height = metadata.get("width"), metadata.get("height")
        if type(width) is not int or type(height) is not int or min(width, height) < 1:
            raise ValueError("invalid image dimensions")
        dimensions.add((width, height))
        is_stale = row.get("current_scene") is not True
        is_degraded = row.get("depth_status") != "available"
        stale += int(is_stale)
        degraded += int(is_degraded and not is_stale)
        timings = row.get("latencies_ms")
        if not isinstance(timings, dict):
            raise ValueError("missing timings")
        if processed == 1:
            first_end = end
            warmup = {
                "latencies_ms": timings,
                "current_scene": row.get("current_scene"),
                "depth_status": row.get("depth_status"),
            }
        elif not is_stale and not is_degraded:
            for stage in stages:
                samples[stage].append(number(timings.get(stage)))
            valid += 1
    elapsed = (
        last_end - first_end if last_end is not None and first_end is not None else None
    )
    return {
        "protocol": "live_performance_v1",
        "stage_clock": next(iter(stage_clocks), None),
        "result_count": processed,
        "valid_post_warmup_count": valid,
        "stale_count_including_warmup": stale,
        "depth_degraded_count_including_warmup": degraded,
        "terminal_fault_count": fault,
        "warmup_first_result": warmup,
        "actual_dimensions": [list(size) for size in sorted(dimensions)],
        "post_warmup_wall_seconds": elapsed,
        "valid_processing_fps": valid / elapsed if elapsed and elapsed > 0 else None,
        "loop_to_last_dispatch_seconds": (
            last_end - loop_start
            if last_end is not None and loop_start is not None
            else None
        ),
        "summary_ms": {stage: statistics(values) for stage, values in samples.items()},
        "unmeasured": [
            "native_camera_fps",
            "audible_latency",
            "accuracy",
            "metric_error",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """离线读取本次 JSONL 并独占写汇总；损坏记录返回失败，不接触真实设备。"""
    parser = argparse.ArgumentParser(description="Summarize live performance JSONL")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = summarize_rows(read_jsonl(args.input))
        with args.output.open("x", encoding="utf-8") as target:
            json.dump(report, target, allow_nan=False, indent=2)
            target.write("\n")
        print(json.dumps(report, allow_nan=False))
    except (OSError, ValueError) as exc:
        logging.error("Performance summary failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
