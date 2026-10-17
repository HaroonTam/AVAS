"""用户显式执行的独立相机启动诊断；不加载模型、不播音、不保存图像。"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import monotonic


def emit(stage: str, **details: str | int | bool) -> None:
    """立即输出阶段及主机单调时刻（秒），原始画面不进入日志。"""
    print(
        json.dumps({"stage": stage, "monotonic_s": monotonic(), **details}), flush=True
    )


def probe_worker(index: int, width: int, height: int) -> int:
    """用与生产相同的默认 OpenCV 后端打开一帧；原生调用卡住由父进程终止。"""
    emit("worker_entered")
    capture = None
    try:
        emit("opencv_import_started")
        import cv2
        import numpy as np

        emit("opencv_import_completed", version=cv2.__version__)
        emit("camera_open_started", index=index)
        capture = cv2.VideoCapture(index)
        emit("camera_open_returned", opened=bool(capture.isOpened()))
        if not capture.isOpened():
            return 2
        emit("resolution_set_started", width=width, height=height)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        emit("resolution_set_completed", backend=str(capture.getBackendName()))
        emit("first_read_started")
        ok, image = capture.read()
        if not ok or not isinstance(image, np.ndarray) or image.ndim != 3:
            emit("first_read_failed")
            return 2
        emit("first_frame", width=int(image.shape[1]), height=int(image.shape[0]))
        return 0
    except Exception as exc:
        emit("error", error_type=type(exc).__name__, message=str(exc))
        return 2
    finally:
        if capture is not None:
            emit("release_started")
            capture.release()
            emit("release_completed")


def decode_output(value: bytes | str | None) -> str:
    """兼容正常进程输出与 TimeoutExpired 的字节输出，不伪造缺失日志。"""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def run_probe(
    index: int, width: int, height: int, timeout_s: float
) -> dict[str, object]:
    """隔离运行诊断子进程；超时终止并保留已输出阶段，超时不是相机根因结论。"""
    started = monotonic()
    timed_out = False
    exit_code: int | None = None
    command = [
        sys.executable,
        "-u",
        "-m",
        "app.camera_probe",
        "--worker",
        "--index",
        str(index),
        "--width",
        str(width),
        "--height",
        str(height),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        stdout, stderr = decode_output(result.stdout), decode_output(result.stderr)
        exit_code = result.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout, stderr = decode_output(exc.stdout), decode_output(exc.stderr)
    events: list[dict[str, object]] = []
    unparsed: list[str] = []
    for line in stdout.splitlines():
        try:
            value: object = json.loads(line)
        except ValueError:
            unparsed.append(line)
            continue
        if isinstance(value, dict):
            events.append(value)
        else:
            unparsed.append(line)
    return {
        "protocol": "isolated_camera_startup_probe_v1",
        "index": index,
        "requested_width": width,
        "requested_height": height,
        "backend": "opencv_default_same_selection_as_production",
        "timeout_s": timeout_s,
        "started_monotonic_s": started,
        "ended_monotonic_s": monotonic(),
        "timed_out": timed_out,
        "worker_exit_code": exit_code,
        "first_frame_received": any(
            item.get("stage") == "first_frame" for item in events
        ),
        "events": events,
        "unparsed_stdout": unparsed,
        "stderr": stderr,
        "limitation": "isolated_no_models_no_speech_not_live_performance",
    }


def main(argv: list[str] | None = None) -> int:
    """在独占报告路径预留成功后启动诊断；默认最多等待 15 秒，不改生产配置。"""
    parser = argparse.ArgumentParser(
        description="Explicit single-frame camera startup probe"
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (
        args.index < 0
        or min(args.width, args.height) < 1
        or not 1 <= args.timeout <= 30
    ):
        parser.error("invalid camera dimensions, index or timeout (1..30 seconds)")
    if args.worker:
        return probe_worker(args.index, args.width, args.height)
    if args.output is None:
        parser.error("--output is required")
    with args.output.open("x", encoding="utf-8") as target:
        report = run_probe(args.index, args.width, args.height, args.timeout)
        json.dump(report, target, indent=2, allow_nan=False)
        target.write("\n")
    print(json.dumps(report, allow_nan=False))
    return (
        0 if report["worker_exit_code"] == 0 and report["first_frame_received"] else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
