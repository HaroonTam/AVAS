"""固定 Metric 候选的显式下载与离线诊断；不接入生产，不生成 MetricEvidence。"""

import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

import numpy as np

from app.e2_capture import ROOT
from app.e2_offline import digest, frozen_paths, load_sample, mapping, read_image
from app.fusion.config import FusionConfig, load_fusion_config
from app.fusion.depth_fusion import filtered_statistics, sampling_region
from app.vision.depth_estimator import DepthMap, load_depth_config
from app.vision.detector import Image, select_device

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
REVISION = "8078d68a9c75a972131914f6afd0c1723be0da7f"
FILES = {
    "config.json": "0b1d9fc591693d16b864249a9cfcf5264c8c00be999c1643fa1c64dc105d55f8",
    "preprocessor_config.json": (
        "533b16a60445d7cab5086d39b45f92be45624b977972c62d5984d93e98366063"
    ),
    "model.safetensors": (
        "e990eb82fbf11b05b7813261196a2b841bdcf5a05f64396724a8987fa90504a3"
    ),
    "README.md": "521e7905b3df074a81f3c7e79a959e9226777b4a5765de1697a816d1965e927e",
}
MODEL_DIR = ROOT / "models/depth-anything-v2-metric-indoor-small-hf"


def verify_model(directory: Path) -> None:
    """验证全部文件的固定身份，拒绝相对模型或被修改的配置。"""
    for name, expected in FILES.items():
        if digest(directory / name) != expected:
            raise ValueError(f"model identity mismatch: {name}")


def download_model(directory: Path) -> None:
    """用户显式触发下载；逐文件验证，保留失败的 part 文件，不覆盖已有文件。"""
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        destination = directory / name
        if destination.exists():
            if digest(destination) != expected:
                raise ValueError(f"existing file mismatch: {destination}")
            continue
        partial = directory / (name + ".part")
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"
        checksum = hashlib.sha256()
        with partial.open("xb") as output:
            with urllib.request.urlopen(url, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    checksum.update(chunk)
        if checksum.hexdigest() != expected:
            raise ValueError(f"download hash mismatch: {partial}")
        partial.rename(destination)
        print(f"Verified: {name}")
    verify_model(directory)


class MetricBackend:
    def __init__(self, directory: Path, device: str, image_size: int) -> None:
        """仅加载已校验的本地米制权重，拒绝远程代码和隐式下载。"""
        verify_model(directory)
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._processor = AutoImageProcessor.from_pretrained(
            str(directory),
            local_files_only=True,
            trust_remote_code=False,
            backend="pil",
        )
        self._model = AutoModelForDepthEstimation.from_pretrained(
            str(directory),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
        if (
            self._model.config.model_type != "depth_anything"
            or self._model.config.depth_estimation_type != "metric"
            or self._model.config.max_depth != 20
        ):
            raise ValueError("expected pinned indoor metric configuration")
        self.device = select_device(
            device, torch.cuda.is_available(), torch.backends.mps.is_available()
        )
        self._model.to(self.device).eval()
        self._size = image_size
        self.torch_version = str(torch.__version__)

    def predict(self, image: Image) -> DepthMap:
        """按相对基线相同的 PIL 预处理与 bicubic 对齐返回原图米制预测。"""
        import torch

        inputs = self._processor(
            images=np.ascontiguousarray(image[:, :, ::-1]),
            return_tensors="pt",
            size={"height": self._size, "width": self._size},
        ).to(self.device)
        with torch.inference_mode():
            raw = self._model(**inputs).predicted_depth
            if not isinstance(raw, torch.Tensor) or raw.ndim != 3 or raw.shape[0] != 1:
                raise RuntimeError("unexpected metric output")
            resized = torch.nn.functional.interpolate(
                raw.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            )[0, 0]
            result: DepthMap = resized.float().cpu().numpy()
        return result


def metric_summary(
    values: DepthMap, bbox: tuple[int, int, int, int], fusion: FusionConfig
) -> dict[str, object]:
    """复用生产统计核；米制输入只保留正且不超过模型上限的有限像素。"""
    x1, y1, x2, y2 = sampling_region(bbox, fusion)
    region = values[y1:y2, x1:x2]
    count = int(region.size)
    if not count:
        raise ValueError("empty metric region")
    valid = region[np.isfinite(region) & (region > 0) & (region <= 20)].astype(
        np.float64
    )
    status, retained, fraction, value, spread = filtered_statistics(
        valid, count, fusion
    )
    return {
        "status": status,
        "roi": [x1, y1, x2, y2],
        "sample_count": count,
        "valid_count": int(valid.size),
        "retained_count": retained,
        "valid_fraction": fraction,
        "predicted_depth_m": value,
        "iqr_m": spread,
        "units": "m",
        "support_status": "unvalidated_model_prediction",
    }


def run_metric(manifest: Path, output: Path, directory: Path, config: Path) -> None:
    """在冻结原图与诊断框上推理一次，保留失败样本；不拟合、不改变风险状态。"""
    if output.exists():
        raise FileExistsError("output already exists")
    frozen = mapping(json.loads(manifest.read_text(encoding="utf-8")))
    if frozen.get("roi_policy") != "board_lower_half_v1":
        raise ValueError("expected reviewed board_lower_half_v1 manifest")
    fusion = load_fusion_config(config)
    samples = [
        load_sample(path, fusion, digest(config))
        for path in frozen_paths(manifest, ROOT / "outputs")
    ]
    if len({s.image_hash for s in samples}) != len(samples):
        raise ValueError("duplicate images")
    if len({s.measurement.installation_id for s in samples}) != 1:
        raise ValueError("mixed installations")
    depth_config = load_depth_config(config)
    backend = MetricBackend(directory, depth_config.device, depth_config.image_size)
    output.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []
    for sample in samples:
        x1, y1, x2, y2 = sample.roi
        bbox = (x1, (y1 + y2 + 1) // 2, x2, y2)
        row: dict[str, object] = {
            "sample_id": sample.path.parent.name,
            "record_sha256": sample.record_hash,
            "image_sha256": sample.image_hash,
            "truth_m": sample.measurement.distance_m,
            "analysis_roi": bbox,
            "metric_evidence": None,
        }
        try:
            image = read_image(sample.image_path)
            values = backend.predict(image)
            if values.shape != image.shape[:2]:
                raise ValueError("metric output grid mismatch")
            row["metric_summary"] = metric_summary(values, bbox, fusion)
        except (RuntimeError, ValueError, OSError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    report = {
        "schema": "e2_metric_v1",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "files_sha256": FILES,
        "device": backend.device,
        "torch_version": backend.torch_version,
        "image_size": depth_config.image_size,
        "manifest_sha256": digest(manifest),
        "config_sha256": digest(config),
        "runner_sha256": digest(Path(__file__)),
        "code_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "sample_count": len(rows),
        "samples": rows,
        "dataset_role": "single_scene_development_not_independent_test",
        "roi_policy": "board_lower_half_v1",
    }
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(output / "report.json")
    if any("error" in row for row in rows):
        raise RuntimeError("some samples failed; inspect saved report")


def main(argv: list[str] | None = None) -> int:
    """下载和推理为互斥的显式操作；默认不联网、不运行模型。"""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--download", action="store_true")
    mode.add_argument("--infer", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/system.ini")
    args = parser.parse_args(argv)
    if args.infer and (args.manifest is None or args.output is None):
        parser.error("--infer requires --manifest and --output")
    try:
        if args.download:
            download_model(args.model_dir)
        else:
            run_metric(args.manifest, args.output, args.model_dir, args.config)
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"E2 metric failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
