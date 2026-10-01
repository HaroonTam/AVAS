import ast
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.config import DetectionConfig, load_config
from app.main import main
from app.vision.detector import (
    Detection,
    Image,
    Yolo11Detector,
    parse_detections,
    select_device,
)


class FakeBackend:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def names(self) -> dict[int, str]:
        return {0: "person"}

    def predict(self, image: Image) -> list[list[float]]:
        self.calls += 1
        return [[-2.5, 1.3, 9.2, 12.1, 0.9, 0.0]]


class DetectionTests(unittest.TestCase):
    def test_default_config_from_another_working_directory(self) -> None:
        """在其他工作目录启动时仍加载项目配置，不加载同名的本地配置。"""
        project_config = Path(__file__).resolve().parents[1] / "configs/system.ini"
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            local_config = Path(directory) / "configs/system.ini"
            local_config.parent.mkdir()
            local_config.write_text("invalid configuration", encoding="utf-8")
            try:
                os.chdir(directory)
                with (
                    patch("app.main.load_config", wraps=load_config) as loader,
                    self.assertLogs(level="ERROR") as logs,
                ):
                    self.assertEqual(main(["--image", "missing.png"]), 1)
                loader.assert_called_once_with(project_config)
                self.assertIn("Image missing", logs.output[0])
            finally:
                os.chdir(original_directory)

    def test_explicit_config_remains_relative_to_working_directory(self) -> None:
        """显式配置路径仍以工作目录为基准，不被项目默认配置覆盖。"""
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            custom_config = Path(directory) / "custom.ini"
            custom_config.write_text(
                "[detection]\nweights=custom.pt\nconfidence=0.6\n"
                "image_size=320\ndevice=cpu\n",
                encoding="utf-8",
            )
            try:
                os.chdir(directory)
                with (
                    patch("app.main.load_config", wraps=load_config) as loader,
                    self.assertLogs(level="ERROR") as logs,
                ):
                    self.assertEqual(
                        main(["--config", "custom.ini", "--image", "missing.png"]), 1
                    )
                loader.assert_called_once_with(Path("custom.ini"))
                self.assertIn("Image missing", logs.output[0])
            finally:
                os.chdir(original_directory)

    def test_clipping_and_original_coordinates(self) -> None:
        result = parse_detections(
            [[-2.5, 1.3, 9.2, 12.1, 0.9, 0]], {0: "person"}, 10, 10, 0.35
        )
        self.assertEqual(result, (Detection("person", 0.9, (0, 1, 10, 10)),))

    def test_threshold_empty_and_external_boxes(self) -> None:
        rows = [
            [0, 0, 5, 5, 0.35, 0],
            [0, 0, 5, 5, 0.349, 0],
            [12, 12, 15, 15, 0.9, 0],
        ]
        self.assertEqual(len(parse_detections(rows, {0: "person"}, 10, 10, 0.35)), 1)
        self.assertEqual(parse_detections([], {0: "person"}, 10, 10, 0.35), ())

    def test_malformed_output_is_not_an_empty_scene(self) -> None:
        bad_rows = [
            [0, 0, 5, 5, float("nan"), 0],
            [0, 0, float("inf"), 5, 0.8, 0],
            [0, 0, 5, 5, 1.1, 0],
            [5, 0, 0, 5, 0.8, 0],
            [0, 0, 5, 5, 0.8, 1],
            [0, 0, 5, 5, 0.8, 0.5],
            [0, 0, 5, 5, 0.8],
        ]
        for row in bad_rows:
            with self.subTest(row=row), self.assertRaises(ValueError):
                parse_detections([row], {0: "person"}, 10, 10, 0.35)

    def test_backend_reused_and_actual_labels_exposed(self) -> None:
        backend = FakeBackend()
        detector = Yolo11Detector(DetectionConfig(Path("unused.pt")), backend)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        detector.detect(frame)
        detector.detect(frame)
        self.assertEqual(backend.calls, 2)
        self.assertEqual(detector.supported_labels, ("person",))
        self.assertNotIn("stairs", detector.supported_labels)

    def test_invalid_images_rejected_before_backend(self) -> None:
        backend = FakeBackend()
        detector = Yolo11Detector(DetectionConfig(Path("unused.pt")), backend)
        for shape in [(0, 10, 3), (10, 10), (10, 10, 4)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                detector.detect(np.zeros(shape, dtype=np.uint8))
        self.assertEqual(backend.calls, 0)

    def test_missing_weights_fail_without_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                Yolo11Detector(DetectionConfig(Path(directory) / "yolo11n.pt"))

    def test_device_selection_and_reported_fallback(self) -> None:
        self.assertEqual(select_device("auto", True, False), "cuda")
        self.assertEqual(select_device("auto", False, True), "mps")
        self.assertEqual(select_device("auto", False, False), "cpu")
        with self.assertLogs("app.vision.detector", level="WARNING"):
            self.assertEqual(select_device("cuda", False, False), "cpu")

    def test_config_validation(self) -> None:
        for confidence in [-0.1, 1.1, float("nan")]:
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                DetectionConfig(Path("unused.pt"), confidence=confidence)
        with self.assertRaises(ValueError):
            DetectionConfig(Path("unused.pt"), image_size=0)
        with self.assertRaises(ValueError):
            DetectionConfig(Path("unused.pt"), device="gpu")

    def test_config_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.ini"
            path.write_text(
                "[detection]\nweights=models/yolo11n.pt\nconfidence=0.4\n"
                "image_size=640\ndevice=cpu\n",
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.weights, Path(directory) / "models/yolo11n.pt")
            self.assertEqual(config.confidence, 0.4)

    def test_cli_reports_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertLogs(level="ERROR"):
                self.assertEqual(
                    main(
                        [
                            "--config",
                            str(Path(directory) / "missing.ini"),
                            "--image",
                            "missing.png",
                        ]
                    ),
                    1,
                )

    def test_all_functions_have_complete_annotations(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for folder in ("app", "tests"):
            for path in (root / folder).rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    self.assertNotIsInstance(node, ast.Lambda, str(path))
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        with self.subTest(path=path, function=node.name):
                            self.assertIsNotNone(node.returns)
                            args = [
                                *node.args.posonlyargs,
                                *node.args.args,
                                *node.args.kwonlyargs,
                            ]
                            if node.args.vararg:
                                args.append(node.args.vararg)
                            if node.args.kwarg:
                                args.append(node.args.kwarg)
                            for arg in args:
                                if arg.arg not in {"self", "cls"}:
                                    self.assertIsNotNone(arg.annotation, arg.arg)
