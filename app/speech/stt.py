"""按次启用的 Windows 本地命令识别；不保存或上传音频。"""

import base64
import json
import os
import subprocess
from dataclasses import dataclass
from math import isfinite
from threading import Event
from time import monotonic
from typing import Literal

COMMANDS: tuple[tuple[str, str], ...] = (
    ("描述周围", "描述周围"),
    ("当前风险", "当前风险"),
    ("寻找椅子", "寻找 椅子"),
    ("寻找人", "寻找 人"),
    ("寻找自行车", "寻找 自行车"),
    ("寻找汽车", "寻找 汽车"),
    ("寻找公交车", "寻找 公交车"),
    ("寻找摩托车", "寻找 摩托车"),
) + tuple(
    (f"{direction}的{label}在哪里", f"{direction}的{label}在哪里")
    for direction in ("左侧", "前方", "右侧")
    for label in ("椅子", "人", "自行车", "汽车", "公交车", "摩托车")
)


@dataclass(frozen=True)
class Recognition:
    status: Literal["recognized", "unrecognized", "timeout", "cancelled", "unavailable"]
    command: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class SttConfig:
    timeout_s: float = 10.0
    min_confidence: float = 0.7

    def __post_init__(self) -> None:
        """校验单次总等待秒数和识别分数阈值；阈值不是安全概率。"""
        if (
            isinstance(self.timeout_s, bool)
            or not isfinite(self.timeout_s)
            or not 1 <= self.timeout_s <= 30
        ):
            raise ValueError("STT timeout_s must be within [1, 30]")
        if (
            isinstance(self.min_confidence, bool)
            or not isfinite(self.min_confidence)
            or not 0 <= self.min_confidence <= 1
        ):
            raise ValueError("STT min_confidence must be within [0, 1]")


def parse_recognition(payload: bytes, config: SttConfig) -> Recognition:
    """收窄子进程 JSON，只接受已知命令及有限分数；畸形输出不执行请求。"""
    if len(payload) > 4096:
        return Recognition("unavailable")
    try:
        data: object = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeError, ValueError):
        return Recognition("unavailable")
    if not isinstance(data, dict):
        return Recognition("unavailable")
    if data.get("status") == "unrecognized":
        return Recognition("unrecognized")
    if data.get("status") != "recognized":
        return Recognition("unavailable")
    text, confidence = data.get("text"), data.get("confidence")
    if not isinstance(text, str) or type(confidence) not in {int, float}:
        return Recognition("unavailable")
    try:
        score = float(confidence)
    except OverflowError:
        return Recognition("unavailable")
    if not isfinite(score) or not 0 <= score <= 1:
        return Recognition("unavailable")
    command = dict(COMMANDS).get(text)
    if command is None or score < config.min_confidence:
        return Recognition("unrecognized")
    return Recognition("recognized", command, score)


def recognition_script(config: SttConfig) -> str:
    """构造固定本地语法，中文词表作为 Base64 数据传入而非可执行语句。"""
    phrases = json.dumps([phrase for phrase, _ in COMMANDS], ensure_ascii=False)
    encoded = base64.b64encode(phrases.encode("utf-8")).decode("ascii")
    return (
        "$ErrorActionPreference='Stop'; "
        "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); "
        "$engine=$null; try { Add-Type -AssemblyName System.Speech; "
        "$installed=@([System.Speech.Recognition.SpeechRecognitionEngine]"
        "::InstalledRecognizers() "
        "| Where-Object { $_.Culture.Name -eq 'zh-CN' }); "
        "if ($installed.Count -eq 0) { throw 'No zh-CN recognizer' }; "
        "$engine=[System.Speech.Recognition.SpeechRecognitionEngine]"
        "::new($installed[0]); "
        "$phrases=[Text.Encoding]::UTF8.GetString("
        f"[Convert]::FromBase64String('{encoded}')) | ConvertFrom-Json; "
        "$choices=[System.Speech.Recognition.Choices]::new([string[]]$phrases); "
        "$builder=[System.Speech.Recognition.GrammarBuilder]::new(); "
        "$builder.Culture=$installed[0].Culture; $builder.Append($choices); "
        "$engine.LoadGrammar([System.Speech.Recognition.Grammar]::new($builder)); "
        "$engine.SetInputToDefaultAudioDevice(); "
        f"$result=$engine.Recognize([TimeSpan]::FromSeconds({config.timeout_s})); "
        "if ($null -eq $result) { @{status='unrecognized'} "
        "| ConvertTo-Json -Compress } "
        "else { @{status='recognized';text=$result.Text;"
        "confidence=[double]$result.Confidence} "
        "| ConvertTo-Json -Compress } "
        "} catch { @{status='unavailable'} | ConvertTo-Json -Compress } "
        "finally { if ($null -ne $engine) { $engine.Dispose() } }"
    )


class WindowsCommandRecognizer:
    def __init__(self, config: SttConfig) -> None:
        """只保存设置，构造不打开麦克风；非 Windows 明确拒绝。"""
        if os.name != "nt":
            raise RuntimeError("local command STT requires Windows")
        self.config = config

    def recognize(self, stop: Event) -> Recognition:
        """在独立进程识别一次；外层总超时涵盖启动，关闭时杀死子进程释放麦克风。"""
        if stop.is_set():
            return Recognition("cancelled")
        encoded = base64.b64encode(
            recognition_script(self.config).encode("utf-16-le")
        ).decode("ascii")
        process: subprocess.Popen[bytes] | None = None
        deadline = monotonic() + self.config.timeout_s
        try:
            process = subprocess.Popen(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-EncodedCommand",
                    encoded,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            while True:
                if stop.is_set():
                    return Recognition("cancelled")
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return Recognition("timeout")
                try:
                    output, _ = process.communicate(timeout=min(0.1, remaining))
                except subprocess.TimeoutExpired:
                    continue
                if stop.is_set():
                    return Recognition("cancelled")
                if monotonic() > deadline:
                    return Recognition("timeout")
                if process.returncode != 0:
                    return Recognition("unavailable")
                return parse_recognition(output, self.config)
        except OSError:
            return Recognition("unavailable")
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=1)
