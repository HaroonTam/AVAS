"""在 Windows PowerShell 中验证真实进程退出码；以无设备子进程替换采集命令。"""

import json
import os
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


@unittest.skipUnless(
    os.name == "nt" and shutil.which("powershell"), "Windows PowerShell required"
)
class MeasureLiveScriptTests(unittest.TestCase):
    def test_redirected_exit_and_empty_run_rejected(self) -> None:
        """执行生产脚本但替换启动参数；非零退出码保留，零退出的空采集也拒绝通过。"""
        root = Path(__file__).resolve().parents[1]
        for exit_code in (7, 0):
            with self.subTest(exit_code=exit_code):
                command = r"""
function Start-Process {
    param($FilePath, $ArgumentList, $WorkingDirectory, $WindowStyle,
          [switch]$PassThru, $RedirectStandardOutput, $RedirectStandardError)
    $configIndex = [array]::IndexOf($ArgumentList, '--config')
    $expected = '"' + (Resolve-Path 'configs/system.ini').Path + '"'
    if ($configIndex -lt 0 -or $ArgumentList[$configIndex + 1] -ne $expected) {
        throw 'Missing or unquoted config path'
    }
    $probe = '"import time; time.sleep(0.5); raise SystemExit(EXIT_CODE)"'
    $PSBoundParameters['ArgumentList'] = @('-c', $probe)
    Microsoft.PowerShell.Management\Start-Process @PSBoundParameters
}
& ./scripts/measure_live.ps1 -Python 'PYTHON_PATH' -Frames 2
""".replace("EXIT_CODE", str(exit_code)).replace(
                    "PYTHON_PATH", sys.executable.replace("'", "''")
                )
                result = subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-Command",
                        command,
                    ],
                    cwd=root,
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
                stdout = result.stdout.decode("utf-8", errors="replace")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    f"live_exit_code={exit_code} summary_exit_code=0",
                    stdout,
                    result.stderr.decode("utf-8", errors="replace"),
                )
                self.assertIn("measurement_complete=False", stdout)
                match = re.search(r"run_directory=(.+)", stdout)
                self.assertIsNotNone(match)
                if match is None:
                    self.fail(stdout)
                directory = Path(match.group(1).strip())
                (directory / "SYNTHETIC_TEST_ONLY.txt").write_text(
                    "Synthetic process test only; no devices or models used.\n",
                    encoding="utf-8",
                )
                metadata = json.loads(
                    (directory / "run.json").read_text(encoding="utf-8-sig")
                )
                self.assertEqual(metadata["live_exit_code"], exit_code)
                self.assertEqual(Path(metadata["config"]), root / "configs/system.ini")
                self.assertEqual((directory / "frames.jsonl").read_bytes(), b"")
