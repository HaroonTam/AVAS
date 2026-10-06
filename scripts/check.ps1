# 统一本地校验入口：合成测试、Ruff 静态检查与格式检查。
# 用法：pwsh -File scripts/check.ps1 [-Python <解释器路径>]
# 默认按 AVAS_PYTHON、D:\python\python.exe、.venv\Scripts\python.exe、python 的顺序选择解释器。

param(
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $Python) {
    $candidates = @($env:AVAS_PYTHON, "D:\python\python.exe", (Join-Path $root ".venv\Scripts\python.exe"))
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) { $Python = $candidate; break }
    }
}
if (-not $Python) { $Python = "python" }
Write-Output "interpreter: $Python"

# 测试自带项目内临时目录；禁用字节码写入，避免产生未被跟踪的缓存。
$env:PYTHONDONTWRITEBYTECODE = "1"

Write-Output "[1/3] unittest discover -s tests -q"
& $Python -m unittest discover -s tests -q
if ($LASTEXITCODE -ne 0) { throw "unittest failed with exit code $LASTEXITCODE" }

Write-Output "[2/3] ruff check ."
& $Python -m ruff check --no-cache .
if ($LASTEXITCODE -ne 0) { throw "ruff check failed with exit code $LASTEXITCODE" }

Write-Output "[3/3] ruff format --check ."
& $Python -m ruff format --check .
if ($LASTEXITCODE -ne 0) { throw "ruff format check failed with exit code $LASTEXITCODE" }

Write-Output "ALL CHECKS PASSED"
