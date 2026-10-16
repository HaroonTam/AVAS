# 用户显式执行的真机测量：保持生产语音启用，不保存图像或音频。
param(
    [string]$Python = "D:\python\python.exe",
    [ValidateRange(2, 100000)][int]$Frames = 600
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$runName = "live_perf_" + (Get-Date -Format "yyyyMMdd_HHmmss") + "_" + [guid]::NewGuid().ToString("N")
$relative = Join-Path "outputs" $runName
$directory = New-Item -ItemType Directory -Path $relative
$samples = [System.Collections.Generic.List[object]]::new()
$revision = git rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw "Cannot record Git revision" }
$changes = @(git status --porcelain -- app configs scripts pyproject.toml)
$process = Start-Process -FilePath $Python -ArgumentList @(
    "-m", "app.live", "--max-frames", "$Frames",
    "--performance-output", "$relative/resources.json"
) -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "$relative/frames.jsonl" `
    -RedirectStandardError "$relative/stderr.log"
# Windows PowerShell 5.1 重定向输出时必须在进程存活期间缓存句柄，否则退出码可能丢失。
$null = $process.Handle
$timer = [System.Diagnostics.Stopwatch]::StartNew()
try {
    while (-not $process.HasExited) {
        $process.Refresh()
        if (-not $process.HasExited) {
            $samples.Add([pscustomobject]@{
                elapsed_s = $timer.Elapsed.TotalSeconds
                working_set_bytes = $process.WorkingSet64
                lifetime_peak_working_set_bytes = $process.PeakWorkingSet64
            })
        }
        Start-Sleep -Milliseconds 100
    }
    $process.WaitForExit()
    $exitCode = $process.ExitCode
    $samples | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath "$relative/memory_samples.json" -Encoding UTF8
    $workingPeak = $null
    $lifetimePeak = $null
    if ($samples.Count -gt 0) {
        $workingPeak = ($samples | Measure-Object -Property working_set_bytes -Maximum).Maximum
        $lifetimePeak = ($samples | Measure-Object -Property lifetime_peak_working_set_bytes -Maximum).Maximum
    }
    $metadata = [ordered]@{
        revision = "$revision"
        tracked_scope_changes = $changes
        requested_frames = $Frames
        interpreter = $Python
        live_exit_code = $exitCode
        memory_scope = "main_python_process_only_startup_through_exit"
        sample_interval_requested_ms = 100
        sample_count = $samples.Count
        sampled_working_set_peak_bytes = $workingPeak
        observed_lifetime_peak_working_set_bytes = $lifetimePeak
        limitation = "excludes_camera_and_speech_children; polling_can_miss_final_peak"
    }
    $metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath "$relative/run.json" -Encoding UTF8
    & $Python -m app.live_performance --input "$relative/frames.jsonl" --output "$relative/summary.json"
    $summaryExit = $LASTEXITCODE
    $complete = $false
    if ($summaryExit -eq 0 -and $null -ne $exitCode -and $exitCode -eq 0) {
        $summary = Get-Content -LiteralPath "$relative/summary.json" -Raw | ConvertFrom-Json
        $complete = ($summary.result_count -eq $Frames -and
            $summary.valid_post_warmup_count -gt 0 -and
            $summary.terminal_fault_count -eq 0)
    }
    Write-Output "run_directory=$($directory.FullName)"
    Write-Output "live_exit_code=$exitCode summary_exit_code=$summaryExit"
    Write-Output "measurement_complete=$complete"
    Get-Content -LiteralPath "$relative/run.json"
    if (Test-Path -LiteralPath "$relative/resources.json") {
        Get-Content -LiteralPath "$relative/resources.json"
    }
    Get-Content -LiteralPath "$relative/stderr.log"
    if (-not $complete) { throw "Measurement incomplete; preserve this run for review" }
} finally {
    $process.Dispose()
}
