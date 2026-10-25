# E2 边界补采：仅由用户在终端启动；Preview 不访问相机或调用 Python。
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('scene09', 'scene10', 'scene11', 'scene12')]
    [string]$Group,
    [ValidateRange(1, 12)]
    [int]$StartPlacement = 1,
    [ValidateRange(1, 12)]
    [int]$EndPlacement = 1,
    [string]$Conditions = '',
    [string]$Python = 'D:\python\python.exe',
    [switch]$Preview
)

$ErrorActionPreference = 'Stop'
$e2Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $e2Root
$e2PlanPath = if ($Group -eq 'scene11') {
    'configs/e2_boundary_validation.json'
} elseif ($Group -eq 'scene12') {
    'configs/e2_boundary_fit_replacement.json'
} else {
    'configs/e2_boundary_fit.json'
}
$e2Plan = Get-Content -LiteralPath $e2PlanPath -Raw -Encoding UTF8 | ConvertFrom-Json
$e2Distances = @(
    foreach ($e2TargetDistance in $e2Plan.distance_targets_m) {
        for ($e2Repeat = 0; $e2Repeat -lt $e2Plan.independent_placements_per_distance; $e2Repeat++) {
            $e2TargetDistance
        }
    }
)
if ($StartPlacement -gt $EndPlacement -or $EndPlacement -gt $e2Distances.Count) {
    throw "起止编号错误：$Group 共 $($e2Distances.Count) 条。"
}
if ($Preview) {
    for ($e2Index = $StartPlacement; $e2Index -le $EndPlacement; $e2Index++) {
        Write-Output "$Group / $('p{0:D2}' -f $e2Index) / 目标约 $($e2Distances[$e2Index - 1]) 米 / $e2PlanPath"
    }
    return
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "找不到 Python：$Python"
}
if ([string]::IsNullOrWhiteSpace($Conditions)) {
    $e2Previous = @(Get-ChildItem -Path 'outputs/e2_*/record.json' -File | ForEach-Object {
        $e2Record = Get-Content -LiteralPath $_.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($e2Record.collection.collection_id -eq $e2Plan.collection_id -and
            $e2Record.collection.group_id -eq $Group) {
            $e2Record
        }
    })
    if ($e2Previous.Count -gt 0) {
        $Conditions = [string]$e2Previous[0].measurement.conditions
        Write-Host "沿用本组条件：$Conditions"
    }
}
while ([string]::IsNullOrWhiteSpace($Conditions)) {
    $Conditions = (Read-Host "$Group 实际背景、光照、反光情况及与此前布景的区别").Trim()
}
for ($e2Index = $StartPlacement; $e2Index -le $EndPlacement; $e2Index++) {
    $e2Placement = 'p{0:D2}' -f $e2Index
    Write-Host "准备 $Group / $e2Placement，目标约 $($e2Distances[$e2Index - 1]) 米" -ForegroundColor Cyan
    Write-Host '请移开后重新摆放目标，保持正面竖直朝向镜头，并重新测量实际距离。'
    do {
        $e2DistanceText = (Read-Host "$e2Placement 实际距离（米，用小数点）").Trim()
        $e2ParsedDistance = 0.0
        $e2DistanceValid = [double]::TryParse(
            $e2DistanceText,
            [Globalization.NumberStyles]::Float,
            [Globalization.CultureInfo]::InvariantCulture,
            [ref]$e2ParsedDistance
        )
        $e2DistanceValid = $e2DistanceValid -and $e2ParsedDistance -gt 0 -and
            -not [double]::IsInfinity($e2ParsedDistance) -and -not [double]::IsNaN($e2ParsedDistance)
        if (-not $e2DistanceValid) {
            Write-Host '请输入正数实际读数，例如 0.405；不能为空或只填单位。' -ForegroundColor Yellow
        }
    } while (-not $e2DistanceValid)
    $e2DistanceText = $e2ParsedDistance.ToString('R', [Globalization.CultureInfo]::InvariantCulture)
    & $Python -m app.e2_capture --save-local --collection-plan $e2PlanPath --group-id $Group --installation-id omen_h0265_v1 --placement-id $e2Placement --target-id A02 --target-material '纸箱,黄色' --target-width-m 0.243 --target-height-m 0.271 --distance-m $e2DistanceText --instrument '卷尺,0.1cm' --conditions $Conditions
    if ($LASTEXITCODE -ne 0) {
        throw "采集在 $Group / $e2Placement 停止，退出码 $LASTEXITCODE。请保留输出，从未保存的编号续采。"
    }
    Write-Host "$Group / $e2Placement 已完成。" -ForegroundColor Green
}
