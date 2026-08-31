$ErrorActionPreference = "Stop"

$repo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$required = @(
    "src\pipeline.py",
    "src\running_metrics_v1\calculate_running.py",
    "src\running_metrics_v1\build_rotation_dynamic_calibration.py",
    "tests\test_frame_domain.py",
    "tests\test_running_metrics.py",
    "artifacts\pipeline\tracking_mot.txt",
    "artifacts\pipeline\event_index.json",
    "artifacts\metric_running\full_run\dynamic_calibration_45x25.json",
    "artifacts\metric_running\full_run\metrics\player_running_summary.csv",
    "artifacts\metric_running\full_run\metrics\player_running_timeseries.csv",
    "large_artifacts\pipeline\tracking_vis_720p_review.mp4",
    "large_artifacts\metric_running\full_running_dynamic_45x25_720p_review.mp4"
)

$missing = @()
foreach ($relative in $required) {
    $path = Join-Path $repo $relative
    if (-not (Test-Path -LiteralPath $path)) {
        $missing += $relative
    }
}
if ($missing.Count -gt 0) {
    throw "Missing required files: $($missing -join ', ')"
}

$limit = 2GB
Get-ChildItem -LiteralPath (Join-Path $repo "large_artifacts") -Recurse -File | ForEach-Object {
    if ($_.Length -ge $limit) {
        throw "GitHub Free/Pro LFS per-file limit exceeded: $($_.FullName)"
    }
}

$motFirst = Get-Content -LiteralPath (Join-Path $repo "artifacts\pipeline\tracking_mot.txt") -TotalCount 1
$motLast = Get-Content -LiteralPath (Join-Path $repo "artifacts\pipeline\tracking_mot.txt") -Tail 1
if (-not $motFirst.StartsWith("1,")) {
    throw "MOT does not start at frame 1"
}
if (-not $motLast.StartsWith("62204,")) {
    throw "MOT does not end at frame 62204"
}

Write-Host "Release structure OK"
Write-Host "Repository: $repo"
Write-Host "Required files: $($required.Count)"
Write-Host "MOT coverage: 1..62204"
