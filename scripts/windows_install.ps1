param(
  [ValidateSet("AUTO","GPU","CPU","WEB")]
  [string]$Mode = "AUTO",
  [string]$TorchIndex = ""
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$InstallMarker = Join-Path $Root ".venv\football_insight_install.mode"
$BootstrapDir = Join-Path $Root "runtime\bootstrap"
$VcInstaller = Join-Path $BootstrapDir "vc_redist.x64.exe"
$VcUrl = "https://aka.ms/vc14/vc_redist.x64.exe"

function Step($Text) { Write-Host "`n[Football Insight] $Text" -ForegroundColor Cyan }
function Fail($Text) {
  Write-Host "`n[ERROR] $Text" -ForegroundColor Red
  if (Test-Path $InstallMarker) { Remove-Item $InstallMarker -Force -ErrorAction SilentlyContinue }
  exit 1
}

function Get-CompatiblePython {
  $Candidates = @()
  $PythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
  if ($PythonCommand -and $PythonCommand.Source) { $Candidates += $PythonCommand.Source }
  if (Get-Command py.exe -ErrorAction SilentlyContinue) {
    foreach ($v in @("3.12","3.11")) {
      try {
        $candidate = (& py "-$v" -c "import sys;print(sys.executable)" 2>$null | Select-Object -Last 1)
        if ($LASTEXITCODE -eq 0 -and $candidate) {
          $candidate = $candidate.ToString().Trim()
          if ($candidate -and (Test-Path $candidate)) { $Candidates += $candidate }
        }
      } catch {}
    }
  }
  foreach ($candidate in ($Candidates | Select-Object -Unique)) {
    try {
      $versionLine = (& $candidate --version 2>&1 | Select-Object -Last 1)
      if ($LASTEXITCODE -ne 0 -or -not $versionLine) { continue }
      $versionText = $versionLine.ToString().Trim()
      if ($versionText -match '^Python\s+3\.(11|12)\.\d+') {
        $bits = (& $candidate -c "import struct;print(struct.calcsize('P')*8)" 2>$null | Select-Object -Last 1)
        if ($bits -and $bits.ToString().Trim() -eq "64") {
          return [PSCustomObject]@{ Path = $candidate; Version = $versionText }
        }
      }
    } catch {}
  }
  return $null
}

function Ensure-VCRuntime([string]$Py) {
  Step "检查 Microsoft Visual C++ x64 运行库"
  & $Py scripts\windows_torch_probe.py --vc-only *> $null
  if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Microsoft Visual C++ runtime is ready." -ForegroundColor Green
    return
  }

  Write-Host "[INFO] Missing/incomplete VC++ runtime. Installing the latest Microsoft x64 redistributable..." -ForegroundColor Yellow
  New-Item -ItemType Directory -Force -Path $BootstrapDir | Out-Null
  if (-not (Test-Path $VcInstaller)) {
    try {
      Invoke-WebRequest -UseBasicParsing -Uri $VcUrl -OutFile $VcInstaller
    } catch {
      Fail "无法下载 Microsoft Visual C++ x64 运行库。请联网后重试，或手动安装最新 vc_redist.x64.exe。"
    }
  }
  try {
    $proc = Start-Process -FilePath $VcInstaller -ArgumentList @("/install","/quiet","/norestart") -Wait -PassThru
    if ($proc.ExitCode -ne 0 -and $proc.ExitCode -ne 1638 -and $proc.ExitCode -ne 3010) {
      Fail "Microsoft Visual C++ runtime 安装失败，exit code=$($proc.ExitCode)"
    }
  } catch {
    Fail "Microsoft Visual C++ runtime 安装失败：$($_.Exception.Message)"
  }
  Start-Sleep -Seconds 2
  & $Py scripts\windows_torch_probe.py --vc-only
  if ($LASTEXITCODE -ne 0) {
    Fail "Visual C++ runtime 安装后仍不完整。请重启 Windows 后再次运行 REPAIR_WINDOWS.bat。"
  }
  Write-Host "[OK] Microsoft Visual C++ runtime installed/updated." -ForegroundColor Green
}

function Get-GpuInfo {
  if (-not (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) { return $null }
  try {
    $line = (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
    if (-not $line) { return $null }
    $parts = $line.ToString().Split(',')
    return [PSCustomObject]@{
      Name = $parts[0].Trim()
      Driver = if ($parts.Count -gt 1) { $parts[1].Trim() } else { "unknown" }
    }
  } catch { return $null }
}

function Test-Torch([string]$Py, [bool]$ExpectCuda) {
  if ($ExpectCuda) { & $Py scripts\windows_torch_probe.py --expect-cuda | Out-Host }
  else { & $Py scripts\windows_torch_probe.py | Out-Host }
  $ProbeExitCode = $LASTEXITCODE
  return ($ProbeExitCode -eq 0)
}

function Remove-Torch([string]$Py) {
  $PreviousErrorActionPreference = $ErrorActionPreference
  try {
    # Windows PowerShell 5.1 promotes harmless native stderr warnings (for an
    # optional package that is not installed) to NativeCommandError under Stop.
    $ErrorActionPreference = "Continue"
    & $Py -m pip uninstall -y torch torchvision torchaudio 2>$null | Out-Host
    $UninstallExitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $PreviousErrorActionPreference
  }
  if ($UninstallExitCode -ne 0) {
    throw "Unable to remove the existing PyTorch packages (exit code $UninstallExitCode)."
  }
}

function Install-TorchPlan([string]$Py, [string]$Index, [string]$TorchVersion, [string]$VisionVersion, [bool]$ExpectCuda) {
  Write-Host "[INFO] Trying torch=$TorchVersion torchvision=$VisionVersion" -ForegroundColor Cyan
  Write-Host "[INFO] Wheel channel: $Index" -ForegroundColor DarkGray
  Remove-Torch $Py
  & $Py -m pip install --no-cache-dir "torch==$TorchVersion" "torchvision==$VisionVersion" --index-url $Index
  if ($LASTEXITCODE -ne 0) { return $false }
  return (Test-Torch $Py $ExpectCuda)
}

Step "检测 Windows Python"
$PythonInfo = Get-CompatiblePython
if (-not $PythonInfo) { Fail "未找到 64 位 Python 3.11/3.12。" }
$BasePython = $PythonInfo.Path
Write-Host "[OK] $($PythonInfo.Version) (64-bit)" -ForegroundColor Green
Write-Host "[OK] executable: $BasePython" -ForegroundColor DarkGray
if (Test-Path $InstallMarker) { Remove-Item $InstallMarker -Force -ErrorAction SilentlyContinue }

Step "创建独立环境 .venv"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
  & $BasePython -m venv .venv
  if ($LASTEXITCODE -ne 0) { Fail "创建 .venv 失败" }
}
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$venvVersion = (& $Py --version 2>&1 | Select-Object -Last 1).ToString().Trim()
if ($venvVersion -notmatch '^Python\s+3\.(11|12)\.\d+') { Fail "现有 .venv 不是 Python 3.11/3.12。请删除 .venv 后重试。" }
Write-Host "[OK] venv: $venvVersion" -ForegroundColor Green

& $Py -m pip install -U pip wheel setuptools
if ($LASTEXITCODE -ne 0) { Fail "升级 pip 失败" }

Step "安装 Web / 视频基础环境"
& $Py -m pip install -r requirements-web.txt
if ($LASTEXITCODE -ne 0) { Fail "基础依赖安装失败" }
if ($Mode -eq "WEB") {
  Set-Content -Encoding ASCII -Path $InstallMarker -Value "WEB|$venvVersion"
  Write-Host "WEB mode ready. Existing/demo results can be shown; fresh AI inference is disabled." -ForegroundColor Yellow
  exit 0
}

Ensure-VCRuntime $Py

if ($Mode -eq "AUTO") {
  if (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue) { $Mode = "GPU" } else { $Mode = "CPU" }
  Write-Host "[INFO] AUTO resolved to: $Mode"
}

$Gpu = Get-GpuInfo
if ($Mode -eq "GPU") {
  Step "检测 NVIDIA GPU / 驱动"
  if (-not $Gpu) { Fail "选择了 GPU 模式，但 nvidia-smi 不可用。请更新 NVIDIA 驱动，或选择 CPU 模式。" }
  Write-Host "[OK] GPU: $($Gpu.Name)" -ForegroundColor Green
  Write-Host "[OK] Driver: $($Gpu.Driver)" -ForegroundColor Green
}

Step "验证已有 PyTorch"
$existingOk = Test-Torch $Py ($Mode -eq "GPU")
if ($existingOk) {
  Write-Host "[OK] Existing PyTorch is healthy; no re-download needed." -ForegroundColor Green
} else {
  Write-Host "[WARN] Existing PyTorch is not usable. A pinned official wheel will be installed." -ForegroundColor Yellow
  $ok = $false
  if ($TorchIndex) {
    $ok = Install-TorchPlan $Py $TorchIndex "2.7.1" "0.22.1" ($Mode -eq "GPU")
  } elseif ($Mode -eq "CPU") {
    $ok = Install-TorchPlan $Py "https://download.pytorch.org/whl/cpu" "2.7.1" "0.22.1" $false
  } else {
    $isBlackwell = $Gpu.Name -match '(RTX\s*50|Blackwell|B200|B100|GB10|GB20|PRO\s+6000)'
    $plans = if ($isBlackwell) {
      @(
        @("https://download.pytorch.org/whl/cu128","2.7.1","0.22.1","CUDA 12.8"),
        @("https://download.pytorch.org/whl/cu126","2.7.1","0.22.1","CUDA 12.6")
      )
    } else {
      @(
        @("https://download.pytorch.org/whl/cu126","2.7.1","0.22.1","CUDA 12.6"),
        @("https://download.pytorch.org/whl/cu118","2.7.1","0.22.1","CUDA 11.8"),
        @("https://download.pytorch.org/whl/cu128","2.7.1","0.22.1","CUDA 12.8")
      )
    }
    foreach ($plan in $plans) {
      Step "安装并验证 PyTorch $($plan[3])"
      if (Install-TorchPlan $Py $plan[0] $plan[1] $plan[2] $true) { $ok = $true; break }
    }
  }
  if (-not $ok) {
    Write-Host "[INFO] Diagnostic file: runtime\diagnostics\windows_torch_probe.json" -ForegroundColor Yellow
    Fail "PyTorch/CUDA 仍不可用。请把 runtime\diagnostics\windows_torch_probe.json 发给开发者；通常需要更新 NVIDIA 驱动或重启 Windows。"
  }
}

Step "安装完整 AI 分析依赖"
& $Py -m pip install -r requirements-ai.txt
if ($LASTEXITCODE -ne 0) { Fail "AI 依赖安装失败" }

Step "最终验证 PyTorch / CUDA"
if (-not (Test-Torch $Py ($Mode -eq "GPU"))) {
  Fail "最终 PyTorch/CUDA 验证失败。诊断文件：runtime\diagnostics\windows_torch_probe.json"
}

Step "运行系统检查"
& $Py scripts\system_check.py
if ($LASTEXITCODE -ne 0) { Fail "系统检查未通过，请查看上方 FAIL 项。" }

Set-Content -Encoding ASCII -Path $InstallMarker -Value "$Mode|$venvVersion"
Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "Football Insight Windows runtime is ready." -ForegroundColor Green
Write-Host "Next: double-click START_WINDOWS.bat" -ForegroundColor Green
Write-Host "============================================================`n" -ForegroundColor Green
