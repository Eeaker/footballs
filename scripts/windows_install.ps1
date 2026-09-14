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
$PrivatePythonDir = Join-Path $Root "runtime\python"
$PrivatePython = Join-Path $PrivatePythonDir "python.exe"
$PythonInstallerName = "python-3.12.10-amd64.exe"
$PythonInstallerSha256 = "67B5635E80EA51072B87941312D00EC8927C4DB9BA18938F7AD2D27B328B95FB"
$PythonInstallerUrls = @(
  "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe",
  "https://npmmirror.com/mirrors/python/3.12.10/python-3.12.10-amd64.exe",
  "https://mirrors.huaweicloud.com/python/3.12.10/python-3.12.10-amd64.exe"
)
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}
$ProgressPreference = "SilentlyContinue"

function Step($Text) { Write-Host "`n[Football Insight] $Text" -ForegroundColor Cyan }
function Fail($Text) {
  Write-Host "`n[ERROR] $Text" -ForegroundColor Red
  if (Test-Path $InstallMarker) { Remove-Item $InstallMarker -Force -ErrorAction SilentlyContinue }
  exit 1
}

function Test-CompatiblePythonExe([string]$Exe) {
  if (-not $Exe -or -not (Test-Path -LiteralPath $Exe)) { return $null }
  try {
    $versionLine = (& $Exe --version 2>&1 | Select-Object -Last 1)
    if ($LASTEXITCODE -ne 0 -or -not $versionLine) { return $null }
    $versionText = $versionLine.ToString().Trim()
    if ($versionText -notmatch '^Python\s+3\.(11|12)\.\d+') { return $null }
    $bits = (& $Exe -c "import struct;print(struct.calcsize('P')*8)" 2>$null | Select-Object -Last 1)
    if (-not $bits -or $bits.ToString().Trim() -ne "64") { return $null }
    return [PSCustomObject]@{ Path = $Exe; Version = $versionText }
  } catch {
    return $null
  }
}

function Get-PrivatePython {
  return (Test-CompatiblePythonExe $PrivatePython)
}

function Ensure-Pip([string]$Exe) {
  & $Exe -m pip --version *> $null
  if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] pip: $Exe -m pip" -ForegroundColor Green
    return
  }
  Write-Host "[INFO] pip is missing. Bootstrapping with ensurepip (system pip is not required)." -ForegroundColor Yellow
  & $Exe -m ensurepip --upgrade
  if ($LASTEXITCODE -ne 0) { Fail "ensurepip 失败。本机不需要预装 pip，但官方 Python 必须能自举 pip。" }
  & $Exe -m pip --version *> $null
  if ($LASTEXITCODE -ne 0) { Fail "pip 自举后仍不可用。" }
  Write-Host "[OK] pip bootstrapped: $Exe -m pip" -ForegroundColor Green
}

function Get-OfficialFile([string]$OutFile, [string[]]$Urls, [string]$ExpectedSha256) {
  if (Test-Path -LiteralPath $OutFile) {
    $hash = (Get-FileHash -LiteralPath $OutFile -Algorithm SHA256).Hash
    if ($hash.ToUpperInvariant() -eq $ExpectedSha256.ToUpperInvariant()) { return }
    Remove-Item -LiteralPath $OutFile -Force
  }
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutFile) | Out-Null
  foreach ($url in $Urls) {
    Write-Host "[INFO] Downloading $url" -ForegroundColor DarkGray
    try {
      Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $OutFile
      $hash = (Get-FileHash -LiteralPath $OutFile -Algorithm SHA256).Hash
      if ($hash.ToUpperInvariant() -eq $ExpectedSha256.ToUpperInvariant()) { return }
      Write-Host "[WARN] Checksum mismatch from $url" -ForegroundColor Yellow
    } catch {
      Write-Host "[WARN] Download failed from $url" -ForegroundColor Yellow
    }
    if (Test-Path -LiteralPath $OutFile) { Remove-Item -LiteralPath $OutFile -Force }
  }
  Fail "无法下载校验通过的 $([IO.Path]::GetFileName($OutFile))。请联网后重试。"
}

function Install-PrivatePython {
  Step "本机不依赖系统 Python/pip，正在安装独立 Python 3.12.10"
  $installer = Join-Path $BootstrapDir $PythonInstallerName
  Get-OfficialFile $installer $PythonInstallerUrls $PythonInstallerSha256
  if (Test-Path -LiteralPath $PrivatePythonDir) {
    Remove-Item -LiteralPath $PrivatePythonDir -Recurse -Force -ErrorAction SilentlyContinue
  }
  New-Item -ItemType Directory -Force -Path $PrivatePythonDir | Out-Null
  try {
    $proc = Start-Process -FilePath $installer -ArgumentList @(
      "/quiet","InstallAllUsers=0","PrependPath=0","Include_test=0","Include_doc=0",
      "Include_launcher=0","Include_tcltk=0","Shortcuts=0","AssociateFiles=0",
      "Include_pip=1","CompileAll=0","TargetDir=$PrivatePythonDir"
    ) -Wait -PassThru
    if ($proc.ExitCode -ne 0 -and $proc.ExitCode -ne 3010) {
      Fail "私有 Python 安装失败，exit code=$($proc.ExitCode)"
    }
  } catch {
    Fail "私有 Python 安装失败：$($_.Exception.Message)"
  }
  $info = Test-CompatiblePythonExe $PrivatePython
  if (-not $info) {
    Fail "私有 Python 安装后仍不可用。请删除 runtime\python 后重试，或检查杀毒软件拦截。"
  }
  Write-Host "[OK] Private Python: $($info.Version)" -ForegroundColor Green
  Write-Host "[OK] location: $PrivatePython" -ForegroundColor DarkGray
  return $info
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

Step "从零检查 Python / pip（不使用本机环境）"
$PythonInfo = Get-PrivatePython
if (-not $PythonInfo) { $PythonInfo = Install-PrivatePython }
$BasePython = $PythonInfo.Path
Write-Host "[OK] $($PythonInfo.Version) (64-bit)" -ForegroundColor Green
Write-Host "[OK] executable: $BasePython" -ForegroundColor DarkGray
Ensure-Pip $BasePython
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
Ensure-Pip $Py

& $Py -m pip install -U pip wheel setuptools
if ($LASTEXITCODE -ne 0) { Fail "升级 pip 失败。请确认已联网；不需要本机预装 pip。" }

Step "安装 Web / 视频基础环境"
& $Py -m pip install -r requirements-web.txt
if ($LASTEXITCODE -ne 0) { Fail "基础依赖安装失败" }
if ($Mode -eq "WEB") {
  Set-Content -Encoding ASCII -Path $InstallMarker -Value "WEB|$venvVersion"
  Write-Host "WEB mode ready. Existing/demo results can be shown; fresh AI inference is disabled." -ForegroundColor Yellow
  exit 0
}

Ensure-VCRuntime $Py

$RequestedMode = $Mode
if ($Mode -eq "AUTO") {
  if (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue) { $Mode = "GPU" } else { $Mode = "CPU" }
  Write-Host "[INFO] AUTO resolved to: $Mode"
}

$Gpu = Get-GpuInfo
if ($Mode -eq "GPU") {
  Step "检测 NVIDIA GPU / 驱动"
  if (-not $Gpu) {
    if ($RequestedMode -eq "AUTO") {
      Write-Host "[WARN] 检测到 NVIDIA 命令但无法读取 GPU，自动回退到 CPU。" -ForegroundColor Yellow
      $Mode = "CPU"
    } else {
      Fail "选择了 GPU 模式，但 nvidia-smi 不可用。请更新 NVIDIA 驱动，或重新运行一键部署让系统自动选择 CPU。"
    }
  } else {
    Write-Host "[OK] GPU: $($Gpu.Name)" -ForegroundColor Green
    Write-Host "[OK] Driver: $($Gpu.Driver)" -ForegroundColor Green
  }
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
  if (-not $ok -and $RequestedMode -eq "AUTO" -and $Mode -eq "GPU") {
    Write-Host "[WARN] GPU PyTorch 不可用，自动回退到 CPU，保证一键部署能完成。" -ForegroundColor Yellow
    $Mode = "CPU"
    $ok = Install-TorchPlan $Py "https://download.pytorch.org/whl/cpu" "2.7.1" "0.22.1" $false
  }
  if (-not $ok) {
    Write-Host "[INFO] Diagnostic file: runtime\diagnostics\windows_torch_probe.json" -ForegroundColor Yellow
    Fail "PyTorch/CUDA 仍不可用。请把 runtime\diagnostics\windows_torch_probe.json 发给开发者；通常需要更新 NVIDIA 驱动、联网后重试，或重启 Windows。"
  }
}

Step "安装完整 AI 分析依赖"
& $Py -m pip install -r requirements-ai.txt
if ($LASTEXITCODE -ne 0) { Fail "AI 依赖安装失败" }

Step "最终验证 PyTorch / CUDA"
if (-not (Test-Torch $Py ($Mode -eq "GPU"))) {
  if ($RequestedMode -eq "AUTO" -and $Mode -eq "GPU") {
    Write-Host "[WARN] GPU 最终验证失败，自动回退到 CPU。" -ForegroundColor Yellow
    $Mode = "CPU"
    if (-not (Install-TorchPlan $Py "https://download.pytorch.org/whl/cpu" "2.7.1" "0.22.1" $false)) {
      Fail "CPU PyTorch 回退仍失败。诊断文件：runtime\diagnostics\windows_torch_probe.json"
    }
  } else {
    Fail "最终 PyTorch/CUDA 验证失败。诊断文件：runtime\diagnostics\windows_torch_probe.json"
  }
}

Step "运行系统检查"
& $Py scripts\system_check.py
if ($LASTEXITCODE -ne 0) { Fail "系统检查未通过，请查看上方 FAIL 项。" }

Step "准备默认检测模型"
& $Py scripts\install_default_model.py
if ($LASTEXITCODE -ne 0) {
  Write-Host "[WARN] 默认模型未下载完成。可稍后双击 DOWNLOAD_MODEL_WINDOWS.bat，或在系统状态页上传 yolov8x.pt。" -ForegroundColor Yellow
}

try {
  $desktop = [Environment]::GetFolderPath("Desktop")
  if ($desktop) {
    $wsh = New-Object -ComObject WScript.Shell
    $lnk = $wsh.CreateShortcut((Join-Path $desktop "Football Insight.lnk"))
    $lnk.TargetPath = Join-Path $Root "DEPLOY_ONE_CLICK_WINDOWS.bat"
    $lnk.WorkingDirectory = $Root
    $lnk.WindowStyle = 1
    $lnk.Description = "Football Insight"
    $lnk.Save()
    Write-Host "[OK] Desktop shortcut: Football Insight.lnk" -ForegroundColor Green
  }
} catch {
  Write-Host "[INFO] Desktop shortcut was skipped." -ForegroundColor DarkGray
}

Set-Content -Encoding ASCII -Path $InstallMarker -Value "$Mode|$venvVersion"
Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "Football Insight Windows runtime is ready. Docker is not required." -ForegroundColor Green
Write-Host "One-click deploy will start the app automatically." -ForegroundColor Green
Write-Host "============================================================`n" -ForegroundColor Green
