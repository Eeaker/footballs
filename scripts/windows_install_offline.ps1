$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Wheelhouse = Join-Path $Root "wheelhouse"
$InstallMarker = Join-Path $Root ".venv\football_insight_install.mode"
$PrivatePythonDir = Join-Path $Root "runtime\python"
$PrivatePython = Join-Path $PrivatePythonDir "python.exe"
$PythonInstallerName = "python-3.12.10-amd64.exe"
$PythonInstallerSha256 = "67B5635E80EA51072B87941312D00EC8927C4DB9BA18938F7AD2D27B328B95FB"
if (-not (Test-Path $Wheelhouse)) { Write-Host "[ERROR] wheelhouse 不存在" -ForegroundColor Red; exit 1 }

function Fail($Text) {
  Write-Host "`n[ERROR] $Text" -ForegroundColor Red
  if (Test-Path $InstallMarker) { Remove-Item $InstallMarker -Force -ErrorAction SilentlyContinue }
  exit 1
}
function Step($Text) { Write-Host "`n[Football Insight] $Text" -ForegroundColor Cyan }

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

function Ensure-Pip([string]$Exe) {
  & $Exe -m pip --version *> $null
  if ($LASTEXITCODE -eq 0) { return }
  & $Exe -m ensurepip --upgrade
  if ($LASTEXITCODE -ne 0) { Fail "ensurepip 失败。离线包不依赖本机 pip。" }
  & $Exe -m pip --version *> $null
  if ($LASTEXITCODE -ne 0) { Fail "pip 自举后仍不可用。" }
}

function Install-PrivatePythonFromWheelhouse {
  $installer = Join-Path $Wheelhouse $PythonInstallerName
  if (-not (Test-Path -LiteralPath $installer)) {
    Fail "wheelhouse 中没有 $PythonInstallerName。请在联网电脑重新运行 PREPARE_OFFLINE_WINDOWS.bat。"
  }
  $hash = (Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash
  if ($hash.ToUpperInvariant() -ne $PythonInstallerSha256.ToUpperInvariant()) {
    Fail "$PythonInstallerName 校验失败。请重新运行 PREPARE_OFFLINE_WINDOWS.bat。"
  }
  Step "从 wheelhouse 安装私有 Python 3.12.10"
  if (Test-Path -LiteralPath $PrivatePythonDir) {
    Remove-Item -LiteralPath $PrivatePythonDir -Recurse -Force -ErrorAction SilentlyContinue
  }
  New-Item -ItemType Directory -Force -Path $PrivatePythonDir | Out-Null
  $proc = Start-Process -FilePath $installer -ArgumentList @(
    "/quiet","InstallAllUsers=0","PrependPath=0","Include_test=0","Include_doc=0",
    "Include_launcher=0","Include_tcltk=0","Shortcuts=0","AssociateFiles=0",
    "Include_pip=1","CompileAll=0","TargetDir=$PrivatePythonDir"
  ) -Wait -PassThru
  if ($proc.ExitCode -ne 0 -and $proc.ExitCode -ne 3010) {
    Fail "私有 Python 安装失败，exit code=$($proc.ExitCode)"
  }
  $info = Test-CompatiblePythonExe $PrivatePython
  if (-not $info) { Fail "私有 Python 安装后仍不可用。" }
  return $info
}

$PythonInfo = Test-CompatiblePythonExe $PrivatePython
if (-not $PythonInfo) { $PythonInfo = Install-PrivatePythonFromWheelhouse }
$BasePython = $PythonInfo.Path
Write-Host "[OK] $($PythonInfo.Version)" -ForegroundColor Green
Ensure-Pip $BasePython
if (Test-Path $InstallMarker) { Remove-Item $InstallMarker -Force -ErrorAction SilentlyContinue }

Step "创建本地 .venv"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
  & $BasePython -m venv .venv
  if ($LASTEXITCODE -ne 0) { Fail "创建 .venv 失败" }
}
$Py = Join-Path $Root ".venv\Scripts\python.exe"
Ensure-Pip $Py

Step "安装/更新 Microsoft Visual C++ x64 运行库"
$VcInstaller = Join-Path $Wheelhouse "vc_redist.x64.exe"
if (Test-Path $VcInstaller) {
  $proc = Start-Process -FilePath $VcInstaller -ArgumentList @("/install","/quiet","/norestart") -Wait -PassThru
  if ($proc.ExitCode -ne 0 -and $proc.ExitCode -ne 1638 -and $proc.ExitCode -ne 3010) { Fail "VC++ runtime 安装失败" }
}

Step "从本地 wheelhouse 安装全部依赖"
& $Py -m pip install --no-index --find-links $Wheelhouse pip wheel setuptools
if ($LASTEXITCODE -ne 0) { Fail "离线基础工具安装失败" }
& $Py -m pip install --no-index --find-links $Wheelhouse -r requirements-web.txt -r requirements-ai.txt "torch==2.7.1" "torchvision==0.22.1"
if ($LASTEXITCODE -ne 0) { Fail "离线依赖安装失败；请确认 wheelhouse 与目标 Python 版本/架构一致。" }

Step "验证 PyTorch"
& $Py scripts\windows_torch_probe.py
if ($LASTEXITCODE -ne 0) { Fail "PyTorch 导入失败。请查看 runtime\diagnostics\windows_torch_probe.json" }

Step "执行正式系统检查"
& $Py scripts\system_check.py
if ($LASTEXITCODE -ne 0) { Fail "系统检查未通过" }

$venvVersion = (& $Py --version 2>&1 | Select-Object -Last 1).ToString().Trim()
Set-Content -Encoding ASCII -Path $InstallMarker -Value "OFFLINE|$venvVersion"
Write-Host "`n[OK] Football Insight 离线环境安装完成。" -ForegroundColor Green
if (-not (Test-Path "models\yolov8x.pt")) {
  Write-Host "[WARN] models\yolov8x.pt 仍不存在。无网机器请提前把模型文件复制进来。" -ForegroundColor Yellow
}
