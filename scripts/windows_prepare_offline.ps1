param(
  [ValidateSet("GPU","CPU")]
  [string]$Mode = "GPU"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Wheelhouse = Join-Path $Root "wheelhouse"
$PythonInstallerName = "python-3.12.10-amd64.exe"
$PythonInstallerSha256 = "67B5635E80EA51072B87941312D00EC8927C4DB9BA18938F7AD2D27B328B95FB"
$PythonInstallerUrls = @(
  "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe",
  "https://npmmirror.com/mirrors/python/3.12.10/python-3.12.10-amd64.exe",
  "https://mirrors.huaweicloud.com/python/3.12.10/python-3.12.10-amd64.exe"
)
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}
$ProgressPreference = "SilentlyContinue"
New-Item -ItemType Directory -Force -Path $Wheelhouse | Out-Null

function Fail($Text) { Write-Host "`n[ERROR] $Text" -ForegroundColor Red; exit 1 }
function Step($Text) { Write-Host "`n[Football Insight] $Text" -ForegroundColor Cyan }

$PrivatePythonDir = Join-Path $Root "runtime\python"
$PrivatePython = Join-Path $PrivatePythonDir "python.exe"

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

function Get-CompatiblePython {
  $Candidates = @()
  if (Test-Path -LiteralPath $PrivatePython) { $Candidates += $PrivatePython }
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
    $info = Test-CompatiblePythonExe $candidate
    if ($info) { return $info }
  }
  return $null
}

function Install-PrivatePython([string]$Installer) {
  Step "安装私有 Python 3.12.10 用于打包离线依赖"
  if (Test-Path -LiteralPath $PrivatePythonDir) {
    Remove-Item -LiteralPath $PrivatePythonDir -Recurse -Force -ErrorAction SilentlyContinue
  }
  New-Item -ItemType Directory -Force -Path $PrivatePythonDir | Out-Null
  $proc = Start-Process -FilePath $Installer -ArgumentList @(
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

$PythonInfo = Get-CompatiblePython
if (-not $PythonInfo) {
  $bootstrapInstaller = Join-Path $Wheelhouse $PythonInstallerName
  $pythonOk = $false
  if (Test-Path -LiteralPath $bootstrapInstaller) {
    $hash = (Get-FileHash -LiteralPath $bootstrapInstaller -Algorithm SHA256).Hash
    if ($hash.ToUpperInvariant() -eq $PythonInstallerSha256.ToUpperInvariant()) { $pythonOk = $true }
  }
  if (-not $pythonOk) {
    foreach ($url in $PythonInstallerUrls) {
      Write-Host "[INFO] Downloading $url" -ForegroundColor DarkGray
      try {
        Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $bootstrapInstaller
        $hash = (Get-FileHash -LiteralPath $bootstrapInstaller -Algorithm SHA256).Hash
        if ($hash.ToUpperInvariant() -eq $PythonInstallerSha256.ToUpperInvariant()) { $pythonOk = $true; break }
      } catch {}
      if (Test-Path -LiteralPath $bootstrapInstaller) { Remove-Item -LiteralPath $bootstrapInstaller -Force }
    }
  }
  if (-not $pythonOk) { Fail "未找到 Python 3.11/3.12，且无法下载官方安装包。" }
  $PythonInfo = Install-PrivatePython $bootstrapInstaller
}
$BasePython = $PythonInfo.Path
Write-Host "[OK] $($PythonInfo.Version)" -ForegroundColor Green

Step "升级 pip 下载器"
& $BasePython -m pip install -U pip wheel setuptools
if ($LASTEXITCODE -ne 0) { Fail "pip 升级失败" }

$TorchIndex = if ($Mode -eq "GPU") { "https://download.pytorch.org/whl/cu126" } else { "https://download.pytorch.org/whl/cpu" }
Step "下载 Windows 依赖到 wheelhouse ($Mode)"
& $BasePython -m pip download --dest $Wheelhouse --extra-index-url $TorchIndex -r requirements-web.txt -r requirements-ai.txt "torch==2.7.1" "torchvision==0.22.1"
if ($LASTEXITCODE -ne 0) { Fail "依赖下载失败" }
try { Invoke-WebRequest -UseBasicParsing -Uri "https://aka.ms/vc14/vc_redist.x64.exe" -OutFile (Join-Path $Wheelhouse "vc_redist.x64.exe") } catch { Fail "VC++ runtime 下载失败" }

Step "下载官方 Python 3.12.10 安装包到 wheelhouse"
$pythonInstaller = Join-Path $Wheelhouse $PythonInstallerName
$pythonOk = $false
if (Test-Path -LiteralPath $pythonInstaller) {
  $hash = (Get-FileHash -LiteralPath $pythonInstaller -Algorithm SHA256).Hash
  if ($hash.ToUpperInvariant() -eq $PythonInstallerSha256.ToUpperInvariant()) { $pythonOk = $true }
  else { Remove-Item -LiteralPath $pythonInstaller -Force }
}
if (-not $pythonOk) {
  foreach ($url in $PythonInstallerUrls) {
    Write-Host "[INFO] Downloading $url" -ForegroundColor DarkGray
    try {
      Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $pythonInstaller
      $hash = (Get-FileHash -LiteralPath $pythonInstaller -Algorithm SHA256).Hash
      if ($hash.ToUpperInvariant() -eq $PythonInstallerSha256.ToUpperInvariant()) { $pythonOk = $true; break }
    } catch {}
    if (Test-Path -LiteralPath $pythonInstaller) { Remove-Item -LiteralPath $pythonInstaller -Force }
  }
}
if (-not $pythonOk) { Fail "Python 3.12.10 安装包下载或校验失败" }

$VersionLine = (& $BasePython --version 2>&1 | Select-Object -Last 1).ToString().Trim()
$Meta = @{
  schema_version = 1
  mode = $Mode
  torch_index = $TorchIndex
  python = $VersionLine
  generated_at = (Get-Date).ToString("o")
  wheel_count = @(Get-ChildItem $Wheelhouse -File).Count
}
$Meta | ConvertTo-Json -Depth 3 | Set-Content -Encoding UTF8 (Join-Path $Wheelhouse "OFFLINE_MANIFEST.json")
Write-Host "`n[OK] wheelhouse 已生成：$($Meta.wheel_count) 个文件" -ForegroundColor Green
Write-Host "如需完全无网运行，请另外把 models\yolov8x.pt 放入系统 models\ 目录。" -ForegroundColor Yellow
