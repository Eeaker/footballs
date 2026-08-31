$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Wheelhouse = Join-Path $Root "wheelhouse"
$InstallMarker = Join-Path $Root ".venv\football_insight_install.mode"
if (-not (Test-Path $Wheelhouse)) { Write-Host "[ERROR] wheelhouse 不存在" -ForegroundColor Red; exit 1 }

function Fail($Text) {
  Write-Host "`n[ERROR] $Text" -ForegroundColor Red
  if (Test-Path $InstallMarker) { Remove-Item $InstallMarker -Force -ErrorAction SilentlyContinue }
  exit 1
}
function Step($Text) { Write-Host "`n[Football Insight] $Text" -ForegroundColor Cyan }

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
        return [PSCustomObject]@{ Path = $candidate; Version = $versionText }
      }
    } catch {}
  }
  return $null
}

$PythonInfo = Get-CompatiblePython
if (-not $PythonInfo) { Fail "未找到 Python 3.11/3.12。Python 本体需提前安装。" }
$BasePython = $PythonInfo.Path
Write-Host "[OK] $($PythonInfo.Version)" -ForegroundColor Green
if (Test-Path $InstallMarker) { Remove-Item $InstallMarker -Force -ErrorAction SilentlyContinue }

Step "创建本地 .venv"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
  & $BasePython -m venv .venv
  if ($LASTEXITCODE -ne 0) { Fail "创建 .venv 失败" }
}
$Py = Join-Path $Root ".venv\Scripts\python.exe"


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
