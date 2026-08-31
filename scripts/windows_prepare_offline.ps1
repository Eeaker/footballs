param(
  [ValidateSet("GPU","CPU")]
  [string]$Mode = "GPU"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Wheelhouse = Join-Path $Root "wheelhouse"
New-Item -ItemType Directory -Force -Path $Wheelhouse | Out-Null

function Fail($Text) { Write-Host "`n[ERROR] $Text" -ForegroundColor Red; exit 1 }
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
if (-not $PythonInfo) { Fail "未找到 Python 3.11/3.12。" }
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
if ($LASTEXITCODE -ne 0) { Fail "依赖下载失败" }

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
