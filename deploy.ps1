$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$envFile = Join-Path $root 'deploy\.env'
$example = Join-Path $root 'deploy\.env.example'

function Test-DockerEngine {
  # Windows PowerShell 5.1 promotes native stderr to NativeCommandError when
  # ErrorActionPreference is Stop. Let cmd.exe own both streams and inspect only
  # the exit code so an engine that is still starting remains a normal state.
  & $env:ComSpec /d /c 'docker info 1>nul 2>nul'
  return ($LASTEXITCODE -eq 0)
}

function Find-DockerDesktop {
  $candidates = @(
    (Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'),
    (Join-Path $env:LOCALAPPDATA 'Docker\Docker Desktop.exe')
  )
  foreach ($candidate in $candidates) {
    if ($candidate -and (Test-Path -LiteralPath $candidate)) { return $candidate }
  }
  return $null
}

function Start-DockerDesktop {
  $dockerDesktop = Find-DockerDesktop
  if (-not $dockerDesktop) {
    throw '已检测到 Docker 命令，但未找到 Docker Desktop。请重新安装 Docker Desktop 后重试。'
  }

  Write-Host 'Docker 引擎尚未运行，正在自动启动 Docker Desktop……' -ForegroundColor Yellow
  Start-Process -FilePath $dockerDesktop -WindowStyle Hidden
  $deadline = (Get-Date).AddMinutes(3)
  while ((Get-Date) -lt $deadline) {
    if (Test-DockerEngine) {
      Write-Progress -Activity '正在启动 Docker Desktop' -Completed
      Write-Host 'Docker Desktop 已就绪。' -ForegroundColor Green
      return
    }
    $remaining = [Math]::Max(0, [int]($deadline - (Get-Date)).TotalSeconds)
    Write-Progress -Activity '正在启动 Docker Desktop' -Status "等待 Linux 引擎就绪（剩余约 $remaining 秒）"
    Start-Sleep -Seconds 2
  }
  Write-Progress -Activity '正在启动 Docker Desktop' -Completed
  throw 'Docker Desktop 启动超时。请打开 Docker Desktop 完成 WSL2/许可初始化，必要时重启 Windows，然后再次双击部署文件。'
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    throw '未检测到 Docker Desktop，且系统没有 winget，无法自动安装。请安装 Docker Desktop 后重试；无需安装 Python。'
  }
  Write-Host '未检测到 Docker Desktop，正在自动安装……' -ForegroundColor Yellow
  winget install --id Docker.DockerDesktop -e --accept-package-agreements --accept-source-agreements
  if ($LASTEXITCODE -ne 0) { throw 'Docker Desktop 自动安装失败，请检查网络或管理员权限。' }
  $dockerBin = Join-Path $env:ProgramFiles 'Docker\Docker\resources\bin'
  if (Test-Path -LiteralPath $dockerBin) { $env:Path = "$dockerBin;$env:Path" }
  if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker Desktop 已安装，但当前 Windows 需要刷新环境或重启。重启后再次双击部署文件即可继续。'
  }
}
if (-not (Test-DockerEngine)) { Start-DockerDesktop }
if (-not (Test-Path $envFile)) {
  $pg = [Convert]::ToBase64String((1..24 | ForEach-Object { Get-Random -Maximum 256 })) -replace '[/+=]','x'
  $redis = [Convert]::ToBase64String((1..24 | ForEach-Object { Get-Random -Maximum 256 })) -replace '[/+=]','y'
  $minio = [Convert]::ToBase64String((1..24 | ForEach-Object { Get-Random -Maximum 256 })) -replace '[/+=]','z'
  (Get-Content -Raw $example).Replace('change-this-postgres-password',$pg).Replace('change-this-redis-password',$redis).Replace('change-this-minio-password',$minio) | Set-Content -Encoding UTF8 $envFile
}
$composeArgs = @('compose', '--env-file', $envFile, '-f', (Join-Path $root 'deploy\compose.yaml'))
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  $composeArgs += @('-f', (Join-Path $root 'deploy\compose.gpu.yaml'))
}
$composeArgs += @('up', '-d', '--build')
& docker @composeArgs
if ($LASTEXITCODE -ne 0) { throw '部署失败，请运行 docker compose logs app 查看原因。' }
Start-Process "http://localhost:8000"
Write-Host '部署完成：系统已打开。本机不需要安装 Python。' -ForegroundColor Green
