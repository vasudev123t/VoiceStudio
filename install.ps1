<#
.SYNOPSIS
  VoiceStudio one-command source setup for native Windows.

.DESCRIPTION
  Installs the prerequisites (Git, Node 22+, Bun, uv, Rust, ffmpeg and the
  Visual Studio C++ build tools when missing) with winget, falling back to
  Chocolatey; clones the repository when needed; runs `bun install` and
  `bun run setup:api`; installs Applio for voice conversion (RVC) with its own
  run-install.bat plus `core.py prerequisites`; optionally downloads the
  starter models through the backend's model API; boots the backend once to
  verify it; and prints a summary. Safe to re-run: finished steps are skipped.

  Execution policy: if scripts are blocked, run it with
    powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
  or allow it for this window only:
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

  From the web (flags go after the script block):
    irm https://raw.githubusercontent.com/vasudev123t/VoiceStudio/main/install.ps1 | iex
    & ([scriptblock]::Create((irm https://raw.githubusercontent.com/vasudev123t/VoiceStudio/main/install.ps1))) -Yes

  Model licences: OmniVoice weights CC-BY-NC-4.0 (non-commercial, ~2.4 GB),
  KittenTTS Apache-2.0 (~80 MB), faster-whisper-small MIT (~0.5 GB). Applio MIT.

.EXAMPLE
  .\install.ps1 -Yes
.EXAMPLE
  .\install.ps1 -Cpu -SkipApplio -Models kittentts -NoLaunch
#>
[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingWriteHost', '', Justification = 'Interactive installer: colored console output is the UI.')]
[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseShouldProcessForStateChangingFunctions', '', Justification = 'Internal helpers of an installer the user runs explicitly.')]
[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSReviewUnusedParameter', 'Port', Justification = 'Read inside Start-Backend.')]
[CmdletBinding()]
param(
  # Answer yes to every prompt (including model downloads).
  [switch]$Yes,
  # Force CPU-only PyTorch (default: CUDA when nvidia-smi finds a GPU).
  [switch]$Cpu,
  # Force CUDA PyTorch.
  [switch]$Cuda,
  # Do not install Applio (voice conversion).
  [switch]$SkipApplio,
  # Do not download models.
  [switch]$SkipModels,
  # Comma list: omnivoice,kittentts,whisper-small, all, none.
  [string]$Models = '',
  # Clone/use VoiceStudio here (default: this checkout, else .\VoiceStudio).
  [string]$Dir = '',
  # Repository to clone.
  [string]$Repo = 'https://github.com/vasudev123t/VoiceStudio',
  # Branch to clone.
  [string]$Branch = '',
  # VoiceStudio data dir (default: OMNIVOICE_DATA_DIR or %APPDATA%\OmniVoice).
  [string]$DataDir = '',
  # Applio location (default: <data dir>\tools\applio, found automatically).
  [string]$ApplioDir = '',
  # Port for the verification backend (default: 3900 or the next free one).
  [int]$Port = 0,
  # Skip winget/choco installs.
  [switch]$SkipSystem,
  # Do not start `bun run dev` at the end.
  [switch]$NoLaunch,
  [switch]$Help
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch { Write-Verbose 'TLS 1.2 already enabled' }

$NodeMajorMin = 22
$DefaultBunVersion = '1.4.2'
$ApplioRepo = 'https://github.com/IAHispano/Applio'
$ModelCatalog = [ordered]@{
  'omnivoice'     = @{ Repo = 'k2-fsa/OmniVoice'; Size = '~2.4 GB'; License = 'weights CC-BY-NC-4.0 (non-commercial)' }
  'kittentts'     = @{ Repo = 'KittenML/kitten-tts-mini-0.8'; Size = '~80 MB'; License = 'Apache-2.0' }
  'whisper-small' = @{ Repo = 'Systran/faster-whisper-small'; Size = '~0.5 GB'; License = 'MIT' }
}

if ($Help) {
  Get-Help -Detailed $MyInvocation.MyCommand.Path -ErrorAction SilentlyContinue
  if (-not $MyInvocation.MyCommand.Path) { Write-Output 'See docs/install/full-setup.md for flags.' }
  return
}
if ($Cpu -and $Cuda) { throw 'Pick one of -Cpu or -Cuda.' }

# -- logging -------------------------------------------------------------
$script:StepNo = 0
function Write-Step([string]$Message) { $script:StepNo++; Write-Host ''; Write-Host ("==> [{0}] {1}" -f $script:StepNo, $Message) -ForegroundColor Cyan }
function Write-Info([string]$Message) { Write-Host "    $Message" }
function Write-Ok([string]$Message) { Write-Host '    OK ' -ForegroundColor Green -NoNewline; Write-Host $Message }
function Write-Warn2([string]$Message) { Write-Host "    ! $Message" -ForegroundColor Yellow }

function Confirm-Step([string]$Question, [bool]$DefaultYes = $true) {
  if ($Yes) { Write-Info "$Question -> yes (-Yes)"; return $true }
  $hint = if ($DefaultYes) { '[Y/n]' } else { '[y/N]' }
  if (-not [Environment]::UserInteractive -or [Console]::IsInputRedirected) {
    Write-Info "$Question -> $(if ($DefaultYes) { 'yes' } else { 'no' }) (non-interactive; pass -Yes to accept)"
    return $DefaultYes
  }
  $answer = Read-Host "    $Question $hint"
  if ([string]::IsNullOrWhiteSpace($answer)) { return $DefaultYes }
  return $answer.Trim().ToLower().StartsWith('y')
}

function Test-Command([string]$Name) { [bool](Get-Command $Name -ErrorAction SilentlyContinue) }

# Run a native command; throw on a non-zero exit code (PowerShell 5.1 does not).
function Invoke-Native {
  param([Parameter(Mandatory)][string]$FilePath, [string[]]$ArgumentList = @())
  Write-Host ("    $ {0} {1}" -f $FilePath, ($ArgumentList -join ' ')) -ForegroundColor DarkGray
  & $FilePath @ArgumentList
  if ($LASTEXITCODE -ne 0) { throw "$FilePath exited with code $LASTEXITCODE" }
}

# Run a native command for its stdout only, ignoring stderr and failures.
# (Windows PowerShell 5.1 turns redirected stderr into terminating errors
# under ErrorActionPreference=Stop, so relax it for this call.)
function Invoke-Quiet([scriptblock]$Command) {
  $ErrorActionPreference = 'Continue'
  try { & $Command 2>$null } catch { $null }
}

function Update-SessionPath {
  $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
  $user = [Environment]::GetEnvironmentVariable('Path', 'User')
  $extra = @(
    (Join-Path $env:USERPROFILE '.bun\bin'),
    (Join-Path $env:USERPROFILE '.local\bin'),
    (Join-Path $env:USERPROFILE '.cargo\bin'),
    (Join-Path $env:ProgramFiles 'Git\cmd'),
    (Join-Path $env:ProgramFiles 'nodejs')
  ) | Where-Object { Test-Path $_ }
  $parts = @($extra) + @(($machine + ';' + $user) -split ';') | Where-Object { $_ } | Select-Object -Unique
  $env:Path = $parts -join ';'
}

$script:HaveWinget = Test-Command 'winget'
$script:HaveChoco = Test-Command 'choco'

# winget first, Chocolatey as the fallback.
function Install-Prereq([string]$WingetId, [string]$ChocoId, [string[]]$WingetExtra = @(), [string[]]$ChocoExtra = @()) {
  if ($SkipSystem) { throw "$WingetId is missing and -SkipSystem is set; install it yourself." }
  $done = $false
  if ($script:HaveWinget) {
    $wargs = @('install', '--id', $WingetId, '-e', '--source', 'winget', '--accept-source-agreements', '--accept-package-agreements', '--silent') + $WingetExtra
    Write-Host "    $ winget $($wargs -join ' ')" -ForegroundColor DarkGray
    & winget @wargs
    # -1978335189 = already installed / no applicable upgrade.
    if ($LASTEXITCODE -eq 0 -or $LASTEXITCODE -eq -1978335189) { $done = $true }
    else { Write-Warn2 "winget failed for $WingetId (exit $LASTEXITCODE)" }
  }
  if (-not $done -and $ChocoId) {
    if (-not $script:HaveChoco) { throw "Could not install ${WingetId}: winget failed or is missing and Chocolatey is not installed (https://chocolatey.org/install)." }
    Invoke-Native choco (@('install', $ChocoId, '-y', '--no-progress') + $ChocoExtra)
    $done = $true
  }
  if (-not $done) { throw "Could not install $WingetId" }
  Update-SessionPath
}

function Get-NodeMajor {
  if (-not (Test-Command 'node')) { return 0 }
  $v = (& node --version) -replace '^v', ''
  return [int]($v.Split('.')[0])
}

function Test-VCToolset {
  $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
  if (-not (Test-Path $vswhere)) { return $false }
  $path = & $vswhere -products * -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  return -not [string]::IsNullOrWhiteSpace($path)
}

function Test-PortBusy([int]$P) {
  $client = New-Object Net.Sockets.TcpClient
  try { $client.Connect('127.0.0.1', $P); return $true } catch { return $false } finally { $client.Close() }
}

function Get-Prop($Object, [string]$Name, $Default = $null) {
  if ($null -ne $Object -and $Object.PSObject.Properties.Name -contains $Name) { return $Object.$Name }
  return $Default
}

# -- main ----------------------------------------------------------------
Write-Host "VoiceStudio source setup (Windows $([Environment]::OSVersion.Version), $env:PROCESSOR_ARCHITECTURE)" -ForegroundColor Green
if ($PSVersionTable.PSVersion.Major -lt 5) { throw 'PowerShell 5.1 or newer is required.' }

Write-Step 'Prerequisites (winget, choco fallback)'
Update-SessionPath
if (-not $script:HaveWinget -and -not $script:HaveChoco -and -not $SkipSystem) {
  throw 'Neither winget (App Installer from the Microsoft Store) nor Chocolatey is available.'
}

if (Test-Command 'git') { Write-Ok (& git --version) } else { Install-Prereq 'Git.Git' 'git'; Write-Ok (& git --version) }

if ((Get-NodeMajor) -ge $NodeMajorMin) { Write-Ok "Node $(& node --version)" }
else {
  Write-Info "Node $NodeMajorMin+ not found; installing Node.js LTS"
  Install-Prereq 'OpenJS.NodeJS.LTS' 'nodejs-lts'
  if ((Get-NodeMajor) -lt $NodeMajorMin) { throw "Node $NodeMajorMin+ still not on PATH (open a new terminal and re-run)." }
  Write-Ok "Node $(& node --version)"
}

if (Test-Command 'uv') { Write-Ok (& uv --version) } else { Install-Prereq 'astral-sh.uv' 'uv'; Write-Ok (& uv --version) }

if (Test-Command 'ffmpeg') { Write-Ok 'ffmpeg present' } else { Install-Prereq 'Gyan.FFmpeg' 'ffmpeg'; Write-Ok 'ffmpeg installed' }

if (Test-VCToolset) { Write-Ok 'Visual Studio C++ build tools present' }
elseif ($SkipSystem) { Write-Warn2 'Visual Studio C++ build tools not found (-SkipSystem); Rust and native modules may fail to build.' }
elseif (Confirm-Step 'Install the Visual Studio 2022 C++ build tools (~6 GB, needed by Rust and native modules)?' $true) {
  Install-Prereq 'Microsoft.VisualStudio.2022.BuildTools' 'visualstudio2022buildtools' `
    @('--override', '--wait --quiet --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended') `
    @('--package-parameters', '--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive --norestart')
  Write-Ok 'Visual Studio C++ build tools installed'
}
else { Write-Warn2 'Skipping the C++ build tools; Rust and native modules may fail to build.' }

if ((Test-Command 'cargo') -and (Test-Command 'rustc')) { Write-Ok (& rustc --version) }
else {
  Install-Prereq 'Rustlang.Rustup' 'rustup.install'
  if (Test-Command 'rustup') { & rustup default stable | Out-Null }
  Update-SessionPath
  if (-not (Test-Command 'cargo')) { throw 'cargo not on PATH after installing rustup (open a new terminal and re-run).' }
  Write-Ok (& rustc --version)
}

# -- repository ----------------------------------------------------------
Write-Step 'VoiceStudio source'
function Test-VoiceStudioRepo([string]$Path) {
  return $Path -and (Test-Path (Join-Path $Path 'package.json')) -and (Test-Path (Join-Path $Path 'backend')) -and (Test-Path (Join-Path $Path 'pyproject.toml'))
}
if (-not $Dir) {
  if ($PSScriptRoot -and (Test-VoiceStudioRepo $PSScriptRoot)) { $Dir = $PSScriptRoot }
  elseif (Test-VoiceStudioRepo (Get-Location).Path) { $Dir = (Get-Location).Path }
  else { $Dir = Join-Path (Get-Location).Path 'VoiceStudio' }
}
$Dir = [IO.Path]::GetFullPath($Dir)
if (Test-VoiceStudioRepo $Dir) { Write-Ok "using $Dir" }
elseif ((Test-Path $Dir) -and (Get-ChildItem -Force $Dir | Select-Object -First 1)) { throw "$Dir exists and is not a VoiceStudio checkout; pick another -Dir." }
else {
  $cloneArgs = @('clone', '--depth', '1')
  if ($Branch) { $cloneArgs += @('--branch', $Branch) }
  Invoke-Native git ($cloneArgs + @($Repo, $Dir))
  Write-Ok "cloned $Repo"
}
Set-Location $Dir

# Bun, pinned to package.json's packageManager.
$BunVersion = $DefaultBunVersion
$pkgJson = Get-Content (Join-Path $Dir 'package.json') -Raw | ConvertFrom-Json
$pm = Get-Prop $pkgJson 'packageManager' ''
if ($pm -match '^bun@(.+)$') { $BunVersion = $Matches[1] }
$haveBun = if (Test-Command 'bun') { (& bun --version).Trim() } else { '' }
if ($haveBun -eq $BunVersion) { Write-Ok "Bun $haveBun" }
else {
  Write-Info "Installing Bun $BunVersion (have: $(if ($haveBun) { $haveBun } else { 'none' }))"
  try {
    & ([scriptblock]::Create((Invoke-RestMethod 'https://bun.sh/install.ps1'))) -Version $BunVersion
  } catch {
    Write-Warn2 "bun.sh installer failed ($($_.Exception.Message)); trying the package manager"
    Install-Prereq 'Oven-sh.Bun' 'bun' @('--version', $BunVersion) @('--version', $BunVersion)
  }
  Update-SessionPath
  $haveBun = (& bun --version).Trim()
  if ($haveBun -ne $BunVersion) { Write-Warn2 "Bun $haveBun is active (wanted $BunVersion)" } else { Write-Ok "Bun $haveBun" }
}

# -- data dir / variant --------------------------------------------------
$CustomDataDir = [bool]$DataDir
if (-not $DataDir) { $DataDir = if ($env:OMNIVOICE_DATA_DIR) { $env:OMNIVOICE_DATA_DIR } else { Join-Path $env:APPDATA 'OmniVoice' } }
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
$DataDir = [IO.Path]::GetFullPath($DataDir)
$ManagedApplioDir = Join-Path $DataDir 'tools\applio'
if (-not $ApplioDir) { $ApplioDir = if ($env:OMNIVOICE_APPLIO_DIR) { $env:OMNIVOICE_APPLIO_DIR } else { $ManagedApplioDir } }
$ApplioDir = [IO.Path]::GetFullPath($ApplioDir)

$GpuName = ''
if (Test-Command 'nvidia-smi') {
  $GpuName = [string](Invoke-Quiet { & nvidia-smi --query-gpu=name --format=csv,noheader } | Select-Object -First 1)
  if ($LASTEXITCODE -ne 0) { $GpuName = '' }
}
$Variant = if ($Cpu) { 'cpu' } elseif ($Cuda) { 'cuda' } elseif ($GpuName) { 'cuda' } else { 'cpu' }

Write-Step 'JavaScript workspace (bun install)'
Invoke-Native bun @('install')

Write-Step "Python backend (bun run setup:api, PyTorch: $Variant)"
$env:OMNIVOICE_TORCH_VARIANT = $Variant
Invoke-Native bun @('run', 'setup:api')
$RepoPython = Join-Path $Dir '.venv\Scripts\python.exe'
$TorchVersion = Invoke-Quiet { & $RepoPython -c 'import torch; print(torch.__version__)' }
Write-Ok "backend environment ready (torch $TorchVersion)"

# -- Applio --------------------------------------------------------------
function Get-ApplioPython {
  foreach ($rel in @('env\python.exe', '.venv\Scripts\python.exe', 'env\Scripts\python.exe')) {
    $p = Join-Path $ApplioDir $rel
    if (Test-Path $p) { return $p }
  }
  return $null
}
function Test-ApplioHealthy {
  if (-not (Test-Path (Join-Path $ApplioDir 'core.py'))) { return $false }
  $py = Get-ApplioPython
  if (-not $py) { return $false }
  Push-Location $ApplioDir
  try { Invoke-Quiet { & $py -c 'import torch, faiss, librosa, soundfile' } | Out-Null; return ($LASTEXITCODE -eq 0) } finally { Pop-Location }
}

$ApplioReady = 'skipped'
Write-Step 'Applio (voice conversion)'
if ($SkipApplio) { Write-Info 'skipped (-SkipApplio)' }
else {
  Write-Info "target: $ApplioDir (MIT; ~6 GB with CUDA wheels, plus ~0.6 GB base models)"
  if (Test-ApplioHealthy) { Write-Ok 'Applio already installed' }
  else {
    if (-not (Test-Path (Join-Path $ApplioDir 'core.py'))) {
      if ((Test-Path $ApplioDir) -and (Get-ChildItem -Force $ApplioDir | Select-Object -First 1)) { throw "$ApplioDir exists but is not an Applio checkout; pick another -ApplioDir." }
      New-Item -ItemType Directory -Force -Path (Split-Path $ApplioDir) | Out-Null
      Invoke-Native git @('clone', '--depth', '1', $ApplioRepo, $ApplioDir)
    }
    Write-Info "running Applio's run-install.bat (Miniconda in %USERPROFILE%\Miniconda3, env in $ApplioDir\env)"
    # Applio's installer creates a conda env; accept the channel ToS non-interactively
    # and feed NUL so its closing `pause` returns.
    $env:CONDA_PLUGINS_AUTO_ACCEPT_TOS = 'yes'
    Push-Location $ApplioDir
    try {
      & cmd.exe /d /c 'run-install.bat < NUL'
      if ($LASTEXITCODE -ne 0) { throw "run-install.bat exited with code $LASTEXITCODE" }
    } finally { Pop-Location }
    if (-not (Test-ApplioHealthy)) { throw "Applio's environment is not importable (torch/faiss/librosa); see the output above." }
    Write-Ok 'Applio environment ready'
  }
  $ApplioPython = Get-ApplioPython
  Write-Info 'downloading Applio base models (core.py prerequisites; skips files already present)'
  Push-Location $ApplioDir
  try { Invoke-Native $ApplioPython @('core.py', 'prerequisites', '--no-pretraineds-hifigan', '--models', '--exe') } finally { Pop-Location }
  Write-Ok 'Applio prerequisites present'
  if ($ApplioDir -ne $ManagedApplioDir) {
    $env:OMNIVOICE_APPLIO_DIR = $ApplioDir
    if (Confirm-Step "Save OMNIVOICE_APPLIO_DIR=$ApplioDir as a user environment variable?" $true) {
      [Environment]::SetEnvironmentVariable('OMNIVOICE_APPLIO_DIR', $ApplioDir, 'User')
      Write-Ok 'OMNIVOICE_APPLIO_DIR saved (new terminals pick it up)'
    }
  }
}

# -- backend helpers -----------------------------------------------------
$script:Backend = $null
$script:BackendPort = 0
$script:BackendLog = Join-Path ([IO.Path]::GetTempPath()) 'voicestudio-backend.log'
function Start-Backend {
  if ($script:Backend) { return }
  $p = if ($Port -gt 0) { $Port } else { 3900 }
  if ($Port -gt 0) { if (Test-PortBusy $p) { throw "Port $p is in use; pick another -Port." } }
  else { while (Test-PortBusy $p) { $p++; if ($p -gt 3999) { throw 'No free port in 3900-3999.' } } }
  $script:BackendPort = $p
  $env:OMNIVOICE_PORT = "$p"
  if ($CustomDataDir) { $env:OMNIVOICE_DATA_DIR = $DataDir }
  Write-Info "starting backend on 127.0.0.1:$p (log: $script:BackendLog)"
  $script:Backend = Start-Process -FilePath 'uv' -WorkingDirectory $Dir -PassThru -NoNewWindow `
    -ArgumentList @('run', '--no-sync', 'uvicorn', 'main:app', '--app-dir', 'backend', '--host', '127.0.0.1', '--port', "$p") `
    -RedirectStandardOutput $script:BackendLog -RedirectStandardError "$script:BackendLog.err"
  for ($i = 0; $i -lt 180; $i++) {
    try { Invoke-RestMethod "http://127.0.0.1:$p/health" -TimeoutSec 5 | Out-Null; Write-Ok "backend up after ${i}s"; return } catch { Start-Sleep -Seconds 1 }
    if ($script:Backend.HasExited) {
      Get-Content "$script:BackendLog.err" -Tail 40 -ErrorAction SilentlyContinue | Write-Host
      $script:Backend = $null
      throw "Backend exited during startup (logs: $script:BackendLog[.err])."
    }
  }
  throw 'Backend did not answer /health within 180s.'
}
function Stop-Backend {
  if (-not $script:Backend) { return }
  # `uv` spawns uvicorn as a child; Windows has no signals, so kill the tree.
  $backendId = $script:Backend.Id
  Invoke-Quiet { & taskkill.exe /PID $backendId /T /F } | Out-Null
  $script:Backend = $null
}
function Invoke-Api([string]$Path, [string]$Method = 'GET', $Body = $null) {
  $uri = "http://127.0.0.1:$script:BackendPort$Path"
  if ($null -ne $Body) { return Invoke-RestMethod -Uri $uri -Method $Method -ContentType 'application/json' -Body ($Body | ConvertTo-Json -Compress) -TimeoutSec 30 }
  return Invoke-RestMethod -Uri $uri -Method $Method -TimeoutSec 30
}
function Test-ModelInstalled([string]$RepoId) {
  $list = Invoke-Api '/models'
  $m = @($list.models) | Where-Object { $_.repo_id -eq $RepoId } | Select-Object -First 1
  return [bool]($m -and (Get-Prop $m 'installed' $false))
}

$Installed = New-Object System.Collections.Generic.List[string]
$VcDevice = ''
try {
  # -- models ------------------------------------------------------------
  Write-Step 'Models'
  if ($SkipModels) { Write-Info 'skipped (-SkipModels)' }
  else {
    $selected = @()
    if (-not $Models) {
      Write-Info 'Starter models (downloaded from Hugging Face through the backend):'
      foreach ($k in $ModelCatalog.Keys) { Write-Info ('  {0,-14} {1,-8} {2}' -f $k, $ModelCatalog[$k].Size, $ModelCatalog[$k].License) }
      if (Confirm-Step 'Download OmniVoice (~2.4 GB, CC-BY-NC weights) and KittenTTS (~80 MB, Apache-2.0)?' $true) { $selected += @('omnivoice', 'kittentts') }
      if (-not $Yes -and (Confirm-Step 'Also download faster-whisper-small (~0.5 GB, MIT) for transcription?' $false)) { $selected += 'whisper-small' }
    }
    elseif ($Models -ne 'none') {
      $list = if ($Models -eq 'all') { @($ModelCatalog.Keys) } else { $Models.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ } }
      foreach ($k in $list) { if (-not $ModelCatalog.Contains($k)) { throw "Unknown model '$k' (use omnivoice, kittentts, whisper-small)." } }
      $summary = ($list | ForEach-Object { "$_ ($($ModelCatalog[$_].Size), $($ModelCatalog[$_].License))" }) -join '; '
      if (Confirm-Step "Download: ${summary}?" $true) { $selected = @($list) }
    }
    if ($selected.Count -eq 0) { Write-Info "no models selected; install them later from the app's Model Store" }
    else {
      Start-Backend
      foreach ($k in $selected) {
        $repoId = $ModelCatalog[$k].Repo
        if (Test-ModelInstalled $repoId) { Write-Ok "$repoId already installed"; $Installed.Add($repoId); continue }
        Write-Info "installing $repoId ($($ModelCatalog[$k].Size))"
        $resp = Invoke-Api '/models/install' 'POST' @{ repo_id = $repoId; target = 'local' }
        $status = Get-Prop $resp 'status' ''
        if (@('install_started', 'already_running', 'already_installed') -notcontains $status) { throw "Unexpected install response for ${repoId}: $($resp | ConvertTo-Json -Compress)" }
        $last = ''
        while ($true) {
          Start-Sleep -Seconds 3
          $jobs = @((Invoke-Api '/models/install/status').jobs) | Where-Object { $_.repo_id -eq $repoId }
          $job = $jobs | Select-Object -First 1
          $state = Get-Prop $job 'state' ''
          if ($state -eq 'failed') { throw "Download failed for ${repoId}: $($job | ConvertTo-Json -Compress)" }
          if ($state -ne 'downloading') { break }
          $done = [double](Get-Prop $job 'downloaded_bytes' 0); $total = [double](Get-Prop $job 'total_bytes' 0)
          $now = '{0:N0}/{1:N0} MB' -f ($done / 1e6), ($total / 1e6)
          if ($now -ne $last) { Write-Info "  $now"; $last = $now }
        }
        if (-not (Test-ModelInstalled $repoId)) { throw "$repoId did not finish installing (see $script:BackendLog)." }
        Write-Ok "$repoId installed"
        $Installed.Add($repoId)
      }
    }
  }

  # -- verify ------------------------------------------------------------
  Write-Step 'Verify backend'
  Start-Backend
  $health = Invoke-Api '/health'
  Write-Ok "/health: $(Get-Prop $health 'status' ($health | ConvertTo-Json -Compress))"
  $vc = Invoke-Api '/voice-conversion/status'
  $VcDevice = Get-Prop $vc 'device' ''
  $applioInfo = Get-Prop $vc 'applio' $null
  $ApplioReady = [string](Get-Prop $applioInfo 'ready' $false)
  Write-Ok "/voice-conversion/status: backend=$(Get-Prop $vc 'backend' '') device=$VcDevice applio_ready=$ApplioReady"
  if (-not $SkipApplio -and $ApplioReady -ne 'True') { Write-Warn2 "Applio is at $ApplioDir but the backend does not see it: $($applioInfo | ConvertTo-Json -Compress)" }
}
finally {
  Stop-Backend
}
Write-Ok 'verification backend stopped'

# -- summary -------------------------------------------------------------
Write-Step 'Summary'
$launchEnv = @()
if ($CustomDataDir) { $launchEnv += "`$env:OMNIVOICE_DATA_DIR='$DataDir'" }
if (-not $SkipApplio -and $ApplioDir -ne $ManagedApplioDir) { $launchEnv += "`$env:OMNIVOICE_APPLIO_DIR='$ApplioDir'" }
$launch = (@("cd '$Dir'") + $launchEnv + @('bun run dev')) -join '; '
Write-Info "OS            Windows $([Environment]::OSVersion.Version) ($env:PROCESSOR_ARCHITECTURE)"
Write-Info "Node          $(& node --version)   Bun $(& bun --version)   $(& uv --version)"
Write-Info "Rust          $(& rustc --version)"
Write-Info "PyTorch       $Variant (torch $TorchVersion)$(if ($GpuName) { "  GPU: $GpuName" })"
Write-Info "VC device     $(if ($VcDevice) { $VcDevice } else { 'n/a' })"
Write-Info "Repository    $Dir"
Write-Info "Data dir      $DataDir"
Write-Info "Applio        $(if ($SkipApplio) { 'skipped' } else { "$ApplioDir (backend sees it: $ApplioReady)" })"
Write-Info "Models        $(if ($Installed.Count) { $Installed -join ', ' } else { 'none downloaded (use the in-app Model Store)' })"
Write-Host ''
Write-Host '    Launch: ' -ForegroundColor Green -NoNewline; Write-Host $launch

if (-not $NoLaunch -and (Confirm-Step 'Start VoiceStudio now (bun run dev)?' $true)) {
  Invoke-Native bun @('run', 'dev')
}
