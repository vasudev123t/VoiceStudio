#!/usr/bin/env bash
# VoiceStudio: one-command source setup for Linux, macOS and WSL2.
#
# Installs the build prerequisites and toolchains (Node 22+, Bun, uv, Rust),
# clones the repository when needed, installs the JS workspace and the Python
# backend, sets up Applio for voice conversion (RVC), optionally downloads the
# starter models through the backend's own model API, and verifies a backend
# boot. Safe to re-run: every step checks what is already present, so an
# interrupted run (closed terminal, dropped connection) can just be started
# again with the same flags.
#
#   curl -fsSL https://raw.githubusercontent.com/vasudev123t/VoiceStudio/main/install.sh | bash
#   curl -fsSL https://raw.githubusercontent.com/vasudev123t/VoiceStudio/main/install.sh | bash -s -- --yes
#   ./install.sh --help
#
# Native Windows: use install.ps1 (Git Bash hands off to it automatically).
# This is the source/dev setup; scripts/install.sh installs packaged releases.

set -Eeuo pipefail

# ── defaults ─────────────────────────────────────────────────────────────
REPO_URL="https://github.com/vasudev123t/VoiceStudio"
BRANCH=""
APPLIO_REPO="https://github.com/IAHispano/Applio"
ASSUME_YES=0
TORCH_VARIANT="auto"
SKIP_APPLIO=0
SKIP_MODELS=0
SKIP_SYSTEM=0
MODELS=""
TARGET_DIR=""
NO_LAUNCH=0
DATA_DIR_ARG=""
APPLIO_DIR_ARG=""
PORT_ARG=""
CACHE_DIR_ARG=""
APPLIO_INSTALL_ATTEMPTS=3
APPLIO_RETRY_DELAY=${APPLIO_RETRY_DELAY:-10}
NODE_MAJOR_MIN=22
DEFAULT_BUN_VERSION="1.4.2"

# Starter models: key|repo id|size|licence
MODEL_CATALOG=(
  "omnivoice|k2-fsa/OmniVoice|~2.4 GB|weights CC-BY-NC-4.0 (non-commercial)"
  "kittentts|KittenML/kitten-tts-mini-0.8|~80 MB|Apache-2.0"
  "whisper-small|Systran/faster-whisper-small|~0.5 GB|MIT"
)

# ── logging ──────────────────────────────────────────────────────────────
if [[ -t 1 && "${NO_COLOR:-}" == "" ]]; then
  C_BLUE=$'\033[1;34m'; C_GREEN=$'\033[1;32m'; C_YELLOW=$'\033[1;33m'
  C_RED=$'\033[1;31m'; C_DIM=$'\033[2m'; C_RESET=$'\033[0m'
else
  C_BLUE=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_DIM=""; C_RESET=""
fi
STEP_NO=0
step() { STEP_NO=$((STEP_NO + 1)); printf '\n%s==> [%d] %s%s\n' "$C_BLUE" "$STEP_NO" "$*" "$C_RESET"; }
info() { printf '    %s\n' "$*"; }
ok()   { printf '    %s✓%s %s\n' "$C_GREEN" "$C_RESET" "$*"; }
warn() { printf '    %s!%s %s\n' "$C_YELLOW" "$C_RESET" "$*" >&2; }
die()  { printf '\n%s✗ %s%s\n' "$C_RED" "$*" "$C_RESET" >&2; exit 1; }
run()  { printf '    %s$ %s%s\n' "$C_DIM" "$*" "$C_RESET"; "$@"; }

on_error() {
  local code=$? line=${1:-?}
  printf '\n%s✗ install.sh failed (exit %s) near line %s.%s\n' "$C_RED" "$code" "$line" "$C_RESET" >&2
  printf '  Fix the error above and re-run; finished steps are skipped.\n' >&2
}
trap 'on_error $LINENO' ERR

usage() {
  cat <<'USAGE'
VoiceStudio full source setup (Linux, macOS, WSL2)

Usage: install.sh [options]

  -y, --yes            Answer yes to every prompt (including model downloads)
      --cpu            Install CPU-only PyTorch
      --cuda           Install NVIDIA CUDA PyTorch
      --rocm           Install AMD ROCm PyTorch (Linux)
                       (default: auto; CUDA when nvidia-smi finds a GPU)
      --skip-applio    Do not install Applio (voice conversion)
      --skip-models    Do not download any models
      --models LIST    Comma list: omnivoice,kittentts,whisper-small, all, none
                       (default: omnivoice,kittentts; whisper-small is offered)
      --dir PATH       Clone/use VoiceStudio at PATH
                       (default: this checkout, else ./VoiceStudio)
      --repo URL       Repository to clone (default: vasudev123t/VoiceStudio)
      --branch NAME    Branch to clone or check out (default: repo default)
      --data-dir PATH  VoiceStudio data dir (default: OMNIVOICE_DATA_DIR or
                       the platform default, ~/.omnivoice on Linux)
      --applio-dir PATH  Applio location (default: <data dir>/tools/applio,
                       which the app finds without extra settings)
      --cache-dir PATH Put the Hugging Face, uv and pip download caches under
                       PATH (sets HF_HOME, UV_CACHE_DIR, PIP_CACHE_DIR)
      --port N         Port for the verification backend (default: 3900, or
                       the next free port)
      --skip-system    Skip OS package installs (no sudo)
      --no-launch      Do not start `bun run dev` at the end
  -h, --help           Show this help

Model licences: OmniVoice weights are CC-BY-NC-4.0 (non-commercial),
KittenTTS is Apache-2.0, faster-whisper-small is MIT. Applio is MIT.

Slow connections: UV_HTTP_TIMEOUT and UV_HTTP_RETRIES default to 600 s and 5
for this run (values you set are kept); Applio's dependency download is
retried up to 3 times. An interrupted run can be re-run safely.
USAGE
}

# ── args ─────────────────────────────────────────────────────────────────
need_value() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$1 needs a value (see --help)"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes) ASSUME_YES=1 ;;
    --cpu) TORCH_VARIANT="cpu" ;;
    --cuda) TORCH_VARIANT="cuda" ;;
    --rocm) TORCH_VARIANT="rocm" ;;
    --skip-applio) SKIP_APPLIO=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --skip-system) SKIP_SYSTEM=1 ;;
    --models) need_value "$@"; MODELS="$2"; shift ;;
    --models=*) MODELS="${1#*=}" ;;
    --dir) need_value "$@"; TARGET_DIR="$2"; shift ;;
    --dir=*) TARGET_DIR="${1#*=}" ;;
    --repo) need_value "$@"; REPO_URL="$2"; shift ;;
    --repo=*) REPO_URL="${1#*=}" ;;
    --branch) need_value "$@"; BRANCH="$2"; shift ;;
    --branch=*) BRANCH="${1#*=}" ;;
    --data-dir) need_value "$@"; DATA_DIR_ARG="$2"; shift ;;
    --data-dir=*) DATA_DIR_ARG="${1#*=}" ;;
    --applio-dir) need_value "$@"; APPLIO_DIR_ARG="$2"; shift ;;
    --applio-dir=*) APPLIO_DIR_ARG="${1#*=}" ;;
    --port) need_value "$@"; PORT_ARG="$2"; shift ;;
    --port=*) PORT_ARG="${1#*=}" ;;
    --cache-dir) need_value "$@"; CACHE_DIR_ARG="$2"; shift ;;
    --cache-dir=*) CACHE_DIR_ARG="${1#*=}" ;;
    --no-launch) NO_LAUNCH=1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1 (see --help)" ;;
  esac
  shift
done
if [[ -n "$PORT_ARG" && ! "$PORT_ARG" =~ ^[0-9]+$ ]]; then die "--port must be a number"; fi

# ── helpers ──────────────────────────────────────────────────────────────
have() { command -v "$1" >/dev/null 2>&1; }

tty_ok() { { : </dev/tty; } 2>/dev/null; }

# confirm "question" [default y|n]; --yes answers yes, no terminal answers the default.
confirm() {
  local question=$1 default=${2:-n} hint answer
  if [[ $ASSUME_YES == 1 ]]; then info "$question -> yes (--yes)"; return 0; fi
  if [[ $default == y ]]; then hint="[Y/n]"; else hint="[y/N]"; fi
  if ! tty_ok; then
    info "$question -> $default (no terminal; pass --yes to accept)"
    [[ $default == y ]]; return
  fi
  read -r -p "    $question $hint " answer </dev/tty || answer=""
  answer=${answer:-$default}
  [[ $answer =~ ^[Yy] ]]
}

SUDO=()
as_root() {
  if [[ $(id -u) -eq 0 ]]; then "$@"; return; fi
  if [[ ${#SUDO[@]} -eq 0 ]]; then
    have sudo || die "sudo is required to install system packages (or re-run with --skip-system)."
    SUDO=(sudo)
  fi
  run "${SUDO[@]}" "$@"
}

version_major() { sed -E 's/^[^0-9]*([0-9]+).*/\1/' <<<"$1"; }

prepend_path() { case ":$PATH:" in *":$1:"*) ;; *) PATH="$1:$PATH"; export PATH ;; esac; }

port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

# Slow-connection defaults for every uv call (uv's own are 30 s / 3 retries)
# and for setup:api's optional cuDNN 8 download; user-set values win.
apply_network_defaults() {
  export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
  export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-5}"
  export OMNIVOICE_CUDNN8_TIMEOUT="${OMNIVOICE_CUDNN8_TIMEOUT:-1800}"
  info "UV_HTTP_TIMEOUT=$UV_HTTP_TIMEOUT UV_HTTP_RETRIES=$UV_HTTP_RETRIES OMNIVOICE_CUDNN8_TIMEOUT=$OMNIVOICE_CUDNN8_TIMEOUT"
  if [[ -n "$CACHE_DIR_ARG" ]]; then
    mkdir -p "$CACHE_DIR_ARG"
    CACHE_DIR="$(cd "$CACHE_DIR_ARG" && pwd)"
    export HF_HOME="$CACHE_DIR/huggingface" UV_CACHE_DIR="$CACHE_DIR/uv" PIP_CACHE_DIR="$CACHE_DIR/pip"
    info "caches in $CACHE_DIR: HF_HOME=$HF_HOME UV_CACHE_DIR=$UV_CACHE_DIR"
  fi
}

# ── platform detection ───────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"
DISTRO="" ; DISTRO_NAME="" ; PKG="" ; IS_WSL=0
case "$OS" in
  Linux)
    if [[ -r /etc/os-release ]]; then
      # shellcheck disable=SC1091
      . /etc/os-release
      DISTRO="${ID:-}"; DISTRO_NAME="${PRETTY_NAME:-$DISTRO}"
    fi
    if grep -qi microsoft /proc/version 2>/dev/null; then IS_WSL=1; prepend_path /usr/lib/wsl/lib; fi
    if have apt-get; then PKG=apt; elif have dnf; then PKG=dnf; elif have pacman; then PKG=pacman; fi
    ;;
  Darwin) DISTRO="macos"; DISTRO_NAME="macOS $(sw_vers -productVersion 2>/dev/null || true)"; PKG=brew ;;
  MINGW*|MSYS*|CYGWIN*)
    # Git Bash on Windows: the native installer is the supported path.
    here="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" 2>/dev/null && pwd || true)"
    printf 'Windows (Git Bash) detected; handing off to install.ps1.\n'
    ps_args=()
    [[ $ASSUME_YES == 1 ]] && ps_args+=(-Yes)
    case "$TORCH_VARIANT" in cpu) ps_args+=(-Cpu) ;; cuda) ps_args+=(-Cuda) ;; esac
    [[ $SKIP_APPLIO == 1 ]] && ps_args+=(-SkipApplio)
    [[ $SKIP_MODELS == 1 ]] && ps_args+=(-SkipModels)
    [[ -n "$MODELS" ]] && ps_args+=(-Models "$MODELS")
    [[ -n "$TARGET_DIR" ]] && ps_args+=(-Dir "$TARGET_DIR")
    [[ -n "$DATA_DIR_ARG" ]] && ps_args+=(-DataDir "$DATA_DIR_ARG")
    [[ -n "$APPLIO_DIR_ARG" ]] && ps_args+=(-ApplioDir "$APPLIO_DIR_ARG")
    [[ -n "$CACHE_DIR_ARG" ]] && ps_args+=(-CacheDir "$CACHE_DIR_ARG")
    [[ $NO_LAUNCH == 1 ]] && ps_args+=(-NoLaunch)
    if [[ -n "$here" && -f "$here/install.ps1" ]]; then
      exec powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$here/install.ps1")" ${ps_args[@]+"${ps_args[@]}"}
    fi
    die "Run in PowerShell instead:  irm https://raw.githubusercontent.com/vasudev123t/VoiceStudio/main/install.ps1 | iex"
    ;;
  *) die "Unsupported OS: $OS" ;;
esac

# ── 1. system packages ───────────────────────────────────────────────────
pkg_installed() {
  case "$PKG" in
    apt) dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q "install ok installed" ;;
    dnf) rpm -q --whatprovides "$1" >/dev/null 2>&1 ;;
    pacman) pacman -Qq "$1" >/dev/null 2>&1 || pacman -Qqg "$1" >/dev/null 2>&1 ;;
    brew) brew list --formula "$1" >/dev/null 2>&1 ;;
  esac
}

# AppImage runtime (optional): prints the package to add, "" when installed, "?" when unknown.
apt_fuse_pkg() {
  local p
  for p in libfuse2t64 libfuse2; do pkg_installed "$p" && return 0; done
  for p in libfuse2t64 libfuse2; do apt-cache show "$p" >/dev/null 2>&1 && { printf '%s' "$p"; return 0; }; done
  printf '?'
}

install_system_packages() {
  local where="$DISTRO_NAME, $ARCH"; [[ $IS_WSL == 1 ]] && where+=", WSL2"
  step "System packages ($where)"
  if [[ $SKIP_SYSTEM == 1 ]]; then info "skipped (--skip-system)"; return; fi
  local wanted=() missing=() p
  case "$PKG" in
    apt)
      wanted=(build-essential pkg-config libasound2-dev libxdo-dev libxtst-dev libx11-dev
              libxkbcommon-dev libwayland-dev libssl-dev binutils zsync git curl ca-certificates
              unzip xz-utils ffmpeg)
      ;;
    dnf)
      wanted=(gcc gcc-c++ make pkgconf-pkg-config alsa-lib-devel libxdo-devel libXtst-devel
              libX11-devel libxkbcommon-devel wayland-devel openssl-devel binutils zsync git curl
              unzip xz ffmpeg fuse-libs)
      ;;
    pacman)
      wanted=(base-devel alsa-lib xdotool libxtst libx11 libxkbcommon wayland openssl binutils
              zsync git curl unzip xz ffmpeg fuse2)
      ;;
    brew)
      if ! xcode-select -p >/dev/null 2>&1; then
        info "Installing the Xcode Command Line Tools (a dialog opens; finish it there)…"
        xcode-select --install >/dev/null 2>&1 || true
        until xcode-select -p >/dev/null 2>&1; do sleep 10; done
      fi
      ok "Xcode Command Line Tools: $(xcode-select -p)"
      if ! have brew; then
        for p in /opt/homebrew/bin/brew /usr/local/bin/brew; do
          [[ -x $p ]] && eval "$("$p" shellenv)" && break
        done
      fi
      if ! have brew; then
        confirm "Homebrew is needed for git/ffmpeg/pkg-config. Install Homebrew?" y \
          || die "Install Homebrew (https://brew.sh) or re-run with --skip-system."
        NONINTERACTIVE=$ASSUME_YES /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        for p in /opt/homebrew/bin/brew /usr/local/bin/brew; do
          [[ -x $p ]] && eval "$("$p" shellenv)" && break
        done
      fi
      wanted=(git ffmpeg pkg-config)
      ;;
    *)
      warn "No supported package manager (apt/dnf/pacman/brew). Install the packages from docs/install/linux.md yourself."
      return
      ;;
  esac
  for p in "${wanted[@]}"; do pkg_installed "$p" || missing+=("$p"); done
  local fuse=""
  [[ $PKG == apt ]] && fuse="$(apt_fuse_pkg)"
  [[ -n "$fuse" && "$fuse" != "?" ]] && missing+=("$fuse")
  if [[ ${#missing[@]} -eq 0 ]]; then ok "all ${#wanted[@]} packages present"; return; fi
  info "missing: ${missing[*]}"
  case "$PKG" in
    apt)
      as_root apt-get update -y
      # Package lists may have been empty before the update.
      if [[ "$fuse" == "?" ]]; then fuse="$(apt_fuse_pkg)"; [[ -n "$fuse" && "$fuse" != "?" ]] && missing+=("$fuse"); fi
      as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing[@]}"
      ;;
    dnf) as_root dnf install -y "${missing[@]}" ;;
    pacman) as_root pacman -S --needed --noconfirm "${missing[@]}" ;;
    brew) run brew install "${missing[@]}" ;;
  esac
  ok "system packages installed"
}

# ── 2. toolchains ────────────────────────────────────────────────────────
load_tool_envs() {
  prepend_path "$HOME/.local/bin"
  # shellcheck source=/dev/null
  [[ -f "$HOME/.cargo/env" ]] && . "$HOME/.cargo/env"
  [[ -d "$HOME/.bun/bin" ]] && prepend_path "$HOME/.bun/bin"
  local fnm_dir
  for fnm_dir in "${FNM_DIR:-}" "$HOME/.local/share/fnm" "$HOME/.fnm"; do
    [[ -n "$fnm_dir" && -x "$fnm_dir/fnm" ]] && prepend_path "$fnm_dir"
  done
  if have fnm; then eval "$(fnm env --shell bash)"; fi
}

node_ok() { have node && [[ $(version_major "$(node --version)") -ge $NODE_MAJOR_MIN ]]; }

ensure_node() {
  if node_ok; then ok "Node $(node --version)"; return; fi
  info "Node ${NODE_MAJOR_MIN}+ not found (have: $(node --version 2>/dev/null || echo none)); installing a user-level Node."
  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
  if ! have fnm && [[ -s "$NVM_DIR/nvm.sh" ]]; then
    # shellcheck disable=SC1091
    . "$NVM_DIR/nvm.sh"
    run nvm install "$NODE_MAJOR_MIN"
    run nvm alias default "$NODE_MAJOR_MIN"
  else
    if ! have fnm; then
      run bash -c "curl -fsSL https://fnm.vercel.app/install | bash -s -- --install-dir \"\$HOME/.local/share/fnm\""
      load_tool_envs
    fi
    have fnm || die "fnm install failed"
    run fnm install "$NODE_MAJOR_MIN"
    run fnm default "$NODE_MAJOR_MIN"
    eval "$(fnm env --shell bash)"
    fnm use "$NODE_MAJOR_MIN" >/dev/null
  fi
  hash -r
  node_ok || die "Node ${NODE_MAJOR_MIN}+ still not on PATH"
  ok "Node $(node --version)"
}

wanted_bun_version() {
  local pkg="$1/package.json" v=""
  [[ -f $pkg ]] && v="$(sed -nE 's/.*"packageManager": *"bun@([0-9][^"]*)".*/\1/p' "$pkg" | head -n1)"
  printf '%s' "${v:-$DEFAULT_BUN_VERSION}"
}

ensure_bun() {
  local want=$1
  if have bun && [[ "$(bun --version)" == "$want" ]]; then ok "Bun $(bun --version)"; return; fi
  if [[ -x "$HOME/.bun/bin/bun" && "$("$HOME/.bun/bin/bun" --version)" == "$want" ]]; then
    prepend_path "$HOME/.bun/bin"; hash -r; ok "Bun $want (~/.bun)"; return
  fi
  info "Installing Bun $want (packageManager pin; have: $(bun --version 2>/dev/null || echo none))"
  run bash -c "curl -fsSL https://bun.sh/install | bash -s \"bun-v$want\""
  prepend_path "$HOME/.bun/bin"; hash -r
  [[ "$(bun --version)" == "$want" ]] || die "Bun $want not active (got $(bun --version 2>/dev/null))"
  ok "Bun $(bun --version)"
}

ensure_uv() {
  if have uv; then ok "$(uv --version)"; return; fi
  run bash -c "curl -LsSf https://astral.sh/uv/install.sh | sh"
  # shellcheck source=/dev/null
  [[ -f "$HOME/.local/bin/env" ]] && . "$HOME/.local/bin/env"
  prepend_path "$HOME/.local/bin"; hash -r
  have uv || die "uv install failed"
  ok "$(uv --version)"
}

ensure_rust() {
  if have cargo && have rustc; then ok "$(rustc --version)"; return; fi
  run bash -c "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"
  # shellcheck source=/dev/null
  . "$HOME/.cargo/env"
  have cargo || die "rustup install failed"
  ok "$(rustc --version)"
}

# ── 3. repository ────────────────────────────────────────────────────────
is_voicestudio_repo() { [[ -f "$1/package.json" && -d "$1/backend" && -f "$1/pyproject.toml" ]]; }

resolve_repo() {
  step "VoiceStudio source"
  local script_dir=""
  if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  fi
  if [[ -z "$TARGET_DIR" ]]; then
    if [[ -n "$script_dir" ]] && is_voicestudio_repo "$script_dir"; then TARGET_DIR="$script_dir"
    elif is_voicestudio_repo "$PWD"; then TARGET_DIR="$PWD"
    else TARGET_DIR="$PWD/VoiceStudio"; fi
  fi
  mkdir -p "$(dirname "$TARGET_DIR")"
  TARGET_DIR="$(cd "$(dirname "$TARGET_DIR")" && pwd)/$(basename "$TARGET_DIR")"
  if is_voicestudio_repo "$TARGET_DIR"; then
    ok "using $TARGET_DIR"
    if [[ -n "$BRANCH" && -d "$TARGET_DIR/.git" ]]; then
      local cur; cur="$(git -C "$TARGET_DIR" rev-parse --abbrev-ref HEAD)"
      [[ "$cur" == "$BRANCH" ]] || warn "checkout is on '$cur', not '$BRANCH'; leaving it as is"
    fi
  elif [[ -e "$TARGET_DIR" && -n "$(ls -A "$TARGET_DIR" 2>/dev/null)" ]]; then
    die "$TARGET_DIR exists and is not a VoiceStudio checkout; pick another --dir"
  else
    local args=(clone --depth 1)
    [[ -n "$BRANCH" ]] && args+=(--branch "$BRANCH")
    run git "${args[@]}" "$REPO_URL" "$TARGET_DIR"
    ok "cloned $REPO_URL"
  fi
  cd "$TARGET_DIR"
}

# ── 4. data dir / backend helpers ────────────────────────────────────────
resolve_data_dir() {
  if [[ -n "$DATA_DIR_ARG" ]]; then DATA_DIR="$DATA_DIR_ARG"
  elif [[ -n "${OMNIVOICE_DATA_DIR:-}" ]]; then DATA_DIR="$OMNIVOICE_DATA_DIR"
  elif [[ "$OS" == Darwin ]]; then DATA_DIR="$HOME/Library/Application Support/OmniVoice"
  else DATA_DIR="$HOME/.omnivoice"; fi
  CUSTOM_DATA_DIR=0
  [[ -n "$DATA_DIR_ARG" ]] && CUSTOM_DATA_DIR=1
  mkdir -p "$DATA_DIR"
  DATA_DIR="$(cd "$DATA_DIR" && pwd)"
  MANAGED_APPLIO_DIR="$DATA_DIR/tools/applio"
  APPLIO_DIR="${APPLIO_DIR_ARG:-${OMNIVOICE_APPLIO_DIR:-$MANAGED_APPLIO_DIR}}"
}

repo_python() { printf '%s' "$TARGET_DIR/.venv/bin/python"; }

# json_field <json> <python expression over `d`>
json_field() { "$(repo_python)" -c "import json,sys; d=json.loads(sys.argv[1]); print($2)" "$1" 2>/dev/null || true; }

BACKEND_PID=""
BACKEND_PORT=""
BACKEND_LOG=""

backend_env() {
  local e=("OMNIVOICE_PORT=$BACKEND_PORT")
  [[ $CUSTOM_DATA_DIR == 1 || -n "${OMNIVOICE_DATA_DIR:-}" ]] && e+=("OMNIVOICE_DATA_DIR=$DATA_DIR")
  [[ "$APPLIO_DIR" != "$MANAGED_APPLIO_DIR" && $SKIP_APPLIO == 0 ]] && e+=("OMNIVOICE_APPLIO_DIR=$APPLIO_DIR")
  printf '%s\n' "${e[@]}"
}

start_backend() {
  [[ -n "$BACKEND_PID" ]] && return 0
  local p=${PORT_ARG:-3900}
  if [[ -z "$PORT_ARG" ]]; then
    while port_busy "$p"; do p=$((p + 1)); [[ $p -gt 3999 ]] && die "no free port in 3900-3999"; done
  elif port_busy "$p"; then
    die "port $p is in use; pick another --port"
  fi
  BACKEND_PORT=$p
  BACKEND_LOG="$(mktemp "${TMPDIR:-/tmp}/voicestudio-backend.XXXXXX")"
  local envs=() line
  while IFS= read -r line; do envs+=("$line"); done < <(backend_env)
  info "starting backend on 127.0.0.1:$BACKEND_PORT (log: $BACKEND_LOG)"
  env "${envs[@]}" uv run --no-sync uvicorn main:app --app-dir backend --host 127.0.0.1 \
    --port "$BACKEND_PORT" >"$BACKEND_LOG" 2>&1 &
  BACKEND_PID=$!
  local i
  for ((i = 0; i < 180; i++)); do
    if curl -fsS "http://127.0.0.1:$BACKEND_PORT/health" >/dev/null 2>&1; then
      ok "backend up after ${i}s"; return 0
    fi
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
      tail -n 40 "$BACKEND_LOG" >&2 || true
      BACKEND_PID=""
      die "backend exited during startup (log: $BACKEND_LOG)"
    fi
    sleep 1
  done
  tail -n 40 "$BACKEND_LOG" >&2 || true
  die "backend did not answer /health within 180s"
}

stop_backend() {
  [[ -z "$BACKEND_PID" ]] && return 0
  # `uv run` spawns uvicorn as a child: stop the child first, then uv.
  pkill -TERM -P "$BACKEND_PID" 2>/dev/null || true
  kill -TERM "$BACKEND_PID" 2>/dev/null || true
  local i
  for ((i = 0; i < 20; i++)); do kill -0 "$BACKEND_PID" 2>/dev/null || break; sleep 0.5; done
  pkill -KILL -P "$BACKEND_PID" 2>/dev/null || true
  kill -KILL "$BACKEND_PID" 2>/dev/null || true
  wait "$BACKEND_PID" 2>/dev/null || true
  BACKEND_PID=""
}
trap 'stop_backend' EXIT
trap 'stop_backend; exit 130' INT TERM

api() { curl -fsS --max-time 30 "http://127.0.0.1:$BACKEND_PORT$1" "${@:2}"; }

# ── 5. Applio (voice conversion) ─────────────────────────────────────────
applio_python_path() {
  local rel
  for rel in .venv/bin/python env/bin/python; do
    [[ -x "$APPLIO_DIR/$rel" ]] && { printf '%s' "$APPLIO_DIR/$rel"; return 0; }
  done
  return 1
}

applio_healthy() {
  local py
  [[ -f "$APPLIO_DIR/core.py" && -d "$APPLIO_DIR/rvc" ]] || return 1
  py="$(applio_python_path)" || return 1
  (cd "$APPLIO_DIR" && "$py" -c "import torch, faiss, librosa, soundfile" >/dev/null 2>&1)
}

# PyTorch index for Applio's requirements, as run-install.sh picks it
# (CUDA 12.8 on Linux, PyPI on macOS) or the matching CPU/ROCm build.
applio_index_args() {
  case "$OS:$RESOLVED_VARIANT" in
    Darwin:*) ;;
    *:cpu) printf '%s\n' --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match ;;
    *:rocm) printf '%s\n' --extra-index-url https://download.pytorch.org/whl/rocm6.4 --index-strategy unsafe-best-match ;;
    *) printf '%s\n' --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match ;;
  esac
}

# Install Applio's requirements into its env with its own interpreter, up to
# APPLIO_INSTALL_ATTEMPTS times (a slow link can time out on one big wheel).
applio_install_deps() {
  local py attempt idx=() line
  py="$(applio_python_path)" || return 1
  while IFS= read -r line; do idx+=("$line"); done < <(applio_index_args)
  for ((attempt = 1; attempt <= APPLIO_INSTALL_ATTEMPTS; attempt++)); do
    info "installing Applio's dependencies (attempt $attempt/$APPLIO_INSTALL_ATTEMPTS, UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-30})"
    (cd "$APPLIO_DIR" && run uv pip install --python "$py" -r requirements.txt ${idx[@]+"${idx[@]}"}) \
      || warn "Applio dependency install failed (attempt $attempt/$APPLIO_INSTALL_ATTEMPTS)"
    applio_healthy && return 0
    ((attempt < APPLIO_INSTALL_ATTEMPTS)) && sleep "$APPLIO_RETRY_DELAY"
  done
  return 1
}

install_applio() {
  step "Applio (voice conversion)"
  if [[ $SKIP_APPLIO == 1 ]]; then info "skipped (--skip-applio)"; return; fi
  info "target: $APPLIO_DIR  (MIT; ~2 GB on CPU, ~6 GB with CUDA, plus ~0.6 GB base models)"
  if applio_healthy; then
    ok "Applio already installed ($(git -C "$APPLIO_DIR" rev-parse --short HEAD 2>/dev/null || echo '?'))"
  else
    if [[ ! -e "$APPLIO_DIR/core.py" ]]; then
      if [[ -e "$APPLIO_DIR" && -n "$(ls -A "$APPLIO_DIR" 2>/dev/null)" ]]; then
        die "$APPLIO_DIR exists but is not an Applio checkout; pick another --applio-dir"
      fi
      mkdir -p "$(dirname "$APPLIO_DIR")"
      run git clone --depth 1 "$APPLIO_REPO" "$APPLIO_DIR"
    fi
    local variant=$RESOLVED_VARIANT
    if applio_python_path >/dev/null; then
      # A previous run made the env but its download died part way: retry the
      # dependency install instead of rebuilding the env.
      info "Applio's environment exists but is incomplete; reinstalling its dependencies"
    elif [[ "$OS" == Darwin || "$variant" == cuda ]]; then
      # A venv without an interpreter makes run-install.sh prompt and then launch the web UI.
      [[ -d "$APPLIO_DIR/.venv" ]] && { warn "removing incomplete Applio venv"; rm -rf "$APPLIO_DIR/.venv"; }
      info "running Applio's run-install.sh"
      # run-install.sh calls `clear` under `set -e`; give it a terminal type.
      local term=${TERM:-xterm}; [[ $term == dumb ]] && term=xterm
      (cd "$APPLIO_DIR" && TERM=$term bash ./run-install.sh </dev/null) \
        || warn "Applio's run-install.sh failed (exit $?); retrying its dependency step"
    else
      # run-install.sh always pulls CUDA 12.8 wheels on Linux; mirror its
      # steps with the matching PyTorch index instead.
      rm -rf "$APPLIO_DIR/.venv"
      info "mirroring run-install.sh with the $variant PyTorch index"
      (cd "$APPLIO_DIR" && run uv venv .venv --python 3.12 \
        && run uv pip install --python .venv/bin/python python-ffmpeg) \
        || die "could not create Applio's environment in $APPLIO_DIR/.venv"
    fi
    applio_python_path >/dev/null || die "Applio's installer did not create an environment in $APPLIO_DIR; see the output above"
    applio_healthy || applio_install_deps \
      || die "Applio's environment is not importable (torch/faiss/librosa) after $APPLIO_INSTALL_ATTEMPTS dependency installs; check the connection and re-run (finished downloads are cached)"
    ok "Applio environment ready"
  fi
  local py; py="$(applio_python_path)"
  info "downloading Applio base models (core.py prerequisites; skips files already present)"
  (cd "$APPLIO_DIR" && run "$py" core.py prerequisites --no-pretraineds-hifigan --models --exe)
  ok "Applio prerequisites present"
}

# ── 6. models ────────────────────────────────────────────────────────────
catalog_field() { # key fieldno
  local entry
  for entry in "${MODEL_CATALOG[@]}"; do
    [[ "${entry%%|*}" == "$1" ]] && { cut -d'|' -f"$2" <<<"$entry"; return 0; }
  done
  return 1
}

SELECTED_MODELS=()
select_models() {
  local list=$MODELS key
  if [[ -z "$list" ]]; then
    printf '\n    Starter models (downloaded from Hugging Face through the backend):\n'
    for key in omnivoice kittentts whisper-small; do
      printf '      %-14s %-8s %s\n' "$key" "$(catalog_field "$key" 3)" "$(catalog_field "$key" 4)"
    done
    if confirm "Download OmniVoice (~2.4 GB, CC-BY-NC weights) and KittenTTS (~80 MB, Apache-2.0)?" y; then
      SELECTED_MODELS+=(omnivoice kittentts)
    fi
    if [[ $ASSUME_YES == 0 ]] && confirm "Also download faster-whisper-small (~0.5 GB, MIT) for transcription?" n; then
      SELECTED_MODELS+=(whisper-small)
    fi
    return
  fi
  [[ $list == none ]] && return
  [[ $list == all ]] && list="omnivoice,kittentts,whisper-small"
  local requested=()
  IFS=',' read -r -a requested <<<"$list"
  for key in "${requested[@]}"; do
    key="${key// /}"
    catalog_field "$key" 2 >/dev/null || die "unknown model '$key' (use omnivoice, kittentts, whisper-small)"
    SELECTED_MODELS+=("$key")
  done
  local summary=""
  for key in "${SELECTED_MODELS[@]}"; do summary+=" $key ($(catalog_field "$key" 3), $(catalog_field "$key" 4));"; done
  confirm "Download:${summary%;}?" y || SELECTED_MODELS=()
}

INSTALLED_MODELS=()
download_models() {
  step "Models"
  if [[ $SKIP_MODELS == 1 ]]; then info "skipped (--skip-models)"; return; fi
  select_models
  if [[ ${#SELECTED_MODELS[@]} -eq 0 ]]; then info "no models selected; install them later from the app's Model Store"; return; fi
  start_backend
  local key repo resp state models_json installed
  models_json="$(api /models || true)"
  for key in "${SELECTED_MODELS[@]}"; do
    repo="$(catalog_field "$key" 2)"
    installed="$(json_field "$models_json" "next((m['installed'] for m in d['models'] if m['repo_id']=='$repo'), False)")"
    if [[ "$installed" == True ]]; then ok "$repo already installed"; INSTALLED_MODELS+=("$repo"); continue; fi
    info "installing $repo ($(catalog_field "$key" 3))"
    resp="$(api /models/install -X POST -H 'Content-Type: application/json' \
      -d "{\"repo_id\":\"$repo\",\"target\":\"local\"}")" || die "POST /models/install failed for $repo"
    state="$(json_field "$resp" "d.get('status','')")"
    [[ "$state" == install_started || "$state" == already_running || "$state" == already_installed ]] \
      || die "unexpected install response for $repo: $resp"
    local last="" now
    while :; do
      sleep 3
      resp="$(api /models/install/status || echo '{}')"
      state="$(json_field "$resp" "next((j.get('state','') for j in d.get('jobs',[]) if j.get('repo_id')=='$repo'), '')")"
      if [[ "$state" == failed ]]; then
        die "download failed for $repo: $(json_field "$resp" "next((j.get('error') or j.get('message') or j for j in d['jobs'] if j.get('repo_id')=='$repo'), '')")"
      fi
      [[ "$state" == downloading ]] || break
      now="$(json_field "$resp" "next((f\"{j.get('downloaded_bytes',0)/1e6:.0f}/{(j.get('total_bytes') or 0)/1e6:.0f} MB\" for j in d['jobs'] if j.get('repo_id')=='$repo'), '')")"
      [[ -n "$now" && "$now" != "$last" ]] && info "  $now"
      last=$now
    done
    models_json="$(api /models || true)"
    installed="$(json_field "$models_json" "next((m['installed'] for m in d['models'] if m['repo_id']=='$repo'), False)")"
    [[ "$installed" == True ]] || die "$repo did not finish installing (see $BACKEND_LOG)"
    ok "$repo installed"
    INSTALLED_MODELS+=("$repo")
  done
}

# ── 7. verify ────────────────────────────────────────────────────────────
VC_READY="skipped"; VC_DEVICE=""; HEALTH=""
verify_backend() {
  step "Verify backend"
  start_backend
  HEALTH="$(api /health)" || die "/health failed"
  ok "/health: $(json_field "$HEALTH" "d.get('status', d)")"
  local vc
  vc="$(api /voice-conversion/status)" || die "/voice-conversion/status failed"
  VC_DEVICE="$(json_field "$vc" "d.get('device')")"
  VC_READY="$(json_field "$vc" "d['applio']['ready']")"
  ok "/voice-conversion/status: backend=$(json_field "$vc" "d.get('backend')") device=$VC_DEVICE applio_ready=$VC_READY"
  if [[ $SKIP_APPLIO == 0 && "$VC_READY" != True ]]; then
    warn "Applio is installed at $APPLIO_DIR but the backend does not see it: $(json_field "$vc" "d['applio']")"
  fi
  stop_backend
  ok "verification backend stopped"
}

# ── main ─────────────────────────────────────────────────────────────────
# Wrapped in a function so `curl … | bash` parses the whole file before any
# step runs (children cannot swallow the rest of the script from stdin).
main() {
printf '%sVoiceStudio source setup%s  (%s %s)\n' "$C_GREEN" "$C_RESET" "$OS" "$ARCH"

install_system_packages

step "Toolchains"
load_tool_envs
ensure_node
ensure_uv
ensure_rust

resolve_repo
ensure_bun "$(wanted_bun_version "$TARGET_DIR")"
resolve_data_dir

RESOLVED_VARIANT=$TORCH_VARIANT
if [[ $RESOLVED_VARIANT == auto ]]; then
  if [[ "$OS" == Darwin ]]; then RESOLVED_VARIANT="native"
  elif have nvidia-smi && nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then RESOLVED_VARIANT="cuda"
  else RESOLVED_VARIANT="cpu"; fi
fi
GPU_NAME=""
if have nvidia-smi; then GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"; fi

step "Download settings"
apply_network_defaults

step "JavaScript workspace (bun install)"
run bun install

step "Python backend (bun run setup:api, PyTorch: $RESOLVED_VARIANT)"
# setup-api understands auto|cpu|cuda|rocm; macOS always uses native wheels.
setup_variant=$RESOLVED_VARIANT; [[ $setup_variant == native ]] && setup_variant=auto
OMNIVOICE_TORCH_VARIANT=$setup_variant run bun run setup:api
ok "backend environment ready ($("$(repo_python)" -c 'import torch; print("torch", torch.__version__)' 2>/dev/null || echo 'torch ?'))"

install_applio
download_models
verify_backend

# summary
launch_env=()
[[ $CUSTOM_DATA_DIR == 1 ]] && launch_env+=("OMNIVOICE_DATA_DIR=$DATA_DIR")
[[ $SKIP_APPLIO == 0 && "$APPLIO_DIR" != "$MANAGED_APPLIO_DIR" ]] && launch_env+=("OMNIVOICE_APPLIO_DIR=$APPLIO_DIR")
# Downloaded models live in HF_HOME, so the app must see the same value.
[[ -n "$CACHE_DIR_ARG" ]] && launch_env+=("HF_HOME=$HF_HOME" "UV_CACHE_DIR=$UV_CACHE_DIR")
local env_words="" kv
for kv in ${launch_env[@]+"${launch_env[@]}"}; do env_words+="$(printf '%q' "$kv") "; done
local launch_cmd models_text="none downloaded (use the in-app Model Store)"
launch_cmd="cd $(printf '%q' "$TARGET_DIR") && ${env_words}bun run dev"
[[ ${#INSTALLED_MODELS[@]} -gt 0 ]] && models_text="${INSTALLED_MODELS[*]}"
local wsl_note=""; [[ $IS_WSL == 1 ]] && wsl_note=" WSL2"

step "Summary"
info "OS            $DISTRO_NAME ($ARCH)$wsl_note"
info "Node          $(node --version)   Bun $(bun --version)   $(uv --version)"
info "Rust          $(rustc --version 2>/dev/null || echo n/a)"
info "PyTorch       $RESOLVED_VARIANT ($("$(repo_python)" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '?'))${GPU_NAME:+  GPU: $GPU_NAME}"
info "VC device     ${VC_DEVICE:-n/a}"
info "Repository    $TARGET_DIR"
info "Data dir      $DATA_DIR"
[[ -n "$CACHE_DIR_ARG" ]] && info "Caches        $CACHE_DIR (HF_HOME, UV_CACHE_DIR)"
if [[ $SKIP_APPLIO == 1 ]]; then info "Applio        skipped"
else info "Applio        $APPLIO_DIR (backend sees it: $VC_READY)"; fi
info "Models        $models_text"
[[ -n "$env_words" ]] && info "Environment   export ${env_words% }"
printf '\n    %sLaunch:%s %s\n' "$C_GREEN" "$C_RESET" "$launch_cmd"

if [[ $NO_LAUNCH == 0 ]]; then
  if confirm "Start VoiceStudio now (bun run dev)?" y; then
    trap - EXIT
    exec env ${launch_env[@]+"${launch_env[@]}"} bun run dev
  fi
fi
return 0
}

# Tests source this file with VOICESTUDIO_INSTALL_SOURCE_ONLY=1 to reach the helpers.
if [[ "${VOICESTUDIO_INSTALL_SOURCE_ONLY:-}" == 1 ]] && (return 0 2>/dev/null); then return 0; fi
main
