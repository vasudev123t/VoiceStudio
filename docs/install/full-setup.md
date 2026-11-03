# One-command source setup

`install.sh` (Linux, macOS, WSL2) and `install.ps1` (native Windows) in the
repository root set up a complete source checkout: prerequisites, toolchains,
the JavaScript workspace, the Python backend, Applio for
[voice conversion](../voice-conversion.md), optional starter models, and a
verification boot of the backend. They are for running from source
(`bun run dev`). To install a packaged release instead, use the
[shell installer](script.md) (`scripts/install.sh` / `scripts/install.ps1`).

Both scripts are safe to re-run: each step checks what is already there, so a
second run only repeats the quick checks.

## Quick start

Linux, macOS, WSL2:

```sh
curl -fsSL https://raw.githubusercontent.com/vasudev123t/VoiceStudio/main/install.sh | bash
# non-interactive, accept the model downloads:
curl -fsSL https://raw.githubusercontent.com/vasudev123t/VoiceStudio/main/install.sh | bash -s -- --yes
# from a checkout:
./install.sh --help
```

Windows (PowerShell):

```powershell
irm https://raw.githubusercontent.com/vasudev123t/VoiceStudio/main/install.ps1 | iex
# with flags:
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/vasudev123t/VoiceStudio/main/install.ps1))) -Yes
# from a checkout (if scripts are blocked by the execution policy):
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -Yes
```

`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` allows scripts
for the current window only.

## What the scripts do

1. **System packages.**
   - Linux: installs the build dependencies from
     [Building from source](linux.md#building-from-source) with
     apt, dnf or pacman, plus git, curl, unzip, ffmpeg and the AppImage FUSE
     runtime. Only missing packages are installed, and `sudo` is used only then.
   - macOS: installs the Xcode Command Line Tools and, through Homebrew, git,
     ffmpeg and pkg-config.
   - Windows: installs Git, ffmpeg and the Visual Studio 2022 C++ build tools
     (asks first) with winget, falling back to Chocolatey.
2. **Toolchains.** Each one is installed only when missing:
   - Node 22+: via fnm (or an existing nvm) at user level; winget `OpenJS.NodeJS.LTS` on Windows.
   - Bun: the `packageManager` version from `package.json`.
   - uv.
   - Rust: via rustup.
3. **Source.** Uses the checkout the script lives in or the current folder.
   Otherwise it clones `--repo` into `--dir` (default `./VoiceStudio`).
4. **Install.** Runs `bun install`, then
   `OMNIVOICE_TORCH_VARIANT=<cpu|cuda|rocm> bun run setup:api`.
   - The default PyTorch variant is CUDA when `nvidia-smi` finds an NVIDIA GPU, and CPU otherwise.
   - macOS always uses the native (MPS) wheels.
5. **Applio.** Clones [Applio](https://github.com/IAHispano/Applio) to
   `<data dir>/tools/applio`, which the backend finds without further settings.
   - Windows: runs Applio's own `run-install.bat`. It installs Miniconda in
     `%USERPROFILE%\Miniconda3` and CUDA 12.8 wheels.
   - Linux with CUDA, and macOS: runs Applio's own `run-install.sh`.
   - Linux with `--cpu` or `--rocm`: `run-install.sh` always pulls CUDA wheels,
     so the script repeats its steps (`uv venv .venv --python 3.12`, then
     `uv pip install -r requirements.txt`) with the CPU or ROCm PyTorch index.
   - If Applio's environment exists but does not import torch, faiss and
     librosa (for example after a download timed out, which `run-install.bat`
     still reports as a success), the script re-runs only the dependency step
     with the environment's own interpreter, up to 3 times. On Windows that is
     `env\python.exe -m uv pip install -r requirements.txt --python env\python.exe`
     with the CUDA 12.8 index (CPU index with `-Cpu`). A re-run of the script
     goes straight to this step instead of rebuilding the environment.
   - After installing, it runs `core.py prerequisites --no-pretraineds-hifigan --models --exe`
     from the Applio folder to fetch the base models.
   - With a custom `--applio-dir` / `-ApplioDir`, the scripts print
     `OMNIVOICE_APPLIO_DIR`. `install.ps1` can also save it as a user
     environment variable.
6. **Models.** Asks before downloading (`--yes` accepts), then installs through
   the app's own `POST /models/install` API and waits for completion:

   | Key | Model | Size | Licence |
   |---|---|---|---|
   | `omnivoice` | `k2-fsa/OmniVoice` | ~2.4 GB | weights CC-BY-NC-4.0 (non-commercial) |
   | `kittentts` | `KittenML/kitten-tts-mini-0.8` | ~80 MB | Apache-2.0 |
   | `whisper-small` | `Systran/faster-whisper-small` | ~0.5 GB | MIT |

   The default is `omnivoice,kittentts`. `whisper-small` is offered
   interactively, or you can pass it with `--models`.
7. **ASR models (Audio Edit).** Downloads the speech recognisers Audio Edit
   runs locally, through the backend's own APIs, so each one lands where the
   app looks for it (Hugging Face cache, or `<data dir>/audio_edit/asr_models`
   for the IndicWhisper archive). Models already present are skipped. Default
   is all of them (about 10 GB); pick with `--asr-models` / `-AsrModels`:

   | Key | Model | Download | Installed with |
   |---|---|---|---|
   | `whisper-large-v3` | `Systran/faster-whisper-large-v3` (Apple Silicon: `mlx-community/whisper-large-v3-mlx`), the active Whisper engine's default | ~2.9 GB | `POST /models/install` |
   | `indicconformer_kn` | `ai4bharat/indic-conformer-600m-multilingual` (**gated**) | ~2.4 GB | `POST /audio-edit/asr/engines/indicconformer_kn/download` |
   | `indicwhisper_kn` | AI4Bharat IndicWhisper Kannada (Vistaar zip) | ~1.4 GB zip, ~3 GB unpacked | `POST /audio-edit/asr/engines/indicwhisper_kn/download` |
   | `vasista22_whisper_kn` | `vasista22/whisper-kannada-medium` | ~3.1 GB | `POST /audio-edit/asr/engines/vasista22_whisper_kn/download` |

   **HF_TOKEN.** Only IndicConformer is gated. Accept its terms at
   <https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual>,
   create a read token at <https://huggingface.co/settings/tokens>, and make
   it visible with `export HF_TOKEN=hf_...` (PowerShell:
   `$env:HF_TOKEN='hf_...'`), `hf auth login`, or Settings → Hugging Face
   token. Without one the script still installs the other models, prints how
   to fix it, and exits with code 3 so the missing model is not overlooked.

   To fetch only these models into an existing setup:
   `./install.sh --asr-only` (Windows: `.\install.ps1 -AsrOnly`).
   `--with-cloud-keys-help` / `-WithCloudKeysHelp` also lists the cloud ASR
   keys (Sarvam, Smallest.ai, Google, Azure, OpenAI, Deepgram, AssemblyAI,
   Bhashini); they are saved in Settings → Models → ASR.
8. **Verify.** Starts the backend on port 3900, or on the next free port if
   3900 is busy, checks `/health` and `/voice-conversion/status`, and stops it.
9. **Summary.** Prints:
   - the tool versions, PyTorch variant, GPU and voice-conversion device
   - the data dir and the Applio path
   - the installed and failed ASR models
   - the launch command and next steps, then offers to start `bun run dev`

## Flags

| install.sh | install.ps1 | Meaning |
|---|---|---|
| `-y`, `--yes` | `-Yes` | Answer yes to every prompt, including model downloads |
| `--cpu` / `--cuda` / `--rocm` | `-Cpu` / `-Cuda` | PyTorch variant (default auto; ROCm is Linux only) |
| `--skip-applio` | `-SkipApplio` | Do not install Applio |
| `--skip-models` | `-SkipModels` | Do not download models (starter and ASR) |
| `--asr-only` | `-AsrOnly` | Only download the ASR models into an existing setup |
| `--asr-models LIST` | `-AsrModels LIST` | `whisper-large-v3,indicconformer_kn,indicwhisper_kn,vasista22_whisper_kn`, `all` (default) or `none` |
| `--with-cloud-keys-help` | `-WithCloudKeysHelp` | Also print where the cloud ASR API keys go |
| `--models LIST` | `-Models LIST` | `omnivoice,kittentts,whisper-small`, `all` or `none` (in PowerShell, quoted or not) |
| `--dir PATH` | `-Dir PATH` | Checkout location |
| `--repo URL` | `-Repo URL` | Repository to clone (default `vasudev123t/VoiceStudio`) |
| `--branch NAME` | `-Branch NAME` | Branch to clone |
| `--data-dir PATH` | `-DataDir PATH` | Data dir (default `OMNIVOICE_DATA_DIR` or the platform default) |
| `--applio-dir PATH` | `-ApplioDir PATH` | Applio location (default `<data dir>/tools/applio`) |
| `--cache-dir PATH` | `-CacheDir PATH` | Put the download caches under PATH: sets `HF_HOME` (`PATH/huggingface`, where models go), `UV_CACHE_DIR` and `PIP_CACHE_DIR` for the run |
| `--port N` | `-Port N` | Port for the verification backend |
| `--skip-system` | `-SkipSystem` | Skip OS package installs (no sudo/winget) |
| `--no-launch` | `-NoLaunch` | Do not start `bun run dev` at the end |
| `-h`, `--help` | `-Help` | Show help |

On Git Bash, `install.sh` hands off to `install.ps1` with the same flags.

## Disk and time

- A CPU setup needs about 5 GB for the backend environment and about 2 GB for Applio.
- CUDA wheels add several GB to each of those.
- The starter models add about 2.5 GB.
- The ASR models add about 10 GB (`--asr-models none` skips them).
- The first run takes 10–30 minutes, depending on the network. A re-run takes seconds.

## Slow connections and interrupted runs

- uv's HTTP timeout defaults to 30 s, which is too short for the multi-GB CUDA
  wheels on a slow link. For the run, the scripts set `UV_HTTP_TIMEOUT=600`
  and `UV_HTTP_RETRIES=5`, plus `OMNIVOICE_CUDNN8_TIMEOUT=1800` for the
  optional cuDNN 8 download in `setup:api`. Values you set yourself are kept.
- A run that stops part way (closed terminal, lost connection, sleep) can be
  started again with the same flags. Every step checks what is already there,
  and uv reuses the wheels it already downloaded. Long downloads run inside the
  terminal that started the script, so keep that window open. On Windows, a
  separate window started with `start powershell` survives closing the editor
  or terminal tab that launched it.

## Installing on another drive

When the system drive is short of space, put the checkout, data, Applio and
the download caches on another drive. Example for `D:` on Windows:

```powershell
.\install.ps1 -Yes -Models omnivoice,kittentts `
  -Dir D:\VoiceStudio -DataDir D:\VoiceStudioData\data `
  -ApplioDir D:\VoiceStudioData\applio -CacheDir D:\VoiceStudioData\cache
```

`-CacheDir` / `--cache-dir` sets `HF_HOME`, `UV_CACHE_DIR` and `PIP_CACHE_DIR`
for the run. Models are stored under `HF_HOME`, so the app must see the same
value later: `install.ps1` offers to save `HF_HOME` and `UV_CACHE_DIR` as user
environment variables, and `install.sh` adds them to the printed launch
command. Setting `HF_HOME` and `UV_CACHE_DIR` yourself before the run works the
same way. Applio's Miniconda still goes to `%USERPROFILE%\Miniconda3`
(about 0.5 GB), because `run-install.bat` fixes that path.
