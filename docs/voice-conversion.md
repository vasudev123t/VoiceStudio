# Voice conversion (RVC / Applio)

Voice conversion turns **your own performance** into another voice. You record a
line, and an RVC (Retrieval-based Voice Conversion) model re-voices it. Timing,
pitch contour, breaths, laughs, hesitations and emphasis all come from your
take, so micro-expressions survive. TTS often flattens them.

Open it from the sidebar: **Voice conversion** (under the voice workspaces) or
the command palette. The workspace has three tabs:

| Tab | What it does |
|---|---|
| Convert | Record or upload a take, pick an RVC voice, tune the controls, convert, A/B against the original, export. Recent conversions stay in history. |
| Voices | Import a trained `.pth` model plus its `.index`, rename or delete voices. |
| Train | Train a new voice from 10–30 minutes of clean audio, with live epoch progress and log. |

The clone workspace also has a **Convert with RVC voice** pill next to audio
quality. When it is on, every clone take is synthesized, then converted with the
chosen RVC voice before the take is watermarked and saved (hybrid TTS → RVC).
Voice Design auditions stay raw.

Only convert into voices you own or have explicit permission to use. All output
goes through VoiceStudio's synthetic-audio watermark like every other take.

## How it runs

VoiceStudio does **not** bundle RVC. Conversion and training run through
[Applio](https://github.com/IAHispano/Applio) (MIT), installed as a separate
checkout with its own Python environment. VoiceStudio starts one persistent
Applio worker process, keeps the current voice warm, unloads the previous model
when you switch voices (to free VRAM), and serializes conversion with TTS on the
GPU pool so an 8 GB card is not shared by two models at once.

If Applio is missing, the workspace shows a setup banner. Nothing is downloaded
until you press **Install Applio** and confirm.

The legacy `rvc-python` package is still supported as a fallback backend if it is
installed in VoiceStudio's own environment, but Applio is preferred.

## Install Applio on Windows (RTX 30/40 series)

Pick one:

1. **From the app**: Voice conversion → **Install Applio** → confirm. VoiceStudio
   clones Applio into `<data dir>/tools/applio`, creates a Python 3.12 venv with
   `uv`, installs Applio's `requirements.txt` with the CUDA 12.8 PyTorch wheels, then
   runs `core.py prerequisites` (RMVPE/FCPE/ContentVec). Git must be installed.
   Expect about 6 GB and 10–30 minutes.
2. **Manually** with Applio's own installer:
   ```powershell
   git clone https://github.com/IAHispano/Applio.git C:\AI\Applio
   cd C:\AI\Applio
   .\run-install.bat
   ```
   Then in VoiceStudio choose **Choose Applio folder** and pick `C:\AI\Applio`
   (the folder must contain `core.py` and `rvc\infer\infer.py`).
3. **Environment variables** (also works for the web UI / headless backend):
   ```powershell
   setx OMNIVOICE_APPLIO_DIR C:\AI\Applio
   # optional, if the interpreter is not env\python.exe or .venv\Scripts\python.exe
   setx OMNIVOICE_APPLIO_PYTHON C:\AI\Applio\env\python.exe
   ```

The status line then reads e.g. `Ready · Applio on cuda:0`.

## Recommended settings

| Control | Start with | Notes |
|---|---|---|
| Pitch | 0 | +12 male→female, −12 female→male. |
| Pitch detection | RMVPE | Crepe is smoother but slower; FCPE is fast. Applio does not offer harvest/pm. |
| Index rate | 0.5–0.75 | Higher = closer to the target timbre, lower = clearer. Only applies when the voice has an `.index`. |
| Protect | 0.33 | Keeps breaths, laughs and voiceless consonants from your take. Lower protects more; 0.5 turns protection off. |
| Volume envelope | 0–0.25 for acting | 0 keeps your original loudness contour (best for expressive delivery); 1 uses the model's. |
| Pitch smoothing | 3 | rvc-python backend only; Applio does not expose it. |
| Light noise cleanup | off | Applio's post-conversion cleanup. |

### Performance-transfer workflow

1. Record your line dry (no reverb), close to the mic, at 44.1/48 kHz.
2. Act it fully: laughs, breaths, whispers. RVC keeps them; it does not invent them.
3. Convert with protect 0.33 and volume envelope near 0.
4. A/B against the original. If consonants smear, lower protect; if the timbre is
   too "you", raise index rate.

## Training a voice (8 GB GPU)

- **Data**: 10–30 minutes of clean, dry speech from one speaker, 48 kHz WAV/FLAC
  if possible. Cover the range you need (calm, emphatic, laughing, whispering).
  Remove music and other speakers.
- **Defaults** (tuned for an RTX 3070 Laptop 8 GB): 48 kHz, RMVPE, batch size 8,
  300 epochs, checkpoint every 25 epochs, HiFi-GAN vocoder, pretrained G/D.
  If you hit CUDA out-of-memory, use batch size 4 and/or memory-saving mode
  (gradient checkpointing).
- 200–400 epochs is typical for 10–30 minutes. More is not always better; Applio
  saves weights at each checkpoint so you can compare.
- Training runs Applio's `preprocess → extract → train → index` stages. When it
  finishes, the voice is registered automatically and appears in **Voices**.
  The heavy intermediate feature folders are removed afterwards.
- Pretrained G/D weights are downloaded by Applio from Hugging Face on the first
  training run.

Desktop users can point training at a dataset folder directly (one-shot folder
capability); the web UI uploads the files instead.

## Where things are stored

| What | Location |
|---|---|
| Registered voices | `<data dir>/rvc_voices/<id>/model.pth`, `model.index`, `meta.json` |
| Training work dirs | `<data dir>/rvc_training/<job>` |
| Conversions | `<outputs dir>/vc_<id>.wav` (+ untouched `vc_<id>_src.*`), index in `<data dir>/rvc_conversions.json` (last 50) |
| Managed Applio | `<data dir>/tools/applio` |

Imports copy files; your original `.pth`/`.index` are never modified.

## Environment variables

| Variable | Meaning |
|---|---|
| `OMNIVOICE_APPLIO_DIR` | Applio checkout to use (overrides the app setting). |
| `OMNIVOICE_APPLIO_PYTHON` | Interpreter for that checkout. |
| `OMNIVOICE_RVC_DEVICE` | Force `cpu`, `cuda:0` or `mps` (default: auto, CUDA first). |
| `OMNIVOICE_RVC_VOICES_DIR` | Alternate voice registry folder. |
| `OMNIVOICE_RVC_TRAINING_DIR` | Alternate training work folder. |
| `OMNIVOICE_RVC_MODEL` / `OMNIVOICE_RVC_INDEX` / `OMNIVOICE_RVC_VOICE` | Legacy global post-processing hook for dubbing: convert every dubbed segment with this model (or registered voice id). |

## API

All routes live under `/voice-conversion` (see the OpenAPI page in Settings):
`GET status`, `PUT settings`, `POST setup` (requires `confirm: true`),
`GET/POST voices…`, `POST convert` (multipart), `GET/DELETE history`,
`POST train`, `GET jobs`, `GET train/{id}`, `POST train/{id}/cancel`.
`POST /generate` accepts `rvc_voice_id` (plus optional `rvc_pitch`,
`rvc_f0_method`, `rvc_index_rate`, `rvc_protect`) for the TTS → RVC pipeline.
Paths are never accepted over the API; desktop folder choices use one-shot
capabilities minted by the Electron main process.

## Applio facts this integration relies on

Verified against Applio `main` (commit `5171e1d`):

- `core.py` is a click CLI; the stages are `preprocess`, `extract`, `train` and `index`, and `prerequisites` downloads the base models.
- `core.py` can **exit 0 even when a stage fails**. It only prints e.g. "Training failed for model …". VoiceStudio therefore checks both the printed failure lines and the expected artifacts (`sliced_audios/`, `extracted/`, `<name>_<epoch>e_<step>s.pth`, `<name>.index`).
- `APPLIO_LOGS_DIR` relocates Applio's `logs/` folder; VoiceStudio uses it to keep each training job in its own work dir.
- `--gpu -` trains on the CPU (very slow, for testing only).
- Inference uses `rvc.infer.infer.VoiceConverter.convert_audio(...)`; there is no `filter_radius` and no harvest/pm in current Applio. The output sample rate is the model's (`32/40/48 kHz`).
