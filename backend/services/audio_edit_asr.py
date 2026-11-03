"""Audio Edit speech recognition: language and engine choice, local and cloud.

The Audio Edit transcript drives every edit, so the user picks the spoken
language and the recogniser before transcribing:

* ``whisper`` — the app's active ASR backend (Settings → Models → ASR) with
  the language forced when one is picked, so Kannada audio is never decoded
  as Telugu by Whisper's language detector.
* Kannada-specialised local models — AI4Bharat IndicConformer, AI4Bharat
  IndicWhisper (Vistaar) and ``vasista22/whisper-kannada-medium``. They are
  downloaded on request into ``<data>/audio_edit/asr_models``.
* Cloud recognisers — Sarvam, Smallest.ai, Google Cloud, Azure, OpenAI,
  Deepgram, AssemblyAI and Bhashini. Keys live encrypted in the settings
  store (``settings_store.set_secret``); environment variables are honoured
  as a fallback.

Every engine returns ``{"language", "words": [{word, start, end}]}``. Engines
without word timings get word times spread over their phrase or chunk
timings (``spread_words``), which is accurate to the phrase.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import shutil
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger("omnivoice.audio_edit.asr")

ASR_SR = 16000
HTTP_TIMEOUT_S = 300

# ── languages ────────────────────────────────────────────────────────────

#: (code, English name). ``auto`` lets the engine detect the language.
LANGUAGES: tuple[tuple[str, str], ...] = (
    ("auto", "Auto-detect"),
    ("kn", "Kannada"), ("en", "English"), ("hi", "Hindi"), ("ta", "Tamil"),
    ("te", "Telugu"), ("ml", "Malayalam"), ("mr", "Marathi"), ("bn", "Bengali"),
    ("gu", "Gujarati"), ("pa", "Punjabi"), ("or", "Odia"), ("ur", "Urdu"),
    ("as", "Assamese"), ("ne", "Nepali"), ("sa", "Sanskrit"),
    ("es", "Spanish"), ("fr", "French"), ("de", "German"), ("it", "Italian"),
    ("pt", "Portuguese"), ("nl", "Dutch"), ("ru", "Russian"), ("ar", "Arabic"),
    ("tr", "Turkish"), ("ja", "Japanese"), ("ko", "Korean"), ("zh", "Chinese"),
)
LANGUAGE_CODES = frozenset(code for code, _ in LANGUAGES)
_WHISPER_NAMES = {code: name.lower() for code, name in LANGUAGES if code != "auto"}
_WHISPER_NAMES["or"] = "odia"  # not in Whisper; kept so a forced pick fails loudly
_BCP47_IN = {"en": "en-IN", "or": "od-IN"}


def bcp47(code: str, *, india: bool = True) -> str:
    if code in _BCP47_IN:
        return _BCP47_IN[code] if india else ("en-US" if code == "en" else "or-IN")
    return f"{code}-IN" if india and code in _INDIC else {
        "en": "en-US", "es": "es-ES", "fr": "fr-FR", "de": "de-DE", "it": "it-IT", "pt": "pt-BR",
        "nl": "nl-NL", "ru": "ru-RU", "ar": "ar-SA", "tr": "tr-TR", "ja": "ja-JP", "ko": "ko-KR",
        "zh": "zh-CN"}.get(code, f"{code}-IN")


_INDIC = frozenset({"kn", "hi", "ta", "te", "ml", "mr", "bn", "gu", "pa", "or", "ur", "as", "ne", "sa", "en"})


class AsrError(Exception):
    """Mapped onto ``AudioEditError`` by the caller (same code/message/status)."""

    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


# ── provider keys ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class KeyField:
    id: str
    label: str
    secret: bool = True
    env: tuple[str, ...] = ()
    placeholder: str = ""


@dataclass(frozen=True)
class Provider:
    id: str
    label: str
    fields: tuple[KeyField, ...]
    docs_url: str


PROVIDERS: dict[str, Provider] = {p.id: p for p in (
    Provider("sarvam", "Sarvam AI", (KeyField("api_key", "API subscription key", env=("SARVAM_API_KEY",)),),
             "https://dashboard.sarvam.ai"),
    Provider("smallest", "Smallest.ai", (KeyField("api_key", "API key", env=("SMALLEST_API_KEY",)),),
             "https://console.smallest.ai"),
    Provider("google", "Google Cloud Speech-to-Text",
             (KeyField("api_key", "API key", env=("GOOGLE_STT_API_KEY", "GOOGLE_API_KEY")),),
             "https://console.cloud.google.com/apis/credentials"),
    Provider("azure", "Azure AI Speech", (
        KeyField("api_key", "Subscription key", env=("AZURE_SPEECH_KEY",)),
        KeyField("region", "Region", secret=False, env=("AZURE_SPEECH_REGION",), placeholder="centralindia"),
    ), "https://portal.azure.com"),
    Provider("openai", "OpenAI", (KeyField("api_key", "API key", env=("OPENAI_API_KEY",)),),
             "https://platform.openai.com/api-keys"),
    Provider("deepgram", "Deepgram", (KeyField("api_key", "API key", env=("DEEPGRAM_API_KEY",)),),
             "https://console.deepgram.com"),
    Provider("assemblyai", "AssemblyAI", (KeyField("api_key", "API key", env=("ASSEMBLYAI_API_KEY",)),),
             "https://www.assemblyai.com/dashboard"),
    Provider("bhashini", "Bhashini (ULCA)", (
        KeyField("api_key", "ULCA API key", env=("BHASHINI_API_KEY",)),
        KeyField("user_id", "User ID", secret=False, env=("BHASHINI_USER_ID",)),
    ), "https://bhashini.gov.in/ulca/user/register"),
)}


def _secret_name(provider: str, field_id: str) -> str:
    return f"audio_edit_asr.{provider}.{field_id}"


def _store_get(name: str) -> Optional[str]:
    from services import settings_store

    return settings_store.get_secret(name)


def _store_set(name: str, value: str) -> None:
    from services import settings_store

    settings_store.set_secret(name, value)


def provider_value(provider: str, field_id: str) -> Optional[str]:
    """Stored value, else the first non-empty environment fallback."""
    p = PROVIDERS[provider]
    f = next(f for f in p.fields if f.id == field_id)
    stored = _store_get(_secret_name(provider, field_id))
    if stored:
        return stored
    for env in f.env:
        if os.environ.get(env):
            return os.environ[env]
    return None


def _mask(value: str) -> str:
    return "••••" + value[-4:] if len(value) > 8 else "••••"


def provider_status(provider: str) -> dict:
    p = PROVIDERS[provider]
    fields = []
    for f in p.fields:
        stored = _store_get(_secret_name(provider, f.id))
        env = next((e for e in f.env if os.environ.get(e)), None)
        value = stored or (os.environ.get(env) if env else None)
        fields.append({
            "id": f.id, "label": f.label, "secret": f.secret, "placeholder": f.placeholder,
            "has_value": bool(value), "source": "settings" if stored else ("env" if env else None),
            "value": (value or "") if not f.secret else None,
            "masked": _mask(value) if (value and f.secret) else None,
        })
    return {"id": p.id, "label": p.label, "docs_url": p.docs_url, "fields": fields,
            "configured": all(fl["has_value"] for fl in fields)}


def list_providers() -> dict:
    return {"providers": [provider_status(pid) for pid in PROVIDERS]}


def set_provider_keys(provider: str, values: object) -> dict:
    """Save fields for *provider*. A missing field is left alone, ``""`` clears it."""
    if provider not in PROVIDERS:
        raise AsrError("invalid_asr_provider", "Unknown speech recognition provider.", 404)
    if not isinstance(values, dict):
        raise AsrError("invalid_request", "fields must be an object.", 422)
    known = {f.id for f in PROVIDERS[provider].fields}
    for key, value in values.items():
        if key not in known:
            raise AsrError("invalid_request", f"Unknown field {key!r}.", 422)
        if value is not None and not isinstance(value, str):
            raise AsrError("invalid_request", f"{key} must be a string.", 422)
        if value is None:
            continue
        value = value.strip()
        if len(value) > 4096:
            raise AsrError("invalid_request", f"{key} is too long.", 422)
        _store_set(_secret_name(provider, key), value)
    return provider_status(provider)


# ── engines ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Engine:
    id: str
    kind: str                         # "local" | "cloud"
    label: str
    description: str
    languages: Optional[frozenset[str]] = None   # None = any listed language
    provider: Optional[str] = None               # cloud key owner
    repo: Optional[str] = None                   # HF repo for local downloads
    url: Optional[str] = None                    # direct archive for local downloads
    size_gb: Optional[float] = None
    gated: bool = False
    needs: tuple[str, ...] = ()                  # importable modules the engine needs
    word_timestamps: bool = True


_KN = frozenset({"kn"})
#: Pulse pre-recorded single-language codes (no Kannada yet), plus auto-detect.
_PULSE = frozenset({"en", "hi", "de", "es", "ru", "it", "fr", "nl", "pt", "zh", "ja", "ko", "auto"})
_SARVAM = frozenset({"auto", "hi", "bn", "kn", "ml", "mr", "or", "pa", "ta", "te", "en", "gu", "as",
                     "ur", "ne", "sa"})
_BHASHINI = frozenset({"hi", "bn", "kn", "ml", "mr", "or", "pa", "ta", "te", "en", "gu", "as", "ur",
                       "ne", "sa"})

ENGINES: dict[str, Engine] = {e.id: e for e in (
    Engine("whisper", "local", "Whisper (active ASR engine)",
           "The ASR engine picked in Settings → Models → ASR, with the language forced when set."),
    Engine("indicconformer_kn", "local", "AI4Bharat IndicConformer (Kannada)",
           "AI4Bharat IndicConformer 600M multilingual, decoded as Kannada. Gated on Hugging Face: "
           "accept its terms and save a Hugging Face token first.",
           languages=_KN, repo="ai4bharat/indic-conformer-600m-multilingual", size_gb=2.4, gated=True,
           needs=("transformers", "onnxruntime", "torchaudio"), word_timestamps=False),
    Engine("indicwhisper_kn", "local", "AI4Bharat IndicWhisper (Kannada)",
           "AI4Bharat Vistaar IndicWhisper fine-tuned for Kannada (1.4 GB download).",
           languages=_KN, url="https://indicwhisper.objectstore.e2enetworks.net/kannada_models.zip",
           size_gb=1.4, needs=("transformers",)),
    Engine("vasista22_whisper_kn", "local", "Whisper Kannada medium (vasista22)",
           "vasista22/whisper-kannada-medium, Whisper medium fine-tuned on Kannada.",
           languages=_KN, repo="vasista22/whisper-kannada-medium", size_gb=3.1, needs=("transformers",)),
    Engine("sarvam", "cloud", "Sarvam Saaras", "Sarvam saaras:v3 speech-to-text (Indian languages).",
           languages=_SARVAM, provider="sarvam", word_timestamps=False),
    Engine("smallest", "cloud", "Smallest.ai Pulse", "Smallest.ai Pulse pre-recorded transcription.",
           languages=_PULSE, provider="smallest"),
    Engine("google", "cloud", "Google Cloud Speech-to-Text", "Google Cloud Speech-to-Text v1.",
           provider="google"),
    Engine("azure", "cloud", "Azure AI Speech", "Azure AI Speech fast transcription.", provider="azure"),
    Engine("openai", "cloud", "OpenAI Whisper API", "OpenAI whisper-1 with word timestamps.",
           provider="openai"),
    Engine("deepgram", "cloud", "Deepgram", "Deepgram Nova pre-recorded transcription.",
           provider="deepgram"),
    Engine("assemblyai", "cloud", "AssemblyAI", "AssemblyAI Universal pre-recorded transcription.",
           provider="assemblyai"),
    Engine("bhashini", "cloud", "Bhashini", "Government of India Bhashini ULCA ASR pipeline.",
           languages=_BHASHINI, provider="bhashini", word_timestamps=False),
)}
DEFAULT_ENGINE = "whisper"


def models_root() -> Path:
    from services import audio_edit

    return audio_edit.root() / "asr_models"


def _engine_dir(engine_id: str) -> Path:
    return models_root() / engine_id


def _module_missing(names: tuple[str, ...]) -> Optional[str]:
    import importlib.util

    for name in names:
        try:
            if importlib.util.find_spec(name) is None:
                return name
        except (ImportError, ValueError):
            return name
    return None


def _hf_installed(repo: str) -> Optional[str]:
    """Local snapshot path of *repo* in the HF cache (no network)."""
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo, local_files_only=True)
    except Exception:  # noqa: BLE001 — not cached, or huggingface_hub missing
        return None


def _archive_model_dir(engine_id: str) -> Optional[Path]:
    """The extracted checkpoint folder (the one holding config.json)."""
    base = _engine_dir(engine_id)
    if not (base / ".complete").is_file():
        return None
    for cfg in sorted(base.rglob("config.json")):
        if (cfg.parent / "preprocessor_config.json").is_file() or any(cfg.parent.glob("*.bin")) \
                or any(cfg.parent.glob("*.safetensors")):
            return cfg.parent
    return None


def local_model_path(engine: Engine) -> Optional[str]:
    if engine.id == "whisper":
        return "active"
    if engine.repo:
        return _hf_installed(engine.repo)
    if engine.url:
        found = _archive_model_dir(engine.id)
        return str(found) if found else None
    return None


# ── downloads ────────────────────────────────────────────────────────────

@dataclass
class Download:
    engine: str
    status: str = "running"            # running | succeeded | failed
    progress: float = 0.0
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"engine": self.engine, "status": self.status, "progress": round(self.progress, 3),
                "error": self.error, "started_at": self.started_at}


_DOWNLOADS: dict[str, Download] = {}
_DL_LOCK = threading.Lock()
RUN_DOWNLOAD_INLINE = False


def _hf_token() -> Optional[str]:
    try:
        from services import token_resolver

        found = token_resolver.resolve()
        return found.token if found else None
    except Exception:  # noqa: BLE001
        return None


_HF_IGNORE = ["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model.ot"]


def _download_hf(engine: Engine, dl: Download) -> None:
    from huggingface_hub import snapshot_download

    dl.progress = 0.05
    # transformers loads the PyTorch weights; skip the Flax/TF copies some
    # repos also ship (vasista22/whisper-kannada-medium: +2.9 GB of msgpack).
    snapshot_download(engine.repo, token=_hf_token(), ignore_patterns=_HF_IGNORE)


def _download_archive(engine: Engine, dl: Download) -> None:
    import requests

    base = _engine_dir(engine.id)
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    archive = base / "model.zip.part"
    with requests.get(engine.url, stream=True, timeout=60) as res:
        res.raise_for_status()
        total = int(res.headers.get("content-length") or 0)
        done = 0
        with open(archive, "wb") as fh:
            for chunk in res.iter_content(1024 * 1024):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    dl.progress = min(0.9, 0.9 * done / total)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            target = (base / member).resolve()
            if not str(target).startswith(str(base.resolve())):
                raise AsrError("asr_download_failed", "The model archive has unsafe paths.", 502)
        zf.extractall(base)
    archive.unlink(missing_ok=True)
    (base / ".complete").write_text(str(time.time()), encoding="utf-8")


def _run_download(engine: Engine, dl: Download) -> None:
    try:
        (_download_hf if engine.repo else _download_archive)(engine, dl)
        if not local_model_path(engine):
            raise AsrError("asr_download_failed", "The download finished but no model was found.", 502)
        dl.status, dl.progress = "succeeded", 1.0
    except Exception as exc:  # noqa: BLE001 — reported through the status endpoint
        logger.warning("Audio Edit ASR download for %s failed: %s", engine.id, exc)
        msg = getattr(exc, "message", None) or f"{type(exc).__name__}: {exc}"
        if engine.gated and ("401" in msg or "403" in msg or "gated" in msg.lower()):
            msg = (f"{engine.repo} is gated. Accept its terms on huggingface.co and save a "
                   "Hugging Face token in Settings, then retry.")
        dl.status, dl.error = "failed", msg[:500]


def start_download(engine_id: str) -> dict:
    engine = ENGINES.get(engine_id)
    if engine is None or engine.kind != "local" or engine.id == "whisper":
        raise AsrError("invalid_asr_engine", "That engine has nothing to download.", 404)
    with _DL_LOCK:
        current = _DOWNLOADS.get(engine_id)
        if current and current.status == "running":
            return current.to_dict()
        dl = Download(engine_id)
        _DOWNLOADS[engine_id] = dl
    if RUN_DOWNLOAD_INLINE:
        _run_download(engine, dl)
    else:
        threading.Thread(target=_run_download, args=(engine, dl), daemon=True,
                         name=f"audio-edit-asr-dl-{engine_id}").start()
    return dl.to_dict()


# ── engine listing ───────────────────────────────────────────────────────

def engine_status(engine: Engine) -> dict:
    out: dict[str, Any] = {
        "id": engine.id, "kind": engine.kind, "label": engine.label, "description": engine.description,
        "languages": sorted(engine.languages) if engine.languages else None,
        "word_timestamps": engine.word_timestamps, "provider": engine.provider,
        "size_gb": engine.size_gb, "gated": engine.gated,
        "installed": True, "configured": True, "available": True, "reason": None, "download": None,
    }
    if engine.kind == "cloud":
        configured = provider_status(engine.provider)["configured"]
        out.update(configured=configured, available=configured,
                   reason=None if configured else "key_missing")
        return out
    if engine.id == "whisper":
        return out
    installed = local_model_path(engine) is not None
    missing = _module_missing(engine.needs)
    with _DL_LOCK:
        dl = _DOWNLOADS.get(engine.id)
    out.update(installed=installed, available=installed and not missing,
               reason=None if installed and not missing else ("not_installed" if not installed
                                                              else f"missing_module:{missing}"),
               download=dl.to_dict() if dl else None)
    return out


def engines_overview() -> dict:
    return {
        "default_engine": DEFAULT_ENGINE,
        "languages": [{"code": c, "name": n} for c, n in LANGUAGES],
        "engines": [engine_status(e) for e in ENGINES.values()],
    }


def validate_options(raw: object) -> dict:
    """``{engine, language}`` from a request body (missing body = defaults)."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise AsrError("invalid_request", "The transcription options must be an object.", 422)
    engine_id = raw.get("engine") or DEFAULT_ENGINE
    language = raw.get("language") or "auto"
    if engine_id not in ENGINES:
        raise AsrError("invalid_asr_engine", f"Unknown speech recogniser {engine_id!r}.", 422)
    if language not in LANGUAGE_CODES:
        raise AsrError("invalid_asr_language", f"Unknown language {language!r}.", 422)
    engine = ENGINES[engine_id]
    if engine.languages is not None and language not in engine.languages:
        if language == "auto" and len(engine.languages) == 1:
            language = next(iter(engine.languages))
        else:
            raise AsrError("asr_language_unsupported",
                           f"{engine.label} does not transcribe that language.", 422)
    return {"engine": engine_id, "language": language}


def check_ready(options: dict) -> None:
    """Fail fast (before a job starts) when a key or model is missing."""
    status = engine_status(ENGINES[options["engine"]])
    if status["available"]:
        return
    if status["reason"] == "key_missing":
        raise AsrError("asr_key_missing",
                       f"Add your {PROVIDERS[status['provider']].label} key in Settings → Models → ASR.", 409)
    if status["reason"] == "not_installed":
        raise AsrError("asr_model_missing", f"Download {status['label']} first.", 409)
    raise AsrError("asr_model_missing",
                   f"{status['label']} needs the Python package {status['reason'].split(':', 1)[-1]}.", 409)


# ── audio helpers ────────────────────────────────────────────────────────

def load_mono_16k(path: Path) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    mono = data.mean(axis=1)
    if sr != ASR_SR:
        from services import audio_edit_dsp as dsp

        mono = dsp.resample(mono[None, :], int(sr), ASR_SR)[0]
    return np.ascontiguousarray(mono, dtype=np.float32)


def chunk_bounds(audio: np.ndarray, max_s: float, sr: int = ASR_SR) -> list[tuple[int, int]]:
    """Split into windows of at most *max_s*, cutting at the quietest 50 ms in
    the last third of each window so words are rarely cut in half."""
    n = len(audio)
    limit = int(max_s * sr)
    if n <= limit:
        return [(0, n)]
    hop = int(0.05 * sr)
    out, start = [], 0
    while start < n:
        end = min(n, start + limit)
        if end < n:
            lo = start + (2 * limit) // 3
            frames = [(float(np.mean(audio[i:i + hop] ** 2)), i) for i in range(lo, end - hop, hop)]
            if frames:
                end = min(frames)[1] + hop // 2
        out.append((start, end))
        start = end
    return out


def wav_bytes(audio: np.ndarray, sr: int = ASR_SR) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def spread_words(text: str, start: float, end: float) -> list[dict]:
    """Word times for an untimed phrase, proportional to word length."""
    tokens = [t for t in str(text or "").split() if t]
    if not tokens:
        return []
    end = max(end, start + 0.05 * len(tokens))
    weights = np.array([max(1, len(t)) + 1 for t in tokens], dtype=np.float64)
    edges = start + np.concatenate([[0.0], np.cumsum(weights)]) / weights.sum() * (end - start)
    out = []
    for i, token in enumerate(tokens):
        a, b = float(edges[i]), float(edges[i + 1])
        out.append({"word": token, "start": round(a, 3), "end": round(b - 0.01 if b - 0.01 > a else b, 3)})
    return out


def _speech_bounds(audio: np.ndarray, sr: int = ASR_SR) -> tuple[float, float]:
    """First and last loud frame, so spread words skip leading/trailing silence."""
    if not len(audio):
        return 0.0, 0.0
    hop = max(1, int(0.02 * sr))
    energy = np.array([np.sqrt(np.mean(audio[i:i + hop] ** 2)) for i in range(0, len(audio), hop)])
    loud = np.nonzero(energy > max(1e-4, 0.1 * float(energy.max() or 0)))[0]
    if not len(loud):
        return 0.0, len(audio) / sr
    return loud[0] * hop / sr, min(len(audio), (loud[-1] + 1) * hop) / sr


def _offset(words: list[dict], offset: float) -> list[dict]:
    return [{"word": w["word"], "start": round(w["start"] + offset, 3), "end": round(w["end"] + offset, 3)}
            for w in words]


ChunkFn = Callable[[np.ndarray, float], tuple[list[dict], Optional[str]]]


def transcribe_chunked(audio: np.ndarray, max_s: float, fn: ChunkFn,
                       check_cancel: Callable[[], None] = lambda: None,
                       on_progress: Callable[[float], None] = lambda p: None) -> tuple[list[dict], Optional[str]]:
    """Run *fn(chunk, chunk_start_s)* per chunk; *fn* returns words relative to the chunk."""
    words: list[dict] = []
    detected = None
    bounds = chunk_bounds(audio, max_s)
    for i, (a, b) in enumerate(bounds):
        check_cancel()
        part = audio[a:b]
        if len(part) < int(0.2 * ASR_SR):
            continue
        got, lang = fn(part, a / ASR_SR)
        detected = detected or lang
        words.extend(_offset(got, a / ASR_SR))
        on_progress((i + 1) / len(bounds))
    return words, detected


def _untimed(text: str, part: np.ndarray) -> list[dict]:
    lo, hi = _speech_bounds(part)
    return spread_words(text, lo, hi)


# ── local engines ────────────────────────────────────────────────────────

def _whisper_active(path: Path, language: Optional[str]) -> dict:
    from services.asr_backend import (
        PyTorchWhisperBackend, get_active_asr_backend, transcribe_request_kwargs)

    backend = get_active_asr_backend()
    if language:
        kwargs = transcribe_request_kwargs(backend, {"language": language})
        if "language" not in kwargs:
            # The active engine cannot be told the language (Parakeet, Moonshine, ...):
            # use transformers Whisper, which can, rather than risk a wrong detection.
            logger.info("Audio Edit: %s cannot force a language; using PyTorch Whisper", backend.id)
            backend = PyTorchWhisperBackend()
            kwargs = {"language": language}
    else:
        kwargs = {}
    result = backend.transcribe(str(path), word_timestamps=True, **kwargs)
    if isinstance(result, dict) and language:
        result = {**result, "language": language}
    return result


_HF_PIPES: dict[str, Any] = {}
_HF_LOCK = threading.Lock()


def _hf_whisper_pipe(model_path: str):
    with _HF_LOCK:
        pipe = _HF_PIPES.get(model_path)
        if pipe is not None:
            return pipe
        _HF_PIPES.clear()
        import torch
        from transformers import pipeline

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        pipe = pipeline("automatic-speech-recognition", model=model_path, torch_dtype=dtype, device=device)
        _HF_PIPES[model_path] = pipe
        return pipe


def release_local_models() -> None:
    with _HF_LOCK:
        _HF_PIPES.clear()
        _CONFORMER.clear()


def _hf_whisper(model_path: str, audio: np.ndarray, language: str, check_cancel) -> dict:
    pipe = _hf_whisper_pipe(model_path)
    generate_kwargs = {"task": "transcribe"}
    if language and language != "auto":
        generate_kwargs["language"] = _WHISPER_NAMES.get(language, language)

    def run(part: np.ndarray, ts) -> dict:
        return pipe({"raw": part, "sampling_rate": ASR_SR}, return_timestamps=ts,
                    generate_kwargs=dict(generate_kwargs), chunk_length_s=30, batch_size=8)

    # Fine-tunes often ship without alignment heads, so word timestamps can
    # fail; phrase timestamps then give the word times by spreading.
    try:
        out = run(audio, "word")
        words = [{"word": c["text"].strip(), "start": c["timestamp"][0], "end": c["timestamp"][1]}
                 for c in out.get("chunks") or []
                 if c.get("text", "").strip() and c["timestamp"][0] is not None]
        for w in words:
            if w["end"] is None:
                w["end"] = w["start"] + 0.3
        if words:
            return {"language": language, "words": words}
    except Exception as exc:  # noqa: BLE001
        logger.info("word timestamps unavailable for %s (%s); using phrase timings", model_path, exc)
    check_cancel()
    out = run(audio, True)
    words: list[dict] = []
    total = len(audio) / ASR_SR
    for c in out.get("chunks") or []:
        a, b = c.get("timestamp") or (None, None)
        a = 0.0 if a is None else float(a)
        b = total if b is None else float(b)
        words.extend(spread_words(c.get("text", ""), a, b))
    if not words and out.get("text"):
        lo, hi = _speech_bounds(audio)
        words = spread_words(out["text"], lo, hi)
    return {"language": language, "words": words}


_CONFORMER: dict[str, Any] = {}


def _indicconformer(model_path: str, audio: np.ndarray, language: str, check_cancel, on_progress) -> dict:
    import torch

    with _HF_LOCK:
        model = _CONFORMER.get(model_path)
        if model is None:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            _CONFORMER.clear()
            _CONFORMER[model_path] = model

    def fn(part: np.ndarray, _start: float):
        with torch.no_grad():
            text = model(torch.from_numpy(part)[None, :], language, "ctc")
        if isinstance(text, (list, tuple)):
            text = " ".join(str(t) for t in text)
        return _untimed(str(text), part), language

    words, _ = transcribe_chunked(audio, 15.0, fn, check_cancel, on_progress)
    return {"language": language, "words": words}


# ── cloud engines ────────────────────────────────────────────────────────

def _http():
    import requests

    return requests


def _raise_for(res, provider: str) -> None:
    if res.status_code < 400:
        return
    if res.status_code in (401, 403):
        raise AsrError("asr_auth_failed",
                       f"{PROVIDERS[provider].label} rejected the key. Check it in Settings → Models → ASR.", 502)
    body = (res.text or "")[:300]
    raise AsrError("asr_failed", f"{PROVIDERS[provider].label} returned HTTP {res.status_code}: {body}", 502)


def _key(provider: str, field_id: str = "api_key") -> str:
    value = provider_value(provider, field_id)
    if not value:
        raise AsrError("asr_key_missing",
                       f"Add your {PROVIDERS[provider].label} key in Settings → Models → ASR.", 409)
    return value


def _openai(audio, language, check_cancel, on_progress) -> dict:
    key = _key("openai")

    def fn(part, _start):
        data = {"model": "whisper-1", "response_format": "verbose_json",
                "timestamp_granularities[]": "word"}
        if language != "auto":
            data["language"] = language
        res = _http().post("https://api.openai.com/v1/audio/transcriptions",
                           headers={"Authorization": f"Bearer {key}"}, data=data,
                           files={"file": ("audio.wav", wav_bytes(part), "audio/wav")}, timeout=HTTP_TIMEOUT_S)
        _raise_for(res, "openai")
        body = res.json()
        words = [{"word": w["word"], "start": float(w["start"]), "end": float(w["end"])}
                 for w in body.get("words") or []]
        return words or _untimed(body.get("text", ""), part), body.get("language")

    # 25 MB upload cap: 10 min of 16 kHz PCM16 is ~19 MB.
    words, detected = transcribe_chunked(audio, 600.0, fn, check_cancel, on_progress)
    return {"language": language if language != "auto" else _iso(detected), "words": words}


def _deepgram(audio, language, check_cancel, on_progress) -> dict:
    key = _key("deepgram")
    params = {"model": "nova-3" if language in ("auto", "en", "es", "fr", "de", "hi", "ru", "pt", "ja",
                                                "it", "nl") else "nova-2",
              "smart_format": "true", "punctuate": "true"}
    if language == "auto":
        params["detect_language"] = "true"
    else:
        params["language"] = language
    res = _http().post("https://api.deepgram.com/v1/listen", params=params,
                       headers={"Authorization": f"Token {key}", "Content-Type": "audio/wav"},
                       data=wav_bytes(audio), timeout=HTTP_TIMEOUT_S)
    _raise_for(res, "deepgram")
    channel = (res.json().get("results") or {}).get("channels", [{}])[0]
    alt = (channel.get("alternatives") or [{}])[0]
    words = [{"word": w.get("punctuated_word") or w["word"], "start": float(w["start"]), "end": float(w["end"])}
             for w in alt.get("words") or []]
    on_progress(1.0)
    return {"language": language if language != "auto" else channel.get("detected_language"), "words": words}


def _assemblyai(audio, language, check_cancel, on_progress) -> dict:
    key = _key("assemblyai")
    req = _http()
    headers = {"authorization": key}
    up = req.post("https://api.assemblyai.com/v2/upload", headers=headers, data=wav_bytes(audio),
                  timeout=HTTP_TIMEOUT_S)
    _raise_for(up, "assemblyai")
    body: dict[str, Any] = {"audio_url": up.json()["upload_url"], "punctuate": True, "format_text": True}
    if language == "auto":
        body["language_detection"] = True
    else:
        body["language_code"] = language
    start = req.post("https://api.assemblyai.com/v2/transcript", headers=headers, json=body, timeout=60)
    _raise_for(start, "assemblyai")
    tid = start.json()["id"]
    deadline = time.time() + 1800
    while time.time() < deadline:
        check_cancel()
        poll = req.get(f"https://api.assemblyai.com/v2/transcript/{tid}", headers=headers, timeout=60)
        _raise_for(poll, "assemblyai")
        data = poll.json()
        if data.get("status") == "completed":
            words = [{"word": w["text"], "start": w["start"] / 1000.0, "end": w["end"] / 1000.0}
                     for w in data.get("words") or []]
            return {"language": data.get("language_code") or language, "words": words}
        if data.get("status") == "error":
            raise AsrError("asr_failed", f"AssemblyAI: {data.get('error')}", 502)
        on_progress(0.5)
        time.sleep(2)
    raise AsrError("asr_failed", "AssemblyAI did not finish within 30 minutes.", 504)


def _google(audio, language, check_cancel, on_progress) -> dict:
    key = _key("google")
    lang = "en-US" if language == "auto" else bcp47(language)

    def fn(part, _start):
        body = {"config": {"encoding": "LINEAR16", "sampleRateHertz": ASR_SR, "languageCode": lang,
                           "enableWordTimeOffsets": True, "enableAutomaticPunctuation": True},
                "audio": {"content": base64.b64encode((np.clip(part, -1, 1) * 32767).astype("<i2")
                                                      .tobytes()).decode("ascii")}}
        res = _http().post("https://speech.googleapis.com/v1/speech:recognize", params={"key": key},
                           json=body, timeout=HTTP_TIMEOUT_S)
        _raise_for(res, "google")
        words = []
        for r in res.json().get("results") or []:
            alt = (r.get("alternatives") or [{}])[0]
            for w in alt.get("words") or []:
                words.append({"word": w["word"], "start": _gsec(w.get("startTime")),
                              "end": _gsec(w.get("endTime"))})
        return words, None

    # Synchronous recognize accepts up to 60 s.
    words, _ = transcribe_chunked(audio, 55.0, fn, check_cancel, on_progress)
    return {"language": None if language == "auto" else language, "words": words}


def _gsec(value) -> float:
    if not value:
        return 0.0
    return float(str(value).rstrip("s"))


def _azure(audio, language, check_cancel, on_progress) -> dict:
    key = _key("azure")
    region = _key("azure", "region")
    definition: dict[str, Any] = {"locales": [bcp47(language)] if language != "auto"
                                  else ["en-IN", "hi-IN", "kn-IN", "ta-IN", "te-IN"]}
    res = _http().post(
        f"https://{region}.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe",
        params={"api-version": "2024-11-15"}, headers={"Ocp-Apim-Subscription-Key": key},
        files={"audio": ("audio.wav", wav_bytes(audio), "audio/wav"),
               "definition": (None, json.dumps(definition), "application/json")},
        timeout=HTTP_TIMEOUT_S)
    _raise_for(res, "azure")
    words, detected = [], None
    for phrase in res.json().get("phrases") or []:
        detected = detected or phrase.get("locale")
        got = [{"word": w["text"], "start": w["offsetMilliseconds"] / 1000.0,
                "end": (w["offsetMilliseconds"] + w["durationMilliseconds"]) / 1000.0}
               for w in phrase.get("words") or []]
        if not got:
            a = phrase.get("offsetMilliseconds", 0) / 1000.0
            got = spread_words(phrase.get("text", ""), a, a + phrase.get("durationMilliseconds", 0) / 1000.0)
        words.extend(got)
    on_progress(1.0)
    return {"language": language if language != "auto" else _iso(detected), "words": words}


def _sarvam(audio, language, check_cancel, on_progress) -> dict:
    key = _key("sarvam")

    def fn(part, _start):
        data = {"model": "saaras:v3", "mode": "transcribe", "with_timestamps": "true",
                "language_code": "unknown" if language == "auto" else bcp47(language)}
        res = _http().post("https://api.sarvam.ai/speech-to-text", headers={"api-subscription-key": key},
                           data=data, files={"file": ("audio.wav", wav_bytes(part), "audio/wav")},
                           timeout=HTTP_TIMEOUT_S)
        _raise_for(res, "sarvam")
        body = res.json()
        ts = body.get("timestamps") or {}
        phrases = list(zip(ts.get("words") or [], ts.get("start_time_seconds") or [],
                           ts.get("end_time_seconds") or []))
        words = [w for text, a, b in phrases for w in spread_words(text, float(a), float(b))]
        return words or _untimed(body.get("transcript", ""), part), _iso(body.get("language_code"))

    # The REST endpoint takes clips under 30 s.
    words, detected = transcribe_chunked(audio, 28.0, fn, check_cancel, on_progress)
    return {"language": language if language != "auto" else detected, "words": words}


def _smallest(audio, language, check_cancel, on_progress) -> dict:
    key = _key("smallest")

    def fn(part, _start):
        res = _http().post("https://api.smallest.ai/waves/v1/stt/",
                           params={"model": "pulse", "language": language, "word_timestamps": "true"},
                           headers={"Authorization": f"Bearer {key}", "Content-Type": "application/octet-stream"},
                           data=wav_bytes(part), timeout=HTTP_TIMEOUT_S)
        _raise_for(res, "smallest")
        body = res.json()
        words = [{"word": w["word"], "start": float(w["start"]), "end": float(w["end"])}
                 for w in body.get("words") or [] if w.get("start") is not None]
        return words or _untimed(body.get("transcription", ""), part), body.get("detected_language")

    words, detected = transcribe_chunked(audio, 300.0, fn, check_cancel, on_progress)
    return {"language": language if language != "auto" else detected, "words": words}


_BHASHINI_PIPELINE = "64392f96daac500b55c543cd"


def _bhashini(audio, language, check_cancel, on_progress) -> dict:
    key = _key("bhashini")
    user = _key("bhashini", "user_id")
    if language == "auto":
        raise AsrError("asr_language_unsupported", "Bhashini needs the language set.", 422)
    req = _http()
    cfg = req.post("https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline",
                   headers={"userID": user, "ulcaApiKey": key},
                   json={"pipelineTasks": [{"taskType": "asr", "config": {"language": {"sourceLanguage": language}}}],
                         "pipelineRequestConfig": {"pipelineId": _BHASHINI_PIPELINE}}, timeout=60)
    _raise_for(cfg, "bhashini")
    cfg = cfg.json()
    endpoint = cfg["pipelineInferenceAPIEndPoint"]
    auth = endpoint["inferenceApiKey"]
    service = cfg["pipelineResponseConfig"][0]["config"][0]["serviceId"]

    def fn(part, _start):
        res = req.post(endpoint["callbackUrl"], headers={auth["name"]: auth["value"]}, json={
            "pipelineTasks": [{"taskType": "asr", "config": {
                "language": {"sourceLanguage": language}, "serviceId": service,
                "audioFormat": "wav", "samplingRate": ASR_SR}}],
            "inputData": {"audio": [{"audioContent": base64.b64encode(wav_bytes(part)).decode("ascii")}]},
        }, timeout=HTTP_TIMEOUT_S)
        _raise_for(res, "bhashini")
        out = res.json()["pipelineResponse"][0]["output"]
        return _untimed(" ".join(o.get("source", "") for o in out), part), language

    words, _ = transcribe_chunked(audio, 25.0, fn, check_cancel, on_progress)
    return {"language": language, "words": words}


def _iso(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    code = str(code).split("-")[0].lower()
    return "or" if code == "od" else code


#: Cloud engine id → adapter name (looked up at call time so tests can fake one).
_CLOUD = {"openai": "_openai", "deepgram": "_deepgram", "assemblyai": "_assemblyai", "google": "_google",
          "azure": "_azure", "sarvam": "_sarvam", "smallest": "_smallest", "bhashini": "_bhashini"}


# ── entry point ──────────────────────────────────────────────────────────

def transcribe(path: Path, options: dict, *, check_cancel: Callable[[], None] = lambda: None,
               on_progress: Callable[[float], None] = lambda p: None) -> dict:
    """Transcribe *path* with ``options = {engine, language}`` (validated)."""
    engine = ENGINES[options["engine"]]
    language = options["language"]
    forced = None if language == "auto" else language
    if engine.id == "whisper":
        return _whisper_active(path, forced)
    check_ready(options)
    audio = load_mono_16k(path)
    if engine.kind == "cloud":
        return globals()[_CLOUD[engine.id]](audio, language, check_cancel, on_progress)
    model_path = local_model_path(engine)
    if engine.id == "indicconformer_kn":
        return _indicconformer(model_path, audio, language, check_cancel, on_progress)
    return _hf_whisper(model_path, audio, language, check_cancel)
