"""Voice Dataset Builder (review item 4).

Turns long, messy recordings of one speaker into a clean, labelled dataset
for RVC training, index building and TTS fine-tuning:

    decode → [vocal separation] → VAD segmentation → speaker filter
    → transcription → quality scoring → nonverbal (laugh/breath) tagging
    → ``<DATA_DIR>/voice_datasets/<id>/`` with clips + manifest

Every stage degrades instead of failing when its model is absent, and the
summary says exactly what ran:

* separation — HTDemucs from the HF cache (``adefossez/HTDemucs``
  safetensors, built from its embedded config) or the torch-hub ``.th``;
  skipped with a warning when neither is on disk. Never downloads.
* VAD — Silero v6 ONNX bundled with faster-whisper (always local).
* speaker filter — SpeechBrain ECAPA when ``speechbrain/spkrec-ecapa-voxceleb``
  is in the HF cache, else MFCC statistics with a robust outlier rule.
* transcription — faster-whisper (``Systran/faster-whisper-large-v3`` by
  default) from the HF cache only (``local_files_only``).
* quality — DSP: SNR estimate, clipping, level, ASR confidence.
* tagging — breath (broadband, low-level gaps) and laugh (ASR markers, or
  rhythmic 3.5–7 Hz bursts with few recognised words) heuristics; snippets
  are exported to ``nonverbal/`` for the nonverbal library.

Layout of a dataset::

    dataset.json      summary: options, stages run, stats, warnings
    manifest.jsonl    one line per segment (accepted and rejected + reasons)
    metadata.csv      LJSpeech-style ``clips/<file>|<text>`` (accepted only)
    clips/*.wav       accepted clips, mono 16-bit at the source rate (≤48 kHz)
    nonverbal/*.wav   tagged breath/laugh snippets (+ nonverbal.jsonl)
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

ANALYSIS_SR = 16_000
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")
_LAUGH_TEXT_RE = re.compile(
    r"(\(|\[)\s*(laugh|laughs|laughter|chuckle|giggle)|\b(ha){2,}\b|\bha ha\b|\b(he){2,}\b|ಹಹ|ಹ್ಹ",
    re.I,
)


# ── Paths & ids ──────────────────────────────────────────────────────────────

def datasets_root() -> Path:
    override = os.environ.get("OMNIVOICE_DATASETS_DIR", "").strip()
    if override:
        root = Path(override)
    else:
        from core.config import DATA_DIR

        root = Path(DATA_DIR) / "voice_datasets"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _check_id(dataset_id: str) -> str:
    if not isinstance(dataset_id, str) or not _ID_RE.match(dataset_id):
        raise KeyError(dataset_id)
    return dataset_id


def dataset_dir(dataset_id: str) -> Path:
    path = datasets_root() / _check_id(dataset_id)
    if not (path / "dataset.json").is_file():
        raise KeyError(dataset_id)
    return path


def dataset_clips_dir(dataset_id: str) -> Path:
    """Accepted clips of a finished dataset (used by RVC index builds)."""
    clips = dataset_dir(dataset_id) / "clips"
    if not clips.is_dir() or not any(clips.glob("*.wav")):
        raise KeyError(dataset_id)
    return clips


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "dataset").lower()).strip("-")[:40]
    return s or "dataset"


# ── Options / results ───────────────────────────────────────────────────────

@dataclass
class BuildOptions:
    name: str = "Dataset"
    language: Optional[str] = None       # ISO code for ASR ("kn", "en"); None = detect
    separate: bool = True
    transcribe: bool = True
    speaker_filter: bool = True
    min_clip_s: float = 1.5
    max_clip_s: float = 12.0
    min_quality: float = 0.45
    speaker_threshold: Optional[float] = None  # method default when None
    tag_nonverbal: bool = True
    asr_model: str = "Systran/faster-whisper-large-v3"
    free_gpu: bool = True
    device: Optional[str] = None

    def validate(self) -> None:
        if not (0.5 <= self.min_clip_s < self.max_clip_s <= 30.0):
            raise ValueError("Need 0.5 ≤ min_clip_s < max_clip_s ≤ 30.")
        if not (0.0 <= self.min_quality <= 1.0):
            raise ValueError("min_quality must be between 0 and 1.")
        if self.speaker_threshold is not None and not (-1.0 <= self.speaker_threshold <= 1.0):
            raise ValueError("speaker_threshold must be between -1 and 1.")


@dataclass
class Segment:
    source: str
    start: float
    end: float
    index: int = 0
    speaker_sim: Optional[float] = None
    speaker_ok: bool = True
    text: str = ""
    asr_conf: Optional[float] = None
    language: Optional[str] = None
    quality: Optional[float] = None
    metrics: dict = field(default_factory=dict)
    tags: list = field(default_factory=list)
    accepted: bool = False
    reasons: list = field(default_factory=list)
    file: Optional[str] = None

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)


# ── Model discovery (never downloads) ───────────────────────────────────────

def _hf_snapshot(repo_id: str, required: tuple[str, ...] = ()) -> Optional[Path]:
    try:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(repo_id, local_files_only=True))
    except Exception:  # noqa: BLE001 — not cached
        return None
    if all((path / name).exists() for name in required):
        return path
    return None


def _htdemucs_weights() -> Optional[Path]:
    snap = _hf_snapshot("adefossez/HTDemucs")
    if snap is not None:
        found = sorted(snap.glob("*.safetensors"))
        if found:
            return found[0]
    try:
        import torch

        hub = Path(torch.hub.get_dir()) / "checkpoints"
        found = sorted(hub.glob("955717e8-*.th"))
        if found:
            return found[0]
    except Exception:  # noqa: BLE001
        pass
    return None


def _ecapa_dir() -> Optional[Path]:
    return _hf_snapshot("speechbrain/spkrec-ecapa-voxceleb",
                        ("hyperparams.yaml", "embedding_model.ckpt"))


def _whisper_dir(repo_id: str) -> Optional[Path]:
    return _hf_snapshot(repo_id, ("model.bin",))


def capabilities(asr_model: str = BuildOptions.asr_model) -> dict:
    def _has(mod: str) -> bool:
        import importlib.util

        return importlib.util.find_spec(mod) is not None

    demucs = _htdemucs_weights() if _has("demucs") else None
    ecapa = _ecapa_dir() if _has("speechbrain") else None
    whisper = _whisper_dir(asr_model) if _has("faster_whisper") else None
    return {
        "separation": {"available": demucs is not None, "model": "htdemucs",
                       "weights": str(demucs) if demucs else None,
                       "missing": None if demucs else "adefossez/HTDemucs (~84 MB)"},
        "vad": {"available": _has("faster_whisper"), "model": "silero-vad v6 (bundled)"},
        "speaker": {"available": True, "method": "ecapa" if ecapa else "mfcc-stats",
                    "missing": None if ecapa else "speechbrain/spkrec-ecapa-voxceleb (~83 MB)"},
        "transcription": {"available": whisper is not None, "model": asr_model,
                          "missing": None if whisper else f"{asr_model}"},
        "quality": {"available": True, "method": "dsp+asr-confidence"},
        "nonverbal": {"available": True, "method": "heuristic"},
    }


# ── Audio helpers ────────────────────────────────────────────────────────────

def _ffmpeg() -> str:
    try:
        from services.ffmpeg_utils import find_ffmpeg

        path = find_ffmpeg()
        if path:
            return path
    except Exception:  # noqa: BLE001
        pass
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg was not found.")
    return path


def decode(path: str, out_wav: Path, sr: int = 44_100) -> Path:
    cmd = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
           "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_f32le", str(out_wav)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not out_wav.is_file():
        raise RuntimeError(f"Could not decode {Path(path).name}: {proc.stderr.strip()[-300:]}")
    return out_wav


def _read(path: Path):
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data), int(sr)


def _resample(x, sr: int, target: int):
    import numpy as np

    if sr == target:
        return x
    import torch
    import torchaudio.functional as F

    return F.resample(torch.from_numpy(np.ascontiguousarray(x)), sr, target).numpy()


def _frame_db(x, sr: int, frame_s: float = 0.02):
    import numpy as np

    hop = max(1, int(sr * frame_s))
    n = len(x) // hop
    if n == 0:
        return np.array([-120.0])
    frames = x[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


# ── Stage: separation ───────────────────────────────────────────────────────

def load_htdemucs(weights: Path, device: str):
    """Build HTDemucs from the HF safetensors (config in its metadata) or a
    torch-hub ``.th`` package — no network."""
    import torch

    if weights.suffix == ".th":
        from demucs.states import load_model

        model = load_model(str(weights))
    else:
        from fractions import Fraction

        from safetensors import safe_open
        from safetensors.torch import load_file

        with safe_open(str(weights), "pt") as f:
            meta = f.metadata() or {}
        kwargs = json.loads(meta.get("kwargs", "{}"))
        seg = kwargs.get("segment")
        if isinstance(seg, dict) and seg.get("_type") == "fraction":
            kwargs["segment"] = Fraction(seg["numerator"], seg["denominator"])
        from demucs.htdemucs import HTDemucs

        model = HTDemucs(*json.loads(meta.get("args", "[]")), **kwargs)
        model.load_state_dict(load_file(str(weights)))
    model.eval()
    return model.to(device) if device else model


def separate_vocals(x, sr: int, model, device: str):
    """Mono float32 at ``sr`` → vocals (mono, same rate)."""
    import numpy as np
    import torch
    from demucs.apply import apply_model

    msr = int(model.samplerate)
    y = _resample(x, sr, msr)
    mix = torch.from_numpy(np.ascontiguousarray(y)).float()[None].repeat(model.audio_channels, 1)
    ref = mix.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        out = apply_model(model, ((mix - mean) / std)[None], device=device, split=True,
                          overlap=0.25, progress=False)[0]
    vocals = (out[model.sources.index("vocals")] * std + mean).mean(0).cpu().numpy()
    return _resample(vocals.astype("float32"), msr, sr)


# ── Stage: VAD ───────────────────────────────────────────────────────────────

def vad_segments(x16, *, min_s: float, max_s: float) -> list[tuple[float, float]]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(threshold=0.5, min_speech_duration_ms=250, max_speech_duration_s=max_s,
                      min_silence_duration_ms=300, speech_pad_ms=120)
    stamps = get_speech_timestamps(x16, vad_options=opts, sampling_rate=ANALYSIS_SR)
    spans = [(s["start"] / ANALYSIS_SR, s["end"] / ANALYSIS_SR) for s in stamps]
    return merge_spans(spans, min_s=min_s, max_s=max_s)


def merge_spans(spans, *, min_s: float, max_s: float, gap_s: float = 0.35):
    """Merge close spans up to ``max_s``, split overlong ones, drop short ones."""
    merged: list[list[float]] = []
    for start, end in sorted(spans):
        if merged and start - merged[-1][1] <= gap_s and end - merged[-1][0] <= max_s:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    out: list[tuple[float, float]] = []
    for start, end in merged:
        length = end - start
        if length > max_s:
            parts = math.ceil(length / max_s)
            step = length / parts
            out += [(start + i * step, start + (i + 1) * step) for i in range(parts)]
        elif length >= min_s:
            out.append((start, end))
    return [(round(a, 3), round(b, 3)) for a, b in out]


# ── Stage: speaker filter ───────────────────────────────────────────────────

class _MfccEmbedder:
    method = "mfcc-stats"
    default_threshold = None  # robust outlier rule instead of a fixed cosine

    def __init__(self):
        import torchaudio

        self._mfcc = torchaudio.transforms.MFCC(
            sample_rate=ANALYSIS_SR, n_mfcc=24,
            melkwargs={"n_fft": 512, "hop_length": 160, "n_mels": 40})

    def embed(self, x16):
        import numpy as np
        import torch

        feats = self._mfcc(torch.from_numpy(np.ascontiguousarray(x16)).float())[1:]  # drop c0
        return torch.cat([feats.mean(1), feats.std(1)]).numpy()


class _EcapaEmbedder:
    method = "ecapa"
    default_threshold = 0.45

    def __init__(self, path: Path, device: str):
        from speechbrain.inference.speaker import EncoderClassifier

        self._enc = EncoderClassifier.from_hparams(
            source=str(path), savedir=str(path), run_opts={"device": device})

    def embed(self, x16):
        import numpy as np
        import torch

        with torch.no_grad():
            emb = self._enc.encode_batch(torch.from_numpy(np.ascontiguousarray(x16))[None])
        return emb.squeeze().cpu().numpy()


def _cos(a, b) -> float:
    import numpy as np

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def speaker_scores(embs: list, ref_emb=None, *, centered: bool) -> list[float]:
    import numpy as np

    mat = np.stack(embs)
    pool = mat if ref_emb is None else np.vstack([mat, ref_emb[None]])
    shift = pool.mean(0) if centered else 0.0
    mat = mat - shift
    target = (np.median(mat, axis=0) if ref_emb is None else ref_emb - shift)
    return [_cos(e, target) for e in mat]


def speaker_keep(sims: list[float], threshold: Optional[float]) -> list[bool]:
    """Fixed threshold (ECAPA) or median − 3·MAD outlier rule (fallback)."""
    import numpy as np

    if not sims:
        return []
    if threshold is not None:
        return [s >= threshold for s in sims]
    arr = np.asarray(sims)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) or 0.05
    floor = med - 3.0 * 1.4826 * mad
    return [s >= floor for s in sims]


# ── Stage: transcription ────────────────────────────────────────────────────

class _Transcriber:
    def __init__(self, path: Path, device: str):
        try:  # CTranslate2 needs the side-loaded cuDNN 8 outside the backend too
            from core.cudnn8 import preload

            preload()
        except Exception:  # noqa: BLE001
            pass
        from faster_whisper import WhisperModel

        compute = "int8_float16" if device.startswith("cuda") else "int8"
        self.model = WhisperModel(str(path), device="cuda" if device.startswith("cuda") else "cpu",
                                  compute_type=compute)

    def __call__(self, x16, language: Optional[str]):
        segs, info = self.model.transcribe(x16, language=language, beam_size=5, vad_filter=False,
                                           condition_on_previous_text=False)
        segs = list(segs)
        text = " ".join(s.text.strip() for s in segs).strip()
        if segs:
            lp = sum(s.avg_logprob * max(s.end - s.start, 0.01) for s in segs) / \
                sum(max(s.end - s.start, 0.01) for s in segs)
            conf = math.exp(min(0.0, lp)) * (1.0 - max(s.no_speech_prob for s in segs) * 0.5)
        else:
            conf = 0.0
        return text, round(conf, 3), getattr(info, "language", language)

    def close(self):
        self.model = None


# ── Stage: quality ──────────────────────────────────────────────────────────

def quality_metrics(clip, sr: int) -> dict:
    import numpy as np

    db = _frame_db(clip, sr)
    peak = float(np.max(np.abs(clip))) if len(clip) else 0.0
    speech = float(np.percentile(db, 90))
    floor = float(np.percentile(db, 10))
    return {
        "peak": round(peak, 4),
        "clip_ratio": round(float(np.mean(np.abs(clip) > 0.99)) if len(clip) else 0.0, 5),
        "level_db": round(speech, 1),
        "noise_db": round(floor, 1),
        "snr_db": round(speech - floor, 1),
    }


def quality_score(m: dict, asr_conf: Optional[float]) -> float:
    snr = min(1.0, max(0.0, (m["snr_db"] - 10.0) / 30.0))
    clipping = 1.0 - min(1.0, m["clip_ratio"] * 200.0)
    level = 1.0 if -32.0 <= m["level_db"] <= -6.0 else 0.5
    parts = [(snr, 0.45), (clipping, 0.2), (level, 0.15)]
    if asr_conf is not None:
        parts.append((asr_conf, 0.2))
    total = sum(w for _, w in parts)
    return round(sum(v * w for v, w in parts) / total, 3)


# ── Stage: nonverbal tagging ────────────────────────────────────────────────

def _flatness(x) -> float:
    import numpy as np

    if len(x) < 256:
        return 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) + 1e-10
    return float(np.exp(np.mean(np.log(spec))) / np.mean(spec))


def laugh_likelihood(clip, sr: int, text: str, duration: float) -> float:
    """0–1. Text markers are decisive; otherwise rhythmic 3.5–7 Hz energy bursts
    combined with few recognised words (laughter has no words)."""
    import numpy as np

    if text and _LAUGH_TEXT_RE.search(text):
        return 0.95
    env = 10 ** (_frame_db(clip, sr, 0.01) / 20)
    if len(env) < 60:
        return 0.0
    env = env - env.mean()
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = int(100 / 7.0), int(100 / 3.5)  # lags for 7 Hz … 3.5 Hz at 100 fps
    periodic = float(ac[lo:hi + 1].max()) if hi < len(ac) else 0.0
    words = len(text.split()) if text else 0
    sparse = 1.0 if words / max(duration, 0.1) < 0.8 else 0.0
    return round(max(0.0, min(1.0, (periodic - 0.25) / 0.5)) * sparse, 3)


def find_breaths(x, sr: int, spans: list[tuple[float, float]], noise_db: float) -> list[tuple]:
    """Breaths sit in the gaps between speech: 0.15–0.9 s, clearly above the
    noise floor but well below speech, spectrally flat (broadband)."""
    import numpy as np

    events = []
    gaps = []
    prev = 0.0
    for start, end in spans:
        if start - prev > 0.2:
            gaps.append((prev, start))
        prev = end
    for g0, g1 in gaps:
        seg = x[int(g0 * sr):int(g1 * sr)]
        db = _frame_db(seg, sr)
        hot = db > noise_db + 8.0
        i = 0
        while i < len(hot):
            if not hot[i]:
                i += 1
                continue
            j = i
            while j < len(hot) and hot[j]:
                j += 1
            t0, t1 = g0 + i * 0.02, g0 + j * 0.02
            if 0.15 <= t1 - t0 <= 0.9:
                piece = x[int(t0 * sr):int(t1 * sr)]
                level = float(np.percentile(_frame_db(piece, sr), 90))
                if _flatness(piece) > 0.12 and level < -24.0:
                    events.append((round(t0, 3), round(t1, 3), round(level, 1)))
            i = j
    return events


# ── Jobs ─────────────────────────────────────────────────────────────────────

class Cancelled(RuntimeError):
    pass


@dataclass
class BuildJob:
    id: str
    dataset_id: str
    name: str
    status: str = "queued"  # queued|running|succeeded|failed|cancelled
    stage: Optional[str] = None
    progress: float = 0.0
    error: Optional[str] = None
    log: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    summary: Optional[dict] = None
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def public(self, log_tail: int = 80) -> dict:
        from dataclasses import fields

        d = {f.name: getattr(self, f.name) for f in fields(self)
             if f.name not in ("cancel", "log")}
        d["log_tail"] = "\n".join(self.log[-log_tail:])
        return d


_JOBS: dict[str, BuildJob] = {}
_JOBS_LOCK = threading.Lock()


def get_job(job_id: str) -> BuildJob:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise KeyError(job_id)
    return job


def list_jobs() -> list[dict]:
    with _JOBS_LOCK:
        jobs = list(_JOBS.values())
    return [j.public(0) for j in sorted(jobs, key=lambda j: -j.created_at)]


def cancel_job(job_id: str) -> BuildJob:
    job = get_job(job_id)
    job.cancel.set()
    return job


def _active() -> Optional[BuildJob]:
    with _JOBS_LOCK:
        return next((j for j in _JOBS.values() if j.status in ("queued", "running")), None)


def start_build(sources: list[Path], options: BuildOptions, *, reference: Optional[Path] = None,
                runner: Optional[Callable] = None) -> BuildJob:
    """Start a build in a thread. ``sources`` are copied into the dataset dir
    by the caller's temp dir, so the job owns its inputs."""
    options.validate()
    if not sources:
        raise ValueError("Add at least one recording.")
    if _active() is not None:
        raise RuntimeError("A dataset build is already running.")
    dataset_id = f"{_slug(options.name)}-{uuid.uuid4().hex[:6]}"
    job = BuildJob(id=uuid.uuid4().hex[:12], dataset_id=dataset_id, name=options.name)
    with _JOBS_LOCK:
        _JOBS[job.id] = job
    work = runner or run_build
    threading.Thread(target=_guarded, args=(work, job, sources, options, reference),
                     daemon=True, name=f"dataset-build-{job.id}").start()
    return job


def _guarded(work, job: BuildJob, sources, options, reference) -> None:
    job.status = "running"
    try:
        job.summary = work(job, sources, options, reference)
        job.status = "succeeded"
        job.progress = 1.0
    except Cancelled:
        job.status = "cancelled"
        job.log.append("Cancelled.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("dataset build failed")
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"
        job.log.append(job.error)
    finally:
        job.finished_at = time.time()
        try:
            from services.model_manager import free_vram

            free_vram()
        except Exception:  # noqa: BLE001
            pass


def _device(options: BuildOptions) -> str:
    if options.device:
        return options.device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def run_build(job: BuildJob, sources: list[Path], options: BuildOptions,
              reference: Optional[Path]) -> dict:
    import numpy as np
    import soundfile as sf

    out = datasets_root() / job.dataset_id
    (out / "clips").mkdir(parents=True, exist_ok=True)
    (out / "nonverbal").mkdir(exist_ok=True)
    work = out / ".work"
    work.mkdir(exist_ok=True)
    device = _device(options)
    caps = capabilities(options.asr_model)
    warnings: list[str] = []
    stages: dict[str, str] = {}

    def step(stage: str, progress: float, msg: str = "") -> None:
        if job.cancel.is_set():
            raise Cancelled()
        job.stage, job.progress = stage, round(progress, 3)
        if msg:
            job.log.append(f"[{stage}] {msg}")

    if options.free_gpu and device.startswith("cuda"):
        try:
            from services.gpu_memory import release_gpu_holders

            freed = release_gpu_holders(reason="dataset build")
            job.log.append("Freed GPU first: " + (", ".join(freed) or "nothing was loaded"))
        except Exception as exc:  # noqa: BLE001
            job.log.append(f"Could not free the GPU first: {exc}")

    # 1. decode
    tracks = []
    for i, src in enumerate(sources):
        step("decode", 0.02 + 0.05 * i / len(sources), f"Decoding {Path(src).name}")
        wav = decode(str(src), work / f"src{i:02d}.wav")
        x, sr = _read(wav)
        tracks.append({"name": Path(src).name, "x": x, "sr": sr})
    ref_x16 = None
    if reference is not None:
        rx, rsr = _read(decode(str(reference), work / "reference.wav"))
        ref_x16 = _resample(rx, rsr, ANALYSIS_SR)

    # 2. separation
    if options.separate and caps["separation"]["available"]:
        step("separate", 0.08, "Loading HTDemucs")
        model = load_htdemucs(Path(caps["separation"]["weights"]), device)
        try:
            for i, t in enumerate(tracks):
                step("separate", 0.08 + 0.2 * i / len(tracks), f"Isolating vocals: {t['name']}")
                t["x"] = separate_vocals(t["x"], t["sr"], model, device)
        finally:
            del model
            _free_cuda()
        stages["separation"] = "htdemucs"
    else:
        stages["separation"] = "skipped"
        if options.separate:
            warnings.append("Vocal separation skipped: HTDemucs weights are not on disk.")

    # 3. VAD
    step("vad", 0.3, "Finding speech")
    segments: list[Segment] = []
    for t in tracks:
        t["x16"] = _resample(t["x"], t["sr"], ANALYSIS_SR)
        spans = vad_segments(t["x16"], min_s=options.min_clip_s, max_s=options.max_clip_s)
        t["spans"] = spans
        segments += [Segment(source=t["name"], start=a, end=b) for a, b in spans]
        job.log.append(f"[vad] {t['name']}: {len(spans)} segments")
    stages["vad"] = "silero-v6"
    if not segments:
        raise RuntimeError("No speech found in the recordings.")
    by_name = {t["name"]: t for t in tracks}

    def clip16(s: Segment):
        x16 = by_name[s.source]["x16"]
        return x16[int(s.start * ANALYSIS_SR):int(s.end * ANALYSIS_SR)]

    # 4. speaker filter
    if options.speaker_filter and (len(segments) > 1 or ref_x16 is not None):
        step("speaker", 0.4, "Checking the speaker")
        emb = None
        if caps["speaker"]["method"] == "ecapa":
            try:
                emb = _EcapaEmbedder(_ecapa_dir(), device)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"ECAPA unavailable ({exc}); using MFCC statistics.")
        emb = emb or _MfccEmbedder()
        embs = [emb.embed(clip16(s)) for s in segments]
        ref_emb = emb.embed(ref_x16) if ref_x16 is not None else None
        sims = speaker_scores(embs, ref_emb, centered=emb.method == "mfcc-stats")
        thr = options.speaker_threshold if options.speaker_threshold is not None \
            else emb.default_threshold
        keep = speaker_keep(sims, thr)
        for s, sim, ok in zip(segments, sims, keep):
            s.speaker_sim, s.speaker_ok = round(sim, 3), bool(ok)
            if not ok:
                s.reasons.append("different speaker")
        stages["speaker"] = emb.method + (" vs reference" if ref_emb is not None else " vs majority")
        del emb
        _free_cuda()
    else:
        stages["speaker"] = "skipped"

    # 5. transcription
    transcriber = None
    if options.transcribe and caps["transcription"]["available"]:
        step("transcribe", 0.5, f"Loading {options.asr_model}")
        transcriber = _Transcriber(_whisper_dir(options.asr_model), device)
        stages["transcription"] = options.asr_model
    elif options.transcribe:
        stages["transcription"] = "skipped"
        warnings.append(f"Transcription skipped: {options.asr_model} is not on disk.")
    else:
        stages["transcription"] = "off"
    try:
        for i, s in enumerate(segments):
            if transcriber is not None and s.speaker_ok:
                step("transcribe", 0.5 + 0.35 * i / len(segments))
                s.text, s.asr_conf, s.language = transcriber(clip16(s), options.language)
    finally:
        if transcriber is not None:
            transcriber.close()
            del transcriber
            _free_cuda()

    # 6. quality + 7. tagging + write
    step("score", 0.87, "Scoring and tagging")
    nonverbal = []
    accepted = 0
    total_s = 0.0
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as manifest, \
            open(out / "metadata.csv", "w", encoding="utf-8") as meta:
        for s in segments:
            t = by_name[s.source]
            sr = t["sr"]
            clip = t["x"][int(s.start * sr):int(s.end * sr)]
            s.metrics = quality_metrics(clip, sr)
            s.quality = quality_score(s.metrics, s.asr_conf)
            if options.tag_nonverbal:
                laugh = laugh_likelihood(clip, sr, s.text, s.duration)
                if laugh >= 0.6:
                    s.tags.append("laugh")
                    s.metrics["laugh"] = laugh
            if s.quality < options.min_quality:
                s.reasons.append("low quality")
            if s.metrics["clip_ratio"] > 0.01:
                s.reasons.append("clipping")
            if stages.get("transcription") not in ("skipped", "off") and s.speaker_ok \
                    and not s.text and "laugh" not in s.tags:
                s.reasons.append("no words recognised")
            s.accepted = not s.reasons
            if s.accepted:
                accepted += 1
                s.index = accepted
                s.file = f"clips/{job.dataset_id}_{accepted:04d}.wav"
                peak = max(float(np.max(np.abs(clip))), 1e-6)
                gain = min(1.0, 0.95 / peak)
                sf.write(str(out / s.file), clip * gain, sr, subtype="PCM_16")
                total_s += s.duration
                text = s.text.replace("|", " ").replace("\n", " ").strip()
                meta.write(f"{s.file}|{text}\n")
                if "laugh" in s.tags:
                    nonverbal.append({"type": "laugh", "file": s.file, "source": s.source,
                                      "start": s.start, "end": s.end,
                                      "confidence": s.metrics.get("laugh")})
            row = asdict(s)
            row["duration"] = s.duration
            manifest.write(json.dumps(row, ensure_ascii=False) + "\n")

    if options.tag_nonverbal:
        n = 0
        for t in tracks:
            noise = float(np.percentile(_frame_db(t["x"], t["sr"]), 10))
            for t0, t1, level in find_breaths(t["x"], t["sr"], t["spans"], noise):
                n += 1
                name = f"nonverbal/breath_{n:03d}.wav"
                sr = t["sr"]
                piece = t["x"][int(t0 * sr):int(t1 * sr)]
                sf.write(str(out / name), piece, sr, subtype="PCM_16")
                nonverbal.append({"type": "breath", "file": name, "source": t["name"],
                                  "start": t0, "end": t1, "level_db": level})
        with open(out / "nonverbal" / "nonverbal.jsonl", "w", encoding="utf-8") as fh:
            for row in nonverbal:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        stages["nonverbal"] = "heuristic"

    reasons: dict[str, int] = {}
    for s in segments:
        for r in s.reasons:
            reasons[r] = reasons.get(r, 0) + 1
    summary = {
        "id": job.dataset_id,
        "name": options.name,
        "created_at": time.time(),
        "sources": [t["name"] for t in tracks],
        "sample_rate": tracks[0]["sr"],
        "options": asdict(options),
        "stages": stages,
        "warnings": warnings,
        "stats": {
            "segments": len(segments),
            "accepted": accepted,
            "rejected": len(segments) - accepted,
            "rejected_by_reason": reasons,
            "accepted_seconds": round(total_s, 1),
            "laughs": sum(1 for r in nonverbal if r["type"] == "laugh"),
            "breaths": sum(1 for r in nonverbal if r["type"] == "breath"),
            "mean_quality": round(float(np.mean([s.quality for s in segments if s.accepted]))
                                  if accepted else 0.0, 3),
        },
    }
    (out / "dataset.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    job.log.append(f"Done: {accepted}/{len(segments)} clips, {round(total_s, 1)} s accepted.")
    return summary


def _free_cuda() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# ── Listing ──────────────────────────────────────────────────────────────────

def list_datasets() -> list[dict]:
    rows = []
    for d in sorted(datasets_root().iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        meta = d / "dataset.json"
        if d.is_dir() and meta.is_file() and _ID_RE.match(d.name):
            try:
                info = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            rows.append({k: info.get(k) for k in ("id", "name", "created_at", "stats",
                                                  "stages", "warnings", "sample_rate")})
    return rows


def load_dataset(dataset_id: str, *, limit: int = 2000) -> dict:
    d = dataset_dir(dataset_id)
    info = json.loads((d / "dataset.json").read_text(encoding="utf-8"))
    segs = []
    with open(d / "manifest.jsonl", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= limit:
                break
            segs.append(json.loads(line))
    nv = []
    nv_path = d / "nonverbal" / "nonverbal.jsonl"
    if nv_path.is_file():
        nv = [json.loads(line) for line in nv_path.read_text(encoding="utf-8").splitlines() if line]
    return {**info, "segments": segs, "nonverbal": nv, "path": str(d)}


def dataset_file(dataset_id: str, rel: str) -> Path:
    """Resolve ``clips/x.wav`` / ``nonverbal/x.wav`` inside a dataset safely."""
    d = dataset_dir(dataset_id)
    parts = Path(rel).parts
    if len(parts) != 2 or parts[0] not in ("clips", "nonverbal") or not parts[1].endswith(".wav") \
            or "/" in parts[1] or "\\" in parts[1] or parts[1].startswith("."):
        raise KeyError(rel)
    path = (d / parts[0] / parts[1]).resolve()
    if path.parent != (d / parts[0]).resolve() or not path.is_file():
        raise KeyError(rel)
    return path
