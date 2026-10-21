"""RVC voice training (Applio pipeline) and Applio setup as background jobs.

Training drives Applio's own CLI (``core.py``, a click group) with Applio's
interpreter, one subprocess per stage, exactly as verified against
IAHispano/Applio ``core.py``:

  preprocess --model-name M --dataset-path D --sample-rate {32000,40000,48000}
             --cpu-cores N --cut-preprocess Automatic --chunk-len 3.0
             --overlap-len 0.3 --normalization-mode none
  extract    --model-name M --f0-method {rmvpe,crepe,fcpe} --cpu-cores N
             --gpu {0|-} --sample-rate SR --embedder-model contentvec
             --include-mutes 2
  train      --model-name M --save-every-epoch K --save-only-latest
             --total-epoch E --sample-rate SR --batch-size B --gpu {0|-}
             --index-algorithm Auto --vocoder HiFi-GAN [--checkpointing]
  index      --model-name M --index-algorithm Auto

``core.py`` reports stage failures as an echoed message and still exits 0, so
each stage is also verified by its artifact (``sliced_audios/``,
``extracted/``, ``M_<E>e_<S>s.pth``, ``M.index``). Every run gets its own
``APPLIO_LOGS_DIR`` (an Applio-supported override) under
``<data>/rvc_training/<job_id>/logs`` so VoiceStudio never touches the user's
Applio ``logs/`` folder. On success the final weights + index are copied into
the voice registry.

Jobs are in-memory (plus ``job.json`` on disk for post-mortems), progress is
pushed on the shared ``/ws/events`` bus as ``kind="voice_conversion"``.
"""
from __future__ import annotations

import collections
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("omnivoice.rvc")

SAMPLE_RATES = (32000, 40000, 48000)
TRAIN_F0_METHODS = ("rmvpe", "crepe", "fcpe")
#: 8 GB VRAM (RTX 3070 Laptop) safe defaults. Batch 8 at 48 kHz HiFi-GAN fits
#: in ~6.5 GB; checkpointing trades speed for headroom when the TTS model is
#: also resident.
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 300
DEFAULT_SAVE_EVERY = 25
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".aac", ".wma"}

_STAGE_WEIGHTS = {"preprocess": (0.0, 0.05), "extract": (0.05, 0.20),
                  "train": (0.20, 0.95), "index": (0.95, 1.0)}
_EPOCH_RE = re.compile(r"\bepoch=(\d+)\b")
_FAIL_RE = re.compile(r"\b(failed|Traceback \(most recent call last\)|Error:)", re.I)
#: The exact messages core.py echoes (with exit status 0) when a stage fails.
_CORE_FAILURES = re.compile(
    r"^(Preprocessing failed for model|Feature extraction failed for model|"
    r"Training failed for model|Index generation failed for model)")
_TERMINAL = {"succeeded", "failed", "cancelled"}

_jobs: dict[str, "Job"] = {}
_jobs_lock = threading.Lock()


class JobConflict(RuntimeError):
    pass


@dataclass
class Job:
    id: str
    kind: str                       # "train" | "setup"
    name: str
    status: str = "queued"          # queued|running|succeeded|failed|cancelled
    stage: Optional[str] = None
    progress: float = 0.0
    epoch: int = 0
    total_epochs: int = 0
    error: Optional[str] = None
    voice_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    settings: dict = field(default_factory=dict)
    log: collections.deque = field(default_factory=lambda: collections.deque(maxlen=4000))
    cancel_event: threading.Event = field(default_factory=threading.Event)
    proc: object = None
    workdir: Optional[str] = None

    def public(self, tail: int = 200) -> dict:
        lines = list(self.log)
        return {
            "id": self.id, "kind": self.kind, "name": self.name, "status": self.status,
            "stage": self.stage, "progress": round(self.progress, 4), "epoch": self.epoch,
            "total_epochs": self.total_epochs, "error": self.error,
            "voice_id": self.voice_id, "created_at": self.created_at,
            "finished_at": self.finished_at, "settings": self.settings,
            "log_tail": lines[-tail:] if tail else [], "log_lines": len(lines),
        }


def _emit(job: Job, **extra) -> None:
    try:
        from core import event_bus

        event_bus.emit("voice_conversion", {
            "job_id": job.id, "job_kind": job.kind, "status": job.status,
            "stage": job.stage, "progress": round(job.progress, 4), "epoch": job.epoch,
            **extra,
        })
    except Exception:  # noqa: BLE001 — events are best-effort
        pass


def training_root() -> Path:
    from core.config import DATA_DIR

    root = Path(os.environ.get("OMNIVOICE_RVC_TRAINING_DIR", "") or Path(DATA_DIR) / "rvc_training")
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_job(job_id: str) -> Optional[Job]:
    with _jobs_lock:
        return _jobs.get(job_id)


def list_jobs() -> list[Job]:
    with _jobs_lock:
        return sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)


def _active(kind: Optional[str] = None) -> Optional[Job]:
    with _jobs_lock:
        for job in _jobs.values():
            if job.status not in _TERMINAL and (kind is None or job.kind == kind):
                return job
    return None


def cancel_job(job_id: str) -> Optional[Job]:
    job = get_job(job_id)
    if job is None:
        return None
    if job.status in _TERMINAL:
        return job
    job.cancel_event.set()
    proc = job.proc
    if proc is not None:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
    return job


def _persist(job: Job) -> None:
    if not job.workdir:
        return
    try:
        data = job.public(tail=0)
        (Path(job.workdir) / "job.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        (Path(job.workdir) / "job.log").write_text("\n".join(job.log), encoding="utf-8")
    except OSError:
        pass


def _model_name(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")[:40]
    return slug or "voice"


# ── Subprocess plumbing ──────────────────────────────────────────────────────


def _run_stage(job: Job, argv: list[str], *, cwd: str, env: dict,
               on_line: Optional[Callable[[str], None]] = None) -> int:
    """Run one stage, streaming merged stdout/stderr into the job log."""
    if job.cancel_event.is_set():
        raise _Cancelled()
    job.log.append(f"$ {' '.join(_redact_argv(argv))}")
    kwargs = dict(cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                  stderr=subprocess.STDOUT)
    try:
        from core.contained_subprocess import spawn_owned

        proc = spawn_owned(argv, **kwargs)
    except Exception:  # noqa: BLE001
        proc = subprocess.Popen(argv, **kwargs)
    job.proc = proc
    try:
        assert proc.stdout is not None
        buf = b""
        while True:
            chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            # tqdm redraws with \r — treat it as a line break for the log.
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for raw in parts:
                line = raw.decode("utf-8", "replace").rstrip()
                if line:
                    job.log.append(line[:2000])
                    if on_line:
                        on_line(line)
        if buf.strip():
            line = buf.decode("utf-8", "replace").rstrip()
            job.log.append(line[:2000])
            if on_line:
                on_line(line)
        rc = proc.wait()
    finally:
        job.proc = None
    if job.cancel_event.is_set():
        raise _Cancelled()
    return rc


def _redact_argv(argv: list[str]) -> list[str]:
    return [a if len(a) < 300 else a[:300] + "…" for a in argv]


class _Cancelled(Exception):
    pass


class StageFailed(RuntimeError):
    pass


def _set_stage(job: Job, stage: str, frac: float = 0.0) -> None:
    lo, hi = _STAGE_WEIGHTS.get(stage, (job.progress, job.progress))
    job.stage = stage
    job.progress = lo + (hi - lo) * max(0.0, min(1.0, frac))
    _emit(job)


# ── Training ─────────────────────────────────────────────────────────────────


@dataclass
class TrainSettings:
    name: str
    sample_rate: int = 48000
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    f0_method: str = "rmvpe"
    save_every: int = DEFAULT_SAVE_EVERY
    cpu_cores: int = 0              # 0 = auto (cpu_count - 1, max 8)
    checkpointing: bool = False     # lower VRAM, slower
    gpu: Optional[str] = None       # None = auto ("0" when CUDA, "-" for CPU)
    #: Initialise from Applio's HiFi-GAN base models (f0G/f0D<sr>). False
    #: trains from scratch — no download, but needs far more epochs.
    pretrained: bool = True

    def validate(self) -> "TrainSettings":
        from services.rvc_registry import clean_name

        self.name = clean_name(self.name)
        if self.sample_rate not in SAMPLE_RATES:
            raise ValueError("sample_rate must be 32000, 40000 or 48000.")
        if not 1 <= self.epochs <= 10000:
            raise ValueError("epochs must be between 1 and 10000.")
        if not 1 <= self.batch_size <= 50:
            raise ValueError("batch_size must be between 1 and 50.")
        if self.f0_method not in TRAIN_F0_METHODS:
            raise ValueError(f"f0_method must be one of: {', '.join(TRAIN_F0_METHODS)}.")
        if not 1 <= self.save_every <= 100:
            raise ValueError("save_every must be between 1 and 100.")
        if self.gpu is not None and not re.fullmatch(r"-|\d+(-\d+)*", self.gpu):
            raise ValueError("gpu must be '-' (CPU) or device ids like '0'.")
        return self


def _auto_gpu() -> str:
    from services.rvc import select_device

    device = select_device()
    if device.startswith("cuda"):
        return device.split(":", 1)[1] if ":" in device else "0"
    if device == "mps":
        return "0"  # Applio's train.py maps any non-"-" id to MPS on Apple
    return "-"


def _cpu_cores(requested: int) -> int:
    if requested and requested > 0:
        return min(64, requested)
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def dataset_files(path: Path) -> list[Path]:
    return [p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix.lower() in AUDIO_EXTS]


def start_training(settings: TrainSettings, *, dataset_dir: Path, job_id: Optional[str] = None,
                   workdir: Optional[Path] = None,
                   runner: Optional[Callable[[Job], None]] = None) -> Job:
    """Validate, create the job and run it on a daemon thread."""
    from services.rvc import applio_dir, applio_python

    settings.validate()
    root = applio_dir()
    py = applio_python(root) if root else None
    if runner is None and (root is None or py is None):
        raise RuntimeError("Training needs Applio. Install it from Voice conversion → Setup "
                           "or set OMNIVOICE_APPLIO_DIR to an Applio checkout.")
    if not dataset_dir.is_dir() or not dataset_files(dataset_dir):
        raise ValueError("The dataset has no audio files (wav, flac, mp3, ogg, m4a).")
    if _active("train") is not None:
        raise JobConflict("A training job is already running. Cancel it or wait for it to finish.")
    job_id = job_id or uuid.uuid4().hex[:12]
    workdir = workdir or training_root() / job_id
    workdir.mkdir(parents=True, exist_ok=True)
    job = Job(id=job_id, kind="train", name=settings.name, total_epochs=settings.epochs,
              settings={**settings.__dict__, "dataset_files": len(dataset_files(dataset_dir))},
              workdir=str(workdir))
    with _jobs_lock:
        _jobs[job.id] = job

    def _target():
        try:
            if runner is not None:
                runner(job)
            else:
                _run_training(job, settings, Path(root), Path(py), dataset_dir, workdir)
        except _Cancelled:
            job.status = "cancelled"
            job.log.append("Cancelled.")
        except Exception as exc:  # noqa: BLE001
            logger.exception("RVC training %s failed", job.id)
            job.status = "failed"
            job.error = str(exc)[:500]
            job.log.append(f"ERROR: {exc}")
        else:
            if job.status not in _TERMINAL:
                job.status = "succeeded"
                job.progress = 1.0
        finally:
            job.finished_at = time.time()
            _persist(job)
            _emit(job, voice_id=job.voice_id)

    threading.Thread(target=_target, name=f"rvc-train-{job.id}", daemon=True).start()
    return job


#: Applio HiFi-GAN base models (IAHispano/Applio Resources/pretrained_v2),
#: byte sizes from the HF CDN. Applio's ``train`` downloads the pair for the
#: chosen sample rate on first use when they are missing.
PRETRAINED_URL = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_v2/"
PRETRAINED_SIZES = {
    "f0G32k.pth": 73_950_049, "f0D32k.pth": 142_875_703,
    "f0G40k.pth": 73_106_273, "f0D40k.pth": 142_875_703,
    "f0G48k.pth": 75_465_569, "f0D48k.pth": 142_875_703,
}


def pretrained_status(applio: Optional[Path] = None) -> dict:
    """Which HiFi-GAN base models are on disk, per sample rate."""
    if applio is None:
        from services.rvc import applio_dir

        applio = applio_dir()
    root = applio
    out = {}
    for sr in SAMPLE_RATES:
        k = f"{sr // 1000}k"
        missing = []
        for kind in ("G", "D"):
            name = f"f0{kind}{k}.pth"
            path = (Path(root) / "rvc" / "models" / "pretraineds" / "hifi-gan" / name) if root else None
            if path is None or not (path.is_file() and path.stat().st_size > 0):
                missing.append({"file": name, "size_bytes": PRETRAINED_SIZES[name],
                                "path": str(path) if path else None, "url": PRETRAINED_URL + name})
        out[str(sr)] = {"present": not missing, "missing": missing,
                        "download_bytes": sum(m["size_bytes"] for m in missing)}
    return out


def build_stage_commands(settings: TrainSettings, *, python: str, applio: str,
                         model_name: str, dataset: str, gpu: str, cores: int) -> dict:
    core = os.path.join(applio, "core.py")
    sr = str(settings.sample_rate)
    train = [python, core, "train", "--model-name", model_name,
             "--save-every-epoch", str(settings.save_every), "--save-only-latest",
             "--total-epoch", str(settings.epochs), "--sample-rate", sr,
             "--batch-size", str(settings.batch_size), "--gpu", gpu,
             "--index-algorithm", "Auto", "--vocoder", "HiFi-GAN"]
    if settings.checkpointing:
        train.append("--checkpointing")
    if not settings.pretrained:
        train.append("--no-pretrained")
    return {
        "preprocess": [python, core, "preprocess", "--model-name", model_name,
                       "--dataset-path", dataset, "--sample-rate", sr,
                       "--cpu-cores", str(cores), "--cut-preprocess", "Automatic",
                       "--chunk-len", "3.0", "--overlap-len", "0.3",
                       "--normalization-mode", "none"],
        "extract": [python, core, "extract", "--model-name", model_name,
                    "--f0-method", settings.f0_method, "--cpu-cores", str(cores),
                    "--gpu", gpu, "--sample-rate", sr, "--embedder-model", "contentvec",
                    "--include-mutes", "2"],
        "train": train,
        "index": [python, core, "index", "--model-name", model_name,
                  "--index-algorithm", "Auto"],
    }


def find_final_weights(exp_dir: Path, model_name: str) -> Optional[Path]:
    """Highest-epoch ``<model>_<E>e_<S>s.pth`` (inference weights, not G_/D_)."""
    pattern = re.compile(rf"^{re.escape(model_name)}_(\d+)e_(\d+)s\.pth$")
    best: Optional[tuple[int, int, Path]] = None
    for p in exp_dir.glob("*.pth"):
        m = pattern.match(p.name)
        if m:
            key = (int(m.group(1)), int(m.group(2)), p)
            if best is None or key[:2] > best[:2]:
                best = key
    return best[2] if best else None


def _run_training(job: Job, settings: TrainSettings, applio: Path, python: Path,
                  dataset_dir: Path, workdir: Path) -> None:
    from services import rvc_registry

    job.status = "running"
    _free_gpu_for_job(job)
    model_name = _model_name(settings.name)
    logs_dir = workdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = logs_dir / model_name
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update({"APPLIO_LOGS_DIR": str(logs_dir), "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8"})
    gpu = settings.gpu or _auto_gpu()
    cmds = build_stage_commands(settings, python=str(python), applio=str(applio),
                                model_name=model_name, dataset=str(dataset_dir), gpu=gpu,
                                cores=_cpu_cores(settings.cpu_cores))
    job.settings["gpu"] = gpu
    job.log.append(f"Applio: {applio}  python: {python}  gpu: {gpu}")
    if settings.pretrained:
        need = pretrained_status(applio).get(str(settings.sample_rate), {})
        if need.get("missing"):
            job.log.append(
                "[note] Applio will download the %d kHz base models first (%s, %.0f MB)."
                % (settings.sample_rate // 1000, ", ".join(m["file"] for m in need["missing"]),
                   need["download_bytes"] / 1e6))
    else:
        job.log.append("[note] Training from scratch (no pretrained base models).")

    def _check(stage: str, rc: int, ok: bool, start_line: int) -> None:
        tail = list(job.log)[start_line:]
        echoed_failure = any(_FAIL_RE.search(line) for line in tail[-15:])
        core_failure = next((line for line in tail if _CORE_FAILURES.match(line)), None)
        if core_failure:
            raise StageFailed(f"{stage} failed: {core_failure}")
        if rc != 0 or not ok:
            hint = next((line for line in reversed(tail) if _FAIL_RE.search(line)), "")
            raise StageFailed(f"{stage} failed{': ' + hint if hint else ''} (exit {rc}).")
        if echoed_failure:
            job.log.append(f"[warn] {stage}: failure text in output but artifacts look fine.")

    # 1. preprocess
    _set_stage(job, "preprocess")
    mark = len(job.log)
    rc = _run_stage(job, cmds["preprocess"], cwd=str(applio), env=env)
    sliced = exp_dir / "sliced_audios"
    _check("preprocess", rc, sliced.is_dir() and any(sliced.iterdir()), mark)

    # 2. extract (f0 + embeddings)
    _set_stage(job, "extract")
    mark = len(job.log)
    rc = _run_stage(job, cmds["extract"], cwd=str(applio), env=env)
    extracted = exp_dir / "extracted"
    _check("extract", rc, extracted.is_dir() and any(extracted.iterdir()), mark)

    # 3. train (Applio also builds the index at the end; we verify below)
    _set_stage(job, "train")
    mark = len(job.log)

    def _on_train_line(line: str) -> None:
        m = _EPOCH_RE.search(line)
        if m:
            job.epoch = int(m.group(1))
            _set_stage(job, "train", job.epoch / max(1, settings.epochs))

    rc = _run_stage(job, cmds["train"], cwd=str(applio), env=env, on_line=_on_train_line)
    weights = find_final_weights(exp_dir, model_name)
    _check("train", rc, weights is not None, mark)

    # 4. index (no-op when train already produced it)
    index = exp_dir / f"{model_name}.index"
    if not index.is_file():
        _set_stage(job, "index")
        mark = len(job.log)
        rc = _run_stage(job, cmds["index"], cwd=str(applio), env=env)
        _check("index", rc, index.is_file(), mark)

    # 5. register
    job.stage = "register"
    voice = rvc_registry.import_from_paths(
        settings.name, weights, index if index.is_file() else None, source="trained",
        extra={"epochs": job.epoch or settings.epochs, "train_job": job.id,
               "train_sample_rate": settings.sample_rate},
    )
    job.voice_id = voice["id"]
    job.log.append(f"Registered voice {voice['id']} ({weights.name}).")
    job.status = "succeeded"
    job.progress = 1.0
    try:
        from core import event_bus

        event_bus.emit("voice_conversion", {"action": "voices_changed", "id": voice["id"]})
    except Exception:  # noqa: BLE001
        pass
    # Keep weights/index (registry has copies) but drop the heavy features.
    if os.environ.get("OMNIVOICE_RVC_KEEP_TRAINING_FILES", "") not in ("1", "true"):
        for heavy in ("sliced_audios", "sliced_audios_16k", "extracted", "f0", "f0_voiced"):
            shutil.rmtree(exp_dir / heavy, ignore_errors=True)


# ── Index (re)build for imported voices ─────────────────────────────────────
#
# Imported third-party voices usually ship only ``model.pth``; without an
# ``.index`` RVC's retrieval step is skipped (index_rate silently becomes 0)
# and the timbre is less close than the model can give. Applio builds the
# FAISS index from ContentVec features of the *target speaker's* audio, so an
# index can be (re)built from any clean recordings of that speaker without
# retraining: preprocess → extract → index at the voice's sample rate. When a
# trained voice's own features are still on disk (OMNIVOICE_RVC_KEEP_TRAINING
# _FILES), only the index stage runs.


def _existing_features(voice_meta: dict) -> Optional[Path]:
    job = voice_meta.get("train_job")
    if not job:
        return None
    logs = training_root() / str(job) / "logs"
    if not logs.is_dir():
        return None
    for exp in logs.iterdir():
        extracted = exp / "extracted"
        if extracted.is_dir() and any(extracted.glob("*.npy")):
            return exp
    return None


def build_index_commands(*, python: str, applio: str, model_name: str, dataset: str,
                         sample_rate: int, gpu: str, cores: int) -> dict:
    core = os.path.join(applio, "core.py")
    sr = str(sample_rate)
    return {
        "preprocess": [python, core, "preprocess", "--model-name", model_name,
                       "--dataset-path", dataset, "--sample-rate", sr,
                       "--cpu-cores", str(cores), "--cut-preprocess", "Automatic",
                       "--chunk-len", "3.0", "--overlap-len", "0.3",
                       "--normalization-mode", "none"],
        # Only the ContentVec features matter for the index; mutes are not
        # needed (they pad the *training* filelist).
        "extract": [python, core, "extract", "--model-name", model_name,
                    "--f0-method", "rmvpe", "--cpu-cores", str(cores),
                    "--gpu", gpu, "--sample-rate", sr, "--embedder-model", "contentvec",
                    "--include-mutes", "0"],
        "index": [python, core, "index", "--model-name", model_name,
                  "--index-algorithm", "Auto"],
    }


def start_index_build(voice_id: str, *, dataset_dir: Optional[Path] = None,
                      job_id: Optional[str] = None,
                      runner: Optional[Callable[[Job], None]] = None) -> Job:
    """(Re)build ``model.index`` for a registered voice as a background job."""
    from services import rvc_registry
    from services.rvc import applio_dir, applio_python

    meta = rvc_registry.voice_meta(voice_id)  # VoiceNotFound → caller maps to 404
    features = _existing_features(meta)
    root = applio_dir()
    py = applio_python(root) if root else None
    if runner is None and (root is None or py is None):
        raise RuntimeError("Building an index needs Applio. Install it from Voice conversion → "
                           "Setup or set OMNIVOICE_APPLIO_DIR to an Applio checkout.")
    if features is None and (dataset_dir is None or not dataset_dir.is_dir()
                             or not dataset_files(dataset_dir)):
        raise ValueError("Add recordings of this voice's speaker (wav, flac, mp3, ogg, m4a) "
                         "to build its index.")
    if _active("train") is not None or _active("index") is not None:
        raise JobConflict("A training or index job is already running.")
    sample_rate = int(meta.get("sample_rate") or 40000)
    if sample_rate not in SAMPLE_RATES:
        sample_rate = min(SAMPLE_RATES, key=lambda sr: abs(sr - sample_rate))
    job_id = job_id or uuid.uuid4().hex[:12]
    workdir = training_root() / f"index-{job_id}"
    workdir.mkdir(parents=True, exist_ok=True)
    job = Job(id=job_id, kind="index", name=meta.get("name") or voice_id, voice_id=voice_id,
              settings={"voice_id": voice_id, "sample_rate": sample_rate,
                        "reuse_features": features is not None,
                        "dataset_files": len(dataset_files(dataset_dir)) if dataset_dir else 0},
              workdir=str(workdir))
    with _jobs_lock:
        _jobs[job.id] = job

    def _target():
        try:
            if runner is not None:
                runner(job)
            else:
                _run_index_build(job, voice_id, Path(root), Path(py), dataset_dir, workdir,
                                 sample_rate, features)
        except _Cancelled:
            job.status = "cancelled"
            job.log.append("Cancelled.")
        except Exception as exc:  # noqa: BLE001
            logger.exception("RVC index build %s failed", job.id)
            job.status = "failed"
            job.error = str(exc)[:500]
            job.log.append(f"ERROR: {exc}")
        else:
            if job.status not in _TERMINAL:
                job.status = "succeeded"
                job.progress = 1.0
        finally:
            job.finished_at = time.time()
            _persist(job)
            _emit(job, voice_id=voice_id)

    threading.Thread(target=_target, name=f"rvc-index-{job.id}", daemon=True).start()
    return job


def _run_index_build(job: Job, voice_id: str, applio: Path, python: Path,
                     dataset_dir: Optional[Path], workdir: Path, sample_rate: int,
                     features: Optional[Path]) -> None:
    from services import rvc_registry

    job.status = "running"
    _free_gpu_for_job(job)
    env = _applio_env(workdir / "logs")
    gpu = _auto_gpu()
    model_name = "index_" + _model_name(job.name)
    if features is not None:
        exp_dir = features
        env["APPLIO_LOGS_DIR"] = str(features.parent)
        model_name = features.name
        job.log.append(f"Reusing training features in {features}")
    else:
        exp_dir = workdir / "logs" / model_name
    cmds = build_index_commands(python=str(python), applio=str(applio), model_name=model_name,
                                dataset=str(dataset_dir or ""), sample_rate=sample_rate,
                                gpu=gpu, cores=_cpu_cores(0))
    job.log.append(f"Applio: {applio}  python: {python}  gpu: {gpu}  sr: {sample_rate}")

    def _check(stage: str, rc: int, ok: bool) -> None:
        if rc != 0 or not ok:
            tail = next((line for line in reversed(job.log) if _FAIL_RE.search(line)), "")
            raise StageFailed(f"{stage} failed{': ' + tail if tail else ''} (exit {rc}).")

    weights = {"preprocess": (0.0, 0.2), "extract": (0.2, 0.8), "index": (0.8, 1.0)}
    if features is None:
        for stage, artifact in (("preprocess", "sliced_audios"), ("extract", "extracted")):
            job.stage, job.progress = stage, weights[stage][0]
            _emit(job)
            rc = _run_stage(job, cmds[stage], cwd=str(applio), env=env)
            out = exp_dir / artifact
            _check(stage, rc, out.is_dir() and any(out.iterdir()))
    job.stage, job.progress = "index", weights["index"][0]
    _emit(job)
    index = exp_dir / f"{model_name}.index"
    if index.is_file() and features is None:
        index.unlink()
    rc = _run_stage(job, cmds["index"], cwd=str(applio), env=env)
    _check("index", rc, index.is_file())
    # A worker may hold the old index open (Windows file locks).
    try:
        from services.rvc import get_service

        get_service().unload()
    except Exception:  # noqa: BLE001
        pass
    rvc_registry.attach_index(voice_id, index, extra={"index_job": job.id})
    job.log.append(f"Index attached to voice {voice_id} ({index.stat().st_size} bytes).")
    job.status, job.progress = "succeeded", 1.0
    try:
        from core import event_bus

        event_bus.emit("voice_conversion", {"action": "voices_changed", "id": voice_id})
    except Exception:  # noqa: BLE001
        pass
    if features is None:
        for heavy in ("sliced_audios", "sliced_audios_16k", "extracted", "f0", "f0_voiced"):
            shutil.rmtree(exp_dir / heavy, ignore_errors=True)


def _applio_env(logs_dir: Path) -> dict:
    logs_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update({"APPLIO_LOGS_DIR": str(logs_dir), "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8"})
    return env


def _free_gpu_for_job(job: Job) -> None:
    """Release VoiceStudio's own GPU holders before an Applio training/index job.

    RVC training on an 8 GB GPU OOMs when the TTS model (~2–4 GB), the Applio
    inference worker and cached ASR sit resident beside it. Opt out with
    ``OMNIVOICE_RVC_TRAIN_KEEP_GPU=1``.
    """
    if os.environ.get("OMNIVOICE_RVC_TRAIN_KEEP_GPU", "").strip() == "1":
        job.log.append("Keeping loaded models on the GPU (OMNIVOICE_RVC_TRAIN_KEEP_GPU=1).")
        return
    try:
        from services.gpu_memory import release_gpu_holders

        released = release_gpu_holders(reason=f"rvc {job.kind} job")
    except Exception as exc:  # noqa: BLE001 — never fail the job over this
        job.log.append(f"Could not free the GPU first: {exc}")
        return
    job.log.append("Freed GPU before starting: " + (", ".join(released) or "nothing was loaded"))


# ── Applio setup ─────────────────────────────────────────────────────────────

APPLIO_REPO = "https://github.com/IAHispano/Applio.git"


def setup_commands(target: Path, *, uv: str, git: str, cuda: bool) -> list[tuple[str, list[str]]]:
    """Non-interactive equivalent of Applio's run-install.{bat,sh}: clone, a
    Python 3.12 ``.venv`` via uv, ``requirements.txt`` with the cu128 torch
    index (CUDA hosts), then the inference/training prerequisites."""
    venv_py = target / (".venv/Scripts/python.exe" if sys.platform == "win32" else ".venv/bin/python")
    req = [uv, "pip", "install", "--python", str(venv_py), "-r", str(target / "requirements.txt")]
    if cuda:
        req += ["--extra-index-url", "https://download.pytorch.org/whl/cu128",
                "--index-strategy", "unsafe-best-match"]
    steps = []
    if not (target / "core.py").is_file():
        steps.append(("clone", [git, "clone", "--depth", "1", APPLIO_REPO, str(target)]))
    steps += [
        ("venv", [uv, "venv", "--python", "3.12", str(target / ".venv")]),
        ("requirements", req),
        ("prerequisites", [str(venv_py), str(target / "core.py"), "prerequisites",
                           "--no-pretraineds-hifigan", "--models", "--exe"]),
    ]
    return steps


def start_setup(target: Optional[Path] = None, *, cuda: Optional[bool] = None) -> Job:
    from services.rvc import managed_applio_dir, select_device

    if _active("setup") is not None:
        raise JobConflict("Applio setup is already running.")
    target = Path(target) if target else managed_applio_dir()
    git = shutil.which("git")
    try:
        from services.sidecar_install import _locate_uv, uv_subprocess_env

        uv = _locate_uv()
        env = uv_subprocess_env(target.parent)
    except Exception:  # noqa: BLE001
        uv, env = shutil.which("uv"), dict(os.environ, UV_NO_CONFIG="1")
    if not git and not (target / "core.py").is_file():
        raise RuntimeError("git is required to download Applio. Install Git for Windows "
                           "(https://git-scm.com) or clone Applio manually.")
    if not uv:
        raise RuntimeError("uv was not found. Install uv (https://docs.astral.sh/uv/) "
                           "or run Applio's run-install.bat manually.")
    if cuda is None:
        cuda = select_device().startswith("cuda") or sys.platform == "win32"
    env.pop("PYTHONPATH", None)
    env.pop("VIRTUAL_ENV", None)
    target.parent.mkdir(parents=True, exist_ok=True)
    steps = setup_commands(target, uv=uv, git=git or "git", cuda=bool(cuda))
    workdir = training_root() / f"setup-{uuid.uuid4().hex[:8]}"
    workdir.mkdir(parents=True, exist_ok=True)
    job = Job(id=workdir.name, kind="setup", name="Applio",
              settings={"target": str(target), "cuda": bool(cuda),
                        "steps": [s for s, _ in steps]}, workdir=str(workdir))
    with _jobs_lock:
        _jobs[job.id] = job

    def _target():
        job.status = "running"
        try:
            for i, (name, argv) in enumerate(steps):
                job.stage = name
                job.progress = i / len(steps)
                _emit(job)
                cwd = str(target) if (target / "core.py").is_file() else str(target.parent)
                rc = _run_stage(job, argv, cwd=cwd, env=env)
                if rc != 0:
                    raise StageFailed(f"{name} failed (exit {rc}).")
            job.status = "succeeded"
            job.progress = 1.0
            from services.rvc import reset_service

            reset_service()
        except _Cancelled:
            job.status = "cancelled"
        except Exception as exc:  # noqa: BLE001
            job.status = "failed"
            job.error = str(exc)[:500]
            job.log.append(f"ERROR: {exc}")
        finally:
            job.finished_at = time.time()
            _persist(job)
            _emit(job)

    threading.Thread(target=_target, name="applio-setup", daemon=True).start()
    return job


def _reset_for_tests() -> None:
    with _jobs_lock:
        _jobs.clear()
