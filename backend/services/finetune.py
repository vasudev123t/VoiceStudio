"""Fine-tuning jobs (review item 6): LoRA adapters trained on a Voice Lab dataset.

Recipes
-------
* ``omnivoice-lora`` — runnable today: OmniVoice (k2-fsa/OmniVoice, on disk)
  with LoRA on its Qwen3-0.6B backbone (``finetune_omnivoice_lora.py``).
  Needs only ``pip install peft``. 8 GB defaults: rank 16, batch 1 × accum 4,
  bf16 backbone, gradient checkpointing, ≤1400 tokens/utterance.
  The finished adapter can be *activated*: it is merged into the shared TTS
  model at its next load (``apply_active_adapter``), and deactivated the same
  way — the base checkpoint is never modified.
* ``voxcpm2-lora`` / ``indicf5-kn`` — preflight only: they report exactly
  which package/weights are missing (downloads need the user's approval).
"""
from __future__ import annotations

import importlib.util
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
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PREF_KEY = "omnivoice_lora_adapter"
TRAINER = Path(__file__).with_name("finetune_omnivoice_lora.py")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")

RECIPES = {
    "omnivoice-lora": {
        "label": "OmniVoice LoRA",
        "base": "k2-fsa/OmniVoice",
        "packages": ["peft"],
        "defaults": {"steps": 600, "rank": 16, "alpha": 32, "learning_rate": 1e-4,
                     "grad_accum": 4, "max_tokens": 1400, "gradient_checkpointing": True},
        "runnable": True,
    },
    "voxcpm2-lora": {
        "label": "VoxCPM2 LoRA",
        "base": "openbmb/VoxCPM2",
        "packages": ["voxcpm", "peft"],
        "defaults": {"steps": 1000, "rank": 32, "alpha": 32, "learning_rate": 1e-4,
                     "grad_accum": 8, "gradient_checkpointing": True},
        "runnable": False,
        "note": "VoxCPM's own LoRA trainer (voxcpm package) will be wired once the "
                "package and weights are approved and installed.",
    },
    "indicf5-kn": {
        "label": "IndicF5 (Kannada) fine-tune",
        "base": "ai4bharat/IndicF5",
        "packages": ["f5_tts"],
        "defaults": {"steps": 2000, "learning_rate": 1e-5, "batch_frames": 1600},
        "runnable": False,
        "note": "IndicF5 already speaks Kannada; fine-tuning uses f5-tts' finetune CLI "
                "once the package and weights are approved and installed.",
    },
}


def root() -> Path:
    override = os.environ.get("OMNIVOICE_FINETUNE_DIR", "").strip()
    if override:
        path = Path(override)
    else:
        from core.config import DATA_DIR

        path = Path(DATA_DIR) / "finetune"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cached(repo_id: str) -> Optional[Path]:
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id, local_files_only=True))
    except Exception:  # noqa: BLE001
        return None


def preflight(recipe_id: str) -> dict:
    recipe = RECIPES.get(recipe_id)
    if recipe is None:
        raise KeyError(recipe_id)
    missing_pkgs = [p for p in recipe["packages"] if importlib.util.find_spec(p) is None]
    base = recipe["base"]
    if recipe_id == "omnivoice-lora":
        try:
            from services.model_manager import resolve_omnivoice_checkpoint

            base = resolve_omnivoice_checkpoint()
        except Exception:  # noqa: BLE001
            pass
    weights = Path(base) if Path(base).is_dir() else _cached(base)
    ready = recipe["runnable"] and not missing_pkgs and weights is not None
    return {"id": recipe_id, "label": recipe["label"], "base": base, "runnable": recipe["runnable"],
            "ready": ready, "missing_packages": missing_pkgs,
            "weights_on_disk": weights is not None, "weights_path": str(weights) if weights else None,
            "defaults": recipe["defaults"], "note": recipe.get("note")}


def recipes() -> list[dict]:
    return [preflight(rid) for rid in RECIPES]


# ── Jobs ─────────────────────────────────────────────────────────────────────

@dataclass
class FinetuneJob:
    id: str
    recipe: str
    name: str
    dataset_id: str
    status: str = "queued"
    stage: Optional[str] = None
    progress: float = 0.0
    step: int = 0
    steps: int = 0
    loss: Optional[float] = None
    peak_vram_mb: Optional[int] = None
    error: Optional[str] = None
    adapter_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    log: list = field(default_factory=list)
    proc: Optional[subprocess.Popen] = field(default=None, repr=False)

    def public(self, log_tail: int = 60) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in ("log", "proc")}
        d["log_tail"] = "\n".join(self.log[-log_tail:])
        return d


_JOBS: dict[str, FinetuneJob] = {}
_LOCK = threading.Lock()


def get_job(job_id: str) -> FinetuneJob:
    job = _JOBS.get(job_id)
    if job is None:
        raise KeyError(job_id)
    return job


def list_jobs() -> list[dict]:
    return [j.public(0) for j in sorted(_JOBS.values(), key=lambda j: -j.created_at)]


def _active() -> Optional[FinetuneJob]:
    return next((j for j in _JOBS.values() if j.status in ("queued", "running")), None)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:32] or "adapter"


def start(recipe_id: str, *, dataset_id: str, name: str, overrides: Optional[dict] = None,
          python: Optional[str] = None) -> FinetuneJob:
    info = preflight(recipe_id)
    if not info["runnable"]:
        raise ValueError(f"{info['label']} is not runnable yet: {info.get('note')}")
    if not info["ready"]:
        what = info["missing_packages"] + ([] if info["weights_on_disk"] else [info["base"]])
        raise ValueError("Missing: " + ", ".join(what))
    from services.dataset_builder import dataset_dir

    ds = dataset_dir(dataset_id)  # KeyError → 404
    with _LOCK:
        if _active() is not None:
            raise RuntimeError("A fine-tuning job is already running.")
        adapter_id = f"{_slug(name)}-{uuid.uuid4().hex[:6]}"
        job = FinetuneJob(id=uuid.uuid4().hex[:12], recipe=recipe_id, name=name,
                          dataset_id=dataset_id, adapter_id=adapter_id)
        _JOBS[job.id] = job
    params = {**info["defaults"], **{k: v for k, v in (overrides or {}).items() if v is not None}}
    try:
        lang = json.loads((ds / "dataset.json").read_text(encoding="utf-8"))["options"].get("language")
    except Exception:  # noqa: BLE001
        lang = None
    out = root() / adapter_id
    out.mkdir(parents=True)
    cfg = {**params, "checkpoint": info["weights_path"] or info["base"], "dataset_dir": str(ds),
           "output_dir": str(out), "language": params.get("language") or lang, "name": name}
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    job.steps = int(params.get("steps", 0))
    threading.Thread(target=_run, args=(job, out, python), daemon=True,
                     name=f"finetune-{job.id}").start()
    return job


def _run(job: FinetuneJob, out: Path, python: Optional[str]) -> None:
    job.status, job.stage = "running", "prepare"
    try:
        from services.gpu_memory import release_gpu_holders

        freed = release_gpu_holders(reason="fine-tune")
        job.log.append("Freed GPU first: " + (", ".join(freed) or "nothing was loaded"))
    except Exception as exc:  # noqa: BLE001
        job.log.append(f"Could not free the GPU first: {exc}")
    script = TRAINER
    env = {k: v for k, v in os.environ.items()
           if not re.search(r"TOKEN|SECRET|PASSWORD|API_KEY|_PIN$", k, re.I)}
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1"})
    repo_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    log_path = out / "train.log"
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.Popen([python or sys.executable, str(script), str(out / "config.json")],
                                    stdout=subprocess.PIPE, stderr=log, env=env, cwd=repo_root,
                                    text=True, encoding="utf-8", errors="replace")
            job.proc = proc
            for line in proc.stdout:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                _apply_event(job, ev)
            rc = proc.wait()
        if job.status == "running":
            if rc == 0 and (out / "adapter" / "adapter_config.json").is_file():
                job.status, job.progress = "succeeded", 1.0
            elif (out / "STOP").exists():
                job.status = "cancelled"
            else:
                job.status = "failed"
                tail = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-5:]
                job.error = job.error or ("Training exited with code %s: %s" % (rc, " | ".join(tail)))
    except Exception as exc:  # noqa: BLE001
        job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
    finally:
        job.proc = None
        job.finished_at = time.time()


def _apply_event(job: FinetuneJob, ev: dict) -> None:
    kind = ev.get("event")
    if kind == "stage":
        job.stage = ev.get("stage")
        job.log.append(json.dumps(ev))
    elif kind == "tokenize":
        job.stage = "tokenize"
        job.progress = 0.1 * ev["done"] / max(1, ev["total"])
    elif kind == "step":
        job.stage, job.step, job.steps = "train", ev["step"], ev["steps"]
        job.loss, job.peak_vram_mb = ev.get("loss_ema"), ev.get("peak_vram_mb")
        job.progress = 0.1 + 0.9 * ev["step"] / max(1, ev["steps"])
        job.log.append(f"step {ev['step']}/{ev['steps']} loss {ev.get('loss')} "
                       f"ema {ev.get('loss_ema')} vram {ev.get('peak_vram_mb')} MB")
    elif kind == "error":
        job.error = ev.get("message")
        job.log.append(job.error or "error")
    elif kind == "cancelled":
        job.status = "cancelled"
    elif kind == "done":
        job.log.append(f"Saved adapter {ev.get('adapter')}")


def cancel(job_id: str) -> FinetuneJob:
    job = get_job(job_id)
    if job.adapter_id:
        (root() / job.adapter_id / "STOP").write_text("stop", encoding="utf-8")
    proc = job.proc
    if proc is not None:
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
    return job


# ── Adapters ─────────────────────────────────────────────────────────────────

def _adapter_dir(adapter_id: str) -> Path:
    if not _ID_RE.match(adapter_id or ""):
        raise KeyError(adapter_id)
    path = root() / adapter_id / "adapter"
    if not (path / "adapter_config.json").is_file():
        raise KeyError(adapter_id)
    return path


def list_adapters() -> list[dict]:
    active = active_adapter_id()
    out = []
    for d in sorted(root().iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        meta = d / "adapter" / "voicestudio.json"
        if meta.is_file() and _ID_RE.match(d.name):
            try:
                info = json.loads(meta.read_text(encoding="utf-8"))
            except ValueError:
                continue
            cfg = {}
            try:
                cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
            out.append({"id": d.name, "name": cfg.get("name") or d.name, **info,
                        "active": d.name == active})
    return out


def active_adapter_id() -> Optional[str]:
    try:
        from core import prefs

        value = prefs.get(PREF_KEY)
    except Exception:  # noqa: BLE001
        return None
    return value if isinstance(value, str) and value else None


def set_active_adapter(adapter_id: Optional[str]) -> Optional[str]:
    from core import prefs

    if adapter_id:
        _adapter_dir(adapter_id)
    prefs.set_(PREF_KEY, adapter_id or "")
    try:  # next generation reloads the shared model with/without the adapter
        from services import model_manager as mm

        mm.unload_shared_model()
    except Exception:  # noqa: BLE001
        logger.warning("could not unload the TTS model after adapter change", exc_info=True)
    return adapter_id


def apply_active_adapter(model) -> Optional[str]:
    """Merge the active LoRA adapter into a freshly loaded OmniVoice model.
    Best-effort: any failure logs and leaves the base model untouched."""
    adapter_id = active_adapter_id()
    if not adapter_id:
        return None
    try:
        path = _adapter_dir(adapter_id)
        from peft import PeftModel

        model.llm = PeftModel.from_pretrained(model.llm, str(path)).merge_and_unload()
        logger.info("Applied LoRA adapter %s to the TTS model.", adapter_id)
        return adapter_id
    except Exception:  # noqa: BLE001
        logger.warning("LoRA adapter %s could not be applied; using the base model.",
                       adapter_id, exc_info=True)
        return None
