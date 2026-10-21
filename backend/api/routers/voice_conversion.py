"""Voice conversion (RVC / Applio) — speech-to-speech that keeps the performance.

The user performs a line (mic or file); a trained RVC voice model re-voices it
while keeping prosody, timing, breaths and laughs. Unlike ``POST /convert``
(ASR → TTS re-synthesis) nothing is re-generated from text, which is what
preserves micro-expressions.

Routes (all local-first; no network unless the user explicitly starts the
Applio setup or a training run, which may fetch Applio's pretrained weights):

* ``GET  /voice-conversion/status``            backend, device, Applio paths
* ``PUT  /voice-conversion/settings``          device / desktop-authorized Applio dir
* ``POST /voice-conversion/setup``             explicit, confirmed Applio install job
* ``GET|POST /voice-conversion/voices[/import]``, ``PATCH|DELETE /voice-conversion/voices/{id}``
* ``POST /voice-conversion/convert``           multipart audio + voice_id + params
* ``GET|DELETE /voice-conversion/history[/{id}]``
* ``POST /voice-conversion/unload``            free the model / VRAM
* ``POST /voice-conversion/voices/{id}/index`` (re)build a voice's FAISS index
* ``POST /voice-conversion/train``, ``GET /voice-conversion/jobs``,
  ``GET /voice-conversion/train/{job_id}``, ``POST /voice-conversion/train/{job_id}/cancel``

Every converted take goes through ``mark_synthetic_async`` (provenance
watermark) before it is written, like all other synthetic audio.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from api.dependencies import require_admin
from core.path_security import portable_filename, upload_suffix

router = APIRouter(prefix="/voice-conversion", tags=["voice-conversion"])
logger = logging.getLogger("omnivoice.rvc")

_MAX_SOURCE_BYTES = 256 * 1024 * 1024
_MAX_DATASET_BYTES = 4 * 1024 * 1024 * 1024
_CHUNK = 1024 * 1024
_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".aac", ".webm", ".wma"}

F0Method = Literal["rmvpe", "crepe", "fcpe", "harvest", "pm"]
TrainF0Method = Literal["rmvpe", "crepe", "fcpe"]
SETUP_DOC = "docs/voice-conversion.md"


def _svc():
    from services.rvc import get_service

    return get_service()


async def _run_blocking(fn, *, what: str, timeout: float):
    """Run on the shared GPU pool (serialises with TTS on small-VRAM GPUs);
    falls back to a thread where the pool is unavailable."""
    try:
        from services.model_manager import run_on_gpu_pool_guarded
    except Exception:  # noqa: BLE001
        return await asyncio.to_thread(fn)
    return await run_on_gpu_pool_guarded(fn, what=what, timeout=timeout)


async def _save_upload(upload: UploadFile, dest: Path, limit: int, what: str) -> int:
    total = 0
    with open(dest, "wb") as out:
        while True:
            chunk = await upload.read(_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise HTTPException(status_code=413, detail=f"{what} is too large.")
            out.write(chunk)
    if total == 0:
        raise HTTPException(status_code=400, detail=f"{what} is empty.")
    return total


def _voice_or_404(voice_id: str) -> dict:
    from services import rvc_registry

    try:
        return rvc_registry.get_voice(voice_id)
    except rvc_registry.VoiceNotFound:
        raise HTTPException(status_code=404, detail="That RVC voice doesn't exist.") from None


def _unavailable(exc: Exception) -> HTTPException:
    return HTTPException(status_code=503, detail={
        "code": "rvc_unavailable", "message": str(exc), "docs": SETUP_DOC,
    })


# ── Status / settings / setup ────────────────────────────────────────────────


@router.get("/status")
def status():
    from services import rvc_training

    data = _svc().status()
    active = next((j for j in rvc_training.list_jobs() if j.status in ("queued", "running")), None)
    data["active_job"] = active.public(tail=0) if active else None
    data["docs"] = SETUP_DOC
    return data


class VoiceConversionSettingsBody(BaseModel):
    device: Optional[Literal["auto", "cpu", "cuda", "cuda:0", "cuda:1", "mps"]] = None
    #: One-shot capability minted by the desktop app's native folder picker
    #: (kind ``applio_dir``). The web API never accepts a raw executable path.
    applio_dir_authorization: Optional[str] = None


@router.put("/settings")
def update_settings(body: VoiceConversionSettingsBody):
    from core import prefs
    from services.rvc import reset_service

    if body.device is not None:
        if body.device == "auto":
            prefs.delete("voice_conversion.device")
        else:
            prefs.set_("voice_conversion.device", body.device)
    if body.applio_dir_authorization is not None:
        from core.path_authorization import PathAuthorizationError, consume

        try:
            raw = consume(body.applio_dir_authorization, "applio_dir").strip()
        except PathAuthorizationError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if not raw:
            prefs.delete("voice_conversion.applio_dir")
        else:
            path = Path(os.path.abspath(os.path.expanduser(raw)))
            if not ((path / "core.py").is_file() and (path / "rvc" / "infer" / "infer.py").is_file()):
                raise HTTPException(status_code=400,
                                    detail="That folder is not an Applio checkout (core.py missing).")
            prefs.set_("voice_conversion.applio_dir", str(path))
    reset_service()
    return status()


class VoiceConversionSetupBody(BaseModel):
    confirm: bool = False
    cuda: Optional[bool] = None


@router.post("/setup", dependencies=[Depends(require_admin)])
def setup_applio(body: VoiceConversionSetupBody):
    """Download + install Applio into ``<data>/tools/applio`` (several GB).

    Never runs implicitly: the caller must pass ``confirm=true``.
    """
    from services import rvc_training

    if not body.confirm:
        raise HTTPException(status_code=400, detail="Applio setup downloads several GB; "
                                                    "pass confirm=true to start it.")
    try:
        job = rvc_training.start_setup(cuda=body.cuda)
    except rvc_training.JobConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return job.public()


@router.post("/unload")
def unload():
    _svc().unload()
    return {"ok": True}


# ── Voice registry ───────────────────────────────────────────────────────────


@router.get("/voices")
def list_voices():
    from services import rvc_registry

    return {"voices": rvc_registry.list_voices()}


@router.post("/voices/import")
async def import_voice(
    name: str = Form(..., min_length=1, max_length=80),
    model: UploadFile = File(...),
    index: Optional[UploadFile] = File(None),
):
    """Import an Applio/RVC-trained voice: ``model`` (.pth) + optional ``index``."""
    from core import event_bus
    from services import rvc_registry

    if (upload_suffix(model.filename) or "").lower() != ".pth":
        raise HTTPException(status_code=415, detail="The model must be a .pth file.")
    if index is not None and index.filename and \
            (upload_suffix(index.filename) or "").lower() != ".index":
        raise HTTPException(status_code=415, detail="The index must be a .index file.")
    with tempfile.TemporaryDirectory(prefix="ov-rvc-import-") as tmp:
        model_tmp = Path(tmp) / "model.pth"
        await _save_upload(model, model_tmp, rvc_registry.MAX_MODEL_BYTES, "Model file")
        index_tmp = None
        if index is not None and index.filename:
            index_tmp = Path(tmp) / "model.index"
            await _save_upload(index, index_tmp, rvc_registry.MAX_INDEX_BYTES, "Index file")
        try:
            voice = await asyncio.to_thread(
                rvc_registry.import_from_paths, name, model_tmp, index_tmp,
                source="import", extra={"original_model": portable_filename(model.filename)},
            )
        except (rvc_registry.InvalidVoiceFile, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    event_bus.emit("voice_conversion", {"action": "voices_changed", "id": voice["id"]})
    return voice


class RvcVoiceRenameBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)


@router.patch("/voices/{voice_id}")
def rename_voice(voice_id: str, body: RvcVoiceRenameBody):
    from core import event_bus
    from services import rvc_registry

    _voice_or_404(voice_id)
    try:
        voice = rvc_registry.rename_voice(voice_id, body.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    event_bus.emit("voice_conversion", {"action": "voices_changed", "id": voice_id})
    return voice


@router.delete("/voices/{voice_id}")
def delete_voice(voice_id: str):
    from core import event_bus
    from services import rvc_registry

    _voice_or_404(voice_id)
    _svc().unload()  # never keep a deleted model resident
    rvc_registry.delete_voice(voice_id)
    event_bus.emit("voice_conversion", {"action": "voices_changed", "id": voice_id})
    return {"ok": True}


@router.post("/voices/{voice_id}/index")
async def build_voice_index(
    voice_id: str,
    files: Optional[List[UploadFile]] = File(None),
    dataset_authorization: Optional[str] = Form(None),
    dataset_id: Optional[str] = Form(None),
):
    """(Re)build the FAISS ``.index`` of a voice from recordings of its speaker.

    Imported voices often ship without an index, which silently turns the
    index-rate knob off. Sources: uploaded clips, a desktop-picked folder, or
    a Voice Dataset Builder dataset (``dataset_id``). A trained voice whose
    training features are still on disk needs no audio at all.
    """
    from services import rvc_registry, rvc_training

    _voice_or_404(voice_id)
    job_id = uuid.uuid4().hex[:12]
    dataset_dir: Optional[Path] = None
    uploads = [f for f in (files or []) if f is not None and f.filename]
    if dataset_authorization:
        from core.path_authorization import PathAuthorizationError, consume

        try:
            dataset_dir = Path(consume(dataset_authorization, "rvc_dataset_dir"))
        except PathAuthorizationError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
    elif dataset_id:
        try:
            from services import dataset_builder

            dataset_dir = dataset_builder.dataset_clips_dir(dataset_id)
        except (ImportError, LookupError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="That dataset doesn't exist.") from exc
    elif uploads:
        dataset_dir = rvc_training.training_root() / f"index-{job_id}" / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        for i, upload in enumerate(uploads):
            ext = (upload_suffix(upload.filename) or "").lower()
            if ext not in rvc_training.AUDIO_EXTS:
                shutil.rmtree(dataset_dir.parent, ignore_errors=True)
                raise HTTPException(status_code=415,
                                    detail=f"Unsupported dataset file: {upload.filename}")
            stem = portable_filename(Path(upload.filename).stem, default="clip", max_bytes=80)
            try:
                total += await _save_upload(upload, dataset_dir / f"{i:04d}_{stem}{ext}",
                                            _MAX_DATASET_BYTES - total, "Dataset")
            except HTTPException:
                shutil.rmtree(dataset_dir.parent, ignore_errors=True)
                raise
    try:
        job = rvc_training.start_index_build(voice_id, dataset_dir=dataset_dir, job_id=job_id)
    except rvc_registry.VoiceNotFound:
        raise HTTPException(status_code=404, detail="That RVC voice doesn't exist.") from None
    except rvc_training.JobConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise _unavailable(exc) from exc
    return job.public()


# ── Conversion ───────────────────────────────────────────────────────────────


@router.post("/convert")
async def convert(
    audio: UploadFile = File(...),
    voice_id: str = Form(...),
    pitch: int = Form(0, ge=-24, le=24),
    f0_method: F0Method = Form("rmvpe"),
    index_rate: float = Form(0.75, ge=0.0, le=1.0),
    protect: float = Form(0.33, ge=0.0, le=0.5),
    filter_radius: int = Form(3, ge=0, le=7),
    rms_mix_rate: float = Form(0.1, ge=0.0, le=1.0),
    clean_audio: bool = Form(False),
    clean_strength: float = Form(12.0, ge=1.0, le=30.0),
    f0_autotune: bool = Form(False),
):
    """Convert ``audio`` with a registered voice.

    The upload is copied untouched (``source_url``) and the converted take is
    written as WAV at the voice model's sample rate (``audio_url``).
    """
    from core.config import OUTPUTS_DIR
    from services import rvc_history
    from services.rvc import (CONVERT_TIMEOUT_S, ConversionParams, RVCConversionError,
                              RVCUnavailable)

    voice = _voice_or_404(voice_id)
    ext = (upload_suffix(audio.filename, ".wav") or ".wav").lower()
    if ext not in _AUDIO_SUFFIXES:
        raise HTTPException(status_code=415, detail="Unsupported audio format.")
    params = ConversionParams(
        pitch=pitch, f0_method=f0_method, index_rate=index_rate, protect=protect,
        filter_radius=filter_radius, rms_mix_rate=rms_mix_rate, clean_audio=clean_audio,
        clean_strength=clean_strength, f0_autotune=f0_autotune,
    )
    entry_id = uuid.uuid4().hex[:12]
    outputs = Path(OUTPUTS_DIR)
    outputs.mkdir(parents=True, exist_ok=True)
    source_name = f"vc_{entry_id}_src{ext}"
    output_name = f"vc_{entry_id}.wav"
    source_path = outputs / source_name
    try:
        await _save_upload(audio, source_path, _MAX_SOURCE_BYTES, "Source audio")
    except BaseException:
        source_path.unlink(missing_ok=True)
        raise

    with tempfile.TemporaryDirectory(prefix="ov-rvc-") as tmp:
        raw_out = os.path.join(tmp, "converted.wav")
        # Converters read the input path; give them a private copy so the
        # kept original can never be touched by a third-party pipeline.
        work_in = os.path.join(tmp, f"input{ext}")
        shutil.copyfile(source_path, work_in)
        try:
            result = await _run_blocking(
                functools.partial(_svc().convert_voice, voice_id, work_in, raw_out, params),
                what="Voice conversion", timeout=CONVERT_TIMEOUT_S + 30,
            )
        except RVCUnavailable as exc:
            source_path.unlink(missing_ok=True)
            raise _unavailable(exc) from exc
        except ValueError as exc:
            source_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RVCConversionError as exc:
            source_path.unlink(missing_ok=True)
            logger.error("RVC conversion failed: %s", exc)
            raise HTTPException(status_code=500,
                                detail=f"Voice conversion failed: {str(exc)[:300]}") from exc
        except Exception:
            source_path.unlink(missing_ok=True)
            raise
        duration = await _mark_and_save(raw_out, outputs / output_name)

    entry = rvc_history.add({
        "id": entry_id,
        "created_at": time.time(),
        "voice_id": voice_id,
        "voice_name": voice["name"],
        "filename": output_name,
        "source_filename": source_name,
        "source_name": portable_filename(audio.filename or "recording", default="recording"),
        "sample_rate": result.sample_rate,
        "duration_s": duration,
        "elapsed_s": result.elapsed_s,
        "backend": result.backend,
        "device": result.device,
        "params": result.params,
        # Without an index the index-rate knob has no effect (RVC skips
        # retrieval); record it so the UI can say why a take sounds less close.
        "index_used": bool(voice.get("has_index")),
    })
    return _entry_public(entry)


async def _mark_and_save(src: str, dest: Path) -> float:
    """Provenance-mark the converted take, then save it as 16-bit WAV."""
    import soundfile as sf
    import torch

    from services.watermark import mark_synthetic_async

    data, sr = await asyncio.to_thread(sf.read, src, dtype="float32", always_2d=True)
    tensor = torch.from_numpy(data.T.copy())
    tensor = await mark_synthetic_async(tensor, int(sr), context="voice_conversion.convert")
    array = tensor.detach().cpu().float().numpy().T
    await asyncio.to_thread(sf.write, str(dest), array, int(sr), subtype="PCM_16")
    return round(array.shape[0] / float(sr), 2)


def _entry_public(entry: dict) -> dict:
    return {
        **entry,
        "audio_url": f"/audio/{entry['filename']}",
        "source_url": f"/audio/{entry['source_filename']}" if entry.get("source_filename") else None,
    }


@router.get("/history")
def history():
    from services import rvc_history

    return {"items": [_entry_public(e) for e in rvc_history.list_entries()]}


@router.delete("/history/{entry_id}")
def delete_history(entry_id: str):
    from services import rvc_history

    if not rvc_history.delete(entry_id):
        raise HTTPException(status_code=404, detail="Conversion not found.")
    return {"ok": True}


@router.delete("/history")
def clear_history():
    from services import rvc_history

    return {"removed": rvc_history.clear()}


# ── Training ─────────────────────────────────────────────────────────────────


@router.post("/train")
async def train(
    name: str = Form(..., min_length=1, max_length=80),
    sample_rate: Literal["32000", "40000", "48000"] = Form("48000"),
    epochs: int = Form(300, ge=1, le=10000),
    batch_size: int = Form(8, ge=1, le=50),
    f0_method: TrainF0Method = Form("rmvpe"),
    save_every: int = Form(25, ge=1, le=100),
    checkpointing: bool = Form(False),
    files: Optional[List[UploadFile]] = File(None),
    dataset_authorization: Optional[str] = Form(None),
    dataset_id: Optional[str] = Form(None, max_length=64),
):
    """Train a new voice with Applio: uploaded clips, a desktop-picked folder,
    or a Voice Lab dataset (``dataset_id``).

    Defaults are sized for an 8 GB GPU (batch 8, 48 kHz, rmvpe). Progress is
    pushed as ``voice_conversion`` events and polled via ``GET /train/{id}``.
    """
    from services import rvc_training

    settings = rvc_training.TrainSettings(
        name=name, sample_rate=int(sample_rate), epochs=epochs, batch_size=batch_size,
        f0_method=f0_method, save_every=min(save_every, epochs), checkpointing=checkpointing,
    )
    try:
        settings.validate()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    uploads = [f for f in (files or []) if f is not None and f.filename]
    if not uploads and not dataset_authorization and not dataset_id:
        raise HTTPException(status_code=400, detail="Add dataset audio files or choose a folder.")
    if rvc_training._active("train") is not None:
        raise HTTPException(status_code=409, detail="A training job is already running.")

    job_id = uuid.uuid4().hex[:12]
    workdir = rvc_training.training_root() / job_id
    if dataset_id and not dataset_authorization and not uploads:
        from services.dataset_builder import dataset_clips_dir

        try:
            dataset_dir = dataset_clips_dir(dataset_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="That dataset doesn't exist.") from None
    elif dataset_authorization:
        from core.path_authorization import PathAuthorizationError, consume

        try:
            dataset_dir = Path(consume(dataset_authorization, "rvc_dataset_dir"))
        except PathAuthorizationError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
    else:
        dataset_dir = workdir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        for i, upload in enumerate(uploads):
            ext = (upload_suffix(upload.filename) or "").lower()
            if ext not in rvc_training.AUDIO_EXTS:
                shutil.rmtree(workdir, ignore_errors=True)
                raise HTTPException(status_code=415,
                                    detail=f"Unsupported dataset file: {upload.filename}")
            stem = portable_filename(Path(upload.filename).stem, default="clip", max_bytes=80)
            try:
                total += await _save_upload(upload, dataset_dir / f"{i:04d}_{stem}{ext}",
                                            _MAX_DATASET_BYTES - total, "Dataset")
            except HTTPException:
                shutil.rmtree(workdir, ignore_errors=True)
                raise
    try:
        job = rvc_training.start_training(settings, dataset_dir=dataset_dir, job_id=job_id,
                                          workdir=workdir)
    except rvc_training.JobConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise _unavailable(exc) from exc
    return job.public()


@router.get("/jobs")
def jobs():
    from services import rvc_training

    return {"jobs": [j.public(tail=0) for j in rvc_training.list_jobs()]}


@router.get("/train/{job_id}")
def train_status(job_id: str, tail: int = Query(200, ge=0, le=4000)):
    from services import rvc_training

    job = rvc_training.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job.public(tail=tail)


@router.post("/train/{job_id}/cancel")
def cancel_train(job_id: str):
    from services import rvc_training

    job = rvc_training.cancel_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job.public(tail=0)
