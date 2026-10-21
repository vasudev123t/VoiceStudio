"""On-disk registry of RVC voice models (Voice conversion feature).

Layout, under the app data dir (``<data>/rvc_voices`` by default, overridable
with ``OMNIVOICE_RVC_VOICES_DIR``)::

    rvc_voices/<voice_id>/model.pth     # RVC generator weights (Applio / RVC WebUI)
    rvc_voices/<voice_id>/model.index   # optional FAISS retrieval index
    rvc_voices/<voice_id>/meta.json     # {id, name, created_at, sample_rate, ...}

Imports always COPY the user's files: the originals (e.g. inside an Applio
``logs/`` folder) are never moved or modified. Voice ids are server-generated
(slug + random suffix) and every path is resolved inside the registry root, so
a client-supplied id can never address files outside it.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Optional

logger = logging.getLogger("omnivoice.rvc")

MODEL_FILENAME = "model.pth"
INDEX_FILENAME = "model.index"
META_FILENAME = "meta.json"

#: RVC generator weights are ~55 MB (v2, fp16) and rarely above 200 MB; the
#: FAISS index of a 30-minute dataset can reach a few hundred MB.
MAX_MODEL_BYTES = 1024 * 1024 * 1024
MAX_INDEX_BYTES = 2 * 1024 * 1024 * 1024
MAX_NAME_CHARS = 80

_VOICE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_lock = threading.RLock()


class VoiceNotFound(LookupError):
    pass


class InvalidVoiceFile(ValueError):
    pass


def voices_root() -> Path:
    override = os.environ.get("OMNIVOICE_RVC_VOICES_DIR", "").strip()
    if override:
        root = Path(override)
    else:
        from core.config import DATA_DIR

        root = Path(DATA_DIR) / "rvc_voices"
    root.mkdir(parents=True, exist_ok=True)
    return root


def clean_name(name: object) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]", "", str(name or "")).strip()
    if not text:
        raise ValueError("Voice name is required.")
    return text[:MAX_NAME_CHARS]


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40]
    return slug or "voice"


def _voice_dir(voice_id: str) -> Path:
    if not isinstance(voice_id, str) or not _VOICE_ID.fullmatch(voice_id):
        raise VoiceNotFound(voice_id)
    root = voices_root().resolve()
    path = (root / voice_id).resolve()
    if path.parent != root:
        raise VoiceNotFound(voice_id)
    return path


def _read_meta(path: Path) -> Optional[dict]:
    try:
        meta = json.loads((path / META_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict) or not (path / MODEL_FILENAME).is_file():
        return None
    return meta


def _write_meta(path: Path, meta: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=path, prefix=".meta-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path / META_FILENAME)


def _public(meta: dict, path: Path) -> dict:
    index = path / INDEX_FILENAME
    model = path / MODEL_FILENAME
    return {
        "id": meta["id"],
        "name": meta.get("name") or meta["id"],
        "created_at": meta.get("created_at"),
        "source": meta.get("source", "import"),
        "sample_rate": meta.get("sample_rate"),
        "version": meta.get("version"),
        "f0": meta.get("f0"),
        "epochs": meta.get("epochs"),
        "has_index": index.is_file(),
        "index_source": (meta.get("index_source") or ("bundled" if index.is_file() else None)),
        "model_bytes": model.stat().st_size if model.is_file() else 0,
        "index_bytes": index.stat().st_size if index.is_file() else 0,
    }


def inspect_model(model_path: os.PathLike | str) -> dict:
    """Best-effort metadata from an RVC checkpoint (no code execution).

    Applio/RVC inference checkpoints are plain dicts whose ``config`` list ends
    with the target sample rate, plus ``version`` (v1/v2) and ``f0`` keys.
    ``weights_only=True`` refuses pickled code; failures just mean "unknown".
    """
    info: dict[str, Any] = {}
    try:
        import torch

        cpt = torch.load(str(model_path), map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 — metadata is optional
        logger.info("RVC model metadata unavailable for %s: %s", model_path, exc)
        return info
    if not isinstance(cpt, dict):
        return info
    config = cpt.get("config")
    if isinstance(config, (list, tuple)) and config and isinstance(config[-1], int):
        info["sample_rate"] = int(config[-1])
    elif isinstance(cpt.get("sr"), (int, str)):
        sr = str(cpt["sr"]).lower().rstrip("k")
        if sr.isdigit():
            info["sample_rate"] = int(sr) * (1000 if int(sr) < 1000 else 1)
    if isinstance(cpt.get("version"), str):
        info["version"] = cpt["version"]
    if "f0" in cpt:
        try:
            info["f0"] = bool(int(cpt["f0"]))
        except (TypeError, ValueError):
            pass
    if "weight" not in cpt and "model" not in cpt:
        raise InvalidVoiceFile(
            "This .pth is not an RVC inference model. Use the small "
            "<name>_<epoch>e_<step>s.pth from Applio, not a G_/D_ training checkpoint."
        )
    return info


def _copy_stream(src: BinaryIO, dest: Path, limit: int, what: str) -> int:
    total = 0
    with open(dest, "wb") as out:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise InvalidVoiceFile(f"{what} is too large.")
            out.write(chunk)
    if total == 0:
        raise InvalidVoiceFile(f"{what} is empty.")
    return total


def create_voice(
    name: str,
    model: BinaryIO,
    index: Optional[BinaryIO] = None,
    *,
    source: str = "import",
    extra: Optional[dict] = None,
) -> dict:
    """Register a voice from open binary streams (upload or local file)."""
    name = clean_name(name)
    root = voices_root()
    voice_id = f"{_slug(name)}-{uuid.uuid4().hex[:6]}"
    staging = Path(tempfile.mkdtemp(dir=root, prefix=".import-"))
    try:
        _copy_stream(model, staging / MODEL_FILENAME, MAX_MODEL_BYTES, "Model file")
        if index is not None:
            _copy_stream(index, staging / INDEX_FILENAME, MAX_INDEX_BYTES, "Index file")
        info = inspect_model(staging / MODEL_FILENAME)
        meta = {
            "id": voice_id,
            "name": name,
            "created_at": time.time(),
            "source": source,
            **info,
            **(extra or {}),
        }
        _write_meta(staging, meta)
        with _lock:
            os.replace(staging, root / voice_id)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    logger.info("RVC voice registered: %s (%s)", voice_id, source)
    return _public(meta, root / voice_id)


def import_from_paths(
    name: str,
    model_path: os.PathLike | str,
    index_path: Optional[os.PathLike | str] = None,
    *,
    source: str = "import",
    extra: Optional[dict] = None,
) -> dict:
    """Copy an existing ``.pth`` (+ optional ``.index``) into the registry."""
    model_path = Path(model_path)
    if model_path.suffix.lower() != ".pth" or not model_path.is_file():
        raise InvalidVoiceFile("Model must be an existing .pth file.")
    idx = Path(index_path) if index_path else None
    if idx is not None and (idx.suffix.lower() != ".index" or not idx.is_file()):
        raise InvalidVoiceFile("Index must be an existing .index file.")
    with open(model_path, "rb") as model_fh:
        if idx is None:
            return create_voice(name, model_fh, None, source=source, extra=extra)
        with open(idx, "rb") as index_fh:
            return create_voice(name, model_fh, index_fh, source=source, extra=extra)


def list_voices() -> list[dict]:
    root = voices_root()
    voices = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        meta = _read_meta(child)
        if meta and meta.get("id") == child.name:
            voices.append(_public(meta, child))
    voices.sort(key=lambda v: v.get("created_at") or 0, reverse=True)
    return voices


def get_voice(voice_id: str) -> dict:
    path = _voice_dir(voice_id)
    meta = _read_meta(path)
    if not meta:
        raise VoiceNotFound(voice_id)
    return _public(meta, path)


def voice_files(voice_id: str) -> tuple[str, Optional[str]]:
    """Absolute ``(model_path, index_path|None)`` of a registered voice."""
    path = _voice_dir(voice_id)
    if not _read_meta(path):
        raise VoiceNotFound(voice_id)
    index = path / INDEX_FILENAME
    return str(path / MODEL_FILENAME), (str(index) if index.is_file() else None)


def attach_index(voice_id: str, index_path: os.PathLike | str, *,
                 extra: Optional[dict] = None) -> dict:
    """Install a (re)built FAISS ``.index`` for a registered voice.

    An existing index is never deleted: it is kept as
    ``model.index.bak-<timestamp>`` next to the new one.
    """
    src = Path(index_path)
    if src.suffix.lower() != ".index" or not src.is_file():
        raise InvalidVoiceFile("Index must be an existing .index file.")
    if src.stat().st_size > MAX_INDEX_BYTES:
        raise InvalidVoiceFile("Index file is too large.")
    with _lock:
        path = _voice_dir(voice_id)
        meta = _read_meta(path)
        if not meta:
            raise VoiceNotFound(voice_id)
        dest = path / INDEX_FILENAME
        fd, tmp = tempfile.mkstemp(dir=path, prefix=".index-", suffix=".tmp")
        os.close(fd)
        try:
            shutil.copyfile(src, tmp)
            if dest.is_file():
                os.replace(dest, path / f"{INDEX_FILENAME}.bak-{int(time.time())}")
            os.replace(tmp, dest)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        meta.update({"index_source": "rebuilt", "index_built_at": time.time(), **(extra or {})})
        _write_meta(path, meta)
    logger.info("RVC voice %s: index attached (%s)", voice_id, src.name)
    return _public(meta, path)


def voice_meta(voice_id: str) -> dict:
    """Raw ``meta.json`` of a registered voice (copy)."""
    path = _voice_dir(voice_id)
    meta = _read_meta(path)
    if not meta:
        raise VoiceNotFound(voice_id)
    return dict(meta)


def rename_voice(voice_id: str, name: str) -> dict:
    name = clean_name(name)
    with _lock:
        path = _voice_dir(voice_id)
        meta = _read_meta(path)
        if not meta:
            raise VoiceNotFound(voice_id)
        meta["name"] = name
        _write_meta(path, meta)
    return _public(meta, path)


def delete_voice(voice_id: str) -> None:
    with _lock:
        path = _voice_dir(voice_id)
        if not _read_meta(path):
            raise VoiceNotFound(voice_id)
        trash = path.with_name(f".deleted-{voice_id}-{uuid.uuid4().hex[:6]}")
        os.replace(path, trash)
    shutil.rmtree(trash, ignore_errors=True)
