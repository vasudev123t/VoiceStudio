"""Real nonverbal library (review item 7).

Libraries are folders of the target speaker's *real* laughs, breaths and
sighs — pulled from a Voice Lab dataset (``nonverbal/`` export) or uploaded —
under ``DATA_DIR/nonverbal_library/<id>/``::

    library.json   {"name", "created_at", "source_dataset", "clips": [...]}
    clips/*.wav

When a library is *active*, ``[laughter]``/``[laugh]``/``[breath]``/``[sigh]``
tags in TTS text are not handed to the model; the text around them is
synthesized as usual and a real clip of that kind is spliced in with fades
and loudness matched to the neighbouring speech (``render``). Tags of a kind
the library has no clips for are left to the model, so nothing regresses.

``restore_regions`` is the performance-transfer post-step for voice
conversion: inside laugh/breath regions of the source performance it keeps
the performer's original sound (RVC turns laughs into pseudo-speech) or swaps
in a library clip.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

PREF_KEY = "nonverbal_library_active"
KINDS = ("laugh", "breath", "sigh")
_TAG_WORDS = {
    "laugh": ("laughter", "laugh", "laughs", "laughing", "chuckle", "chuckles", "giggle", "giggles"),
    "breath": ("breath", "breathe", "breathes", "breathing", "inhale", "inhales", "exhale", "exhales"),
    "sigh": ("sigh", "sighs", "sighing"),
}
_WORD_TO_KIND = {w: k for k, ws in _TAG_WORDS.items() for w in ws}
TAG_RE = re.compile(r"\[\s*(" + "|".join(sorted(_WORD_TO_KIND, key=len, reverse=True)) + r")\s*\]",
                    re.IGNORECASE)
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")
_MAX_CLIPS = 300
# Loudness of the spliced clip relative to the surrounding speech RMS (dB).
_REL_DB = {"laugh": -1.0, "breath": -14.0, "sigh": -8.0}


def root() -> Path:
    override = os.environ.get("OMNIVOICE_NONVERBAL_DIR", "").strip()
    if override:
        path = Path(override)
    else:
        from core.config import DATA_DIR

        path = Path(DATA_DIR) / "nonverbal_library"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _lib_dir(lib_id: str) -> Path:
    if not _ID_RE.match(lib_id or ""):
        raise KeyError(lib_id)
    path = root() / lib_id
    if not (path / "library.json").is_file():
        raise KeyError(lib_id)
    return path


def _read_meta(path: Path) -> dict:
    return json.loads((path / "library.json").read_text(encoding="utf-8"))


def _write_meta(path: Path, meta: dict) -> None:
    tmp = path / "library.json.tmp"
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path / "library.json")


def _clip_stats(path: Path) -> dict:
    import numpy as np
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
    return {"duration": round(len(x) / sr, 3), "sample_rate": int(sr),
            "rms_db": round(20 * np.log10(rms + 1e-12), 1)}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:32] or "library"


def create_library(name: str, *, source_dataset: Optional[str] = None) -> dict:
    lib_id = f"{_slug(name)}-{uuid.uuid4().hex[:6]}"
    path = root() / lib_id
    (path / "clips").mkdir(parents=True)
    meta = {"id": lib_id, "name": name, "created_at": time.time(),
            "source_dataset": source_dataset, "clips": []}
    _write_meta(path, meta)
    return meta


def add_clip(lib_id: str, src: Path, kind: str, *, source: Optional[str] = None) -> dict:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    path = _lib_dir(lib_id)
    meta = _read_meta(path)
    if len(meta["clips"]) >= _MAX_CLIPS:
        raise ValueError(f"A library holds at most {_MAX_CLIPS} clips.")
    stats = _clip_stats(src)
    if not 0.1 <= stats["duration"] <= 8.0:
        raise ValueError("Nonverbal clips must be 0.1–8 s long.")
    name = f"{kind}_{uuid.uuid4().hex[:8]}.wav"
    shutil.copyfile(src, path / "clips" / name)
    clip = {"file": name, "kind": kind, "source": source, **stats}
    meta["clips"].append(clip)
    _write_meta(path, meta)
    return clip


def from_dataset(dataset_id: str, name: Optional[str] = None) -> dict:
    """New library from a Voice Lab dataset's tagged laughs and breaths."""
    from services.dataset_builder import dataset_dir

    ds = dataset_dir(dataset_id)
    rows = []
    nv = ds / "nonverbal" / "nonverbal.jsonl"
    if nv.is_file():
        for line in nv.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    if not rows:
        raise ValueError("That dataset has no tagged laughs or breaths.")
    try:
        ds_name = json.loads((ds / "dataset.json").read_text(encoding="utf-8")).get("name")
    except (OSError, ValueError):
        ds_name = None
    meta = create_library(name or f"{ds_name or dataset_id} nonverbals", source_dataset=dataset_id)
    added = skipped = 0
    for row in rows:
        kind = row.get("type")
        rel = str(row.get("file") or "")
        src = (ds / rel).resolve()
        if kind not in KINDS or not src.is_file() or ds.resolve() not in src.parents:
            skipped += 1
            continue
        try:
            add_clip(meta["id"], src, kind, source=row.get("source"))
            added += 1
        except ValueError:
            skipped += 1
    out = get_library(meta["id"])
    out.update(added=added, skipped=skipped)
    return out


def get_library(lib_id: str) -> dict:
    meta = _read_meta(_lib_dir(lib_id))
    counts = {k: sum(1 for c in meta["clips"] if c["kind"] == k) for k in KINDS}
    return {**meta, "counts": counts, "active": active_library_id() == meta["id"]}


def list_libraries() -> list[dict]:
    out = []
    for d in sorted(root().iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if (d / "library.json").is_file() and _ID_RE.match(d.name):
            try:
                lib = get_library(d.name)
            except (KeyError, ValueError, OSError):
                continue
            lib.pop("clips", None)
            out.append(lib)
    return out


def remove_clip(lib_id: str, file: str) -> None:
    """User-initiated: drop one clip from a library (its copy only)."""
    path = _lib_dir(lib_id)
    meta = _read_meta(path)
    keep = [c for c in meta["clips"] if c["file"] != file]
    if len(keep) == len(meta["clips"]):
        raise KeyError(file)
    meta["clips"] = keep
    _write_meta(path, meta)
    (path / "clips" / Path(file).name).unlink(missing_ok=True)


def clip_path(lib_id: str, file: str) -> Path:
    path = _lib_dir(lib_id)
    if not any(c["file"] == file for c in _read_meta(path)["clips"]):
        raise KeyError(file)
    return path / "clips" / Path(file).name


def active_library_id() -> Optional[str]:
    try:
        from core import prefs

        value = prefs.get(PREF_KEY)
    except Exception:  # noqa: BLE001
        return None
    return value if isinstance(value, str) and value else None


def set_active(lib_id: Optional[str]) -> Optional[str]:
    from core import prefs

    if lib_id:
        _lib_dir(lib_id)
    prefs.set_(PREF_KEY, lib_id or "")
    return lib_id


# ── Splicing ────────────────────────────────────────────────────────────────

def plan(text: str, lib_id: Optional[str] = None) -> Optional[list[tuple[str, str]]]:
    """Split ``text`` into ``("text", s)`` / ``("nv", kind)`` pieces when the
    active library can serve at least one tag in it; otherwise None (the
    caller keeps its normal path and the model renders the tags)."""
    if not text or "[" not in text:
        return None
    lib_id = lib_id or active_library_id()
    if not lib_id:
        return None
    try:
        counts = get_library(lib_id)["counts"]
    except (KeyError, ValueError, OSError):
        return None
    pieces: list[tuple[str, str]] = []
    last, served = 0, False
    for m in TAG_RE.finditer(text):
        kind = _WORD_TO_KIND[m.group(1).lower()]
        if not counts.get(kind):
            continue  # leave this tag inside the text for the model
        pieces.append(("text", text[last:m.start()]))
        pieces.append(("nv", kind))
        last, served = m.end(), True
    if not served:
        return None
    pieces.append(("text", text[last:]))
    return [(k, v) for k, v in pieces if k == "nv" or v.strip()]


def pick_clip(lib_id: str, kind: str, key: str) -> Path:
    """Deterministic per (text, position) choice so re-renders are stable."""
    meta = _read_meta(_lib_dir(lib_id))
    clips = [c for c in meta["clips"] if c["kind"] == kind]
    if not clips:
        raise KeyError(kind)
    i = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % len(clips)
    return _lib_dir(lib_id) / "clips" / clips[i]["file"]


def _load_mono(path: Path, sr: int):
    import numpy as np
    import soundfile as sf

    x, src_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if src_sr != sr:
        from services.dataset_builder import _resample

        x = _resample(x, int(src_sr), sr)
    return np.ascontiguousarray(x, dtype="float32")


def _fade(x, sr: int, ms: float = 12.0):
    import numpy as np

    n = min(len(x) // 2, int(sr * ms / 1000))
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype="float32")
        x = x.copy()
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


def _rms_db(x) -> float:
    import numpy as np

    if x is None or len(x) == 0:
        return -30.0
    # speech level: RMS of the louder half of 20 ms frames (ignores pauses)
    hop = 480
    n = max(1, len(x) // hop)
    frames = x[: n * hop].reshape(n, -1) if len(x) >= hop else x.reshape(1, -1)
    e = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    top = np.sort(e)[len(e) // 2:]
    return float(20 * np.log10(np.mean(top) + 1e-12))


def match_level(clip, kind: str, speech_db: float):
    import numpy as np

    clip_db = _rms_db(clip)
    gain = 10 ** ((speech_db + _REL_DB.get(kind, -6.0) - clip_db) / 20)
    out = clip * min(gain, 8.0)
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 0.97:
        out = out * (0.97 / peak)
    return out.astype("float32")


def render(gen_text: Callable[[str], object], pieces: list[tuple[str, str]], sr: int,
           lib_id: Optional[str] = None, *, key: str = ""):
    """Synthesize text pieces with ``gen_text`` (→ torch tensor, any of
    [N] / [1, N] / [C, N]) and splice real clips at the ``nv`` pieces."""
    import numpy as np
    import torch

    lib_id = lib_id or active_library_id()
    rendered: list = []
    ref = None
    for idx, (kind, value) in enumerate(pieces):
        if kind == "text":
            t = gen_text(value)
            ref = t if ref is None else ref
            rendered.append(("a", t.detach().float().cpu().reshape(-1, t.shape[-1]).mean(0).numpy()))
        else:
            rendered.append(("nv", value, idx))
    speech = [a for k, a, *_ in rendered if k == "a"]
    speech_db = float(np.mean([_rms_db(a) for a in speech])) if speech else -22.0
    parts = []
    for item in rendered:
        if item[0] == "a":
            parts.append(_fade(item[1], sr, 4.0))
            continue
        _, nv_kind, idx = item
        try:
            clip = _load_mono(pick_clip(lib_id, nv_kind, f"{key}|{idx}"), sr)
        except (KeyError, OSError, ValueError, RuntimeError):
            logger.warning("nonverbal clip for %s unavailable; skipped", nv_kind)
            continue
        gap = np.zeros(int(sr * 0.06), dtype="float32")
        parts += [gap, _fade(match_level(clip, nv_kind, speech_db), sr), gap]
    audio = np.concatenate(parts) if parts else np.zeros(1, dtype="float32")
    out = torch.from_numpy(audio)
    if ref is not None:
        out = out.to(dtype=ref.dtype)
        if ref.dim() == 2:
            out = out.unsqueeze(0).expand(ref.shape[0], -1).contiguous()
    return out


# ── Performance transfer (voice conversion post-step) ───────────────────────

def parse_regions(spec: str | None, total_s: float) -> list[tuple[float, float]]:
    """"1.2-2.5, 7-8.1" → [(1.2, 2.5), (7.0, 8.1)] clipped to the audio."""
    out = []
    for part in (spec or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", part)
        if not m:
            raise ValueError(f"Bad region {part!r}; use start-end in seconds, e.g. 1.2-2.5")
        a, b = float(m.group(1)), min(float(m.group(2)), total_s)
        if b - a >= 0.05:
            out.append((a, b))
    return sorted(out)


def detect_nonverbal(x, sr: int) -> list[tuple[float, float, str]]:
    """Conservative laugh detector for an acted performance: 1 s windows with
    rhythmic 3.5–7 Hz bursts *and* breathy (spectrally flat) sound; adjacent
    hits merge. Speech alone is periodic but voiced, so it rarely passes."""
    import numpy as np

    from services.dataset_builder import _flatness, _frame_db, laugh_likelihood

    win, hop = int(sr * 1.0), int(sr * 0.25)
    hits = []
    for start in range(0, max(1, len(x) - win + 1), hop):
        w = x[start:start + win]
        if len(w) < win // 2 or float(np.percentile(_frame_db(w, sr), 90)) < -40:
            continue
        lk = laugh_likelihood(w, sr, "", len(w) / sr)
        flat = np.mean([_flatness(w[i:i + 1024]) for i in range(0, len(w) - 1024, 1024)] or [0])
        if lk >= 0.6 and flat >= 0.18:
            hits.append((start / sr, (start + win) / sr))
    merged: list[list[float]] = []
    for a, b in hits:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(round(a, 2), round(b, 2), "laugh") for a, b in merged]


def restore_regions(source_wav: Path, converted_wav: Path, *, mode: str,
                    regions: list[tuple[float, float]], lib_id: Optional[str] = None,
                    xfade_ms: float = 25.0) -> list[dict]:
    """Rewrite ``converted_wav`` in place: inside each region keep the source
    performance (``keep_source``) or a library laugh (``library``), crossfaded."""
    import numpy as np
    import soundfile as sf

    y, sr = sf.read(str(converted_wav), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    src = _load_mono(source_wav, int(sr))
    n = min(len(y), len(src)) if mode == "keep_source" else len(y)
    xf = max(1, int(sr * xfade_ms / 1000))
    applied = []
    for i, (a, b) in enumerate(regions):
        s0, s1 = int(a * sr), min(int(b * sr), n)
        if s1 - s0 < 2 * xf:
            continue
        if mode == "keep_source":
            fill = src[s0:s1]
            level = _rms_db(y[s0:s1])
            src_level = _rms_db(fill)
            fill = fill * min(4.0, 10 ** ((level - src_level) / 20)) if src_level > -60 else fill
        elif mode == "library":
            lib_id = lib_id or active_library_id()
            if not lib_id:
                raise ValueError("No active nonverbal library.")
            clip = _load_mono(pick_clip(lib_id, "laugh", f"{converted_wav.name}|{i}"), int(sr))
            clip = match_level(clip, "laugh", _rms_db(y))
            fill = np.zeros(s1 - s0, dtype="float32")
            fill[: min(len(clip), len(fill))] = clip[: len(fill)]
            fill = _fade(fill, int(sr), 20.0)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        ramp = np.linspace(0, 1, xf, dtype="float32")
        mix = np.ones(s1 - s0, dtype="float32")
        mix[:xf], mix[-xf:] = ramp, ramp[::-1]
        y[s0:s1] = y[s0:s1] * (1 - mix) + fill.astype("float32") * mix
        applied.append({"start": round(a, 3), "end": round(s1 / sr, 3), "mode": mode})
    peak = float(np.max(np.abs(y))) if len(y) else 0
    if peak > 0.99:
        y = y * (0.99 / peak)
    sf.write(str(converted_wav), y, int(sr), subtype="FLOAT")
    return applied
