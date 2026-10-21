"""A/B take scoring (review item 5): SECS, WER/CER and MOS per take.

Uses the research-grade scorers from ``omnivoice/eval`` when their weights
(``k2-fsa/TTS_eval_models``) are on disk, and clearly-labelled local
fallbacks otherwise, so the UI can always rank takes:

=========  ==========================================  ==========================
metric     preferred (needs TTS_eval_models)           fallback (on disk today)
=========  ==========================================  ==========================
SECS       WavLM-large ECAPA-TDNN (omnivoice SIM-o)    SpeechBrain ECAPA if cached,
                                                       else MFCC-statistics cosine
WER/CER    faster-whisper (cached) vs the take's text  — (skipped without ASR)
MOS        UTMOS22 strong (omnivoice eval)             DSP proxy (SNR/clipping/level)
=========  ==========================================  ==========================

CER is used instead of WER for unsegmented scripts and for Kannada/other
Indic scripts where word segmentation of ASR output is unreliable.
Nothing is downloaded; each metric reports the ``method`` that produced it.
"""
from __future__ import annotations

import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SR = 16_000
EVAL_REPO = "k2-fsa/TTS_eval_models"
_CER_LANGS = {"zh", "ja", "ko", "th", "kn", "ta", "te", "ml", "hi", "bn", "mr", "gu", "pa", "or"}
_SIM_FILES = ("speaker_similarity/wavlm_large_finetune.pth",
              "speaker_similarity/wavlm_large/wavlm_large.pt")
_MOS_FILE = "mos/utmos22_strong_step7459_v1.pt"


# ── Model discovery ─────────────────────────────────────────────────────────

def eval_models_dir() -> Optional[Path]:
    override = os.environ.get("OMNIVOICE_EVAL_MODELS_DIR", "").strip()
    if override and Path(override).is_dir():
        return Path(override)
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(EVAL_REPO, local_files_only=True))
    except Exception:  # noqa: BLE001
        return None


def capabilities() -> dict:
    import importlib.util

    root = eval_models_dir()
    sim_ok = bool(root and all((root / f).exists() for f in _SIM_FILES)
                  and importlib.util.find_spec("s3prl") is not None)
    mos_ok = bool(root and (root / _MOS_FILE).exists())
    from services import dataset_builder as db

    ecapa = db._ecapa_dir() if importlib.util.find_spec("speechbrain") else None
    asr = db._whisper_dir("Systran/faster-whisper-large-v3")
    return {
        "secs": {"method": "wavlm-ecapa" if sim_ok else ("ecapa" if ecapa else "mfcc-stats"),
                 "research_grade": sim_ok,
                 "missing": None if sim_ok else f"{EVAL_REPO} (speaker_similarity, ~1.3 GB) + pip s3prl"},
        "wer": {"method": "faster-whisper-large-v3" if asr else None, "available": bool(asr),
                "missing": None if asr else "Systran/faster-whisper-large-v3"},
        "mos": {"method": "utmos22" if mos_ok else "dsp-proxy", "research_grade": mos_ok,
                "missing": None if mos_ok else f"{EVAL_REPO} (mos/utmos22, ~0.4 GB)"},
    }


# ── Text metrics ─────────────────────────────────────────────────────────────

_PUNCT = re.compile(r"[\W_]+", re.UNICODE)


def normalize_for_wer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", text)  # [laugh] (sigh) tags
    # Keep combining marks (Indic vowel signs) — \W would split words on them.
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        out.append(ch if cat[0] in ("L", "N", "M") else " ")
    return " ".join("".join(out).split())


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def error_rate(reference: str, hypothesis: str, *, unit: str) -> float:
    ref, hyp = normalize_for_wer(reference), normalize_for_wer(hypothesis)
    if unit == "cer":
        r, h = list(ref.replace(" ", "")), list(hyp.replace(" ", ""))
    else:
        r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    return round(edit_distance(r, h) / len(r), 4)


def metric_unit(language: Optional[str]) -> str:
    code = (language or "").strip().lower()
    try:
        from services.text_normalization import _plain_lang_code

        code = _plain_lang_code(language) or code
    except Exception:  # noqa: BLE001
        pass
    return "cer" if code[:2] in _CER_LANGS else "wer"


def asr_language(language: Optional[str]) -> Optional[str]:
    try:
        from services.text_normalization import _plain_lang_code

        code = _plain_lang_code(language)
    except Exception:  # noqa: BLE001
        code = None
    return code[:2] if code and len(code) >= 2 and code != "auto" else None


# ── Scorers ──────────────────────────────────────────────────────────────────

def _load16(path: str):
    from services import dataset_builder as db

    x, sr = db._read(Path(path)) if str(path).lower().endswith(".wav") else (None, None)
    if x is None:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="ov-score-") as tmp:
            wav = db.decode(path, Path(tmp) / "a.wav", sr=SR)
            x, sr = db._read(wav)
    return db._resample(x, sr, SR)


class Scorer:
    """Loads each model once per batch; ``close()`` frees them."""

    def __init__(self, device: Optional[str] = None):
        from services import dataset_builder as db

        self.device = device or db._device(db.BuildOptions())
        self.caps = capabilities()
        self._sim = None
        self._mos = None
        self._asr = None

    # SECS
    def _speaker_model(self):
        if self._sim is not None:
            return self._sim
        method = self.caps["secs"]["method"]
        if method == "wavlm-ecapa":
            import torch

            from omnivoice.eval.models.ecapa_tdnn_wavlm import ECAPA_TDNN_WAVLM

            root = eval_models_dir()
            model = ECAPA_TDNN_WAVLM(feat_dim=1024, channels=512, emb_dim=256, sr=SR,
                                     ssl_model_path=str(root / "speaker_similarity/wavlm_large/"))
            state = torch.load(str(root / _SIM_FILES[0]), map_location="cpu")
            model.load_state_dict(state["model"], strict=False)
            model.to(self.device).eval()

            def embed(x16):
                with torch.no_grad():
                    return model([torch.from_numpy(x16).to(self.device)]).squeeze().cpu().numpy()

            self._sim = ("wavlm-ecapa", embed)
        elif method == "ecapa":
            from services import dataset_builder as db

            enc = db._EcapaEmbedder(db._ecapa_dir(), self.device)
            self._sim = ("ecapa", enc.embed)
        else:
            from services import dataset_builder as db

            enc = db._MfccEmbedder()
            self._sim = ("mfcc-stats", enc.embed)
        return self._sim

    def secs(self, take16, ref16) -> dict:
        from services import dataset_builder as db

        method, embed = self._speaker_model()
        return {"value": round(db._cos(embed(take16), embed(ref16)), 4), "method": method,
                "research_grade": method == "wavlm-ecapa"}

    # MOS
    def mos(self, take16) -> dict:
        if self.caps["mos"]["method"] == "utmos22":
            import torch

            if self._mos is None:
                from omnivoice.eval.models.utmos import UTMOS22Strong

                model = UTMOS22Strong()
                model.load_state_dict(torch.load(str(eval_models_dir() / _MOS_FILE),
                                                 map_location="cpu"))
                self._mos = model.to(self.device).eval()
            with torch.no_grad():
                val = self._mos(torch.from_numpy(take16)[None].to(self.device), SR).item()
            return {"value": round(val, 3), "method": "utmos22", "research_grade": True}
        from services import dataset_builder as db

        q = db.quality_score(db.quality_metrics(take16, SR), None)
        return {"value": round(1.0 + 4.0 * q, 2), "method": "dsp-proxy", "research_grade": False}

    # WER
    def wer(self, take16, text: str, language: Optional[str]) -> Optional[dict]:
        if not text or not self.caps["wer"]["available"]:
            return None
        from services import dataset_builder as db

        if self._asr is None:
            self._asr = db._Transcriber(db._whisper_dir("Systran/faster-whisper-large-v3"),
                                        self.device)
        hyp, conf, _lang = self._asr(take16, asr_language(language))
        unit = metric_unit(language)
        return {"value": error_rate(text, hyp, unit=unit), "unit": unit, "transcript": hyp,
                "asr_conf": conf, "method": self.caps["wer"]["method"]}

    def close(self) -> None:
        self._sim = self._mos = None
        if self._asr is not None:
            self._asr.close()
        self._asr = None
        from services import dataset_builder as db

        db._free_cuda()


def overall(score: dict) -> float:
    """0–1 ranking score: speaker match, intelligibility, naturalness."""
    parts = []
    if score.get("secs"):
        method = score["secs"]["method"]
        lo, hi = (0.2, 0.8) if method in ("wavlm-ecapa", "ecapa") else (0.5, 1.0)
        parts.append((min(1.0, max(0.0, (score["secs"]["value"] - lo) / (hi - lo))), 0.4))
    if score.get("wer"):
        parts.append((max(0.0, 1.0 - score["wer"]["value"]), 0.35))
    if score.get("mos"):
        parts.append((min(1.0, max(0.0, (score["mos"]["value"] - 1.0) / 4.0)), 0.25))
    if not parts:
        return 0.0
    return round(sum(v * w for v, w in parts) / sum(w for _, w in parts), 4)


def score_takes(takes: list[dict], *, reference: Optional[str], device: Optional[str] = None,
                scorer: Optional[Scorer] = None) -> list[dict]:
    """``takes``: [{id, path, text, language}] → per-take scores, ranked."""
    scorer = scorer or Scorer(device)
    results = []
    try:
        ref16 = _load16(reference) if reference else None
        for take in takes:
            row = {"id": take["id"], "secs": None, "wer": None, "mos": None, "error": None}
            try:
                x16 = _load16(take["path"])
                if ref16 is not None:
                    row["secs"] = scorer.secs(x16, ref16)
                row["wer"] = scorer.wer(x16, take.get("text") or "", take.get("language"))
                row["mos"] = scorer.mos(x16)
            except Exception as exc:  # noqa: BLE001 — one bad take must not sink the batch
                logger.warning("take scoring failed for %s", take.get("id"), exc_info=True)
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["overall"] = overall(row)
            results.append(row)
    finally:
        scorer.close()
    ranked = sorted((r for r in results if not r["error"]), key=lambda r: -r["overall"])
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    return results
