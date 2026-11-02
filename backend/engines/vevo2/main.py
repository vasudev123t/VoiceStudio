"""Vevo2 lyric-edit sidecar for Audio Edit (own venv).

Uses Vevo2's AR + flow-matching editing path: the window is the prosody,
style and timbre reference, and the target lyrics replace the original
ones. Runs from the Amphion checkout so its relative config paths
resolve. Whisper "medium" is loaded from the file the installer fetched;
nothing is downloaded at render time.
"""
from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _edit_protocol import serve  # noqa: E402

OUTPUT_SR = 24000


def load():
    os.environ["HF_HUB_OFFLINE"] = "1"
    checkout = Path(os.environ["OMNIVOICE_VEVO2_DIR"])
    os.chdir(checkout)
    sys.path.insert(0, str(checkout))
    import torch
    import whisper

    whisper_dir = checkout / "checkpoints" / "whisper"
    if not (whisper_dir / "medium.pt").is_file():
        raise RuntimeError("Vevo2's Whisper checkpoint is missing. Re-run the Vevo2 install.")
    whisper.load_model = functools.partial(whisper.load_model, download_root=str(whisper_dir))

    from models.svc.vevo2.vevo2_utils import Vevo2InferencePipeline

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        raise RuntimeError("Vevo2 needs a CUDA or Apple GPU.")
    d = checkout / "checkpoints" / "Vevo2"
    return Vevo2InferencePipeline(
        prosody_tokenizer_ckpt_path=str(d / "tokenizer" / "prosody_fvq512_6.25hz"),
        content_style_tokenizer_ckpt_path=str(d / "tokenizer" / "contentstyle_fvq16384_12.5hz"),
        ar_cfg_path=str(d / "contentstyle_modeling" / "posttrained" / "amphion_config.json"),
        ar_ckpt_path=str(d / "contentstyle_modeling" / "posttrained"),
        fmt_cfg_path=str(d / "acoustic_modeling" / "fm_emilia101k_singnet7k_repa" / "config.json"),
        fmt_ckpt_path=str(d / "acoustic_modeling" / "fm_emilia101k_singnet7k_repa"),
        vocoder_cfg_path=str(d / "vocoder" / "config.json"),
        vocoder_ckpt_path=str(d / "vocoder"),
        device=device,
    )


def edit(pipeline, msg: dict, progress) -> dict:
    import soundfile as sf
    import torch

    seed = msg.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))
    progress(0.05)
    with torch.inference_mode():
        wave = pipeline.inference_ar_and_fm(
            target_text=msg["target_text"],
            prosody_wav_path=msg["audio_path"],
            style_ref_wav_path=msg["audio_path"],
            style_ref_wav_text=msg["raw_text"],
            timbre_ref_wav_path=msg["audio_path"],
            use_prosody_code=True,
            flow_matching_steps=32,
        )
    progress(0.95)
    sf.write(msg["out_path"], wave.squeeze().detach().float().cpu().numpy(), OUTPUT_SR, subtype="FLOAT")
    return {"path": msg["out_path"], "sample_rate": OUTPUT_SR}


if __name__ == "__main__":
    _frame_fd = os.dup(1)
    os.dup2(2, 1)
    sys.exit(serve("vevo2", load, edit, os.fdopen(_frame_fd, "wb")))
