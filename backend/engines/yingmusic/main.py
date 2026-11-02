"""YingMusic-Singer-Plus lyric-edit sidecar for Audio Edit (own venv).

Re-sings one vocal window with new lyrics. The window is both the timbre
reference and the melody, so the regenerated clip keeps the singer, the
tune and the window length; the parent splices it back over the span.
Upstream's config.json uses checkout-relative paths, so the sidecar runs
from the checkout. Weights come only from <checkout>/checkpoints.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _edit_protocol import serve  # noqa: E402

DEFAULT_SEED = 666


def load():
    os.environ["HF_HUB_OFFLINE"] = "1"
    checkout = Path(os.environ["OMNIVOICE_YINGMUSIC_DIR"])
    os.chdir(checkout)
    sys.path.insert(0, str(checkout))
    import torch
    from src.YingMusicSinger.infer.YingMusicSinger import YingMusicSinger

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        raise RuntimeError("YingMusic-Singer-Plus needs a CUDA or Apple GPU.")
    model = YingMusicSinger.from_pretrained(str(checkout / "checkpoints"))
    return model.to(device).eval()


def edit(model, msg: dict, progress) -> dict:
    import soundfile as sf
    import torch

    progress(0.05)
    seed = msg.get("seed")
    with torch.inference_mode():
        audio, sr = model(
            ref_audio_path=msg["audio_path"],
            melody_audio_path=msg["audio_path"],
            ref_text=msg["raw_text"],
            target_text=msg["target_text"],
            lrc_align_mode="sentence_level",
            sil_len_to_end=0.5,
            t_shift=0.5,
            nfe_step=32,
            cfg_strength=3.0,
            seed=DEFAULT_SEED if seed is None else int(seed),
        )
    progress(0.95)
    sf.write(msg["out_path"], audio.detach().float().cpu().numpy().T, int(sr), subtype="FLOAT")
    return {"path": msg["out_path"], "sample_rate": int(sr)}


if __name__ == "__main__":
    _frame_fd = os.dup(1)
    os.dup2(2, 1)
    sys.exit(serve("yingmusic", load, edit, os.fdopen(_frame_fd, "wb")))
