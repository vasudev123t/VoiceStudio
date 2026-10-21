"""Applio inference worker — runs INSIDE Applio's own Python environment.

VoiceStudio never imports Applio (or its torch/faiss/librosa pins) into the
backend process. Instead ``services.rvc.ApplioBackend`` starts this script
with Applio's interpreter (``cwd`` = the Applio checkout) and talks to it with
one JSON object per line:

    -> {"op": "ping"}
    <- {"ok": true, "device": "cuda:0", "gpu": "NVIDIA GeForce RTX 3070 Laptop GPU"}
    -> {"op": "convert", "input": "...", "output": "...", "model": "...",
        "index": "...", "params": {...}}
    <- {"ok": true, "sample_rate": 48000}
    -> {"op": "unload"}            # frees the model and CUDA cache
    <- {"ok": true}

Applio's own prints go to stderr so stdout stays a clean protocol channel.
Uses only Applio's public ``rvc.infer.infer.VoiceConverter.convert_audio``.
Stdlib-only apart from Applio itself; keep it importable without VoiceStudio.
"""
import json
import os
import sys
import traceback

_PROTOCOL = sys.stdout
sys.stdout = sys.stderr  # Applio prints progress; never let it corrupt the protocol.


def _reply(payload):
    _PROTOCOL.write(json.dumps(payload) + "\n")
    _PROTOCOL.flush()


def _patch_cleanup_ffmpeg():
    """Point Applio's light cleanup at the ffmpeg VoiceStudio selected.

    Applio's ``audio_cleanup._ffmpeg()`` prefers the ``ffmpeg.exe`` in its own
    root (FFmpeg 4.3.2 on Windows), which lacks ``afftdn gs``. The backend
    probes for a capable binary and passes it as ``OMNIVOICE_RVC_FFMPEG``.
    """
    path = os.environ.get("OMNIVOICE_RVC_FFMPEG", "").strip()
    if not path or not os.path.isfile(path):
        return None
    try:
        from rvc.lib.tools import audio_cleanup
    except Exception:  # noqa: BLE001 — older Applio without the module
        return None
    if hasattr(audio_cleanup, "_ffmpeg"):
        audio_cleanup._ffmpeg = lambda: path
        return path
    return None


def _make_converter(device_override):
    root = os.getcwd()
    if root not in sys.path:
        sys.path.insert(0, root)
    _patch_cleanup_ffmpeg()
    from rvc.configs.config import inference_config
    from rvc.infer.infer import VoiceConverter

    config = inference_config()
    if device_override and device_override != "auto":
        config.device = device_override
        if not device_override.startswith("cuda"):
            config.gpu_name = None
    return VoiceConverter(config=config)


def _free(converter):
    try:
        converter.cleanup_model()
    except Exception:  # noqa: BLE001
        pass
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def main():
    device_override = os.environ.get("OMNIVOICE_RVC_WORKER_DEVICE", "auto")
    try:
        converter = _make_converter(device_override)
    except Exception as exc:  # noqa: BLE001
        _reply({"ok": False, "fatal": True, "error": f"{type(exc).__name__}: {exc}"})
        return 2
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            op = msg.get("op")
            if op == "ping":
                _reply({
                    "ok": True,
                    "device": str(converter.config.device),
                    "gpu": getattr(converter.config, "gpu_name", None),
                })
            elif op == "unload":
                _free(converter)
                _reply({"ok": True})
            elif op == "convert":
                params = msg.get("params") or {}
                kwargs = {
                    "audio_input_path": msg["input"],
                    "audio_output_path": msg["output"],
                    "model_path": msg["model"],
                    "index_path": msg.get("index") or "",
                    "pitch": int(params.get("pitch", 0)),
                    "f0_method": params.get("f0_method", "rmvpe"),
                    "index_rate": float(params.get("index_rate", 0.75)),
                    "volume_envelope": float(params.get("rms_mix_rate", 0.1)),
                    "protect": float(params.get("protect", 0.33)),
                    "f0_autotune": bool(params.get("f0_autotune", False)),
                    "embedder_model": params.get("embedder_model", "contentvec"),
                    "export_format": "WAV",
                    "sid": 0,
                }
                if params.get("clean_audio"):
                    kwargs["cleanup_mode"] = "light"
                    kwargs["cleanup_strength"] = float(params.get("clean_strength", 12))
                converter.convert_audio(**kwargs)
                if not os.path.isfile(msg["output"]):
                    raise RuntimeError("Applio finished without writing an output file.")
                _reply({"ok": True, "sample_rate": converter.tgt_sr})
            elif op == "exit":
                _free(converter)
                _reply({"ok": True})
                return 0
            else:
                _reply({"ok": False, "error": f"unknown op {op!r}"})
        except Exception as exc:  # noqa: BLE001 — report, keep serving
            traceback.print_exc()
            _reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
