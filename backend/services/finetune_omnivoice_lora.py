"""OmniVoice LoRA fine-tuning worker (review item 6). Run as a subprocess:

    python finetune_omnivoice_lora.py <config.json>

Trains rank-r LoRA adapters on the Qwen3 backbone of OmniVoice with the
model's own masked-token loss (``OmniVoice.forward(labels=...)``) and the
upstream ``OmniVoiceSampleProcessor`` (prompt/mask sampling), one utterance
per step with a dense bidirectional mask — so no ``flex_attention`` (which is
what the upstream packed trainer needs and Windows lacks). 8 GB-safe: the
backbone is frozen in bf16/fp16, gradient checkpointing is on, the audio
tokenizer leaves the GPU once the clips are tokenized.

Progress lines on stdout are JSON: {"event": "...", ...}.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path


def emit(event: str, **kw) -> None:
    print(json.dumps({"event": event, **kw}, ensure_ascii=False), flush=True)


def load_clips(dataset_dir: Path, max_clips: int) -> list[tuple[Path, str]]:
    rows = []
    meta = dataset_dir / "metadata.csv"
    for line in meta.read_text(encoding="utf-8").splitlines():
        rel, _, text = line.partition("|")
        if rel and text.strip() and (dataset_dir / rel).is_file():
            rows.append((dataset_dir / rel, text.strip()))
    return rows[:max_clips]


def main(cfg_path: str) -> int:
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.get("seed", 42))

    import torch
    import torchaudio.functional as AF
    import soundfile as sf
    from peft import LoraConfig, get_peft_model

    from omnivoice.data.processor import OmniVoiceSampleProcessor
    from omnivoice.models.omnivoice import OmniVoice, _resolve_language

    torch.manual_seed(cfg.get("seed", 42))
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if device.startswith("cuda") else torch.float32)

    clips = load_clips(Path(cfg["dataset_dir"]), int(cfg.get("max_clips", 2000)))
    if len(clips) < int(cfg.get("min_clips", 3)):
        emit("error", message=f"Need at least {cfg.get('min_clips', 3)} transcribed clips; found {len(clips)}.")
        return 2
    emit("stage", stage="load", clips=len(clips), device=device, dtype=str(dtype))
    model = OmniVoice.from_pretrained(cfg["checkpoint"], device_map=device, dtype=dtype, load_asr=False)
    sr = int(model.sampling_rate)
    hop = int(model.audio_tokenizer.config.hop_length)
    lang_id = _resolve_language(cfg.get("language")) if cfg.get("language") else None

    # 1. tokenize clips once (cached across runs of the same dataset)
    cache = out / "tokens"
    cache.mkdir(exist_ok=True)
    samples = []
    with torch.inference_mode():
        for i, (path, text) in enumerate(clips):
            cached = cache / f"{path.stem}.pt"
            if cached.is_file():
                tokens = torch.load(cached)
            else:
                wav, wsr = sf.read(str(path), dtype="float32", always_2d=False)
                if wav.ndim == 2:
                    wav = wav.mean(axis=1)
                x = torch.from_numpy(wav)[None]
                if wsr != sr:
                    x = AF.resample(x, wsr, sr)
                x = x[:, : x.shape[-1] - (x.shape[-1] % hop)]
                tokens = model.audio_tokenizer.encode(
                    x.unsqueeze(0).to(model.audio_tokenizer.device)).audio_codes.squeeze(0).cpu()
                torch.save(tokens, cached)
            samples.append({"audio_tokens": tokens,
                            "label": {"text": text, "language_id": lang_id or "None"}})
            if i % 10 == 0:
                emit("tokenize", done=i + 1, total=len(clips))
    model.audio_tokenizer.to("cpu")
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # 2. LoRA on the backbone
    lcfg = LoraConfig(r=int(cfg.get("rank", 16)), lora_alpha=int(cfg.get("alpha", 32)),
                      lora_dropout=float(cfg.get("dropout", 0.05)),
                      target_modules=list(cfg.get("target_modules",
                                                  ["q_proj", "k_proj", "v_proj", "o_proj"])))
    for p in model.parameters():
        p.requires_grad_(False)
    model.llm = get_peft_model(model.llm, lcfg)
    for name, p in model.llm.named_parameters():
        if "lora_" in name:
            p.data = p.data.float()  # fp32 master weights for the adapter only
            p.requires_grad_(True)
    if cfg.get("gradient_checkpointing", True):
        model.llm.base_model.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.llm.base_model.model.config.use_cache = False
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    emit("stage", stage="train", trainable_params=n_train)

    c = model.config
    proc = OmniVoiceSampleProcessor(
        text_tokenizer=model.text_tokenizer, num_channels=c.num_audio_codebook,
        audio_mask_id=c.audio_mask_id, prompt_ratio_range=(0.0, 0.3), mask_ratio_range=(0.0, 1.0),
        drop_cond_ratio=0.1, language_ratio=0.8, use_pinyin_ratio=0.0, instruct_ratio=0.0,
        only_instruct_ratio=0.0)
    steps = int(cfg.get("steps", 600))
    accum = int(cfg.get("grad_accum", 4))
    lr = float(cfg.get("learning_rate", 1e-4))
    warm = max(1, int(steps * 0.05))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    max_tokens = int(cfg.get("max_tokens", 1400))
    scaler_needed = dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda") if scaler_needed else None
    t0, ema = time.time(), None
    stop_file = out / "STOP"
    for step in range(steps):
        if stop_file.exists():
            emit("cancelled", step=step)
            break
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accum):
            s = proc(random.choice(samples))
            ids = s["input_ids"][:, :max_tokens]
            labels = s["labels"][:, :max_tokens]
            amask = s["audio_mask"][:max_tokens]
            L = ids.shape[-1]
            attn = torch.ones(1, 1, L, L, dtype=torch.bool, device=device)
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                                dtype=dtype, enabled=device.startswith("cuda")):
                res = model(input_ids=ids[None].to(device), audio_mask=amask[None].to(device),
                            labels=labels[None].to(device), attention_mask=attn)
            loss = res.loss / accum
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total += float(loss.detach())
        if scaler:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        sched.step()
        ema = total if ema is None else 0.95 * ema + 0.05 * total
        if step % 5 == 0 or step == steps - 1:
            mem = torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else 0
            emit("step", step=step + 1, steps=steps, loss=round(total, 4), loss_ema=round(ema, 4),
                 lr=sched.get_last_lr()[0], peak_vram_mb=round(mem), elapsed_s=round(time.time() - t0))
        save_every = int(cfg.get("save_every", 0))
        if save_every and (step + 1) % save_every == 0:
            model.llm.save_pretrained(str(out / f"checkpoint-{step + 1}"))

    adapter = out / "adapter"
    model.llm.save_pretrained(str(adapter))
    (adapter / "voicestudio.json").write_text(json.dumps({
        "recipe": "omnivoice-lora", "base_checkpoint": cfg["checkpoint"],
        "dataset_dir": cfg["dataset_dir"], "clips": len(samples), "steps": steps,
        "rank": lcfg.r, "alpha": lcfg.lora_alpha, "language": cfg.get("language"),
        "final_loss_ema": ema, "created_at": time.time(),
    }, indent=2), encoding="utf-8")
    emit("done", adapter=str(adapter), loss_ema=ema)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
