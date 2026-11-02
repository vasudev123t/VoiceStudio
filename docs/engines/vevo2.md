# Vevo2 (Audio Edit lyric edit)

Vevo2 (Amphion) regenerates a vocal window with new lyrics while copying the
original's prosody, style and timbre.

- Install: Audio Edit → Song → *Install Vevo2*, or `POST /engines/sidecar/vevo2/install`.
  Source `open-mmlab/Amphion@26f6883110181f1dbfe95c70a7c7dbaf4de5f42a`, Python 3.10,
  the inference requirements in `backend/engines/vevo2/requirements.txt` with torch 2.7.1.
- Whisper: Vevo2's content tokenizer uses OpenAI Whisper *medium* (MIT). The installer
  fetches `medium.pt` once into `<checkout>/checkpoints/whisper` and checks its SHA-256;
  renders never download it.
- Weights: `RMSnow/Vevo2@2674843cbaa50aa89ee7ccaf5bb15d6ccf46c6c8`, downloaded by the installer into `<checkout>/checkpoints/Vevo2`.
- Licence: **CC-BY-NC-ND-4.0**, so it is non-commercial and needs a one-time
  acknowledgement. Weights are used unmodified.
