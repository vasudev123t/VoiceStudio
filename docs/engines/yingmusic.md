# YingMusic-Singer-Plus (Audio Edit lyric edit)

YingMusic re-sings a short vocal window with new lyrics, keeping the melody
of the original vocal.

- Install: Audio Edit → Song → *Install YingMusic*, or `POST /engines/sidecar/yingmusic/install`.
  Source `ASLP-lab/YingMusic-Singer-Plus@baa409c2e7e5e775f09b4e92a220808f4827d2cc`,
  Python 3.10, the inference requirements in `backend/engines/yingmusic/requirements.txt`
  with torch 2.7.1.
- System dependency: English and other phoneme lyrics go through `phonemizer`, which
  needs `espeak-ng` on PATH (Windows: the espeak-ng MSI; macOS: `brew install espeak-ng`;
  Debian/Ubuntu: `sudo apt install espeak-ng`).
- Weights: `ASLP-lab/YingMusic-Singer-Plus@9b3f444f2bccd77fbc03e32eb7c334fc96040b8d`,
  downloaded by the installer into `<checkout>/checkpoints`.
- Licence: the card is CC-BY-4.0, but the bundled Stable Audio VAE is under the
  **Stability AI Community License**, so the engine is gated as non-commercial
  and needs a one-time acknowledgement.
