# ACE-Step 1.5 (Audio Edit song repaint)

ACE-Step repaints a time range of a full song mix with new lyrics. Audio Edit
then splits the repainted range back into vocals and splices only the edited
words into your original vocals.

- Install: Audio Edit → Song → *Install ACE-Step*, or `POST /engines/sidecar/ace-step/install`.
  This makes a Python 3.11 venv from `ace-step/ACE-Step-1.5@ca1e85fe9430179831e6bc6be790c332190a3866`.
- Runtime: Audio Edit starts `acestep-api` from that venv on `127.0.0.1` at a free
  port and stops it after the render. An existing server can be used instead by
  setting `OMNIVOICE_ACESTEP_URL`; only loopback URLs are accepted.
- Weights: `ACE-Step/Ace-Step1.5@19671f406d603126926c1b7e2adc169acbcade22`, downloaded
  by the installer into `<checkout>/checkpoints`.
  Only the turbo DiT, the VAE and the Qwen3 text encoder are downloaded. The 5 Hz LM is
  skipped because repaint never uses it, and the server starts with `ACESTEP_INIT_LLM=false`.
- Licence: the model card says **MIT** (the design spec said Apache-2.0). The
  record stays unreviewed, so renders are badged non-commercial until a human
  review clears it in `backend/config/model_licenses.json`.
