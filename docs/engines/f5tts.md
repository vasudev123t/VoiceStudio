# F5-TTS (Audio Edit speech engine)

Audio Edit uses F5-TTS to regenerate the words you mark in a speech recording.
It is not a text-to-speech engine in the picker: it only runs from `/audio-edit`.

- Install: Audio Edit → Speech → *Install F5-TTS*, or `POST /engines/sidecar/f5tts/install`.
  This creates a private Python 3.11 venv with `f5-tts==1.1.22` and `torchcodec`.
- Weights: `SWivid/F5-TTS@84e5a410d9cead4de2f847e7c9369a6440bdfaca`
  (`F5TTS_v1_Base/model_1250000.safetensors`) and the vocoder
  `charactr/vocos-mel-24khz@0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21`.
  Both download on the first render, never before.
- Licence: the code is MIT, the checkpoint is **CC-BY-NC-4.0**. Audio Edit asks
  you to acknowledge this once before the first render (you can revoke it), and
  every F5-TTS render is badged non-commercial.
- Disk: about 6 GB for the venv and 1.4 GB of weights.
