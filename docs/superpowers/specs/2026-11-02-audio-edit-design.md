# Audio Edit: same-voice speech and song inpainting

Status: approved design · 2026-11-02 · spec only, no implementation in this change

## Overview

Audio Edit lets a user fix words inside an existing recording without
re-recording it. The user uploads a speech clip or a song, VoiceStudio
transcribes it with word timestamps, the user selects one or more word spans
and types replacement text, and VoiceStudio regenerates only those spans in
the same voice. The regenerated audio is optionally passed through an RVC
voice model for timbre match, then crossfaded back into the original (speech)
or remixed over the original instrumental (song).

The feature uses **Approach A: one unified session pipeline**. Speech and Song
are two modes of the same session object, the same API and the same stages
(upload → ASR → spans → regenerate → optional RVC → crossfade/remix). Only the
regenerate stage differs by mode and engine.

It ships as a new desktop sidebar page, `/audio-edit`, with three tabs:
**Speech**, **Song** and **History**.

## Goals

1. **Speech inpainting** with F5-TTS `speech_edit`: replace, insert or delete
   words in a speech clip in the speaker's own voice.
2. **Song inpainting** with two selectable engine paths, both shipped:
   - **ACE-Step Cover/Repaint + RVC**: repaint a time range of the song
     with new lyrics, then convert the vocal to the target voice with RVC.
   - **YingMusic / Vevo2 + RVC**: lyric-edit the separated vocal stem
     (melody kept), then RVC for timbre, then remix over the instrumental.
3. **One pipeline** for both modes: a single session model, one router, one
   service, one set of stage endpoints.
4. **Editable spans** chosen from ASR word timestamps, with manual nudging of
   span start/end, and several spans per session.
5. **Seamless joins**: equal-power crossfade at every span boundary for
   speech; stem-level remix with loudness match for song.
6. **Provenance**: every output passes through `mark_synthetic`.
7. **Licence honesty**: non-commercial engines are gated behind an explicit
   personal / non-commercial acknowledgement and recorded with
   `commercial_use: false`.
8. **History**: every session and every render is kept on disk and listed in
   the History tab, re-openable and deletable.

## Non-goals

- Editing other people's voices without consent. The consent/licence gate
  is a click-through acknowledgement, not identity verification.
- Real-time or streaming editing. Every regenerate is a batch job.
- A multitrack DAW. There is no free waveform editing beyond span selection,
  nudging and crossfade length.
- Training new RVC models. Audio Edit only uses RVC models already registered
  through the existing RVC registry; training stays in Voice Lab / RVC.
- Mobile and web. The feature is desktop-only (see Out of scope).

## User flow

The same six stages for both modes. Each stage is a step in the tab and maps
to one API call.

1. **Upload.** The user drops or picks a file (wav, flac, mp3, m4a, ogg;
   max 10 minutes speech, 8 minutes song). The backend normalises it with
   ffmpeg to 44.1 kHz mono (speech) or 44.1 kHz stereo (song), stores it as
   `source.wav` in a new session folder and returns a `session_id`.
   - Song mode additionally runs Demucs (`htdemucs`, two stems) to produce
     `vocals.wav` and `instrumental.wav`. ASR runs on `vocals.wav`.
2. **ASR word timestamps.** The backend transcribes with the existing ASR
   backend (default faster-whisper with `word_timestamps=True`; any installed
   ASR engine that returns word timing is accepted). Result: `words[]` with
   `text`, `start`, `end`, `confidence`, saved as `transcript.json`.
3. **Editable spans.** The UI renders the transcript as clickable words over
   a waveform. The user selects a contiguous word range to create a span and
   types replacement text. Empty replacement = delete; a zero-length span
   between two words = insert. Span edges snap to word boundaries and can be
   nudged in 10 ms steps. Spans cannot overlap; the UI merges or refuses
   overlapping selections. Spans are saved to the session on every change.
4. **Regenerate.** The user picks the engine (Speech: F5-TTS; Song:
   ACE-Step or YingMusic/Vevo2) and clicks Regenerate. The backend renders
   every span as a background job, reporting progress per span. Each span
   render is stored as `renders/<render_id>/span-<n>.wav`.
5. **Optional RVC.** If the user selected an RVC model, each regenerated
   span (speech) or the regenerated vocal (song) is converted with the
   existing RVC service. Off by default for Speech (F5-TTS already clones
   the voice), on by default for Song.
6. **Crossfade / remix.**
   - Speech: each span is spliced into `source.wav` with an equal-power
     crossfade (default 30 ms, range 5–200 ms). Only the regenerated span is
     gain-adjusted, to match the surrounding 2 s of source (±1 LU); source
     samples outside the crossfade windows are never modified.
   - Engines that run at other sample rates (F5-TTS: 24 kHz) are resampled
     to 44.1 kHz before RVC and splicing.
   - Song: the edited vocal stem is rebuilt with crossfades, then mixed over
     `instrumental.wav` with loudness matched to the original vocal stem.
     ACE-Step renders that already contain accompaniment are separated again
     with Demucs so only their vocal is used, which keeps the original
     instrumental bit-exact outside edited spans.
   - The final file goes through `mark_synthetic(..., context="audio_edit")`
     and is saved as `renders/<render_id>/output.wav`.

The user can A/B the output against the source, adjust spans and regenerate
(creating a new render), export the output, or open the session later from
History.

## UI

### Navigation

- New sidebar entry `/audio-edit`, label key `nav.audio_edit`, placed
  directly after `/voice-lab` in the workspace menu, icon `AudioWaveformIcon`
  from lucide-react.
- Route registered in `electron/src/renderer/src/routes/index.ts` with
  `lazyRouteComponent(() => import('@/features/audio-edit/audio-edit-page'), 'AudioEditPage')`,
  copying the `/voice-lab` entry.
- Added to the path lists in `components/app-shell/workspace-menu.tsx`
  (route union, item table, accent map) and
  `components/app-shell/use-workspace-sidebar.ts`.

### Page

`AudioEditPage` copies the Voice Lab tab pattern: a `TABS` constant
(`['speech', 'song', 'history']`), shadcn `Tabs` / `TabsList` /
`TabsTrigger` / `TabsContent`, the tab held in component state and labelled
via `t('audioEdit.tabs.<tab>')`.

### Speech tab

- Upload drop zone → waveform with transcript words overlaid.
- Span list: each row shows original words, replacement text input,
  start/end nudgers, delete button.
- Engine: F5-TTS (only option). If the licence acknowledgement has not been
  given, the Regenerate button is replaced by the licence gate card (see
  Engines & licences).
- Optional RVC model picker (existing RVC registry list, "None" default).
- Crossfade slider (5–200 ms, default 30 ms).
- Regenerate button with per-span progress, then A/B player
  (Original / Edited) and Export.

### Song tab

- Same layout as Speech, plus stem preview (vocals / instrumental) after
  upload.
- Engine radio: **ACE-Step Cover/Repaint** or **YingMusic / Vevo2**.
  Under YingMusic / Vevo2 a sub-select picks YingMusic-Singer-Plus or Vevo2.
- ACE-Step options: repaint strength (0–1, default 0.7), extra seconds of
  context padding (default 2 s each side).
- RVC model picker, on by default when at least one RVC model exists;
  pitch shift (−12…+12 semitones, default 0).
- Remix: vocal gain (−12…+6 dB, default 0), crossfade slider.

### History tab

- Table of sessions, newest first: name (source file name), mode, engine of
  latest render, number of renders, created time, total size.
- Row actions: Open (loads session into its tab), Play latest output,
  Export, Delete session (with confirm).

### Strings and states

All strings live under `audioEdit.*` and `nav.audio_edit` in all 21
`electron/src/renderer/src/i18n/locales/*.json` catalogs with real
translations. Every tab has explicit empty, loading, running, error and
engine-not-installed states; engine-not-installed links to the Models page
for that engine.

## Architecture

### Renderer (Electron)

```
electron/src/renderer/src/features/audio-edit/
  audio-edit-page.tsx        # tabs shell (Voice Lab pattern)
  speech-tab.tsx
  song-tab.tsx
  history-tab.tsx
  transcript-spans.tsx       # word grid + span selection/nudging
  span-list.tsx
  licence-gate.tsx           # NC acknowledgement card
  use-audio-edit-session.ts  # session state, polling, actions
  audio-edit.test.tsx
electron/src/renderer/src/lib/api/audio-edit.ts       # typed client
electron/src/renderer/src/lib/api/audio-edit.test.ts
```

`lib/api/audio-edit.ts` follows `lib/api/voice-lab.ts`: one function per
endpoint, types generated from / checked against the OpenAPI schema.

### Backend

```
backend/api/routers/audio_edit.py   # APIRouter(prefix="/audio-edit", tags=["audio-edit"])
backend/services/audio_edit.py      # session store + pipeline stages
backend/engines/f5tts/main.py       # F5-TTS in its own process
```

- `audio_edit.py` router is registered in `backend/main.py` next to
  `voice_lab` in the router import/include list.
- `services/audio_edit.py` owns: session folder layout, `session.json`
  read/write, stage functions (`ingest`, `separate`, `transcribe`,
  `save_spans`, `render`, `splice`, `remix`), and job submission through the
  existing job queue (`backend/core/job_queue.py`).
- **F5-TTS runs as a separate process** under `backend/engines/f5tts/`,
  following the existing subprocess engine pattern (`*_subprocess/main.py`),
  with its own Python environment so its torch/transformers pins never touch
  the main backend. The service talks to it over the same local IPC the other
  subprocess engines use and can kill it on cancel.
- **ACE-Step** is reached over HTTP at `http://127.0.0.1:8001` (the ACE-Step
  API server, started and stopped by VoiceStudio's engine manager like other
  sidecars). The service calls its cover/repaint endpoint with the source
  audio, the time range (span ± padding), the new lyrics and the strength.
- **YingMusic-Singer-Plus and Vevo2** run as separate engine processes in
  the same way as F5-TTS (`backend/engines/yingmusic/`,
  `backend/engines/vevo2/`), each with an `edit(vocal_wav, raw_text,
  target_text) -> wav` entry point.
- **Demucs and RVC are reused**, not reimplemented: Demucs `htdemucs` via
  the existing separation helper used by dubbing/dataset builder
  (`services/dub_background.py` / `services/dataset_builder.py`), RVC via
  `services/rvc.py` and models from `services/rvc_registry.py`.
- **Provenance**: the final mix and every exported file pass through
  `services/watermark.py::mark_synthetic` (async variant in job code).
- **GPU**: before loading F5-TTS, YingMusic, Vevo2 or asking ACE-Step to
  render, the service calls `services/gpu_memory.py::release_gpu_holders(reason="audio_edit")`.
- **Desktop-only**: the router is mounted only in desktop mode, using the
  same guard as other desktop-only routers; it is not added to
  `mobile/contract/` and not exposed through remote sharing.

## API

All routes under `/audio-edit`. JSON unless noted. Long stages return a
`job_id` and are polled through the existing jobs API.

| Method | Path | Body / query | Returns |
|---|---|---|---|
| POST | `/audio-edit/sessions` | multipart: `file`, `mode` (`speech`\|`song`), `name?` | `Session` (stage `uploaded`), `job_id` for normalise + separation |
| GET | `/audio-edit/sessions` | `mode?`, `limit` (default 50), `offset` | `{items: SessionSummary[], total}` |
| GET | `/audio-edit/sessions/{session_id}` | – | `Session` |
| DELETE | `/audio-edit/sessions/{session_id}` | – | `204` |
| POST | `/audio-edit/sessions/{session_id}/transcribe` | `{engine?, language?}` | `{job_id}`; result `Transcript` |
| PUT | `/audio-edit/sessions/{session_id}/spans` | `{spans: Span[]}` | `Session` (400 on overlap/out of range) |
| POST | `/audio-edit/sessions/{session_id}/render` | `RenderRequest` | `{job_id, render_id}` |
| GET | `/audio-edit/sessions/{session_id}/renders/{render_id}` | – | `Render` |
| GET | `/audio-edit/sessions/{session_id}/audio/{kind}` | `kind` = `source`\|`vocals`\|`instrumental`; or `/renders/{render_id}/audio` | `audio/wav` stream |
| POST | `/audio-edit/licence/ack` | `{engine, accepted: true}` | `{engine, accepted_at}` |
| GET | `/audio-edit/engines` | – | `EngineStatus[]` (installed, running, licence, ack state) |

Schemas:

- `Span { id, start_s, end_s, original_text, replacement_text }`
  (`start_s == end_s` = insert; empty `replacement_text` = delete).
- `RenderRequest { engine: "f5tts"|"ace_step"|"yingmusic"|"vevo2",
  rvc_model_id?: string, rvc_pitch?: int, crossfade_ms: int = 30,
  ace_step?: {strength: float = 0.7, padding_s: float = 2.0},
  vocal_gain_db?: float = 0, seed?: int }`.
- `Render { id, engine, status: queued|running|done|failed|cancelled,
  progress, spans: [{span_id, status, path}], output_path?, error?,
  created_at, commercial_use: bool }`.
- `Session { id, mode, name, stage, duration_s, created_at, transcript?,
  spans, renders[] }`.

Cancellation uses the existing job cancel endpoint; cancelling kills the
engine subprocess call in flight and marks the render `cancelled`.

After the router lands, regenerate the committed contracts:
`scripts/export_openapi.py` (OpenAPI snapshot used by the renderer/fixtures)
and `scripts/dump_api_routes.py` → `tests/fixtures/api_routes.txt`.

## Engines & licences

| Engine | Use | Code licence | Weights licence | `commercial_use` | Gate |
|---|---|---|---|---|---|
| F5-TTS `speech_edit` | Speech regenerate | MIT | CC-BY-NC-4.0 | `false` | Personal / non-commercial ack required |
| ACE-Step (Cover/Repaint) | Song regenerate | Apache-2.0 | Apache-2.0 | `true` | None |
| YingMusic-Singer-Plus | Song lyric edit | CC-BY-4.0 | CC-BY-4.0, VAE under Stability AI Community License | `false` (conservative: Stability licence revenue/registration conditions) | Personal / non-commercial ack required |
| Vevo2 (Amphion) | Song lyric edit | MIT (Amphion) | CC-BY-NC-ND-4.0 | `false` | Personal / non-commercial ack required |
| HTDemucs | Stem separation | MIT | MIT (existing record) | `true` | None |
| RVC models | Timbre | existing | per user model | per existing registry record | Existing RVC rules |

Rules:

- Each new engine gets a record in `backend/config/model_licenses.json`
  and `docs/licensing/model-inventory.md`, and an engine page under
  `docs/engines/`.
- The licence gate is the same pattern as Voice Lab's fine-tuning gate:
  the user must tick "I will use this output for personal, non-commercial
  purposes only" once per engine; the ack is stored with a timestamp in the
  user's settings and can be revoked in Settings.
- The render endpoint returns `403` with code `licence_ack_required` if
  the engine needs an ack that has not been given; the UI shows the gate.
- A render's `commercial_use` is the AND of every engine used in it
  (regenerate engine, RVC model, Demucs). It is stored in the render
  metadata, shown in History as a "Non-commercial" badge, and written into
  the exported file's metadata alongside the `mark_synthetic` watermark.

## Data & GPU

### Data layout

```
VoiceStudioData/data/audio_edit/
  <session_id>/
    session.json        # Session, spans, render index
    source.wav          # normalised upload
    vocals.wav          # song only
    instrumental.wav    # song only
    transcript.json
    renders/
      <render_id>/
        render.json     # RenderRequest, status, engines, commercial_use
        span-<n>.wav    # raw regenerated spans
        span-<n>.rvc.wav
        output.wav      # final, watermarked
```

- `session_id` and `render_id` are UUID4 hex; paths are always built from
  ids, never from user file names, and resolved under the audio_edit root
  (path traversal rejected).
- The original upload file is not kept after normalisation.
- Deleting a session removes its folder; nothing is kept elsewhere.
- No size quota in v1; History shows per-session size so the user can prune.

### GPU

- Only one Audio Edit render runs at a time (job queue concurrency 1 for
  the `audio_edit` job type); further renders queue.
- Before each GPU stage the service calls
  `release_gpu_holders(reason="audio_edit")` so TTS/ASR models held by other
  features are freed.
- Stage order inside a render keeps one big model resident at a time:
  regenerate engine → unload/stop → RVC → unload → splice/remix (CPU).
- ACE-Step's server is stopped after the render when it was started by
  Audio Edit, so its VRAM is returned.
- CPU fallback: F5-TTS and RVC run on CPU when no CUDA/MPS device is
  available, with a "slow on CPU" notice; ACE-Step, YingMusic and Vevo2
  require a GPU and are shown as unavailable without one.

## Error handling

| Case | Behaviour |
|---|---|
| Unsupported / corrupt upload | `400 invalid_audio`; nothing stored |
| Over duration limit | `400 too_long` with the limit in the message |
| ffmpeg or Demucs failure | Session stage `failed`, error text shown, Retry button reruns the stage |
| ASR returns no word timings | `422 no_word_timestamps`, UI suggests another ASR engine |
| Overlapping / out-of-range spans | `400 invalid_spans` listing the offending span ids |
| Engine not installed | `409 engine_not_installed`; UI links to Models |
| Licence ack missing | `403 licence_ack_required`; UI shows gate |
| ACE-Step server not reachable on :8001 | Start it once; if still down after 60 s, render `failed` with `ace_step_unavailable` |
| Engine subprocess crash / OOM | Render `failed`, partial span files kept for debugging, `release_gpu_holders` called, user can retry |
| One span fails | Whole render fails (no partial splice), failed span id shown |
| Cancel | Engine call killed, render `cancelled`, completed span files kept |
| Backend restart mid-render | Render marked `failed` with `interrupted` on startup |

All errors use the existing API error envelope and are logged through the
standard backend logger; no audio or transcript text is logged.

## Testing

Backend (`tests/`):

- `tests/test_audio_edit_service.py`: session create/read/delete, path
  traversal rejection, span validation (overlap, insert, delete, bounds),
  crossfade splice on synthetic sine fixtures (length and boundary
  continuity), loudness match within ±1 LU, `commercial_use` AND logic.
- `tests/test_audio_edit_router.py`: every endpoint with engines faked; 400 /
  403 / 409 / 422 paths; desktop-only mounting.
- `tests/test_audio_edit_engines.py`: F5-TTS, YingMusic, Vevo2 subprocess
  clients and ACE-Step HTTP client against fakes (no model downloads), cancel
  kills the subprocess, `release_gpu_holders` called before each GPU stage.
- Watermark route coverage: `/audio-edit` render output added to
  `tests/test_watermark_route_coverage.py` so `mark_synthetic` is enforced.
- `tests/fixtures/api_routes.txt` and the OpenAPI snapshot regenerated;
  existing route/OpenAPI drift tests must pass.
- Licence records: existing model-licence tests cover the new entries.
- `tests/test_locale_parity.py` passes with the new keys in all 21 locales.

Renderer (Electron):

- `features/audio-edit/audio-edit.test.tsx`: tab switching, span creation
  and nudging, licence gate blocks Regenerate, engine-not-installed state,
  History list/open/delete.
- `lib/api/audio-edit.test.ts`: request shapes and error mapping.
- `use-workspace-sidebar.test.tsx` and `workspace-menu.test.tsx` updated
  for the new entry.
- `locale:check` passes.

Manual acceptance (MSI, Windows, NVIDIA GPU): one speech clip with a
replace, an insert and a delete; one song through each song path with RVC.

## Docs and bookkeeping

- `docs/STRUCTURE.md`: new feature folder, router, service and engine dirs,
  updated counts.
- `CHANGELOG.md` → Unreleased: one quiet one-liner for Audio Edit.
- `docs/engines/f5tts.md`, `docs/engines/ace-step.md`,
  `docs/engines/yingmusic.md`, `docs/engines/vevo2.md`.
- `docs/licensing/model-inventory.md` and `model_licenses.json` entries.

## Out of scope

- Mobile app and `mobile/contract/` (desktop-only).
- Remote/network-shared access to `/audio-edit`.
- Video files and dubbing integration (could feed Dubbing later).
- Batch editing of many files; editing over the public API for third parties.
- Training or fine-tuning F5-TTS, ACE-Step, YingMusic, Vevo2 or RVC.
- Commercial licensing of the non-commercial engines.
- Automatic word-level alignment correction beyond manual nudging.
- Multi-speaker speech edits where spans belong to different speakers
  (v1 edits assume one speaker per session).

## Success criteria

1. `/audio-edit` appears in the sidebar after Voice Lab on macOS, Windows
   and Linux, with Speech, Song and History tabs, in all 21 locales.
2. Speech: on a clean 30 s English clip, replacing a 3-word span with F5-TTS
   produces an output where the unedited regions are sample-identical to
   the source outside the crossfade windows, and the edit is completed in
   under 60 s on an RTX-class GPU.
3. Song: both ACE-Step + RVC and YingMusic/Vevo2 + RVC produce an output
   where the instrumental outside edited spans is unchanged and the edited
   lyric is audible in the target voice.
4. Non-commercial engines cannot render until acknowledged; their renders
   are stored and shown with `commercial_use: false`.
5. Every output file carries the `mark_synthetic` watermark.
6. History lists every session and render, survives an app restart, and
   delete removes the session folder.
7. Backend and renderer tests above pass; route, OpenAPI, locale-parity and
   changelog-style checks are green.
