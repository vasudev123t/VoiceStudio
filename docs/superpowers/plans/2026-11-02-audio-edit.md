# Audio Edit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the `/audio-edit` workspace from `docs/superpowers/specs/2026-11-02-audio-edit-design.md`. The user uploads speech or a song, gets a word-timed transcript, marks word spans, types replacements, and renders an edited file. Speech uses F5-TTS. Songs use ACE-Step 1.5, YingMusic-Singer-Plus or Vevo2, with an optional RVC pass and a remix over the original backing track. Audio outside each edit, beyond its crossfade, stays byte-identical, and every render passes through `mark_synthetic`.

**Architecture:**
- **Router.** `backend/api/routers/audio_edit.py` is a thin, desktop-only FastAPI router at `/audio-edit`. It calls `backend/services/audio_edit.py`, which owns sessions on disk, a background-thread job runner, licence acknowledgements, and the speech and song render pipelines.
- **Pure DSP.** `backend/services/audio_edit_dsp.py` holds splicing, equal-power crossfades, BS.1770 loudness matching and the remix.
- **Engine sidecars.** F5-TTS, YingMusic and Vevo2 each run as a `SubprocessBackend` sidecar in a private venv. Their venvs are installed by the existing sidecar installer and wrapped by `backend/services/audio_edit_engines.py`. They speak one new `edit` op through a shared stdlib protocol module.
- **ACE-Step.** ACE-Step 1.5 runs its own `acestep-api` server on loopback, driven by `backend/services/acestep_client.py`.
- **Frontend.** `features/audio-edit/` provides Speech, Song and History tabs and a word-span transcript editor, over `lib/api/audio-edit.ts`.

**Tech Stack:**
- Backend: Python 3.11, FastAPI, numpy, soundfile, torchaudio (resample and loudness), ffmpeg, and the existing `SubprocessBackend` / `sidecar_install` / HTDemucs / ASR / RVC services.
- Frontend: React 19, TanStack Query and Router, base-ui components, react-i18next, and Vitest through `bunx vp test`.
- Tests: pytest.

## Global Constraints

- Work on `main` in `D:\VoiceStudio`. Push only with `git push origin main` (origin is the `vasudev123t` fork). Never push to `debpalash`, never force-push, never open a PR.
- Commit messages carry no agent credit: no `Co-authored-by` agent line, no "Generated with".
- Do not bump versions. Do not add a required network call: every model download is an explicit, user-started install, and ACE-Step binds to `127.0.0.1` only.
- Every user-visible string goes into all 21 locales under `electron/src/renderer/src/i18n/locales/`: ar, de, en, es, fr, hi, id, it, ja, ko, nl, pl, pt, ru, sv, th, tr, uk, vi, zh-CN, zh-TW. `bun run locale:check` must pass.
- Use `SearchableSelect` (`@shared/components/SearchableSelect`) for every picker. Never use a native `<select>`.
- CHANGELOG entries are single quiet lines of at most 400 characters (`tests/test_changelog_style.py`).
- No JS dependency changes are planned. If one becomes necessary, regenerate `bun.lock`. The working tree already has an unrelated `bun.lock` modification: never stage it in these commits. Stage files by explicit path.
- Licence registry rules (`scripts/check_model_licenses.py`):
  - `commercial_use: true` requires `review_status: "cleared"` with a named human reviewer. No record is cleared here, so every Audio Edit render is badged non-commercial.
  - The UI computes the badge from the registry and never hard-codes it.
- Engine terms:
  - F5-TTS weights (CC-BY-NC-4.0) and Vevo2 (CC-BY-NC-ND-4.0) are non-commercial. YingMusic uses the Stable Audio VAE under the Stability AI Community License.
  - All three need a one-time, revocable in-app acknowledgement before their first render. ACE-Step (MIT) needs no acknowledgement.
- The backend runs one render at a time (`_RENDER_SLOT`), frees VRAM after every job, and kills engine sidecars on cancel.
- Backend tests use `OMNIVOICE_AUDIO_EDIT_DIR=<tmp>` and `audio_edit.RUN_INLINE = True`, so they never touch the real data dir or need a GPU. Engine and ACE-Step tests use stdlib fakes.

### Intentional deviations from the spec (decided while planning)

| Spec said | Plan does | Why |
|---|---|---|
| Jobs on `core/job_queue.py` | A thread runner in `services/audio_edit.py`, following `services/dataset_builder.py` | `core/job_queue.py` is unused in production. The dataset-builder pattern is the live one. |
| New HTDemucs licence record | Reuse the existing `dynamic_assets` record `htdemucs` and append to its `configured_at` | The record already exists. A duplicate id fails the checker. |
| ACE-Step 1.5 is Apache-2.0 | Record it as **MIT** | The model card at `ACE-Step/Ace-Step1.5@19671f406d603126926c1b7e2adc169acbcade22` says MIT. |
| "Commercial OK" badge for ACE-Step and HTDemucs | Both stay `unreviewed` / `commercial_use: false`, so the badge reads Non-commercial | The checker forbids `commercial_use: true` without a human `cleared` review. The badge flips automatically once someone records that review. |
| A zero-length span (`start_s == end_s`) is an insert | Spans always cover at least one word (`start_s < end_s`). To insert, select the word next to the gap and retype it with the new words ("fox" → "fox and the cat"). | Every engine needs a non-empty region to condition on, and one splice rule then covers replace, insert and delete. The UI explains this with `audioEdit.insert_hint`. |
| Licence acks are sticky | Stored in prefs under `audio_edit.licence_acks`, with `DELETE /audio-edit/licence/ack/{engine}` to revoke | This makes revocation explicit and testable. |
| Revoke from Settings | The "Revoke acknowledgement" button lives next to the render button on the Audio Edit page | That is where the acknowledgement is given, and it keeps Settings unchanged. |
| Render status `done` | `succeeded` | Matches the job statuses used by the other job runners and the frontend `isLive` helper. |
| Transcribe with a chosen ASR engine | Uses the active ASR backend; no per-call engine parameter | The existing ASR service has one active backend. Switching it per call would reload models. |
| `GET /engines` returns a list | Returns `{engines, separation, gpu}` | The UI also needs HTDemucs availability and GPU presence on the same screen. |
| Watermark the final file | `mark_synthetic` is applied to each synthetic insert before splicing | Marking the whole file would change untouched samples and break the byte-identity guarantee. |
| Song remix always recombines stems | At 0 dB vocal gain, the original mix is kept and only the edited regions take the residual (`original − old vocals + new vocals`) | This keeps unedited song audio byte-identical. Any other gain uses `dsp.remix`. |
| `DELETE` session returns 204 | Returns `{"ok": true}` with 200 | The shared `apiJson` client expects a JSON body. |
| Every render needs an engine and ack | Delete-only renders (all replacements empty) skip the engine install and ack checks | Deleting words synthesises nothing. |
| One render at a time | Separation, transcription and render all share that one slot (`_RENDER_SLOT`) | One GPU, no contention. |

---

## File map

**Create (backend)**

| Path | Responsibility |
|---|---|
| `backend/services/audio_edit_dsp.py` | Pure numpy/torchaudio DSP: splice, crossfades, loudness, remix, resample, duration estimate |
| `backend/services/audio_edit.py` | Sessions, spans, jobs, licence acks, speech and song render pipelines |
| `backend/services/audio_edit_engines.py` | `EditSidecar(SubprocessBackend)` clients for f5tts, yingmusic and vevo2; install status |
| `backend/services/acestep_client.py` | Loopback HTTP client and process owner for `acestep-api` |
| `backend/engines/_edit_protocol/__init__.py` | Stdlib sidecar loop for the `edit` op (ready, ping, edit, progress, audio, error) |
| `backend/engines/f5tts/__init__.py`, `backend/engines/f5tts/main.py` | F5-TTS speech-edit sidecar |
| `backend/engines/yingmusic/__init__.py`, `backend/engines/yingmusic/main.py` | YingMusic-Singer-Plus lyric-edit sidecar |
| `backend/engines/yingmusic/requirements.txt`, `backend/engines/vevo2/requirements.txt` | Pinned sidecar venv requirements |
| `backend/engines/vevo2/__init__.py`, `backend/engines/vevo2/main.py` | Vevo2 lyric-edit sidecar |
| `backend/api/routers/audio_edit.py` | `/audio-edit` REST surface (16 routes) |
| `docs/engines/f5tts.md`, `docs/engines/ace-step.md`, `docs/engines/yingmusic.md`, `docs/engines/vevo2.md` | Install and licence notes per engine |

**Create (tests)**

| Path | Covers |
|---|---|
| `tests/test_audio_edit_dsp.py` | Splice byte-identity, crossfades, loudness, remix |
| `tests/test_audio_edit_service.py` | Sessions, spans, windows, jobs, licence acks |
| `tests/test_audio_edit_render.py` | Speech and song render pipelines with fake engines, fake ACE-Step and fake RVC |
| `tests/test_audio_edit_engines.py` | Edit protocol round-trip through a fake sidecar; F5-TTS client |
| `tests/test_audio_edit_song_engines.py` | YingMusic and Vevo2 clients; sidecar `extra_files` |
| `tests/test_acestep_client.py` | ACE-Step client against a fake `http.server` |
| `tests/test_audio_edit_router.py` | All 16 routes, error envelope, desktop guard |

**Create (frontend)**, all under `electron/src/renderer/src/`

| Path | Responsibility |
|---|---|
| `lib/api/audio-edit.ts`, `lib/api/audio-edit.test.ts` | Typed client and pure span helpers |
| `features/audio-edit/audio-edit-page.tsx` | Page shell: header, FreeGpuButton, tabs |
| `features/audio-edit/edit-workspace.tsx` | Upload, stage, transcript, spans, render controls, result (shared by both tabs) |
| `features/audio-edit/speech-tab.tsx`, `features/audio-edit/song-tab.tsx` | Mode wrappers around `EditWorkspace` |
| `features/audio-edit/render-controls.tsx` | Engine and RVC pickers, crossfade, vocal gain, ACE-Step options (song-only controls) |
| `features/audio-edit/history-tab.tsx` | Session list with latest render and badge |
| `features/audio-edit/transcript-spans.tsx` | Clickable word transcript that creates spans |
| `features/audio-edit/span-list.tsx` | Span editors with 10 ms nudges |
| `features/audio-edit/licence-gate.tsx` | Acknowledgement gate and commercial badge |
| `features/audio-edit/use-audio-edit-session.ts` | Query keys, engines query, session and job polling hook |
| `features/audio-edit/audio-edit.test.tsx` | Component tests |

**Modify**

| Path | Change |
|---|---|
| `backend/config/model_licenses.json` | Add 5 model records. Append Audio Edit to `htdemucs` and the RVC dynamic asset `configured_at`. |
| `docs/licensing/model-inventory.md` | 56 → 61 configured repositories |
| `backend/services/sidecar_install.py` | Add `SidecarSpec`s `f5tts`, `yingmusic`, `vevo2`, `ace-step`; `ExtraFile` / `extra_files` support |
| `tests/test_sidecar_install.py`, `tests/test_sidecar_stdout_isolation_1428.py` | Expect the four new sidecar specs |
| `backend/main.py` | Import and register the `audio_edit` router |
| `tests/test_watermark_route_coverage.py` | Add `_engine_edit(` and `_ace_repaint(` to `_SYNTH_CALL`, and `services/audio_edit.py` to `_PRODUCERS` |
| `tests/fixtures/api_routes.txt`, `mobile/contract/openapi.json` | Regenerated |
| `electron/src/renderer/src/routes/index.ts` | `audioEditRoute` |
| `electron/src/renderer/src/components/app-shell/workspace-menu.tsx`, `workspace-menu.test.tsx` | `/audio-edit` destination after Voice Lab, and its test |
| `electron/src/renderer/src/components/app-shell/use-workspace-sidebar.ts`, `use-workspace-sidebar.test.tsx` | `/audio-edit` in `SECONDARY_ROUTES`, and its test |
| `electron/src/renderer/src/i18n/locales/*.json` (21 files) | `nav.audio_edit` and the `audioEdit.*` keys |
| `docs/STRUCTURE.md` | Routers 44 → 45; services 111 → 115 |
| `CHANGELOG.md`, `tests/test_changelog_style.py` | One Added line, allowlisted as owner-authored (no `(#N)` ref) |

---

## Task 1: Licence records and engine docs

**Files:**
- Modify: `backend/config/model_licenses.json`, `docs/licensing/model-inventory.md`
- Create: `docs/engines/f5tts.md`, `docs/engines/ace-step.md`, `docs/engines/yingmusic.md`, `docs/engines/vevo2.md`

**Interfaces:**
- Consumes: the registry schema v2, validated by `scripts/check_model_licenses.py` and `services.model_licenses.load_registry`.
- Produces:
  - Model records `SWivid/F5-TTS`, `charactr/vocos-mel-24khz`, `ACE-Step/Ace-Step1.5`, `ASLP-lab/YingMusic-Singer-Plus` and `RMSnow/Vevo2`.
  - Task 7's `render_commercial_use()` reads `commercial_use` from these records, from `htdemucs`, and from the RVC dynamic asset.

The records come first so the source scanner (which captures literal repo ids in `backend/**/*.py`) never sees an unrecorded repo in later tasks.

- [ ] **Step 1: Add the five model records**

Insert these five objects at the end of the `"models"` array in `backend/config/model_licenses.json`, after the current last record. Add a comma after that record, and keep the file's two-space indentation.

```json
    {
      "id": "SWivid/F5-TTS",
      "license": "CC-BY-NC-4.0",
      "commercial_use": false,
      "review_status": "noncommercial",
      "credit": "F5-TTS — Yushen Chen and coauthors (SWivid)",
      "source_url": "https://huggingface.co/SWivid/F5-TTS",
      "evidence_url": "https://huggingface.co/SWivid/F5-TTS/blob/84e5a410d9cead4de2f847e7c9369a6440bdfaca/README.md",
      "revision": "84e5a410d9cead4de2f847e7c9369a6440bdfaca",
      "engines": ["f5tts"],
      "configured_at": "backend/engines/f5tts/main.py; backend/services/audio_edit_engines.py",
      "notes": "Audio Edit speech engine. The f5-tts code is MIT; the F5TTS_v1_Base checkpoint is CC-BY-NC-4.0, so renders are non-commercial and need an in-app acknowledgement.",
      "runtime_revision": "84e5a410d9cead4de2f847e7c9369a6440bdfaca",
      "runtime_pin_scope": "external",
      "runtime_pin_source": "backend/engines/f5tts/main.py",
      "component_closure": "incomplete",
      "commercial_inference": "restricted",
      "commercial_outputs": "unknown",
      "documents": [],
      "artifacts": [],
      "required_components": [],
      "upstream_access": "unknown",
      "audit_observations": {
        "commercial_inference": {
          "status": "noncommercial",
          "scope": "Selected weight/component; all required components and intended use must be assessed.",
          "summary": "Pinned card licenses the checkpoints CC-BY-NC-4.0; the pip package code is MIT."
        },
        "output_terms": {
          "status": "not_specifically_granted_in_reviewed_material",
          "summary": "No output ownership grant identified; non-commercial weights cannot authorize commercial generation.",
          "ownership_or_all_rights_cleared": false
        },
        "sources": [
          {
            "url": "https://huggingface.co/SWivid/F5-TTS/blob/84e5a410d9cead4de2f847e7c9369a6440bdfaca/README.md",
            "checked_at": "2026-11-02",
            "method": "read_primary_source"
          }
        ],
        "open_questions": [
          "Emilia training-data terms and attribution duties",
          "Reference voice and input rights, consent"
        ],
        "identified_components": [
          "F5TTS_v1_Base/model_1250000.safetensors",
          "charactr/vocos-mel-24khz vocoder",
          "f5-tts==1.1.22 pip package"
        ]
      }
    },
    {
      "id": "charactr/vocos-mel-24khz",
      "license": "MIT",
      "commercial_use": false,
      "review_status": "unreviewed",
      "credit": "Vocos — Hubert Siuzdak (charactr)",
      "source_url": "https://huggingface.co/charactr/vocos-mel-24khz",
      "evidence_url": "https://huggingface.co/charactr/vocos-mel-24khz/blob/0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21/README.md",
      "revision": "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21",
      "engines": ["f5tts"],
      "configured_at": "backend/engines/f5tts/main.py",
      "notes": "Vocoder for the F5-TTS Audio Edit sidecar, pinned and loaded from a local snapshot.",
      "runtime_revision": "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21",
      "runtime_pin_scope": "external",
      "runtime_pin_source": "backend/engines/f5tts/main.py",
      "component_closure": "incomplete",
      "commercial_inference": "unknown",
      "commercial_outputs": "unknown",
      "documents": [],
      "artifacts": [],
      "required_components": [],
      "upstream_access": "unknown",
      "audit_observations": {
        "commercial_inference": {
          "status": "permissive_metadata_only",
          "scope": "Selected weight/component; all required components and intended use must be assessed.",
          "summary": "Card metadata states MIT; no separate weight terms reviewed."
        },
        "output_terms": {
          "status": "not_specifically_granted_in_reviewed_material",
          "summary": "No output terms identified.",
          "ownership_or_all_rights_cleared": false
        },
        "sources": [
          {
            "url": "https://huggingface.co/charactr/vocos-mel-24khz/blob/0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21/README.md",
            "checked_at": "2026-11-02",
            "method": "read_primary_source"
          }
        ],
        "open_questions": ["Training-data provenance"],
        "identified_components": ["Vocos mel 24 kHz vocoder weights"]
      }
    },
    {
      "id": "ACE-Step/Ace-Step1.5",
      "license": "MIT",
      "commercial_use": false,
      "review_status": "unreviewed",
      "credit": "ACE-Step 1.5 — ACE Studio and StepFun",
      "source_url": "https://huggingface.co/ACE-Step/Ace-Step1.5",
      "evidence_url": "https://huggingface.co/ACE-Step/Ace-Step1.5/blob/19671f406d603126926c1b7e2adc169acbcade22/README.md",
      "revision": "19671f406d603126926c1b7e2adc169acbcade22",
      "engines": ["ace-step"],
      "configured_at": "backend/services/sidecar_install.py; backend/services/acestep_client.py",
      "notes": "Audio Edit song repaint engine, served by the upstream acestep-api on loopback. The design spec said Apache-2.0; the pinned card says MIT. Commercial use stays false until a human review clears it.",
      "runtime_revision": "19671f406d603126926c1b7e2adc169acbcade22",
      "runtime_pin_scope": "external",
      "runtime_pin_source": "backend/services/sidecar_install.py",
      "component_closure": "incomplete",
      "commercial_inference": "unknown",
      "commercial_outputs": "unknown",
      "documents": [],
      "artifacts": [],
      "required_components": [],
      "upstream_access": "unknown",
      "audit_observations": {
        "commercial_inference": {
          "status": "permissive_metadata_only",
          "scope": "Selected weight/component; all required components and intended use must be assessed.",
          "summary": "Card metadata states MIT for the 1.5 release; bundled LM and text encoder terms not separately reviewed."
        },
        "output_terms": {
          "status": "not_specifically_granted_in_reviewed_material",
          "summary": "No output ownership grant identified.",
          "ownership_or_all_rights_cleared": false
        },
        "sources": [
          {
            "url": "https://huggingface.co/ACE-Step/Ace-Step1.5/blob/19671f406d603126926c1b7e2adc169acbcade22/README.md",
            "checked_at": "2026-11-02",
            "method": "read_primary_source"
          }
        ],
        "open_questions": [
          "Terms of the bundled language model and text encoder",
          "Training-data provenance for music generation"
        ],
        "identified_components": ["ACE-Step 1.5 DiT", "ACE-Step 1.5 LM", "music DCAE / vocoder"]
      }
    },
    {
      "id": "ASLP-lab/YingMusic-Singer-Plus",
      "license": "CC-BY-4.0; Stability AI Community License (Stable Audio VAE)",
      "commercial_use": false,
      "review_status": "noncommercial",
      "credit": "YingMusic-Singer-Plus — ASLP@NPU; VAE by Stability AI",
      "source_url": "https://huggingface.co/ASLP-lab/YingMusic-Singer-Plus",
      "evidence_url": "https://huggingface.co/ASLP-lab/YingMusic-Singer-Plus/blob/9b3f444f2bccd77fbc03e32eb7c334fc96040b8d/README.md",
      "revision": "9b3f444f2bccd77fbc03e32eb7c334fc96040b8d",
      "engines": ["yingmusic"],
      "configured_at": "backend/services/sidecar_install.py; backend/services/audio_edit_engines.py",
      "notes": "Audio Edit lyric-edit engine. The card is CC-BY-4.0 but the bundled Stable Audio VAE is under the Stability AI Community License (LICENSE-STABILITY in the code repo), which caps commercial use, so it is gated as non-commercial.",
      "runtime_revision": "9b3f444f2bccd77fbc03e32eb7c334fc96040b8d",
      "runtime_pin_scope": "external",
      "runtime_pin_source": "backend/services/sidecar_install.py",
      "component_closure": "incomplete",
      "commercial_inference": "restricted",
      "commercial_outputs": "unknown",
      "documents": [],
      "artifacts": [],
      "required_components": [],
      "upstream_access": "unknown",
      "audit_observations": {
        "commercial_inference": {
          "status": "noncommercial",
          "scope": "Selected weight/component; all required components and intended use must be assessed.",
          "summary": "CC-BY-4.0 card plus Stability AI Community License for the VAE component."
        },
        "output_terms": {
          "status": "not_specifically_granted_in_reviewed_material",
          "summary": "No output ownership grant identified.",
          "ownership_or_all_rights_cleared": false
        },
        "sources": [
          {
            "url": "https://huggingface.co/ASLP-lab/YingMusic-Singer-Plus/blob/9b3f444f2bccd77fbc03e32eb7c334fc96040b8d/README.md",
            "checked_at": "2026-11-02",
            "method": "read_primary_source"
          },
          {
            "url": "https://github.com/ASLP-lab/YingMusic-Singer-Plus/blob/baa409c2e7e5e775f09b4e92a220808f4827d2cc/LICENSE-STABILITY",
            "checked_at": "2026-11-02",
            "method": "read_primary_source"
          }
        ],
        "open_questions": [
          "Stability AI Community License revenue threshold applicability",
          "Attribution duties for CC-BY-4.0 outputs"
        ],
        "identified_components": ["YingMusic-Singer-Plus weights", "Stable Audio VAE", "lyric aligner"]
      }
    },
    {
      "id": "RMSnow/Vevo2",
      "license": "CC-BY-NC-ND-4.0",
      "commercial_use": false,
      "review_status": "noncommercial",
      "credit": "Vevo2 — Xueyao Zhang and coauthors (Amphion)",
      "source_url": "https://huggingface.co/RMSnow/Vevo2",
      "evidence_url": "https://huggingface.co/RMSnow/Vevo2/blob/2674843cbaa50aa89ee7ccaf5bb15d6ccf46c6c8/README.md",
      "revision": "2674843cbaa50aa89ee7ccaf5bb15d6ccf46c6c8",
      "engines": ["vevo2"],
      "configured_at": "backend/services/sidecar_install.py; backend/services/audio_edit_engines.py",
      "notes": "Audio Edit lyric-edit engine via Amphion. Weights are CC-BY-NC-ND-4.0: non-commercial, and weights are used unmodified.",
      "runtime_revision": "2674843cbaa50aa89ee7ccaf5bb15d6ccf46c6c8",
      "runtime_pin_scope": "external",
      "runtime_pin_source": "backend/services/sidecar_install.py",
      "component_closure": "incomplete",
      "commercial_inference": "restricted",
      "commercial_outputs": "unknown",
      "documents": [],
      "artifacts": [],
      "required_components": [],
      "upstream_access": "unknown",
      "audit_observations": {
        "commercial_inference": {
          "status": "noncommercial",
          "scope": "Selected weight/component; all required components and intended use must be assessed.",
          "summary": "Pinned card licenses the weights CC-BY-NC-ND-4.0."
        },
        "output_terms": {
          "status": "not_specifically_granted_in_reviewed_material",
          "summary": "No output ownership grant identified; non-commercial weights cannot authorize commercial generation.",
          "ownership_or_all_rights_cleared": false
        },
        "sources": [
          {
            "url": "https://huggingface.co/RMSnow/Vevo2/blob/2674843cbaa50aa89ee7ccaf5bb15d6ccf46c6c8/README.md",
            "checked_at": "2026-11-02",
            "method": "read_primary_source"
          }
        ],
        "open_questions": [
          "Whether edited outputs count as adaptations under ND terms",
          "Reference voice and input rights, consent"
        ],
        "identified_components": [
          "prosody and content-style tokenizers",
          "content-style AR model",
          "flow-matching acoustic model",
          "OpenAI Whisper medium encoder (MIT, fetched by the installer)",
          "vocoder"
        ]
      }
    }
```

- [ ] **Step 2: Append Audio Edit to the two existing dynamic assets**

In `"dynamic_assets"`, make these edits and change nothing else:

- On the record with `"id": "htdemucs"`, change `configured_at` to:
  ```json
  "configured_at": "backend/services/dub_pipeline.py:1537; backend/api/routers/system.py:1364; backend/services/audio_edit.py",
  ```
- On the record whose id is `User-provided Sherpa ONNX / RVC voice models / GPT-SoVITS server / custom MLX and env-overridden repos`, append `; backend/services/audio_edit.py` to the end of its existing `configured_at` string.

- [ ] **Step 3: Update the inventory count**

In `docs/licensing/model-inventory.md`, change `records 56 configured model repositories` to `records 61 configured model repositories`.

- [ ] **Step 4: Write the four engine docs**

`docs/engines/f5tts.md`:

```markdown
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
```

`docs/engines/ace-step.md`:

```markdown
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
```

`docs/engines/yingmusic.md`:

```markdown
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
```

`docs/engines/vevo2.md`:

```markdown
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
```

- [ ] **Step 5: Validate the registry**

Run: `uv run python scripts/check_model_licenses.py`
Expected: `Model licence records valid: 63 models, …`. That is the previous count plus 5, and there must be no error lines.

Run: `uv run pytest tests/test_model_licenses.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add backend/config/model_licenses.json docs/licensing/model-inventory.md docs/engines/f5tts.md docs/engines/ace-step.md docs/engines/yingmusic.md docs/engines/vevo2.md
git commit -m "Audio Edit: licence records and engine docs"
```

## Task 2: Pure DSP module

**Files:**
- Create: `backend/services/audio_edit_dsp.py`
- Test: `tests/test_audio_edit_dsp.py`

**Interfaces:**
- Consumes: numpy, torch and torchaudio (`torchaudio.functional.loudness` and `torchaudio.functional.resample`, both present in torchaudio 2.8).
- Produces, used by Tasks 9 and 10:
  - `as_2d`, `match_channels`, `ms_to_samples`, `equal_power`, `fit_length`
  - `splice(base, sr, start_s, end_s, insert, crossfade_ms)`
  - `apply_splices(base, sr, edits, crossfade_ms)`
  - `loudness_lufs`, `match_loudness`, `context_around`, `db_to_gain`, `remix`, `resample`, `estimate_duration`
  - The constants `SILENCE_LUFS`, `MAX_GAIN_DB`, `CONTEXT_S`

**Splice contract**, which every later task relies on:
- `x = ms_to_samples(crossfade_ms, sr)`.
- An `insert` is `x` samples of lead-in, then the replacement body, then `x` samples of lead-out.
- The lead-in crossfades with `base[a-x:a]` and the lead-out with `base[b:b+x]` (equal power). The body replaces `base[a:b]`.
- Samples outside `[a-x, b+x)` are returned unchanged.
- `insert=None` deletes `[a, b)` with a single crossfade at the seam. The two crossfade halves merge into one, so the output is `(b - a) + x` samples shorter. Speech uses this. Song deletes pass a silent insert instead, so the length stays fixed.

- [ ] **Step 1: Write the failing tests**

`tests/test_audio_edit_dsp.py`:

```python
"""Audio Edit DSP: untouched audio must stay byte-identical."""
from __future__ import annotations

import numpy as np
import pytest

from services import audio_edit_dsp as dsp

SR = 1000
LSR = 16000  # loudness needs a real sample rate for the K-weighting filters


def _noise(n: int, ch: int = 1, seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal((ch, n)) * 0.1).astype(np.float32)


def _sine(seconds: float, amp: float, sr: int = LSR, ch: int = 1) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return np.tile((amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32), (ch, 1))


def test_equal_power_sums_to_unity():
    out, fin = dsp.equal_power(64)
    np.testing.assert_allclose(out ** 2 + fin ** 2, 1.0, atol=1e-6)
    assert out[0] > 0.99 and fin[-1] > 0.99
    assert dsp.equal_power(0)[0].shape == (0,)


def test_as_2d_and_match_channels():
    assert dsp.as_2d(np.zeros(5)).shape == (1, 5)
    stereo = dsp.match_channels(np.ones((1, 4)), 2)
    assert stereo.shape == (2, 4)
    np.testing.assert_allclose(dsp.match_channels(np.array([[1.0, 1.0], [0.0, 0.0]]), 1), [[0.5, 0.5]])
    with pytest.raises(ValueError):
        dsp.as_2d(np.zeros((1, 2, 3)))


def test_splice_replaces_body_and_keeps_outside_identical():
    base = _noise(5000)
    x = dsp.ms_to_samples(30, SR)
    insert = np.full((1, 700 + 2 * x), 0.25, np.float32)
    out = dsp.splice(base, SR, 2.0, 2.5, insert, 30)
    assert out.shape == (1, 5000 - 500 + 700)
    assert np.array_equal(out[:, : 2000 - x], base[:, : 2000 - x])
    assert np.array_equal(out[:, 2700 + x:], base[:, 2500 + x:])
    np.testing.assert_allclose(out[:, 2000:2700], 0.25)


def test_splice_at_file_edges_shrinks_crossfade():
    base = _noise(1000)
    x = dsp.ms_to_samples(30, SR)
    insert = np.full((1, 100 + 2 * x), 0.5, np.float32)
    out = dsp.splice(base, SR, 0.0, 0.1, insert, 30)
    assert out.shape == (1, 1000)
    np.testing.assert_allclose(out[:, :100], 0.5)
    assert np.array_equal(out[:, 100 + x:], base[:, 100 + x:])
    tail = dsp.splice(base, SR, 0.9, 1.0, insert, 30)
    np.testing.assert_allclose(tail[:, 900:], 0.5)
    assert np.array_equal(tail[:, : 900 - x], base[:, : 900 - x])


def test_splice_delete_single_crossfade():
    base = _noise(3000)
    x = dsp.ms_to_samples(20, SR)
    out = dsp.splice(base, SR, 1.0, 1.5, None, 20)
    assert out.shape == (1, 3000 - 500 - x)
    assert np.array_equal(out[:, : 1000 - x], base[:, : 1000 - x])
    assert np.array_equal(out[:, 1000:], base[:, 1500 + x:])


def test_splice_rejects_bad_ranges_and_short_inserts():
    base = _noise(1000)
    with pytest.raises(ValueError):
        dsp.splice(base, SR, 0.8, 1.2, None, 10)
    with pytest.raises(ValueError):
        dsp.splice(base, SR, 0.5, 0.4, None, 10)
    with pytest.raises(ValueError):
        dsp.splice(base, SR, 0.2, 0.3, np.zeros((1, 10), np.float32), 30)


def test_apply_splices_runs_latest_first_and_rejects_overlap():
    base = _noise(6000)
    x = dsp.ms_to_samples(10, SR)
    a = np.full((1, 300 + 2 * x), 0.2, np.float32)
    b = np.full((1, 900 + 2 * x), -0.2, np.float32)
    expected = dsp.splice(dsp.splice(base, SR, 4.0, 4.5, b, 10), SR, 1.0, 1.5, a, 10)
    out = dsp.apply_splices(base, SR, [(1.0, 1.5, a), (4.0, 4.5, b)], 10)
    assert np.array_equal(out, expected)
    assert out.shape[1] == 6000 - 500 + 300 - 500 + 900
    with pytest.raises(ValueError):
        dsp.apply_splices(base, SR, [(1.0, 2.0, None), (1.5, 2.5, None)], 10)


def test_fit_length_pads_and_trims():
    assert dsp.fit_length(np.ones((1, 5)), 8).shape == (1, 8)
    assert dsp.fit_length(np.ones((2, 9)), 4).shape == (2, 4)


def test_loudness_tracks_gain_and_floors_silence():
    quiet = dsp.loudness_lufs(_sine(1.0, 0.1), LSR)
    loud = dsp.loudness_lufs(_sine(1.0, 0.4), LSR)
    assert loud - quiet == pytest.approx(12.04, abs=0.3)
    assert dsp.loudness_lufs(np.zeros((1, LSR), np.float32), LSR) == dsp.SILENCE_LUFS
    short = dsp.loudness_lufs(_sine(0.1, 0.4), LSR)
    assert short == pytest.approx(loud, abs=1.0)


def test_match_loudness_matches_clamps_and_skips_silence():
    context = _sine(2.0, 0.4)
    matched = dsp.match_loudness(_sine(0.5, 0.1), context, LSR)
    assert dsp.loudness_lufs(matched, LSR) == pytest.approx(dsp.loudness_lufs(context, LSR), abs=0.5)
    tiny = dsp.match_loudness(_sine(0.5, 0.01), context, LSR)
    gain_db = 20 * np.log10(np.max(np.abs(tiny)) / 0.01)
    assert gain_db == pytest.approx(dsp.MAX_GAIN_DB, abs=0.1)
    ins = _sine(0.5, 0.1)
    assert np.array_equal(dsp.match_loudness(ins, np.zeros((1, LSR), np.float32), LSR), ins)


def test_context_around_excludes_the_span():
    audio = np.arange(10_000, dtype=np.float32)[None, :]
    ctx = dsp.context_around(audio, SR, 4.0, 5.0, 2.0)
    assert ctx.shape == (1, 4000)
    assert ctx[0, 0] == 2000 and ctx[0, 1999] == 3999 and ctx[0, 2000] == 5000


def test_remix_applies_gain_and_limits_peak():
    vocals = np.full((1, 100), 0.5, np.float32)
    inst = np.full((2, 120), 0.3, np.float32)
    mixed = dsp.remix(vocals, inst, 0.0)
    assert mixed.shape == (2, 120)
    assert np.max(np.abs(mixed)) <= 0.999 + 1e-6
    soft = dsp.remix(vocals, np.zeros((2, 100), np.float32), -6.0)
    assert soft[0, 0] == pytest.approx(0.5 * dsp.db_to_gain(-6.0), rel=1e-5)


def test_resample_changes_length():
    out = dsp.resample(np.zeros((2, 24000), np.float32), 24000, 44100)
    assert out.shape == (2, 44100)
    same = np.ones((1, 10), np.float32)
    assert dsp.resample(same, 16000, 16000) is not None


def test_estimate_duration_scales_and_clamps():
    assert dsp.estimate_duration("hello", "hello", 1.0) == pytest.approx(1.0)
    assert dsp.estimate_duration("hi", "hello there", 1.0) == pytest.approx(3.0 + 1.0)
    assert dsp.estimate_duration("hello there", "a", 1.0) == pytest.approx(0.25)
    assert dsp.estimate_duration("ab", "abcd", 0.5) == pytest.approx(1.0)
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `uv run pytest tests/test_audio_edit_dsp.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'services.audio_edit_dsp'`.

- [ ] **Step 3: Implement `backend/services/audio_edit_dsp.py`**

```python
"""Pure DSP for Audio Edit: splicing, crossfades, loudness, remix.

Arrays are float32 shaped ``(channels, samples)``. Nothing here touches disk,
models or the network, so every function is unit-testable. The splice
contract (see :func:`splice`) is what keeps unedited audio byte-identical.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

#: Loudness reported for digital silence (BS.1770 is -inf there).
SILENCE_LUFS = -120.0
#: Largest correction :func:`match_loudness` will apply, either way.
MAX_GAIN_DB = 12.0
#: Seconds of audio either side of a span used as the loudness reference.
CONTEXT_S = 2.0
#: Below this, loudness is too unreliable to match against.
_MATCH_FLOOR_LUFS = -60.0


def as_2d(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 1:
        return a[None, :]
    if a.ndim != 2:
        raise ValueError(f"audio must be 1-D or 2-D, got shape {a.shape}")
    return a


def match_channels(x, channels: int) -> np.ndarray:
    a = as_2d(x)
    if a.shape[0] == channels:
        return a
    if channels == 1:
        return a.mean(axis=0, keepdims=True).astype(np.float32)
    if a.shape[0] == 1:
        return np.repeat(a, channels, axis=0)
    raise ValueError(f"cannot map {a.shape[0]} channels to {channels}")


def ms_to_samples(ms: float, sr: int) -> int:
    return max(0, int(round(float(ms) * sr / 1000.0)))


def equal_power(n: int) -> tuple[np.ndarray, np.ndarray]:
    """``(fade_out, fade_in)`` of length *n* with ``out**2 + in**2 == 1``."""
    if n <= 0:
        empty = np.zeros(0, np.float32)
        return empty, empty
    t = (np.arange(n, dtype=np.float64) + 0.5) / n
    return (np.cos(t * math.pi / 2).astype(np.float32),
            np.sin(t * math.pi / 2).astype(np.float32))


def fit_length(x, n: int) -> np.ndarray:
    a = as_2d(x)
    if a.shape[1] >= n:
        return a[:, :n].copy()
    return np.pad(a, ((0, 0), (0, n - a.shape[1])))


def splice(base, sr: int, start_s: float, end_s: float,
           insert: Optional[np.ndarray], crossfade_ms: float) -> np.ndarray:
    """Replace ``base[a:b]`` (``a``/``b`` = *start_s*/*end_s* in samples).

    *insert* carries ``x`` samples of lead-in, the body, and ``x`` samples
    of lead-out (``x`` = *crossfade_ms* in samples). The margins
    crossfade with ``base[a-x:a]`` and ``base[b:b+x]``; at a file edge the
    margin shrinks to the room available. ``insert=None`` deletes the range
    with one crossfade at the seam (the output is ``b - a + x`` shorter).
    Samples outside ``[a-x, b+x)`` are copied unchanged.
    """
    base = as_2d(base)
    n = base.shape[1]
    a, b = int(round(start_s * sr)), int(round(end_s * sr))
    if not 0 <= a < b <= n:
        raise ValueError(f"span {start_s:.3f}-{end_s:.3f}s is outside the audio")
    x = ms_to_samples(crossfade_ms, sr)
    if insert is None:
        x = min(x, a, n - b)
        fade_out, fade_in = equal_power(x)
        seam = base[:, a - x:a] * fade_out + base[:, b:b + x] * fade_in
        return np.concatenate([base[:, :a - x], seam, base[:, b + x:]], axis=1).astype(np.float32)
    ins = match_channels(insert, base.shape[0])
    if ins.shape[1] < 2 * x + 1:
        raise ValueError("insert is shorter than its crossfade margins")
    lead, tail = min(x, a), min(x, n - b)
    body = ins[:, x - lead: ins.shape[1] - (x - tail)]
    out_l, in_l = equal_power(lead)
    out_t, in_t = equal_power(tail)
    head = base[:, a - lead:a] * out_l + body[:, :lead] * in_l
    end = body.shape[1]
    tail_mix = body[:, end - tail:] * out_t + base[:, b:b + tail] * in_t
    middle = body[:, lead:end - tail]
    return np.concatenate(
        [base[:, :a - lead], head, middle, tail_mix, base[:, b + tail:]], axis=1,
    ).astype(np.float32)


def apply_splices(base, sr: int,
                  edits: Sequence[tuple[float, float, Optional[np.ndarray]]],
                  crossfade_ms: float) -> np.ndarray:
    """Apply non-overlapping *edits* ``(start_s, end_s, insert|None)``.

    Runs from the latest edit back, so earlier offsets stay valid when an
    insert changes the length.
    """
    ordered = sorted(edits, key=lambda e: e[0])
    for (s0, e0, _), (s1, _e1, _) in zip(ordered, ordered[1:]):
        if s1 < e0:
            raise ValueError(f"edits overlap at {s1:.3f}s")
    out = as_2d(base)
    for start_s, end_s, insert in reversed(ordered):
        out = splice(out, sr, start_s, end_s, insert, crossfade_ms)
    return out


def loudness_lufs(x, sr: int) -> float:
    """Integrated BS.1770 loudness; short clips are tiled to 400 ms."""
    import torch
    import torchaudio

    a = as_2d(x)
    if a.size == 0 or float(np.max(np.abs(a))) < 1e-6:
        return SILENCE_LUFS
    need = int(math.ceil(0.4 * sr))
    if a.shape[1] < need:
        a = np.tile(a, (1, int(math.ceil(need / a.shape[1]))))
    value = float(torchaudio.functional.loudness(torch.from_numpy(np.ascontiguousarray(a)), sr))
    return value if math.isfinite(value) else SILENCE_LUFS


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def match_loudness(insert, context, sr: int) -> np.ndarray:
    """Scale *insert* to *context*'s loudness, by at most ±MAX_GAIN_DB."""
    ins = as_2d(insert)
    target = loudness_lufs(context, sr)
    current = loudness_lufs(ins, sr)
    if target < _MATCH_FLOOR_LUFS or current < _MATCH_FLOOR_LUFS:
        return ins
    gain_db = float(np.clip(target - current, -MAX_GAIN_DB, MAX_GAIN_DB))
    return (ins * db_to_gain(gain_db)).astype(np.float32)


def context_around(audio, sr: int, start_s: float, end_s: float,
                   context_s: float = CONTEXT_S) -> np.ndarray:
    """The *context_s* seconds before and after a span, span excluded."""
    a = as_2d(audio)
    n = a.shape[1]
    s, e = int(round(start_s * sr)), int(round(end_s * sr))
    c = int(round(context_s * sr))
    return np.concatenate([a[:, max(0, s - c):s], a[:, e:min(n, e + c)]], axis=1)


def remix(vocals, instrumental, vocal_gain_db: float) -> np.ndarray:
    """Stereo ``vocals * gain + instrumental``, peak-limited to 0.999."""
    n = max(as_2d(vocals).shape[1], as_2d(instrumental).shape[1])
    v = fit_length(match_channels(vocals, 2), n)
    i = fit_length(match_channels(instrumental, 2), n)
    mix = v * db_to_gain(vocal_gain_db) + i
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.999:
        mix = mix * (0.999 / peak)
    return mix.astype(np.float32)


def resample(x, sr_from: int, sr_to: int) -> np.ndarray:
    a = as_2d(x)
    if sr_from == sr_to:
        return a
    import torch
    import torchaudio

    out = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(a)), sr_from, sr_to)
    return out.numpy().astype(np.float32)


def estimate_duration(original_text: str, replacement_text: str, original_s: float) -> float:
    """Seconds the replacement should take: the UTF-8 length ratio of the
    texts times the original span, clamped to ``[0.25, 3 * original + 1]``."""
    o = max(1, len(original_text.strip().encode("utf-8")))
    r = len(replacement_text.strip().encode("utf-8"))
    estimate = original_s * r / o
    return float(min(max(estimate, 0.25), 3.0 * original_s + 1.0))
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run pytest tests/test_audio_edit_dsp.py -q`
Expected: `14 passed`.

- [ ] **Step 5: Commit**

```powershell
git add backend/services/audio_edit_dsp.py tests/test_audio_edit_dsp.py
git commit -m "Audio Edit: splice, crossfade and loudness DSP"
```

## Task 3: Session store, spans, windows and job registry

**Files:**
- Create: `backend/services/audio_edit.py`
- Test: `tests/test_audio_edit_service.py`

**Interfaces:**
- Consumes:
  - `core.config.DATA_DIR`
  - `services.dataset_builder._ffmpeg()`
  - `soundfile`
  - `services.audio_edit_dsp` (Task 2)
- Produces, used by Tasks 7 to 11:
  - **Constants:** `MODES`, `ENGINES`, `MODE_ENGINES`, `ACK_ENGINES`, `MAX_DURATION_S`, `SAMPLE_RATE`, `CHANNELS`, `AUDIO_KINDS`, `LIVE_STATUSES`, `MAX_SPANS`, `MAX_REPLACEMENT_CHARS`, `SPEECH_CONTEXT_S`, `SONG_CONTEXT_S`, `ALLOWED_UPLOAD_EXTS`, `RUN_INLINE`
  - **Errors:** `AudioEditError(code, message, status)` with `.detail() -> {"code", "message"}`
  - **Paths:** `root()`, `session_dir(sid)`, `render_dir(sid, rid)`, `audio_path(sid, kind)`, `render_audio_path(sid, rid)`
  - **Audio I/O:** `decode_to_wav(src, dst, channels) -> float`, `read_audio(path) -> (ndarray (ch, n), sr)`, `write_audio(path, audio, sr)`
  - **Sessions:** `create_session(mode, name, upload_path, original_name) -> dict`, `load_session(sid) -> dict` (with `"renders"`, newest first), `list_sessions()`, `delete_session(sid)`, `update_session(sid, **changes)`
  - **Renders:** `load_render(sid, rid)`, `update_render(sid, rid, **changes)`
  - **Spans:** `validate_spans(raw, duration_s) -> list[dict]`, `set_spans(sid, raw) -> dict`, `words_from_asr(result) -> list[dict]`
  - **Windows:** `Window(start_s, end_s, raw_text, target_text)`, `window_bounds(spans, idx, duration_s)`, `span_window(span, words, duration_s, context_s, lo_s, hi_s)`
  - **Jobs:** `EditJob`, `_JOBS`, `_JOBS_LOCK`, `get_job(job_id)`, `_job_alive(sid, key)`, `_session_busy(sid)`

**On-disk layout** under `root()`, per session:
- `<sid>/session.json` holds `id, mode, name, stage, error, duration_s, sample_rate, channels, created_at, transcript, spans, render_ids`.
- `<sid>/source.wav` is 44.1 kHz float32: mono for speech, stereo for songs.
- `<sid>/vocals.wav` (mono) and `<sid>/instrumental.wav` (stereo) exist for songs only.
- `<sid>/renders/<rid>/render.json` holds `id, session_id, engine, status, progress, spans[{span_id, status, path}], output_path, error, message, created_at, finished_at, commercial_use, params, span_snapshot`.

**Stage machine:**
- Speech: `uploaded → transcribing → ready`.
- Song: `uploaded → separating → separated → transcribing → ready`.
- A failure returns to the previous stage and sets `error`.
- A stage or render left `running` with no live job reads back as `error`/`status` `"interrupted"`.

- [ ] **Step 1: Write the failing tests**

`tests/test_audio_edit_service.py`:

```python
"""Audio Edit service: sessions, spans, windows, jobs and render pipelines."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from services import audio_edit as ae
from services import audio_edit_dsp as dsp

SR = ae.SAMPLE_RATE


def _fake_decode(src: Path, dst: Path, channels: int) -> float:
    data, sr = sf.read(str(src), always_2d=True, dtype="float32")
    audio = dsp.match_channels(data.T, channels)
    sf.write(str(dst), audio.T, sr, subtype="FLOAT")
    return audio.shape[1] / sr


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIVOICE_AUDIO_EDIT_DIR", str(tmp_path / "audio_edit"))
    monkeypatch.setattr(ae, "decode_to_wav", _fake_decode)
    monkeypatch.setattr(ae, "RUN_INLINE", True)
    ae._JOBS.clear()
    yield tmp_path
    ae._JOBS.clear()


def _upload(tmp_path: Path, seconds: float = 4.0, channels: int = 1, name: str = "take.wav") -> Path:
    rng = np.random.default_rng(1)
    data = (rng.standard_normal((int(seconds * SR), channels)) * 0.1).astype(np.float32)
    path = tmp_path / name
    sf.write(str(path), data, SR, subtype="FLOAT")
    return path


WORDS = [
    {"i": 0, "word": "the", "start": 0.10, "end": 0.30},
    {"i": 1, "word": "quick", "start": 0.35, "end": 0.70},
    {"i": 2, "word": "brown", "start": 0.80, "end": 1.20},
    {"i": 3, "word": "fox", "start": 1.30, "end": 1.60},
    {"i": 4, "word": "jumps", "start": 1.70, "end": 2.10},
    {"i": 5, "word": "high", "start": 2.20, "end": 2.60},
]


def _ready(session_id: str) -> dict:
    return ae.update_session(
        session_id, stage="ready",
        transcript={"language": "en", "text": " ".join(w["word"] for w in WORDS), "words": WORDS},
    )


def test_create_speech_session_writes_source_and_json(store):
    session = ae.create_session("speech", "", _upload(store), "Interview Take.wav")
    d = ae.session_dir(session["id"])
    assert (d / "source.wav").is_file()
    assert session["stage"] == "uploaded" and session["mode"] == "speech"
    assert session["name"] == "Interview Take"
    assert session["duration_s"] == pytest.approx(4.0, abs=1e-3)
    assert session["channels"] == 1 and session["renders"] == []
    assert json.loads((d / "session.json").read_text(encoding="utf-8"))["id"] == session["id"]


def test_create_song_session_is_stereo(store):
    session = ae.create_session("song", "Demo", _upload(store, channels=2), "demo.wav")
    audio, sr = ae.read_audio(ae.session_dir(session["id"]) / "source.wav")
    assert audio.shape[0] == 2 and sr == SR and session["name"] == "Demo"


def test_create_rejects_bad_mode_and_too_long(store, monkeypatch):
    with pytest.raises(ae.AudioEditError) as bad:
        ae.create_session("podcast", "", _upload(store), "a.wav")
    assert bad.value.code == "invalid_mode" and bad.value.status == 422
    monkeypatch.setitem(ae.MAX_DURATION_S, "speech", 1.0)
    with pytest.raises(ae.AudioEditError) as long:
        ae.create_session("speech", "", _upload(store), "a.wav")
    assert long.value.code == "too_long" and long.value.status == 400
    assert list(ae.root().iterdir()) == []


def test_ids_are_validated(store):
    for bad in ("../etc", "ABC", "0" * 31, ""):
        with pytest.raises(ae.AudioEditError) as err:
            ae.session_dir(bad)
        assert err.value.status == 404


def test_list_and_delete_sessions(store):
    first = ae.create_session("speech", "one", _upload(store), "a.wav")
    second = ae.create_session("speech", "two", _upload(store), "b.wav")
    ae.update_session(first["id"], created_at=1.0)
    assert [s["name"] for s in ae.list_sessions()] == ["two", "one"]
    ae.delete_session(first["id"])
    assert [s["id"] for s in ae.list_sessions()] == [second["id"]]
    with pytest.raises(ae.AudioEditError):
        ae.load_session(first["id"])


def test_delete_refuses_busy_session(store):
    session = ae.create_session("speech", "", _upload(store), "a.wav")
    job = ae.EditJob(id="j" * 32, session_id=session["id"], kind="transcribe", status="running")
    ae._JOBS[job.id] = job
    with pytest.raises(ae.AudioEditError) as err:
        ae.delete_session(session["id"])
    assert err.value.code == "busy" and err.value.status == 409


def test_interrupted_stage_is_swept(store):
    session = ae.create_session("song", "", _upload(store, channels=2), "a.wav")
    ae.update_session(session["id"], stage="transcribing")
    swept = ae.load_session(session["id"])
    assert swept["stage"] == "separated" and swept["error"] == "interrupted"


def _span(sid="s1", start=0.35, end=0.70, original="quick", replacement="slow"):
    return {"id": sid, "start_s": start, "end_s": end, "original_text": original, "replacement_text": replacement}


def test_validate_spans_sorts_and_normalises():
    spans = ae.validate_spans([_span("b", 1.3, 1.6, "fox", "cat"), _span("a")], 4.0)
    assert [s["id"] for s in spans] == ["a", "b"]
    assert spans[0] == {"id": "a", "start_s": 0.35, "end_s": 0.7, "original_text": "quick", "replacement_text": "slow"}
    assert ae.validate_spans([_span(replacement="")], 4.0)[0]["replacement_text"] == ""


@pytest.mark.parametrize("raw", [
    [_span(), _span("s2", 0.5, 0.9)],                       # overlap
    [_span(start=3.9, end=4.5)],                            # past the end
    [_span(start=0.7, end=0.35)],                           # reversed
    [_span(replacement="quick")],                           # unchanged
    [_span(), _span()],                                     # duplicate id
    [_span(sid="bad id!")],                                 # id charset
    [_span(replacement="x" * 501)],                         # too long
    [_span(f"s{i}", i * 0.05, i * 0.05 + 0.01) for i in range(51)],  # too many
    "nope",
])
def test_validate_spans_rejects(raw):
    with pytest.raises(ae.AudioEditError) as err:
        ae.validate_spans(raw, 4.0)
    assert err.value.code == "invalid_spans" and err.value.status == 400


def test_set_spans_requires_ready(store):
    session = ae.create_session("speech", "", _upload(store), "a.wav")
    with pytest.raises(ae.AudioEditError) as err:
        ae.set_spans(session["id"], [_span()])
    assert err.value.code == "not_ready" and err.value.status == 409
    _ready(session["id"])
    assert ae.set_spans(session["id"], [_span()])["spans"][0]["replacement_text"] == "slow"


def test_words_from_asr_reads_segments_and_top_level():
    segs = {"segments": [{"words": [{"word": " Hi", "start": 0.0, "end": 0.2},
                                    {"word": "there", "start": 0.25}]},
                         {"words": [{"word": "you", "start": 0.5, "end": 0.7, "score": 0.9}]}]}
    assert ae.words_from_asr(segs) == [
        {"i": 0, "word": "Hi", "start": 0.0, "end": 0.2},
        {"i": 1, "word": "you", "start": 0.5, "end": 0.7},
    ]
    top = {"words": [{"word": "a", "start": 1, "end": 2}]}
    assert ae.words_from_asr(top)[0] == {"i": 0, "word": "a", "start": 1.0, "end": 2.0}
    with pytest.raises(ae.AudioEditError) as err:
        ae.words_from_asr({"segments": [{"text": "no words"}]})
    assert err.value.code == "no_word_timestamps" and err.value.status == 422


def test_span_window_adds_context_words():
    win = ae.span_window(_span("s", 0.80, 1.20, "brown", "red"), WORDS, 4.0, 0.6)
    assert (win.start_s, win.end_s) == (0.35, 1.6)
    assert win.raw_text == "quick brown fox"
    assert win.target_text == "quick red fox"


def test_span_window_respects_bounds_and_edges():
    spans = [_span("a", 0.35, 0.70, "quick", "slow"), _span("b", 1.30, 1.60, "fox", "cat")]
    lo, hi = ae.window_bounds(spans, 1, 4.0)
    assert (lo, hi) == (pytest.approx(1.0), 4.0)
    win = ae.span_window(spans[1], WORDS, 4.0, 2.0, lo, hi)
    assert win.start_s == pytest.approx(1.05) and win.raw_text == "fox jumps high"
    first = ae.span_window(_span("c", 0.10, 0.30, "the", "a"), WORDS, 4.0, 2.0)
    assert first.raw_text.startswith("the")
    lonely = ae.span_window(_span("d", 3.0, 3.5, "", "hello"), WORDS, 4.0, 2.0, 2.7, 4.0)
    assert (lonely.start_s, lonely.end_s) == (2.75, 3.75) and lonely.target_text == "hello"


def test_get_job_unknown_is_404(store):
    with pytest.raises(ae.AudioEditError) as err:
        ae.get_job("0" * 32)
    assert err.value.status == 404
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `uv run pytest tests/test_audio_edit_service.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'services.audio_edit'`.

- [ ] **Step 3: Implement the core of `backend/services/audio_edit.py`**

```python
"""Audio Edit: regenerate chosen words inside speech or sung vocals.

Sessions live under ``<DATA_DIR>/audio_edit/<session_id>/`` (override with
``OMNIVOICE_AUDIO_EDIT_DIR``). Separation, transcription and renders run as
background-thread jobs, like the Voice Lab dataset builder; only one render
holds the GPU at a time. Unedited audio is never re-encoded: each edit is
spliced into the original samples with short equal-power crossfades.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from services.audio_edit_dsp import as_2d

logger = logging.getLogger("omnivoice.audio_edit")

MODES = ("speech", "song")
ENGINES = ("f5tts", "ace_step", "yingmusic", "vevo2")
MODE_ENGINES = {"speech": ("f5tts",), "song": ("ace_step", "yingmusic", "vevo2")}
#: Engines whose weights are non-commercial: the user acknowledges once.
ACK_ENGINES = frozenset({"f5tts", "yingmusic", "vevo2"})
MAX_DURATION_S = {"speech": 600.0, "song": 480.0}
SAMPLE_RATE = 44100
CHANNELS = {"speech": 1, "song": 2}
AUDIO_KINDS = ("source", "vocals", "instrumental")
LIVE_STATUSES = frozenset({"queued", "running"})
MAX_SPANS = 50
MAX_REPLACEMENT_CHARS = 500
SPEECH_CONTEXT_S = 1.5
SONG_CONTEXT_S = 2.0
ALLOWED_UPLOAD_EXTS = frozenset({
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus",
    ".webm", ".mp4", ".mov", ".mkv",
})
#: Tests flip this to run jobs synchronously on the calling thread.
RUN_INLINE = False

_ID_RE = re.compile(r"[0-9a-f]{32}")
_SPAN_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_STAGE_JOB = {"separating": "separate", "transcribing": "transcribe"}
#: Serialises read-modify-write of session.json and render.json.
_STATE_LOCK = threading.RLock()


class AudioEditError(Exception):
    """A user-facing failure with a stable ``code`` the UI translates."""

    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status

    def detail(self) -> dict:
        return {"code": self.code, "message": self.message}


# ── paths ────────────────────────────────────────────────────────────────

def root() -> Path:
    override = os.environ.get("OMNIVOICE_AUDIO_EDIT_DIR")
    if override:
        return Path(override)
    from core.config import DATA_DIR
    return Path(DATA_DIR) / "audio_edit"


def _check_id(value: object, what: str = "session") -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise AudioEditError("not_found", f"Unknown {what}.", 404)
    return value


def session_dir(session_id: str) -> Path:
    base = root().resolve()
    path = (base / _check_id(session_id)).resolve()
    if path.parent != base:
        raise AudioEditError("not_found", "Unknown session.", 404)
    return path


def render_dir(session_id: str, render_id: str) -> Path:
    return session_dir(session_id) / "renders" / _check_id(render_id, "render")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ── audio I/O ────────────────────────────────────────────────────────────

def decode_to_wav(src: Path, dst: Path, channels: int) -> float:
    """Decode any local audio/video file to 44.1 kHz float WAV; returns seconds."""
    import soundfile as sf
    from services.dataset_builder import _ffmpeg

    cmd = [
        _ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-protocol_whitelist", "file", "-y", "-i", str(src),
        "-vn", "-ac", str(channels), "-ar", str(SAMPLE_RATE), "-c:a", "pcm_f32le", str(dst),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AudioEditError("invalid_audio", f"Could not decode the file: {exc}", 400) from exc
    if proc.returncode != 0 or not dst.is_file():
        raise AudioEditError("invalid_audio", "Could not read that file as audio.", 400)
    return float(sf.info(str(dst)).duration)


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    return np.ascontiguousarray(data.T), int(sr)


def write_audio(path: Path, audio, sr: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), as_2d(audio).T, sr, subtype="FLOAT")


# ── jobs registry (runner in Task 8) ─────────────────────────────────────

@dataclass
class EditJob:
    id: str
    session_id: str
    kind: str                      # "separate" | "transcribe" | "render"
    render_id: Optional[str] = None
    status: str = "queued"         # queued | running | succeeded | failed | cancelled
    stage: Optional[str] = None
    progress: float = 0.0
    error: Optional[str] = None    # an AudioEditError code
    message: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    engine: Any = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "id", "session_id", "kind", "render_id", "status", "stage",
            "progress", "error", "message", "created_at")}


_JOBS: dict[str, EditJob] = {}
_JOBS_LOCK = threading.Lock()


def get_job(job_id: str) -> EditJob:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise AudioEditError("not_found", "Unknown job.", 404)
    return job


def _job_alive(session_id: str, key: str) -> bool:
    """A live job for *session_id* whose kind or render id is *key*."""
    with _JOBS_LOCK:
        return any(j.session_id == session_id and j.status in LIVE_STATUSES
                   and key in (j.kind, j.render_id) for j in _JOBS.values())


def _session_busy(session_id: str) -> bool:
    with _JOBS_LOCK:
        return any(j.session_id == session_id and j.status in LIVE_STATUSES
                   for j in _JOBS.values())


# ── sessions and renders ─────────────────────────────────────────────────

def _load_raw(session_id: str) -> dict:
    path = session_dir(session_id) / "session.json"
    if not path.is_file():
        raise AudioEditError("not_found", "Unknown session.", 404)
    return json.loads(path.read_text(encoding="utf-8"))


def _save_session(session: dict) -> None:
    data = {k: v for k, v in session.items() if k != "renders"}
    _write_json(session_dir(data["id"]) / "session.json", data)


def update_session(session_id: str, **changes) -> dict:
    with _STATE_LOCK:
        session = _load_raw(session_id)
        session.update(changes)
        _save_session(session)
        return session


def _save_render(render: dict) -> None:
    _write_json(render_dir(render["session_id"], render["id"]) / "render.json", render)


def load_render(session_id: str, render_id: str) -> dict:
    path = render_dir(session_id, render_id) / "render.json"
    if not path.is_file():
        raise AudioEditError("not_found", "Unknown render.", 404)
    with _STATE_LOCK:
        render = json.loads(path.read_text(encoding="utf-8"))
        if render.get("status") in LIVE_STATUSES and not _job_alive(session_id, render_id):
            render.update(status="failed", error="interrupted",
                          message="The app stopped before this render finished.",
                          finished_at=render.get("finished_at") or time.time())
            _save_render(render)
    return render


def update_render(session_id: str, render_id: str, **changes) -> dict:
    with _STATE_LOCK:
        render = json.loads((render_dir(session_id, render_id) / "render.json").read_text(encoding="utf-8"))
        render.update(changes)
        _save_render(render)
        return render


def load_session(session_id: str) -> dict:
    with _STATE_LOCK:
        session = _load_raw(session_id)
        stage = session.get("stage")
        if stage in _STAGE_JOB and not _job_alive(session_id, _STAGE_JOB[stage]):
            fallback = "uploaded" if stage == "separating" or session["mode"] == "speech" else "separated"
            session.update(stage=fallback, error="interrupted")
            _save_session(session)
        renders = []
        for render_id in reversed(session.get("render_ids", [])):
            try:
                renders.append(load_render(session_id, render_id))
            except (AudioEditError, ValueError, OSError):
                continue
    session["renders"] = renders
    return session


def create_session(mode: str, name: str, upload_path: Path, original_name: str) -> dict:
    if mode not in MODES:
        raise AudioEditError("invalid_mode", "Mode must be speech or song.", 422)
    session_id = uuid.uuid4().hex
    d = root() / session_id
    d.mkdir(parents=True, exist_ok=False)
    try:
        duration = decode_to_wav(Path(upload_path), d / "source.wav", CHANNELS[mode])
        if duration > MAX_DURATION_S[mode]:
            minutes = int(MAX_DURATION_S[mode] // 60)
            raise AudioEditError("too_long", f"{mode.capitalize()} files are limited to {minutes} minutes.", 400)
    except BaseException:
        shutil.rmtree(d, ignore_errors=True)
        raise
    label = (name or "").strip()[:120] or Path(original_name or "").stem[:120] or "Untitled"
    session = {
        "id": session_id, "mode": mode, "name": label, "stage": "uploaded", "error": None,
        "duration_s": round(duration, 3), "sample_rate": SAMPLE_RATE, "channels": CHANNELS[mode],
        "created_at": time.time(), "transcript": None, "spans": [], "render_ids": [],
    }
    _save_session(session)
    return load_session(session_id)


def list_sessions() -> list[dict]:
    base = root()
    if not base.is_dir():
        return []
    sessions = []
    for child in base.iterdir():
        if not (child.is_dir() and _ID_RE.fullmatch(child.name) and (child / "session.json").is_file()):
            continue
        try:
            sessions.append(load_session(child.name))
        except (AudioEditError, ValueError, OSError) as exc:
            logger.warning("skipping unreadable Audio Edit session %s: %s", child.name, exc)
    sessions.sort(key=lambda s: s.get("created_at", 0), reverse=True)
    return sessions


def delete_session(session_id: str) -> None:
    d = session_dir(session_id)
    if not (d / "session.json").is_file():
        raise AudioEditError("not_found", "Unknown session.", 404)
    if _session_busy(session_id):
        raise AudioEditError("busy", "Cancel the running job before deleting this session.", 409)
    shutil.rmtree(d)


def audio_path(session_id: str, kind: str) -> Path:
    if kind not in AUDIO_KINDS:
        raise AudioEditError("not_found", "Unknown audio kind.", 404)
    path = session_dir(session_id) / f"{kind}.wav"
    if not path.is_file():
        raise AudioEditError("not_found", f"This session has no {kind} audio.", 404)
    return path


def render_audio_path(session_id: str, render_id: str) -> Path:
    render = load_render(session_id, render_id)
    if render.get("status") != "succeeded" or not render.get("output_path"):
        raise AudioEditError("not_ready", "This render has no audio yet.", 409)
    path = session_dir(session_id) / render["output_path"]
    if not path.is_file():
        raise AudioEditError("not_found", "The rendered file is missing.", 404)
    return path


# ── spans and transcript ─────────────────────────────────────────────────

def _bad_spans(message: str) -> AudioEditError:
    return AudioEditError("invalid_spans", message, 400)


def validate_spans(raw: object, duration_s: float) -> list[dict]:
    """Normalise and check client spans; sorted by start, never overlapping."""
    if not isinstance(raw, list):
        raise _bad_spans("Spans must be a list.")
    if len(raw) > MAX_SPANS:
        raise _bad_spans(f"At most {MAX_SPANS} spans per session.")
    seen: set[str] = set()
    spans = []
    for item in raw:
        if not isinstance(item, dict):
            raise _bad_spans("Each span must be an object.")
        span_id = item.get("id")
        if not isinstance(span_id, str) or not _SPAN_ID_RE.fullmatch(span_id) or span_id in seen:
            raise _bad_spans("Each span needs a unique id of letters, digits, - or _.")
        try:
            start, end = float(item["start_s"]), float(item["end_s"])
        except (KeyError, TypeError, ValueError):
            raise _bad_spans(f"Span {span_id} needs numeric start_s and end_s.") from None
        if not (0.0 <= start < end <= duration_s + 1e-6):
            raise _bad_spans(f"Span {span_id} is outside the audio.")
        original = str(item.get("original_text") or "").strip()
        replacement = str(item.get("replacement_text") or "").strip()
        if len(replacement) > MAX_REPLACEMENT_CHARS:
            raise _bad_spans(f"Span {span_id}: replacements are limited to {MAX_REPLACEMENT_CHARS} characters.")
        if replacement == original:
            raise _bad_spans(f"Span {span_id} does not change anything.")
        seen.add(span_id)
        spans.append({"id": span_id, "start_s": round(start, 3), "end_s": round(min(end, duration_s), 3),
                      "original_text": original, "replacement_text": replacement})
    spans.sort(key=lambda s: s["start_s"])
    for prev, cur in zip(spans, spans[1:]):
        if cur["start_s"] < prev["end_s"]:
            raise _bad_spans(f"Spans {prev['id']} and {cur['id']} overlap.")
    return spans


def set_spans(session_id: str, raw: object) -> dict:
    with _STATE_LOCK:
        session = _load_raw(session_id)
        if session.get("stage") != "ready" or not session.get("transcript"):
            raise AudioEditError("not_ready", "Transcribe the audio before marking spans.", 409)
        session["spans"] = validate_spans(raw, float(session["duration_s"]))
        _save_session(session)
    return load_session(session_id)


def words_from_asr(result: dict) -> list[dict]:
    """Flatten an ASR result to ``[{i, word, start, end}]`` with timings."""
    raw = result.get("words")
    if not isinstance(raw, list) or not raw:
        raw = [w for seg in result.get("segments") or [] for w in (seg.get("words") or [])]
    words = []
    for w in raw:
        text = str(w.get("word") or "").strip()
        start, end = w.get("start"), w.get("end")
        if not text or not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or end < start:
            continue
        words.append({"i": len(words), "word": text, "start": round(float(start), 3), "end": round(float(end), 3)})
    if not words:
        raise AudioEditError("no_word_timestamps",
                             "The speech recogniser returned no word timings. Pick an engine with word timestamps in Settings.",
                             422)
    return words


@dataclass(frozen=True)
class Window:
    """The audio an engine sees for one span: the span plus context words."""
    start_s: float
    end_s: float
    raw_text: str
    target_text: str


def window_bounds(spans: list[dict], idx: int, duration_s: float) -> tuple[float, float]:
    """Midpoints to the neighbouring spans, so windows never overlap."""
    lo = 0.0 if idx == 0 else (spans[idx - 1]["end_s"] + spans[idx]["start_s"]) / 2
    hi = duration_s if idx == len(spans) - 1 else (spans[idx]["end_s"] + spans[idx + 1]["start_s"]) / 2
    return lo, hi


def span_window(span: dict, words: list[dict], duration_s: float, context_s: float,
                lo_s: float = 0.0, hi_s: Optional[float] = None) -> Window:
    hi_s = duration_s if hi_s is None else hi_s
    start, end = span["start_s"], span["end_s"]
    eps = 1e-3
    before = [w for w in words if w["end"] <= start + eps and w["start"] >= max(lo_s, start - context_s) - eps]
    after = [w for w in words if w["start"] >= end - eps and w["end"] <= min(hi_s, end + context_s) + eps]
    w_start = before[0]["start"] if before else max(lo_s, start - min(context_s, 0.25))
    w_end = after[-1]["end"] if after else min(hi_s, end + min(context_s, 0.25))
    w_start, w_end = max(lo_s, min(w_start, start)), min(hi_s, max(w_end, end))
    b_text = " ".join(w["word"] for w in before)
    a_text = " ".join(w["word"] for w in after)
    raw = " ".join(t for t in (b_text, span["original_text"], a_text) if t)
    target = " ".join(t for t in (b_text, span["replacement_text"], a_text) if t)
    return Window(round(w_start, 3), round(w_end, 3), raw, target)
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run pytest tests/test_audio_edit_service.py -q`
Expected: `22 passed` (13 tests plus 9 parametrised rejects).

- [ ] **Step 5: Commit**

```powershell
git add backend/services/audio_edit.py tests/test_audio_edit_service.py
git commit -m "Audio Edit: session store, spans and windows"
```

## Task 4: Edit sidecar protocol, engine clients and the F5-TTS speech engine

**Files:**
- Create:
  - `backend/engines/_edit_protocol/__init__.py`
  - `backend/services/audio_edit_engines.py`
  - `backend/engines/f5tts/__init__.py` (empty)
  - `backend/engines/f5tts/main.py`
- Modify:
  - `backend/services/sidecar_install.py`: add the `f5tts` spec
  - `tests/test_sidecar_install.py`: the host-gate expectation
  - `tests/test_sidecar_stdout_isolation_1428.py`: `EXPECTED_SIDECARS`
- Test: `tests/test_audio_edit_engines.py`

**Interfaces:**
- Consumes:
  - `services.subprocess_backend.SubprocessBackend`: `_spawn`, `_send`, `_recv_with_timeout`, `_force_kill`, `_touch`, `_stderr_tail_text`, `shutdown`
  - `services.sidecar_install.engine_venv_python`, `SidecarSpec`, `SPECS`, `_GIB`
- Produces:
  - **Wire op.** The request is `{"op": "edit", audio_path, out_path, raw_text, target_text, seed, [parts, fix_duration]}`. The sidecar answers with zero or more `{"op": "progress", "fraction"}` frames, then either `{"op": "audio", "path", "sample_rate", ["regions"]}` or `{"op": "error", "message"}`. Every reply op is already in `PARENT_INBOUND_OPS`.
  - **Protocol helper.** `engines._edit_protocol.serve(engine_name, load, edit, out) -> int`. Here `out` is the private frame stream that the entry point opens.
  - **Engine module.** `services.audio_edit_engines` exposes `INSTALLS`, `DISPLAY_NAMES`, `ENGINE_MODELS`, `EDIT_TIMEOUT_S`, `EngineError`, `engine_python(engine_id)`, `engine_checkout(engine_id)`, `EditSidecar` (`.edit(payload, on_progress=None) -> dict`, `.kill()`), `F5TTSEdit`, `EDITORS`, `get_edit_engine(engine_id)`, `GPU_ENGINES`, `has_gpu()` and `engine_status(engine_id) -> {installed, install_id, needs_gpu, available, reason}`.
  - **Installer.** A new sidecar install id `f5tts` (`POST /engines/sidecar/f5tts/install`).

These clients are deliberately **not** TTS backends. They are never added to the engine registry or picker.

- [ ] **Step 1: Write the failing tests**

`tests/test_audio_edit_engines.py`:

```python
"""Audio Edit engine sidecars: the edit op round-trips through a real child process."""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from services import audio_edit_engines as aee

ENGINES_DIR = Path(__file__).resolve().parents[1] / "backend" / "engines"

FAKE_SIDECAR = textwrap.dedent(f"""
    import os, shutil, sys
    sys.path.insert(0, {str(ENGINES_DIR)!r})
    from _edit_protocol import serve

    def load():
        return {{"loads": 1}}

    def edit(model, msg, progress):
        if msg.get("fail"):
            raise RuntimeError("boom")
        progress(0.5)
        shutil.copyfile(msg["audio_path"], msg["out_path"])
        return {{"path": msg["out_path"], "sample_rate": 44100, "regions": [[0.1, 0.2]],
                 "loads": model["loads"]}}

    _frame_fd = os.dup(1)
    os.dup2(2, 1)
    sys.exit(serve("fake", load, edit, os.fdopen(_frame_fd, "wb")))
""")


@pytest.fixture
def fake_engine(tmp_path):
    script = tmp_path / "fake_sidecar.py"
    script.write_text(FAKE_SIDECAR, encoding="utf-8")

    class FakeEdit(aee.EditSidecar):
        id = "audio-edit-fake"
        display_name = "Fake editor"
        engine_key = "f5tts"

        @classmethod
        def venv_python(cls) -> Path:
            return Path(sys.executable)

        @classmethod
        def sidecar_script(cls) -> Path:
            return script

    engine = FakeEdit()
    yield engine
    engine.shutdown()


def test_edit_round_trip_reports_progress(fake_engine, tmp_path):
    src = tmp_path / "in.wav"
    src.write_bytes(b"RIFF....fake")
    seen: list[float] = []
    reply = fake_engine.edit({"audio_path": str(src), "out_path": str(tmp_path / "out.wav")}, seen.append)
    assert reply["op"] == "audio" and Path(reply["path"]).read_bytes() == b"RIFF....fake"
    assert reply["regions"] == [[0.1, 0.2]]
    assert seen == [0.5]


def test_edit_error_frame_raises_engine_error(fake_engine, tmp_path):
    with pytest.raises(aee.EngineError, match="RuntimeError: boom"):
        fake_engine.edit({"fail": True, "audio_path": "x", "out_path": str(tmp_path / "o.wav")})


def test_kill_then_edit_respawns(fake_engine, tmp_path):
    src = tmp_path / "in.wav"
    src.write_bytes(b"x")
    fake_engine.edit({"audio_path": str(src), "out_path": str(tmp_path / "a.wav")})
    fake_engine.kill()
    assert fake_engine._proc is None
    reply = fake_engine.edit({"audio_path": str(src), "out_path": str(tmp_path / "b.wav")})
    assert reply["loads"] == 1 and Path(reply["path"]).is_file()


def test_missing_venv_reports_not_installed(monkeypatch):
    monkeypatch.delenv("OMNIVOICE_F5TTS_DIR", raising=False)
    status = aee.engine_status("f5tts")
    assert status == {"installed": False, "install_id": "f5tts", "needs_gpu": False,
                      "available": False, "reason": "Install F5-TTS to use it."}
    ok, reason = aee.F5TTSEdit.is_available()
    assert ok is False and "F5-TTS" in reason
    with pytest.raises(aee.EngineError):
        aee.F5TTSEdit.venv_python()


def test_gpu_engines_need_a_gpu(monkeypatch, tmp_path):
    monkeypatch.setattr(aee, "engine_python", lambda engine_id: tmp_path / "python")
    monkeypatch.setattr(aee, "has_gpu", lambda: False)
    assert aee.engine_status("f5tts")["available"] is True
    vevo = aee.engine_status("vevo2")
    assert vevo["installed"] is True and vevo["available"] is False and "GPU" in vevo["reason"]
    monkeypatch.setattr(aee, "has_gpu", lambda: True)
    assert aee.engine_status("vevo2")["available"] is True


def test_get_edit_engine_is_a_singleton():
    assert aee.get_edit_engine("f5tts") is aee.get_edit_engine("f5tts")
    with pytest.raises(KeyError):
        aee.get_edit_engine("ace_step")


def test_f5tts_install_spec():
    from services.sidecar_install import SPECS

    spec = SPECS["f5tts"]
    assert spec.has_source is False and spec.env_var == "OMNIVOICE_F5TTS_DIR"
    assert "--python" in spec.venv_args and "f5-tts==1.1.22" in spec.install_args
    assert spec.docs_path == "docs/engines/f5tts.md"
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `uv run pytest tests/test_audio_edit_engines.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'services.audio_edit_engines'`.

- [ ] **Step 3: Write `backend/engines/_edit_protocol/__init__.py`**

```python
"""Stdlib wire loop shared by the Audio Edit engine sidecars.

Framing is identical to engines/_echo and SubprocessBackend: a 4-byte
big-endian length, then UTF-8 JSON. Ops: ping -> pong; shutdown; and
edit -> zero or more {"op":"progress","fraction":f}, then
{"op":"audio", ...reply} or {"op":"error","message":...}. The model loads
on the first edit, so the ready frame is instant and a sidecar that is
never asked to edit never touches the GPU.
"""
from __future__ import annotations

import json
import struct
import sys
import traceback
from typing import Callable

MAX_FRAME_BYTES = 64 * 1024 * 1024


def send(stream, obj: dict) -> None:
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    stream.write(struct.pack("!I", len(body)))
    stream.write(body)
    stream.flush()


def recv(stream):
    header = stream.read(4)
    if len(header) < 4:
        return None
    (n,) = struct.unpack("!I", header)
    if n > MAX_FRAME_BYTES:
        raise IOError(f"frame too large: {n}")
    body = bytearray()
    while len(body) < n:
        chunk = stream.read(n - len(body))
        if not chunk:
            raise IOError("short read")
        body.extend(chunk)
    return json.loads(bytes(body).decode("utf-8"))


def serve(engine_name: str, load: Callable[[], object],
          edit: Callable[[object, dict, Callable[[float], None]], dict], out) -> int:
    """Run the wire loop. ``out`` is the private frame stream the entry point
    opened after ``_frame_fd = os.dup(1); os.dup2(2, 1)`` (#1428): the stdout
    isolation guard checks every engines/*/main.py for that pair itself."""
    stdin = sys.stdin.buffer
    send(out, {"op": "ready", "engine": engine_name})
    model = None

    def progress(fraction: float) -> None:
        send(out, {"op": "progress", "fraction": max(0.0, min(1.0, float(fraction)))})

    while True:
        try:
            msg = recv(stdin)
        except Exception as exc:
            send(out, {"op": "error", "stage": "recv", "message": f"{type(exc).__name__}: {exc}"})
            return 1
        if msg is None:
            return 0
        op = msg.get("op")
        if op == "ping":
            send(out, {"op": "pong"})
        elif op == "shutdown":
            return 0
        elif op == "edit":
            try:
                if model is None:
                    model = load()
                reply = edit(model, msg, progress)
                send(out, {"op": "audio", **reply})
            except Exception as exc:
                traceback.print_exc()
                send(out, {"op": "error", "stage": "edit",
                           "message": f"{type(exc).__name__}: {exc}"[:2000]})
        else:
            send(out, {"op": "error", "stage": "dispatch", "message": f"unknown op: {op!r}"})
```

- [ ] **Step 4: Write `backend/services/audio_edit_engines.py`**

```python
"""Sidecar clients for the Audio Edit engines (F5-TTS, YingMusic, Vevo2).

Each engine runs ``backend/engines/<dir>/main.py`` in the private venv the
sidecar installer made. The parent sends one ``edit`` frame per span and
reads ``progress`` frames until ``audio`` (the sidecar wrote a WAV and
replies with its path) or ``error``. These are not TTS backends and are
never listed in the engine picker. ACE-Step is an HTTP server instead; see
services/acestep_client.py.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable, Optional

from services.subprocess_backend import SubprocessBackend

_ENGINES_DIR = Path(__file__).resolve().parents[1] / "engines"
#: One span can take minutes on CPU; a wedged sidecar is still reclaimed.
EDIT_TIMEOUT_S = 900.0

#: Audio Edit engine id -> (sidecar_install id, env var of its install dir).
INSTALLS = {
    "f5tts": ("f5tts", "OMNIVOICE_F5TTS_DIR"),
    "ace_step": ("ace-step", "OMNIVOICE_ACESTEP_DIR"),
    "yingmusic": ("yingmusic", "OMNIVOICE_YINGMUSIC_DIR"),
    "vevo2": ("vevo2", "OMNIVOICE_VEVO2_DIR"),
}
DISPLAY_NAMES = {
    "f5tts": "F5-TTS",
    "ace_step": "ACE-Step 1.5",
    "yingmusic": "YingMusic-Singer-Plus",
    "vevo2": "Vevo2",
}
#: Licence-registry model ids each engine's output depends on.
ENGINE_MODELS = {
    "f5tts": ("SWivid/F5-TTS", "charactr/vocos-mel-24khz"),
    "ace_step": ("ACE-Step/Ace-Step1.5",),
    "yingmusic": ("ASLP-lab/YingMusic-Singer-Plus",),
    "vevo2": ("RMSnow/Vevo2",),
}


class EngineError(RuntimeError):
    """An engine failed or is not installed; the message is user-facing."""


def engine_python(engine_id: str) -> Optional[Path]:
    from services.sidecar_install import engine_venv_python
    return engine_venv_python(INSTALLS[engine_id][1])


def engine_checkout(engine_id: str) -> Optional[Path]:
    value = os.environ.get(INSTALLS[engine_id][1])
    return Path(value) if value else None


class EditSidecar(SubprocessBackend):
    """A long-lived engine sidecar that answers ``edit`` frames."""

    engine_key = "f5tts"
    sidecar_dir = "f5tts"
    recv_timeout_s = EDIT_TIMEOUT_S

    @property
    def sample_rate(self) -> int:
        return 44100

    @property
    def supported_languages(self) -> list[str]:
        return ["multi"]

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if engine_python(cls.engine_key) is None:
            return False, f"{DISPLAY_NAMES[cls.engine_key]} is not installed."
        return True, ""

    @classmethod
    def venv_python(cls) -> Path:
        py = engine_python(cls.engine_key)
        if py is None:
            raise EngineError(f"{DISPLAY_NAMES[cls.engine_key]} is not installed.")
        return py

    @classmethod
    def sidecar_script(cls) -> Path:
        return _ENGINES_DIR / cls.sidecar_dir / "main.py"

    def edit(self, payload: dict, on_progress: Optional[Callable[[float], None]] = None) -> dict:
        with self._lock:
            self._spawn()
            self._send({"op": "edit", **payload})
            while True:
                frame = self._recv_with_timeout(self.recv_timeout_s)
                if frame is None:
                    tail = self._stderr_tail_text()
                    raise EngineError(f"{self.display_name} stopped unexpectedly. {tail}".strip())
                op = frame.get("op")
                if op == "progress":
                    if on_progress is not None:
                        on_progress(float(frame.get("fraction") or 0.0))
                    continue
                if op == "error":
                    raise EngineError(str(frame.get("message") or f"{self.display_name} failed."))
                if op == "audio":
                    path = frame.get("path")
                    if not isinstance(path, str) or not Path(path).is_file():
                        raise EngineError(f"{self.display_name} returned no audio file.")
                    self._touch()
                    return frame

    def kill(self) -> None:
        """Hard-stop the sidecar (cancel). Lock-free: an edit may hold _lock."""
        self._force_kill()


class F5TTSEdit(EditSidecar):
    id = "audio-edit-f5tts"
    display_name = "F5-TTS"
    engine_key = "f5tts"
    sidecar_dir = "f5tts"


EDITORS: dict[str, type[EditSidecar]] = {"f5tts": F5TTSEdit}
_INSTANCES: dict[str, EditSidecar] = {}
_INSTANCES_LOCK = threading.Lock()


def get_edit_engine(engine_id: str) -> EditSidecar:
    cls = EDITORS[engine_id]
    with _INSTANCES_LOCK:
        if engine_id not in _INSTANCES:
            _INSTANCES[engine_id] = cls()
        return _INSTANCES[engine_id]


#: Engines that are unusable without CUDA or MPS (F5-TTS falls back to CPU).
GPU_ENGINES = frozenset({"ace_step", "yingmusic", "vevo2"})


def has_gpu() -> bool:
    try:
        import torch
    except ImportError:
        return False
    mps = getattr(torch.backends, "mps", None)
    return bool(torch.cuda.is_available() or (mps is not None and mps.is_available()))


def engine_status(engine_id: str) -> dict:
    """``{installed, install_id, needs_gpu, available, reason}`` for one engine."""
    install_id, _env_var = INSTALLS[engine_id]
    installed = engine_python(engine_id) is not None
    if engine_id == "ace_step" and os.environ.get("OMNIVOICE_ACESTEP_URL"):
        installed = True
    needs_gpu = engine_id in GPU_ENGINES
    reason = None
    if not installed:
        reason = f"Install {DISPLAY_NAMES[engine_id]} to use it."
    elif needs_gpu and not has_gpu():
        reason = f"{DISPLAY_NAMES[engine_id]} needs a CUDA or Apple GPU."
    return {"installed": installed, "install_id": install_id, "needs_gpu": needs_gpu,
            "available": reason is None, "reason": reason}
```

- [ ] **Step 5: Write the F5-TTS sidecar `backend/engines/f5tts/main.py`** (and an empty `backend/engines/f5tts/__init__.py`)

```python
"""F5-TTS speech-edit sidecar for Audio Edit (runs in its own venv).

Implements F5-TTS's speech_edit recipe on one window: each edited part is
replaced by silence of the target length, masked out, and regenerated by
the flow model while the rest of the window conditions it. Replies with
the regenerated window and where each edited part landed in it, so the
parent splices only those regions into the untouched original.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _edit_protocol import serve  # noqa: E402

F5_REPO = "SWivid/F5-TTS"
F5_REVISION = "84e5a410d9cead4de2f847e7c9369a6440bdfaca"
F5_CHECKPOINT = "F5TTS_v1_Base/model_1250000.safetensors"
VOCOS_REPO = "charactr/vocos-mel-24khz"
VOCOS_REVISION = "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"
TARGET_RMS = 0.1
NFE_STEPS = 32
CFG_STRENGTH = 2.0
SWAY_SAMPLING = -1.0


def load() -> dict:
    from importlib.resources import files

    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from hydra.utils import get_class
    from omegaconf import OmegaConf
    from f5_tts.infer.utils_infer import load_checkpoint, load_vocoder
    from f5_tts.model import CFM
    from f5_tts.model.utils import get_tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml")))
    mel = cfg.model.mel_spec
    vocab_char_map, vocab_size = get_tokenizer(str(files("f5_tts").joinpath("infer/examples/vocab.txt")), "custom")
    backbone = get_class(f"f5_tts.model.{cfg.model.backbone}")
    model = CFM(
        transformer=backbone(**cfg.model.arch, text_num_embeds=vocab_size, mel_dim=mel.n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=mel.n_fft, hop_length=mel.hop_length, win_length=mel.win_length,
            n_mel_channels=mel.n_mel_channels, target_sample_rate=mel.target_sample_rate,
            mel_spec_type=mel.mel_spec_type,
        ),
        odeint_kwargs=dict(method="euler"),
        vocab_char_map=vocab_char_map,
    ).to(device)
    checkpoint = hf_hub_download(F5_REPO, F5_CHECKPOINT, revision=F5_REVISION)
    model = load_checkpoint(model, checkpoint, device, dtype=torch.float32, use_ema=True)
    vocos_dir = snapshot_download(VOCOS_REPO, revision=VOCOS_REVISION)
    vocoder = load_vocoder(vocoder_name="vocos", is_local=True, local_path=vocos_dir, device=device)
    return {"model": model, "vocoder": vocoder, "device": device,
            "sr": int(mel.target_sample_rate), "hop": int(mel.hop_length)}


def edit(state: dict, msg: dict, progress) -> dict:
    import soundfile as sf
    import torch
    import torch.nn.functional as F
    import torchaudio
    from f5_tts.model.utils import convert_char_to_pinyin

    sr, hop = state["sr"], state["hop"]
    data, in_sr = sf.read(msg["audio_path"], always_2d=True, dtype="float32")
    audio = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)
    rms = float(torch.sqrt(torch.mean(audio ** 2)))
    if 0.0 < rms < TARGET_RMS:
        audio = audio * TARGET_RMS / rms
    if in_sr != sr:
        audio = torchaudio.functional.resample(audio, in_sr, sr)

    pieces, mask, regions = [], [], []
    offset, cursor = 0.0, 0.0
    for (p0, p1), dur in zip(msg["parts"], msg["fix_duration"]):
        keep_from, part_start = round(offset * sr), round(p0 * sr)
        pieces += [audio[:, keep_from:part_start], torch.zeros(1, round(dur * sr))]
        mask += [torch.ones(1, round((part_start - keep_from) / hop), dtype=torch.bool),
                 torch.zeros(1, round(dur * sr / hop), dtype=torch.bool)]
        new_start = cursor + (p0 - offset)
        regions.append([round(new_start, 4), round(new_start + dur, 4)])
        cursor, offset = new_start + dur, p1
    pieces.append(audio[:, round(offset * sr):])
    audio = torch.cat(pieces, dim=-1)
    edit_mask = torch.cat(mask, dim=-1)
    edit_mask = F.pad(edit_mask, (0, audio.shape[-1] // hop - edit_mask.shape[-1] + 1), value=True)

    progress(0.1)
    with torch.inference_mode():
        generated, _ = state["model"].sample(
            cond=audio.to(state["device"]),
            text=convert_char_to_pinyin([msg["target_text"]]),
            duration=audio.shape[-1] // hop,
            steps=NFE_STEPS, cfg_strength=CFG_STRENGTH, sway_sampling_coef=SWAY_SAMPLING,
            seed=msg.get("seed"), edit_mask=edit_mask.to(state["device"]),
        )
        wave = state["vocoder"].decode(generated.to(torch.float32).permute(0, 2, 1))
    if 0.0 < rms < TARGET_RMS:
        wave = wave * rms / TARGET_RMS
    progress(0.9)
    sf.write(msg["out_path"], wave.squeeze().cpu().numpy(), sr, subtype="FLOAT")
    return {"path": msg["out_path"], "sample_rate": sr, "regions": regions}


if __name__ == "__main__":
    _frame_fd = os.dup(1)
    os.dup2(2, 1)
    sys.exit(serve("f5tts", load, edit, os.fdopen(_frame_fd, "wb")))
```

- [ ] **Step 6: Register the `f5tts` install spec**

In `backend/services/sidecar_install.py`, add this helper just above `SPECS: dict[str, SidecarSpec] = {`:

```python
def _venv_installed(env_var: str) -> Callable[[], bool]:
    """Install probe for the Audio Edit engines: venv plus completion marker."""
    def probe() -> bool:
        return engine_venv_python(env_var) is not None
    return probe
```

Add this entry inside `SPECS`, after the `"voxcpm2"` entry:

```python
    # Audio Edit speech engine (services/audio_edit_engines.py), not a TTS
    # engine. A PyPI package; the CC-BY-NC weights download on first render.
    "f5tts": SidecarSpec(
        engine_id="f5tts",
        display_name="F5-TTS (Audio Edit)",
        repo_url="",
        tarball_url="",
        checkout_dirname="f5tts",
        env_var="OMNIVOICE_F5TTS_DIR",
        probe_module="f5_tts",
        has_source=False,
        venv_args=("--python", "3.11"),
        install_args=("f5-tts==1.1.22", "torchcodec==0.5"),
        torch_pins=("torch==2.7.1", "torchaudio==2.7.1"),
        docs_path="docs/engines/f5tts.md",
        # CUDA torch (~5 GB) + f5-tts; ~1.4 GB of weights on first render.
        required_bytes=8 * _GIB,
        installed_probe=_venv_installed("OMNIVOICE_F5TTS_DIR"),
        host_supported=_no_intel_mac("F5-TTS needs a PyTorch version that has no Intel Mac build."),
    ),
```

`Callable` is already imported in `sidecar_install.py`, because `SidecarSpec` uses it. `engine_venv_python` is defined later in the same module, and the probe looks it up only when called.

- [ ] **Step 7: Update the two guards that enumerate sidecars**

In `tests/test_sidecar_stdout_isolation_1428.py`, add one line to `EXPECTED_SIDECARS` after the `cosyvoice_subprocess` line:

```python
    ENGINES / "f5tts" / "main.py",
```

In `tests/test_sidecar_install.py`, add this constant directly above the `@pytest.mark.parametrize` decorator of `test_installable_engine_ids_follow_the_host`:

```python
#: Audio Edit engines: offered everywhere PyTorch still publishes builds.
AUDIO_EDIT_SPEC_IDS = {"f5tts"}
```

Then, inside the test body, replace

```python
    expected = set(expected) | {"indextts2", "confucius4-tts", "supertonic3"}
```

with

```python
    expected = set(expected) | {"indextts2", "confucius4-tts", "supertonic3"}
    if (platform, machine) != ("darwin", "x86_64"):
        expected |= AUDIO_EDIT_SPEC_IDS
```

- [ ] **Step 8: Run the tests to confirm they pass, including the sidecar guards**

Run: `uv run pytest tests/test_audio_edit_engines.py -q`
Expected: `7 passed`.

Run: `uv run pytest tests/test_sidecar_install.py tests/test_model_licenses.py -q`
Expected: PASS. These guards pin several invariants for every spec, including `--python` in `venv_args`, its own venv, non-overlapping roots and the host gates.

Run: `uv run pytest tests/test_sidecar_stdout_isolation_1428.py -q`
Expected: PASS. `f5tts/main.py` contains `_frame_fd = os.dup(1)` before `os.dup2(2, 1)`.

- [ ] **Step 9: Commit**

```powershell
git add backend/engines/_edit_protocol/__init__.py backend/engines/f5tts/__init__.py backend/engines/f5tts/main.py backend/services/audio_edit_engines.py backend/services/sidecar_install.py tests/test_audio_edit_engines.py tests/test_sidecar_install.py tests/test_sidecar_stdout_isolation_1428.py
git commit -m "Audio Edit: edit sidecar protocol and F5-TTS engine"
```

## Task 5: YingMusic and Vevo2 singing sidecars

**Files:**
- Create:
  - `backend/engines/yingmusic/__init__.py` (empty)
  - `backend/engines/yingmusic/main.py`
  - `backend/engines/yingmusic/requirements.txt`
  - `backend/engines/vevo2/__init__.py` (empty)
  - `backend/engines/vevo2/main.py`
  - `backend/engines/vevo2/requirements.txt`
- Modify:
  - `backend/services/sidecar_install.py`: `ExtraFile`, the `SidecarSpec.extra_files` field, `_ensure_extra_files`, and the `yingmusic` and `vevo2` specs
  - `backend/services/audio_edit_engines.py`: `YingMusicEdit`, `Vevo2Edit`, `EDITORS`
  - `tests/test_sidecar_install.py`: `AUDIO_EDIT_SPEC_IDS`
  - `tests/test_sidecar_stdout_isolation_1428.py`: `EXPECTED_SIDECARS`
- Test: `tests/test_audio_edit_song_engines.py`

**Interfaces:**
- Consumes: `EditSidecar` and the wire op from Task 4, plus `SidecarSpec`, `_no_intel_mac`, `_venv_installed`, `_GIB` and `_StepError` from `sidecar_install`.
- Produces:
  - `EDITORS["yingmusic"]` and `EDITORS["vevo2"]`. Each takes `{audio_path, out_path, raw_text, target_text, seed}` and replies `{path, sample_rate}`. The reply covers the whole window, so it has no `regions`.
  - Install ids `yingmusic` and `vevo2`.
  - `sidecar_install.ExtraFile(url, path, sha256)` and `SidecarSpec.extra_files`. A single pinned file is fetched during the weights step and checked against its SHA-256.

Both upstream repositories are pinned source trees with no `pyproject.toml`. Their own requirement lists either pin a torch that the CUDA 12.8 index does not publish (YingMusic pins `torch==2.6.0`) or leave out most of the imports (Vevo2). Each engine therefore ships a VoiceStudio-owned `requirements.txt` holding only the inference import closure, with torch supplied through `torch_pins`. This is the same pattern as `engines/cosyvoice_subprocess/requirements.txt`.

At construction, Vevo2 calls `whisper.load_model("medium")`, and openai-whisper downloads that checkpoint on first use. To keep renders local-first, the installer fetches the checkpoint once into `<checkout>/checkpoints/whisper/medium.pt` as an `ExtraFile`. The sidecar then points `load_model` at that directory and fails instead of downloading.

- [ ] **Step 1: Write the failing tests**

`tests/test_audio_edit_song_engines.py`:

```python
"""YingMusic and Vevo2: specs, editor wiring, and the pinned extra-file fetch."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from services import audio_edit_engines as aee
from services import sidecar_install as si

ENGINES_DIR = Path(__file__).resolve().parents[1] / "backend" / "engines"


def test_song_editors_are_registered():
    assert aee.EDITORS["yingmusic"].sidecar_dir == "yingmusic"
    assert aee.EDITORS["vevo2"].sidecar_dir == "vevo2"
    assert aee.EDITORS["vevo2"].engine_key == "vevo2"
    assert aee.get_edit_engine("yingmusic").display_name == "YingMusic-Singer-Plus"


def test_yingmusic_spec_pins_source_and_weights():
    spec = si.SPECS["yingmusic"]
    assert spec.source_revision == "baa409c2e7e5e775f09b4e92a220808f4827d2cc"
    assert spec.weights_revision == "9b3f444f2bccd77fbc03e32eb7c334fc96040b8d"
    assert spec.weights_config_names == ("config.json",)
    assert spec.source_manifest == "requirements.txt"
    assert spec.install_args[0] == "-r" and spec.install_args[1].endswith("requirements.txt")
    assert Path(spec.install_args[1]).is_file()


def test_vevo2_spec_fetches_whisper_once():
    spec = si.SPECS["vevo2"]
    assert spec.weights_subdir == "checkpoints/Vevo2"
    assert spec.source_required_path == "models/svc/vevo2/vevo2_utils.py"
    (whisper,) = spec.extra_files
    assert whisper.path == "checkpoints/whisper/medium.pt"
    assert whisper.sha256 in whisper.url
    assert not any("optimizer" in p or "rng_state" in p for p in spec.weights_allow_patterns)


@pytest.fixture
def extra_spec(tmp_path, monkeypatch):
    payload = b"whisper-bytes"
    digest = hashlib.sha256(payload).hexdigest()
    spec = si.SidecarSpec(
        engine_id="extra-test", display_name="Extra", repo_url="", tarball_url="",
        checkout_dirname="extra-test", env_var="OMNIVOICE_EXTRA_TEST_DIR", probe_module="x",
        has_source=False, venv_args=("--python", "3.11"),
        extra_files=(si.ExtraFile(f"https://example.invalid/{digest}/m.pt", "ckpt/m.pt", digest),),
    )
    monkeypatch.setattr(si, "managed_checkout", lambda s: tmp_path)
    calls: list[str] = []

    def fake_stream(url: str, dest: Path) -> None:
        calls.append(url)
        dest.write_bytes(fake_stream.body)

    fake_stream.body = payload
    monkeypatch.setattr(si, "_stream_to_file", fake_stream)
    return spec, calls, fake_stream


def test_extra_file_downloads_and_verifies(extra_spec, tmp_path):
    spec, calls, _ = extra_spec
    si._ensure_extra_files(spec, {"log": []})
    assert (tmp_path / "ckpt" / "m.pt").read_bytes() == b"whisper-bytes"
    si._ensure_extra_files(spec, {"log": []})
    assert len(calls) == 1


def test_extra_file_rejects_a_bad_checksum(extra_spec, tmp_path):
    spec, _, fake_stream = extra_spec
    fake_stream.body = b"tampered"
    with pytest.raises(si._StepError, match="checksum"):
        si._ensure_extra_files(spec, {"log": []})
    assert not (tmp_path / "ckpt" / "m.pt").exists()


@pytest.mark.parametrize("name", ["yingmusic", "vevo2"])
def test_song_sidecars_never_download_at_render(name):
    source = (ENGINES_DIR / name / "main.py").read_text(encoding="utf-8")
    assert "snapshot_download" not in source and "hf_hub_download" not in source
    assert "HF_HUB_OFFLINE" in source
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `uv run pytest tests/test_audio_edit_song_engines.py -q`
Expected: FAIL with `KeyError: 'yingmusic'` and `AttributeError: module 'services.sidecar_install' has no attribute 'ExtraFile'`.

- [ ] **Step 3: Add `ExtraFile` and the verified single-file fetch to `sidecar_install.py`**

Below `class ExtraSource(NamedTuple)`, add:

```python
class ExtraFile(NamedTuple):
    """One pinned file an engine loads at run time that upstream would
    otherwise download on first use (Vevo2's Whisper checkpoint). Fetched
    during the weights step and checked against its SHA-256."""

    url: str     # direct download URL
    path: str    # destination, relative to the checkout
    sha256: str  # expected hex digest
```

Append this field as the last field of `SidecarSpec`:

```python
    # Pinned single files fetched with the weights (see ExtraFile).
    extra_files: tuple[ExtraFile, ...] = ()
```

Add these helpers directly above `def _step_fetch_weights`:

```python
def _stream_to_file(url: str, dest: Path) -> None:
    import httpx

    with open(dest, "wb") as out:
        with httpx.stream("GET", url, follow_redirects=True, timeout=_TARBALL_TIMEOUT_S) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_bytes():
                out.write(chunk)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_extra_files(spec: SidecarSpec, job: dict) -> None:
    checkout = managed_checkout(spec)
    for extra in spec.extra_files:
        dest = checkout / extra.path
        if dest.is_file() and _sha256_file(dest) == extra.sha256:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        _log(job, f"Downloading {extra.path} …")
        try:
            _stream_to_file(extra.url, part)
        except Exception as exc:
            part.unlink(missing_ok=True)
            raise _StepError(
                f"Download of {extra.path} failed: {_redact(exc)}",
                "Re-run the install. Check Settings → Network (proxy) if it keeps failing.",
            ) from exc
        if _sha256_file(part) != extra.sha256:
            part.unlink(missing_ok=True)
            raise _StepError(
                f"{extra.path} failed its checksum; the download was corrupted or changed upstream.",
                "Re-run the install; if it keeps failing, see the engine docs.",
            )
        os.replace(part, dest)
```

In `_step_fetch_weights`, make this the first statement after `step = _job_step(job, "fetch_weights")`:

```python
    _ensure_extra_files(spec, job)
```

Add `import hashlib` to the module's stdlib imports, in alphabetical order. The module does not import it yet.

- [ ] **Step 4: Write the YingMusic sidecar**

`backend/engines/yingmusic/requirements.txt` (the inference import closure of `src/YingMusicSinger/infer/YingMusicSinger.py` at `baa409c2`, with versions taken from upstream's own `requirements.txt`; torch comes from `torch_pins`):

```text
alias-free-torch==0.0.6
cn2an==0.5.23
descript-audio-codec==1.0.0
einops==0.8.1
einops-exts==0.0.4
ema-pytorch==0.7.7
encodec==0.1.1
g2p-en==2.1.0
huggingface-hub==0.35.3
hydra-core==1.3.2
jieba==0.42.1
librosa==0.10.2.post1
omegaconf==2.3.0
phonemizer==3.3.0
py3langid==0.3.0
pypinyin==0.55.0
safetensors==0.6.2
soundfile==0.13.1
tokenizers==0.22.2
torchdiffeq==0.2.5
transformers==4.57.6
vector-quantize-pytorch==1.23.2
x-transformers==2.1.2
zhconv==1.4.3
```

`backend/engines/yingmusic/main.py`:

```python
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
```

- [ ] **Step 5: Write the Vevo2 sidecar**

`backend/engines/vevo2/requirements.txt` (the import closure of `models/svc/vevo2/vevo2_utils.py` at Amphion `26f68831`; torch, torchaudio and torchvision come from `torch_pins`):

```text
accelerate==1.1.1
einops==0.8.0
huggingface-hub==0.26.2
ipython==8.29.0
json5==0.9.28
librosa==0.10.2.post1
openai-whisper==20240930
praat-parselmouth==0.4.5
pyworld==0.3.4
pyyaml==6.0.2
ruamel.yaml==0.18.6
safetensors==0.4.5
scipy==1.14.1
six==1.16.0
soundfile==0.12.1
torchcrepe==0.0.24
torchmetrics==1.6.0
tqdm==4.67.1
transformers==4.46.3
```

`backend/engines/vevo2/main.py`:

```python
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
```

- [ ] **Step 6: Register the two specs**

Below `_COSYVOICE_REQUIREMENTS`, add:

```python
_ENGINES_ROOT = Path(__file__).resolve().parents[1] / "engines"
_YINGMUSIC_REQUIREMENTS = str(_ENGINES_ROOT / "yingmusic" / "requirements.txt")
_VEVO2_REQUIREMENTS = str(_ENGINES_ROOT / "vevo2" / "requirements.txt")
_YINGMUSIC_REV = "baa409c2e7e5e775f09b4e92a220808f4827d2cc"
_AMPHION_REV = "26f6883110181f1dbfe95c70a7c7dbaf4de5f42a"
_WHISPER_MEDIUM_SHA = "345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1"
```

Add these entries inside `SPECS`, after `"f5tts"`:

```python
    # Audio Edit singing engines (services/audio_edit_engines.py).
    "yingmusic": SidecarSpec(
        engine_id="yingmusic",
        display_name="YingMusic-Singer-Plus (Audio Edit)",
        repo_url="https://github.com/ASLP-lab/YingMusic-Singer-Plus.git",
        tarball_url=f"https://github.com/ASLP-lab/YingMusic-Singer-Plus/archive/{_YINGMUSIC_REV}.tar.gz",
        checkout_dirname="YingMusic-Singer-Plus",
        env_var="OMNIVOICE_YINGMUSIC_DIR",
        probe_module="src.YingMusicSinger",
        probe_code=(
            "import sys; sys.path.insert(0, {checkout_repr}); "
            "import src.YingMusicSinger.infer.YingMusicSinger"
        ),
        source_revision=_YINGMUSIC_REV,
        source_manifest="requirements.txt",
        source_required_path="src/YingMusicSinger/infer/YingMusicSinger.py",
        venv_args=("--python", "3.10"),
        install_args=("-r", _YINGMUSIC_REQUIREMENTS),
        torch_pins=("torch==2.7.1", "torchaudio==2.7.1"),
        weights_repo_id="ASLP-lab/YingMusic-Singer-Plus",
        weights_revision="9b3f444f2bccd77fbc03e32eb7c334fc96040b8d",
        weights_config_names=("config.json",),
        weights_allow_patterns=("config.json", "model.safetensors"),
        docs_path="docs/engines/yingmusic.md",
        required_bytes=12 * _GIB,
        installed_probe=_venv_installed("OMNIVOICE_YINGMUSIC_DIR"),
        host_supported=_no_intel_mac("YingMusic needs a PyTorch version that has no Intel Mac build."),
    ),
    "vevo2": SidecarSpec(
        engine_id="vevo2",
        display_name="Vevo2 (Audio Edit)",
        repo_url="https://github.com/open-mmlab/Amphion.git",
        tarball_url=f"https://github.com/open-mmlab/Amphion/archive/{_AMPHION_REV}.tar.gz",
        checkout_dirname="Amphion",
        env_var="OMNIVOICE_VEVO2_DIR",
        probe_module="models.svc.vevo2",
        probe_code=(
            "import sys, os; sys.path.insert(0, {checkout_repr}); os.chdir({checkout_repr}); "
            "import models.svc.vevo2.vevo2_utils"
        ),
        source_revision=_AMPHION_REV,
        source_manifest="README.md",
        source_required_path="models/svc/vevo2/vevo2_utils.py",
        venv_args=("--python", "3.10"),
        install_args=("-r", _VEVO2_REQUIREMENTS),
        torch_pins=("torch==2.7.1", "torchaudio==2.7.1", "torchvision==0.22.1"),
        weights_repo_id="RMSnow/Vevo2",
        weights_revision="2674843cbaa50aa89ee7ccaf5bb15d6ccf46c6c8",
        weights_subdir="checkpoints/Vevo2",
        weights_config_names=("vocoder/config.json",),
        weights_allow_patterns=(
            "tokenizer/*",
            "contentstyle_modeling/posttrained/*.json",
            "contentstyle_modeling/posttrained/*.txt",
            "contentstyle_modeling/posttrained/model.safetensors",
            "acoustic_modeling/fm_emilia101k_singnet7k_repa/*",
            "vocoder/*",
        ),
        extra_files=(
            ExtraFile(
                url=("https://openaipublic.azureedge.net/main/whisper/models/"
                     f"{_WHISPER_MEDIUM_SHA}/medium.pt"),
                path="checkpoints/whisper/medium.pt",
                sha256=_WHISPER_MEDIUM_SHA,
            ),
        ),
        docs_path="docs/engines/vevo2.md",
        required_bytes=16 * _GIB,
        installed_probe=_venv_installed("OMNIVOICE_VEVO2_DIR"),
        host_supported=_no_intel_mac("Vevo2 needs a PyTorch version that has no Intel Mac build."),
    ),
```

The Vevo2 snapshot has no top-level `config.json`. `_weights_floor_ok` checks `wdir / name`, so `weights_config_names=("vocoder/config.json",)` is the file that proves the download.

- [ ] **Step 7: Register the editors**

In `backend/services/audio_edit_engines.py`, replace the `EDITORS` line with:

```python
class YingMusicEdit(EditSidecar):
    id = "audio-edit-yingmusic"
    display_name = "YingMusic-Singer-Plus"
    engine_key = "yingmusic"
    sidecar_dir = "yingmusic"


class Vevo2Edit(EditSidecar):
    id = "audio-edit-vevo2"
    display_name = "Vevo2"
    engine_key = "vevo2"
    sidecar_dir = "vevo2"


EDITORS: dict[str, type[EditSidecar]] = {
    "f5tts": F5TTSEdit,
    "yingmusic": YingMusicEdit,
    "vevo2": Vevo2Edit,
}
```

- [ ] **Step 8: Update the sidecar guards**

In `tests/test_sidecar_stdout_isolation_1428.py`, add these lines after `ENGINES / "f5tts" / "main.py",`:

```python
    ENGINES / "yingmusic" / "main.py",
    ENGINES / "vevo2" / "main.py",
```

In `tests/test_sidecar_install.py`, change the constant to:

```python
AUDIO_EDIT_SPEC_IDS = {"f5tts", "yingmusic", "vevo2"}
```

- [ ] **Step 9: Run the tests to confirm they pass**

Run: `uv run pytest tests/test_audio_edit_song_engines.py tests/test_audio_edit_engines.py -q`
Expected: `14 passed` (7 new tests plus the 7 from Task 4).

Run: `uv run pytest tests/test_sidecar_install.py tests/test_sidecar_stdout_isolation_1428.py tests/test_model_licenses.py -q`
Expected: PASS.

- [ ] **Step 10: Commit**

```powershell
git add backend/engines/yingmusic backend/engines/vevo2 backend/services/sidecar_install.py backend/services/audio_edit_engines.py tests/test_audio_edit_song_engines.py tests/test_sidecar_install.py tests/test_sidecar_stdout_isolation_1428.py
git commit -m "Audio Edit: YingMusic and Vevo2 singing sidecars"
```

## Task 6: ACE-Step 1.5 HTTP client and install spec

**Files:**
- Create: `backend/services/acestep_client.py`
- Modify:
  - `backend/services/sidecar_install.py`: the `ace-step` spec
  - `tests/test_sidecar_install.py`: `AUDIO_EDIT_SPEC_IDS`
- Test: `tests/test_acestep_client.py`

**Interfaces:**
- Consumes: `audio_edit_engines.engine_python("ace_step")` and `engine_checkout("ace_step")`.
- Produces: `services.acestep_client`, which exposes the following.
  - `STARTUP_TIMEOUT_S = 60.0`, `POLL_INTERVAL_S`, `REPAINT_TIMEOUT_S = 900.0`.
  - `AceStepUnavailable(RuntimeError)` with `.code = "ace_step_unavailable"`.
  - `AceStepError(RuntimeError)` and `AceStepCancelled(RuntimeError)`.
  - `is_loopback_url(url) -> bool`.
  - `AceStepClient(work_dir: Path)` with:
    - `.ensure_running() -> str` (the base URL)
    - `.repaint(src_path, start_s, end_s, lyrics, caption, strength, seed, out_path, language="en") -> Path`
    - `.cancel()`
    - `.stop()`
    - `.started_server` (bool)
  - Install id `ace-step`.

ACE-Step is not a stdio sidecar. Its upstream ships a FastAPI server (`acestep.api_server`). Audio Edit starts that server from the ACE-Step venv on `127.0.0.1` at a free port, waits for `GET /health`, and posts a `repaint` task as multipart with the source file attached. That way the server never has to read paths outside its own temp directory. The client polls `POST /query_result` and downloads the file through `GET /v1/audio`, then stops the server only if it started it. Setting `OMNIVOICE_ACESTEP_URL` reuses a server the user runs themselves. Only loopback URLs are accepted, so no audio ever leaves the machine.

- [ ] **Step 1: Write the failing tests**

`tests/test_acestep_client.py`:

```python
"""ACE-Step client against a fake acestep-api server on loopback."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from services import acestep_client as ac


class FakeAce(BaseHTTPRequestHandler):
    healthy = True
    fail = False
    polls = 0
    last_form = b""

    def log_message(self, *args):
        pass

    def _json(self, data, status=200):
        body = json.dumps({"data": data, "code": status, "error": None,
                           "timestamp": 0, "extra": None}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            if FakeAce.healthy:
                self._json({"status": "ok"})
            else:
                self._json({"status": "starting"}, status=503)
        elif self.path.startswith("/v1/audio"):
            body = b"RIFFrepainted"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        if self.path == "/release_task":
            FakeAce.last_form = raw
            self._json({"task_id": "t1", "status": "queued"})
        elif self.path == "/query_result":
            assert json.loads(raw) == {"task_id_list": ["t1"]}
            FakeAce.polls += 1
            if FakeAce.fail:
                self._json([{"task_id": "t1", "status": 2, "result": "[]", "error": "cuda oom"}])
            elif FakeAce.polls < 2:
                self._json([{"task_id": "t1", "status": 0, "result": "[]"}])
            else:
                result = json.dumps([{"file": "/v1/audio?path=%2Ftmp%2Fout.wav", "status": 1}])
                self._json([{"task_id": "t1", "status": 1, "result": result}])
        else:
            self.send_error(404)


@pytest.fixture
def fake_server(monkeypatch):
    FakeAce.healthy, FakeAce.fail, FakeAce.polls, FakeAce.last_form = True, False, 0, b""
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeAce)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OMNIVOICE_ACESTEP_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(ac, "POLL_INTERVAL_S", 0.01)
    yield server
    server.shutdown()


def test_loopback_only():
    assert ac.is_loopback_url("http://127.0.0.1:8001")
    assert ac.is_loopback_url("http://localhost:8001/")
    assert not ac.is_loopback_url("http://192.168.1.4:8001")
    assert not ac.is_loopback_url("https://api.example.com")


def test_remote_url_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("OMNIVOICE_ACESTEP_URL", "http://10.0.0.5:8001")
    with pytest.raises(ac.AceStepUnavailable, match="this computer"):
        ac.AceStepClient(tmp_path).ensure_running()


def test_repaint_round_trip(fake_server, tmp_path):
    src = tmp_path / "mix.wav"
    src.write_bytes(b"RIFFmix")
    client = ac.AceStepClient(tmp_path)
    out = client.repaint(src, 12.5, 18.25, "[verse]\nnew words", "pop, female vocal",
                         0.5, 7, tmp_path / "out.wav")
    assert out.read_bytes() == b"RIFFrepainted"
    form = FakeAce.last_form.decode("latin-1")
    for field, value in [("task_type", "repaint"), ("repainting_start", "12.500"),
                         ("repainting_end", "18.250"), ("chunk_mask_mode", "explicit"),
                         ("batch_size", "1"), ("use_random_seed", "false"), ("seed", "7"),
                         ("audio_format", "wav"), ("audio_cover_strength", "0.5")]:
        assert f'name="{field}"\r\n\r\n{value}\r\n' in form
    assert 'name="src_audio"; filename="mix.wav"' in form
    assert client.started_server is False


def test_failed_task_raises(fake_server, tmp_path):
    FakeAce.fail = True
    src = tmp_path / "mix.wav"
    src.write_bytes(b"x")
    with pytest.raises(ac.AceStepError, match="cuda oom"):
        ac.AceStepClient(tmp_path).repaint(src, 0, 1, "", "", 0.5, 1, tmp_path / "o.wav")


def test_unhealthy_server_times_out(fake_server, monkeypatch, tmp_path):
    FakeAce.healthy = False
    monkeypatch.setattr(ac, "STARTUP_TIMEOUT_S", 0.3)
    with pytest.raises(ac.AceStepUnavailable, match="did not answer"):
        ac.AceStepClient(tmp_path).ensure_running()


def test_cancel_stops_polling(fake_server, tmp_path):
    src = tmp_path / "mix.wav"
    src.write_bytes(b"x")
    client = ac.AceStepClient(tmp_path)
    client.cancel()
    with pytest.raises(ac.AceStepCancelled):
        client.repaint(src, 0, 1, "", "", 0.5, 1, tmp_path / "o.wav")


def test_spawns_local_server_with_safe_env(monkeypatch, tmp_path):
    monkeypatch.delenv("OMNIVOICE_ACESTEP_URL", raising=False)
    monkeypatch.setenv("ACESTEP_API_KEY", "secret")
    monkeypatch.setattr(ac, "engine_python", lambda engine_id: tmp_path / "python")
    monkeypatch.setattr(ac, "engine_checkout", lambda engine_id: tmp_path / "ace")
    seen = {}

    class FakeProc:
        def poll(self):
            return None

        def terminate(self):
            seen["terminated"] = True

        def wait(self, timeout=None):
            return 0

    def fake_popen(cmd, **kwargs):
        seen.update(cmd=cmd, **kwargs)
        return FakeProc()

    monkeypatch.setattr(ac.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(ac.AceStepClient, "_wait_healthy", lambda self: None)
    client = ac.AceStepClient(tmp_path)
    base = client.ensure_running()
    env = seen["env"]
    assert seen["cmd"] == [str(tmp_path / "python"), "-m", "acestep.api_server"]
    assert seen["cwd"] == str(tmp_path / "ace")
    assert env["ACESTEP_API_HOST"] == "127.0.0.1" and base.endswith(env["ACESTEP_API_PORT"])
    assert env["ACESTEP_INIT_LLM"] == "false" and env["HF_HUB_OFFLINE"] == "1"
    assert "ACESTEP_API_KEY" not in env
    assert client.started_server is True
    client.stop()
    assert seen["terminated"] is True and client.started_server is False
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `uv run pytest tests/test_acestep_client.py -q`
Expected: FAIL with `ImportError: cannot import name 'acestep_client' from 'services'`.

- [ ] **Step 3: Write `backend/services/acestep_client.py`**

```python
"""Client for ACE-Step 1.5's local API server (Audio Edit song repaint).

Starts ``python -m acestep.api_server`` from the ACE-Step venv on
127.0.0.1 at a free port (or reuses a loopback server named by
OMNIVOICE_ACESTEP_URL), waits for /health, posts one ``repaint`` task
with the source mix uploaded, polls /query_result and downloads the
result from /v1/audio. Nothing here talks to anything but loopback.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx

from services.audio_edit_engines import engine_checkout, engine_python

STARTUP_TIMEOUT_S = 60.0
POLL_INTERVAL_S = 1.0
REPAINT_TIMEOUT_S = 900.0
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class AceStepUnavailable(RuntimeError):
    code = "ace_step_unavailable"


class AceStepError(RuntimeError):
    """The server accepted the task but it failed."""


class AceStepCancelled(RuntimeError):
    """The render was cancelled while ACE-Step was working."""


def is_loopback_url(url: str) -> bool:
    try:
        host = urlparse(url).hostname
    except ValueError:
        return False
    return host in _LOOPBACK_HOSTS


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _data(resp: httpx.Response):
    resp.raise_for_status()
    body = resp.json()
    if isinstance(body, dict) and "data" in body:
        return body["data"]
    return body


class AceStepClient:
    def __init__(self, work_dir: Path) -> None:
        self._work_dir = Path(work_dir)
        self._proc: Optional[subprocess.Popen] = None
        self._log = None
        self._base: Optional[str] = None
        self._cancelled = threading.Event()

    @property
    def started_server(self) -> bool:
        return self._proc is not None

    def ensure_running(self) -> str:
        url = os.environ.get("OMNIVOICE_ACESTEP_URL", "").strip()
        if url:
            if not is_loopback_url(url):
                raise AceStepUnavailable(
                    "OMNIVOICE_ACESTEP_URL must point at this computer (127.0.0.1 or localhost)."
                )
            self._base = url.rstrip("/")
        elif self._proc is None or self._proc.poll() is not None:
            self._base = self._spawn()
        self._wait_healthy()
        return self._base

    def _spawn(self) -> str:
        py, checkout = engine_python("ace_step"), engine_checkout("ace_step")
        if py is None or checkout is None:
            raise AceStepUnavailable("Install ACE-Step 1.5 to use it.")
        port = _free_port()
        tmp = self._work_dir / "acestep-tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            "ACESTEP_API_HOST": "127.0.0.1",
            "ACESTEP_API_PORT": str(port),
            "ACESTEP_INIT_LLM": "false",
            "ACESTEP_TMPDIR": str(tmp),
            "HF_HUB_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
        }
        env.pop("ACESTEP_API_KEY", None)
        self._log = open(self._work_dir / "acestep-server.log", "ab")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        self._proc = subprocess.Popen(
            [str(py), "-m", "acestep.api_server"],
            cwd=str(checkout), env=env, stdin=subprocess.DEVNULL,
            stdout=self._log, stderr=subprocess.STDOUT, creationflags=flags,
        )
        return f"http://127.0.0.1:{port}"

    def _wait_healthy(self) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            self._check_cancel()
            if self._proc is not None and self._proc.poll() is not None:
                raise AceStepUnavailable(
                    "ACE-Step stopped while starting. See acestep-server.log in the session folder."
                )
            try:
                resp = httpx.get(f"{self._base}/health", timeout=2.0)
                if resp.status_code == 200 and (_data(resp) or {}).get("status") == "ok":
                    return
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(0.25)
        raise AceStepUnavailable(f"ACE-Step did not answer within {int(STARTUP_TIMEOUT_S)} s.")

    def _check_cancel(self) -> None:
        if self._cancelled.is_set():
            raise AceStepCancelled("cancelled")

    def repaint(self, src_path: Path, start_s: float, end_s: float, lyrics: str,
                caption: str, strength: float, seed: int, out_path: Path,
                language: str = "en") -> Path:
        self._check_cancel()
        base = self.ensure_running()
        form = {
            "task_type": "repaint",
            "repainting_start": f"{start_s:.3f}",
            "repainting_end": f"{end_s:.3f}",
            "lyrics": lyrics,
            "prompt": caption,
            "vocal_language": language,
            "audio_cover_strength": f"{strength:g}",
            "chunk_mask_mode": "explicit",
            "use_random_seed": "false",
            "seed": str(int(seed)),
            "batch_size": "1",
            "inference_steps": "8",
            "thinking": "false",
            "use_cot_caption": "false",
            "use_cot_language": "false",
            "audio_format": "wav",
        }
        src_path = Path(src_path)
        with open(src_path, "rb") as fh:
            resp = httpx.post(f"{base}/release_task", data=form,
                              files={"src_audio": (src_path.name, fh, "audio/wav")}, timeout=60.0)
        task_id = (_data(resp) or {}).get("task_id")
        if not task_id:
            raise AceStepError("ACE-Step did not accept the repaint task.")
        file_url = self._poll(base, task_id)
        out_path = Path(out_path)
        with httpx.stream("GET", f"{base}{file_url}", timeout=120.0) as dl:
            dl.raise_for_status()
            with open(out_path, "wb") as out:
                for chunk in dl.iter_bytes():
                    out.write(chunk)
        return out_path

    def _poll(self, base: str, task_id: str) -> str:
        deadline = time.monotonic() + REPAINT_TIMEOUT_S
        while time.monotonic() < deadline:
            self._check_cancel()
            if self._proc is not None and self._proc.poll() is not None:
                raise AceStepError("ACE-Step stopped during the repaint.")
            items = _data(httpx.post(f"{base}/query_result",
                                     json={"task_id_list": [task_id]}, timeout=30.0)) or []
            item = items[0] if items else {}
            status = item.get("status")
            if status == 2:
                raise AceStepError(f"ACE-Step repaint failed: {item.get('error') or 'unknown error'}")
            if status == 1:
                result = item.get("result")
                files = json.loads(result) if isinstance(result, str) else (result or [])
                if not files or not files[0].get("file"):
                    raise AceStepError("ACE-Step returned no audio.")
                return str(files[0]["file"])
            time.sleep(POLL_INTERVAL_S)
        raise AceStepError("ACE-Step repaint timed out.")

    def cancel(self) -> None:
        self._cancelled.set()

    def stop(self) -> None:
        """Stop the server only if this client started it."""
        proc, self._proc = self._proc, None
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        if self._log is not None:
            self._log.close()
            self._log = None
```

- [ ] **Step 4: Register the `ace-step` install spec**

Below the Task 5 constants in `sidecar_install.py`, add:

```python
_ACESTEP_REV = "ca1e85fe9430179831e6bc6be790c332190a3866"
```

Add this entry inside `SPECS`, after `"vevo2"`:

```python
    # Audio Edit song repaint (services/acestep_client.py). Upstream's own
    # pyproject selects the CUDA torch per platform through [tool.uv.sources].
    "ace-step": SidecarSpec(
        engine_id="ace-step",
        display_name="ACE-Step 1.5 (Audio Edit)",
        repo_url="https://github.com/ace-step/ACE-Step-1.5.git",
        tarball_url=f"https://github.com/ace-step/ACE-Step-1.5/archive/{_ACESTEP_REV}.tar.gz",
        checkout_dirname="ACE-Step-1.5",
        env_var="OMNIVOICE_ACESTEP_DIR",
        probe_module="acestep",
        source_revision=_ACESTEP_REV,
        source_required_path="acestep/api_server.py",
        venv_args=("--python", "3.11"),
        install_args=("-e", "{checkout}"),
        weights_repo_id="ACE-Step/Ace-Step1.5",
        weights_revision="19671f406d603126926c1b7e2adc169acbcade22",
        weights_config_names=("config.json",),
        weights_allow_patterns=(
            "config.json",
            "acestep-v15-turbo/*",
            "vae/*",
            "Qwen3-Embedding-0.6B/*",
        ),
        docs_path="docs/engines/ace-step.md",
        required_bytes=20 * _GIB,
        installed_probe=_venv_installed("OMNIVOICE_ACESTEP_DIR"),
        host_supported=_no_intel_mac("ACE-Step 1.5 has no Intel Mac PyTorch build."),
    ),
```

In `tests/test_sidecar_install.py`, change the constant to:

```python
AUDIO_EDIT_SPEC_IDS = {"f5tts", "yingmusic", "vevo2", "ace-step"}
```

Then add this test to `tests/test_acestep_client.py`:

```python
def test_ace_step_install_spec():
    from services.sidecar_install import SPECS

    spec = SPECS["ace-step"]
    assert spec.env_var == "OMNIVOICE_ACESTEP_DIR" and spec.venv_args == ("--python", "3.11")
    assert not any(p.startswith("acestep-5Hz-lm") for p in spec.weights_allow_patterns)
    assert spec.weights_revision == "19671f406d603126926c1b7e2adc169acbcade22"
```

- [ ] **Step 5: Run the tests to confirm they pass**

Run: `uv run pytest tests/test_acestep_client.py -q`
Expected: `8 passed`.

Run: `uv run pytest tests/test_sidecar_install.py tests/test_model_licenses.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add backend/services/acestep_client.py backend/services/sidecar_install.py tests/test_acestep_client.py tests/test_sidecar_install.py
git commit -m "Audio Edit: ACE-Step 1.5 client and install spec"
```

## Task 7: Licence acknowledgements, commercial-use flag and engine overview

**Files:**
- Modify: `backend/services/audio_edit.py` (append the licence section)
- Test: `tests/test_audio_edit_service.py` (append)

**Interfaces:**
- Consumes:
  - `core.prefs.get/set_`
  - `services.model_licenses.load_registry()` and the Task 1 records
  - `audio_edit_engines.ENGINE_MODELS`, `DISPLAY_NAMES`, `engine_status`, `has_gpu`
  - `dataset_builder.capabilities()["separation"]`
- Produces, used by Tasks 9 to 11:
  - Seams `_read_acks()` / `_write_acks(acks)` over the pref `audio_edit.licence_acks` (`{engine: accepted_at}`)
  - `licence_acks() -> dict[str, float]`, `acknowledge_licence(engine) -> {engine, required, accepted_at}`, `revoke_licence(engine)`, `require_licence(engine)` (raises 403 `licence_ack_required`)
  - `HTDEMUCS_RECORD`, `RVC_RECORD`, `_registry_records()`, `render_commercial_use(engine, *, rvc, demucs, records=None) -> bool`
  - `separation_status() -> {available, weights, missing}`
  - `engines_overview() -> {engines: [...], separation, gpu}`

A record counts towards "Commercial OK" only when it says `commercial_use: true` **and** `review_status: "cleared"`. That is the same rule the licence checker enforces, so a stray `true` can never light the badge. A render is commercial only if every record its output depends on passes: the engine's models, HTDemucs for any song, and the RVC dynamic asset when RVC ran.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_audio_edit_service.py`)

```python
# ── Task 7: licences and commercial use ─────────────────────────────────

@pytest.fixture
def acks(monkeypatch):
    saved: dict = {}
    monkeypatch.setattr(ae, "_read_acks", lambda: dict(saved))
    monkeypatch.setattr(ae, "_write_acks", lambda value: (saved.clear(), saved.update(value)))
    return saved


def test_ack_and_revoke(acks):
    assert ae.licence_acks() == {}
    with pytest.raises(ae.AudioEditError) as missing:
        ae.require_licence("f5tts")
    assert missing.value.code == "licence_ack_required" and missing.value.status == 403
    out = ae.acknowledge_licence("f5tts")
    assert out["required"] is True and acks["f5tts"] == out["accepted_at"]
    ae.require_licence("f5tts")
    assert ae.revoke_licence("f5tts")["accepted_at"] is None and "f5tts" not in acks
    assert ae.acknowledge_licence("ace_step") == {"engine": "ace_step", "required": False, "accepted_at": None}
    ae.require_licence("ace_step")
    with pytest.raises(ae.AudioEditError) as bad:
        ae.acknowledge_licence("nope")
    assert bad.value.code == "invalid_engine" and bad.value.status == 422


def test_licence_acks_ignore_junk(acks):
    acks.update({"f5tts": 5, "ace_step": 3, "vevo2": "yes", "other": 1, "yingmusic": True})
    assert ae.licence_acks() == {"f5tts": 5.0}


def test_commercial_use_is_the_and_of_every_record():
    cleared = {"commercial_use": True, "review_status": "cleared"}
    records = {"ACE-Step/Ace-Step1.5": dict(cleared), ae.HTDEMUCS_RECORD: dict(cleared),
               ae.RVC_RECORD: {"commercial_use": False, "review_status": "unreviewed"}}
    assert ae.render_commercial_use("ace_step", rvc=False, demucs=True, records=records) is True
    assert ae.render_commercial_use("ace_step", rvc=True, demucs=True, records=records) is False
    records["ACE-Step/Ace-Step1.5"]["review_status"] = "unreviewed"
    assert ae.render_commercial_use("ace_step", rvc=False, demucs=True, records=records) is False
    assert ae.render_commercial_use("f5tts", rvc=False, demucs=False, records={}) is False


def test_real_registry_badges_every_engine_non_commercial():
    records = ae._registry_records()
    assert ae.HTDEMUCS_RECORD in records and ae.RVC_RECORD in records
    for engine in ae.ENGINES:
        assert ae.render_commercial_use(engine, rvc=False, demucs=False, records=records) is False


def test_engines_overview(acks, monkeypatch):
    from services import audio_edit_engines as aee

    monkeypatch.setattr(aee, "engine_status", lambda e: {
        "installed": True, "install_id": e, "needs_gpu": False, "available": True, "reason": None})
    monkeypatch.setattr(aee, "has_gpu", lambda: False)
    monkeypatch.setattr(ae, "separation_status", lambda: {
        "available": False, "weights": None, "missing": "adefossez/HTDemucs (~84 MB)"})
    acks["vevo2"] = 9.0
    data = ae.engines_overview()
    by_id = {e["id"]: e for e in data["engines"]}
    assert list(by_id) == list(ae.ENGINES)
    assert by_id["f5tts"]["mode"] == "speech" and by_id["vevo2"]["mode"] == "song"
    assert by_id["f5tts"]["display_name"] == "F5-TTS" and by_id["f5tts"]["licence"]
    assert by_id["vevo2"]["acked_at"] == 9.0 and by_id["vevo2"]["needs_ack"] is True
    assert by_id["ace_step"]["needs_ack"] is False and by_id["ace_step"]["acked_at"] is None
    assert all(e["commercial_use"] is False and e["available"] for e in data["engines"])
    assert data["separation"]["available"] is False and data["gpu"] is False
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `uv run pytest tests/test_audio_edit_service.py -q -k "ack or commercial or overview or registry"`
Expected: FAIL with `AttributeError: module 'services.audio_edit' has no attribute '_read_acks'`.

- [ ] **Step 3: Implement** (append to `backend/services/audio_edit.py`)

```python
# ── licences and commercial use ──────────────────────────────────────────

_ACK_PREF = "audio_edit.licence_acks"
#: Licence-registry ids a render depends on beyond the engine's own models.
HTDEMUCS_RECORD = "htdemucs"
RVC_RECORD = ("User-provided Sherpa ONNX / RVC voice models / GPT-SoVITS server / "
              "custom MLX and env-overridden repos")


def _read_acks() -> dict:
    from core import prefs

    value = prefs.get(_ACK_PREF, {})
    return dict(value) if isinstance(value, dict) else {}


def _write_acks(acks: dict) -> None:
    from core import prefs

    prefs.set_(_ACK_PREF, dict(acks))


def _check_engine(engine: object) -> str:
    if engine not in ENGINES:
        raise AudioEditError("invalid_engine", f"Engine must be one of: {', '.join(ENGINES)}.", 422)
    return str(engine)


def licence_acks() -> dict[str, float]:
    """Accepted non-commercial licences: ``{engine: accepted_at}``."""
    return {k: float(v) for k, v in _read_acks().items()
            if k in ACK_ENGINES and isinstance(v, (int, float)) and not isinstance(v, bool)}


def acknowledge_licence(engine: str) -> dict:
    _check_engine(engine)
    if engine not in ACK_ENGINES:
        return {"engine": engine, "required": False, "accepted_at": None}
    with _STATE_LOCK:
        acks = licence_acks()
        acks[engine] = time.time()
        _write_acks(acks)
    return {"engine": engine, "required": True, "accepted_at": acks[engine]}


def revoke_licence(engine: str) -> dict:
    _check_engine(engine)
    with _STATE_LOCK:
        acks = licence_acks()
        acks.pop(engine, None)
        _write_acks(acks)
    return {"engine": engine, "required": engine in ACK_ENGINES, "accepted_at": None}


def require_licence(engine: str) -> None:
    if engine in ACK_ENGINES and engine not in licence_acks():
        from services.audio_edit_engines import DISPLAY_NAMES

        raise AudioEditError("licence_ack_required",
                             f"Acknowledge the {DISPLAY_NAMES[engine]} licence before rendering with it.", 403)


def _registry_records() -> dict[str, dict]:
    from services.model_licenses import load_registry

    registry = load_registry()
    records: dict[str, dict] = {}
    for section in ("models", "dynamic_assets"):
        for record in registry.get(section) or []:
            if isinstance(record, dict) and isinstance(record.get("id"), str):
                records[record["id"]] = record
    return records


def _cleared(record: Optional[dict]) -> bool:
    return bool(record) and record.get("commercial_use") is True and record.get("review_status") == "cleared"


def render_commercial_use(engine: str, *, rvc: bool, demucs: bool,
                          records: Optional[dict] = None) -> bool:
    """True only if every licence record behind the output allows commercial use."""
    from services.audio_edit_engines import ENGINE_MODELS

    records = _registry_records() if records is None else records
    ids = list(ENGINE_MODELS[engine])
    if demucs:
        ids.append(HTDEMUCS_RECORD)
    if rvc:
        ids.append(RVC_RECORD)
    return all(_cleared(records.get(i)) for i in ids)


def separation_status() -> dict:
    """HTDemucs readiness, shared with Voice Lab's dataset builder."""
    from services import dataset_builder

    sep = dataset_builder.capabilities()["separation"]
    return {"available": bool(sep.get("available")), "weights": sep.get("weights"),
            "missing": sep.get("missing")}


def engines_overview() -> dict:
    from services import audio_edit_engines as aee

    records = _registry_records()
    acks = licence_acks()
    engines = []
    for engine in ENGINES:
        mode = "speech" if engine in MODE_ENGINES["speech"] else "song"
        licence = " / ".join(dict.fromkeys(
            str((records.get(i) or {}).get("license") or "unknown") for i in aee.ENGINE_MODELS[engine]))
        engines.append({
            "id": engine, "mode": mode, "display_name": aee.DISPLAY_NAMES[engine],
            **aee.engine_status(engine),
            "licence": licence, "needs_ack": engine in ACK_ENGINES, "acked_at": acks.get(engine),
            "commercial_use": render_commercial_use(engine, rvc=False, demucs=mode == "song", records=records),
        })
    return {"engines": engines, "separation": separation_status(), "gpu": aee.has_gpu()}
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run pytest tests/test_audio_edit_service.py -q`
Expected: `27 passed` (22 from Task 3 plus 5 new).

- [ ] **Step 5: Commit**

```powershell
git add backend/services/audio_edit.py tests/test_audio_edit_service.py
git commit -m "Audio Edit: licence acknowledgements and commercial-use flag"
```

## Task 8: Job runner, stem separation and transcription

**Files:**
- Modify: `backend/services/audio_edit.py` (one import, then append the runner section)
- Test: `tests/test_audio_edit_service.py` (append)

**Interfaces:**
- Consumes:
  - `services.gpu_memory.release_gpu_holders(*, reason)`
  - `services.model_manager.free_vram()`
  - `services.dataset_builder.load_htdemucs(path, device)` and `demucs.apply.apply_model`
  - `services.asr_backend.get_active_asr_backend().transcribe(path, word_timestamps=True)`
  - The Task 3 store and the Task 7 `separation_status()`
- Produces, used by Tasks 9 to 11:
  - `Cancelled`, `_RENDER_SLOT`, `_new_job(session_id, kind, render_id=None)`, `_launch(job, work)`, `_run(job, work)`, `_finish(job)`, `_check_cancel(job)`
  - `cancel_job(job_id) -> EditJob`
  - Seams: `_free_vram()`, `_release_gpu()`, `_asr_transcribe(path) -> dict`
  - `separate_stems(path) -> (vocals (1, n), instrumental (2, n))` at `SAMPLE_RATE`
  - `start_separation(session_id) -> EditJob` and `start_transcription(session_id) -> EditJob`

How the runner works:
- Every job is registered in `_JOBS` **before** the session's stage changes. That way `load_session`'s interrupted-stage sweep never sees a stage without its job.
- Jobs then run on a daemon thread, or inline when `RUN_INLINE` is set (tests).
- They queue on one GPU slot (`_RENDER_SLOT`), so a render, a separation and a transcription never share the card. A job cancelled while it waits for the slot never starts.
- `job.engine` holds a zero-argument kill callable while an engine call is in flight. `cancel_job` sets the event and calls it, so a stuck sidecar or ACE-Step server stops at once.
- When a separate or transcribe job fails or is cancelled, the session stage reverts and `error` is set to the job's code. A render job's outcome is written into `render.json`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_audio_edit_service.py`)

```python
# ── Task 8: runner, separation, transcription ───────────────────────────

@pytest.fixture(autouse=True)
def _no_vram_calls(monkeypatch):
    monkeypatch.setattr(ae, "_free_vram", lambda: None)


def test_transcribe_speech_session(store, monkeypatch):
    calls = []
    monkeypatch.setattr(ae, "_release_gpu", lambda: calls.append("release"))
    monkeypatch.setattr(ae, "_asr_transcribe", lambda path: calls.append(Path(path).name) or {
        "language": "en", "segments": [{"words": [{"word": "hi", "start": 0.1, "end": 0.4}]}]})
    sid = ae.create_session("speech", "", _upload(store), "a.wav")["id"]
    job = ae.start_transcription(sid)
    assert job.status == "succeeded" and job.kind == "transcribe" and job.progress == 1.0
    loaded = ae.load_session(sid)
    assert loaded["stage"] == "ready" and loaded["error"] is None
    assert loaded["transcript"] == {"language": "en", "text": "hi",
                                    "words": [{"i": 0, "word": "hi", "start": 0.1, "end": 0.4}]}
    assert calls == ["release", "source.wav"]
    assert (ae.session_dir(sid) / "transcript.json").is_file()


def test_transcribe_without_word_timings_reverts_stage(store, monkeypatch):
    monkeypatch.setattr(ae, "_release_gpu", lambda: None)
    monkeypatch.setattr(ae, "_asr_transcribe", lambda path: {"segments": [{"text": "hello"}]})
    sid = ae.create_session("speech", "", _upload(store), "a.wav")["id"]
    job = ae.start_transcription(sid)
    assert job.status == "failed" and job.error == "no_word_timestamps"
    loaded = ae.load_session(sid)
    assert loaded["stage"] == "uploaded" and loaded["error"] == "no_word_timestamps"


def test_song_separation_then_transcription(store, monkeypatch):
    monkeypatch.setattr(ae, "_release_gpu", lambda: None)

    def fake_separate(path):
        audio, _ = ae.read_audio(path)
        return audio.mean(axis=0, keepdims=True) * 0.5, audio * 0.5

    seen = []
    monkeypatch.setattr(ae, "separate_stems", fake_separate)
    monkeypatch.setattr(ae, "_asr_transcribe", lambda path: seen.append(Path(path).name) or {"words": WORDS})
    sid = ae.create_session("song", "", _upload(store, channels=2), "s.wav")["id"]
    with pytest.raises(ae.AudioEditError) as early:
        ae.start_transcription(sid)
    assert early.value.code == "not_ready" and early.value.status == 409
    assert ae.start_separation(sid).status == "succeeded"
    assert ae.load_session(sid)["stage"] == "separated"
    assert ae.read_audio(ae.audio_path(sid, "vocals"))[0].shape[0] == 1
    assert ae.read_audio(ae.audio_path(sid, "instrumental"))[0].shape[0] == 2
    assert ae.start_transcription(sid).status == "succeeded"
    assert seen == ["vocals.wav"] and ae.load_session(sid)["stage"] == "ready"


def test_separation_is_for_uploaded_songs_only(store):
    sid = ae.create_session("speech", "", _upload(store), "a.wav")["id"]
    with pytest.raises(ae.AudioEditError) as err:
        ae.start_separation(sid)
    assert err.value.code == "not_ready" and err.value.status == 409


def test_separation_unavailable_fails_job(store, monkeypatch):
    monkeypatch.setattr(ae, "separation_status", lambda: {
        "available": False, "weights": None, "missing": "adefossez/HTDemucs (~84 MB)"})
    sid = ae.create_session("song", "", _upload(store, channels=2), "s.wav")["id"]
    job = ae.start_separation(sid)
    assert job.status == "failed" and job.error == "separation_unavailable"
    assert "HTDemucs" in job.message
    assert ae.load_session(sid)["stage"] == "uploaded"


def test_busy_session_refuses_a_second_job(store):
    sid = ae.create_session("speech", "", _upload(store), "a.wav")["id"]
    ae._new_job(sid, "transcribe").status = "running"
    with pytest.raises(ae.AudioEditError) as err:
        ae.start_transcription(sid)
    assert err.value.code == "busy" and err.value.status == 409


def test_cancel_kills_the_engine_of_live_jobs_only(store):
    sid = ae.create_session("speech", "", _upload(store), "a.wav")["id"]
    killed = []
    live = ae._new_job(sid, "render", render_id="r" * 32)
    live.status, live.engine = "running", lambda: killed.append("live")
    assert ae.cancel_job(live.id).cancel.is_set() and killed == ["live"]
    done = ae._new_job(sid, "transcribe")
    done.status, done.engine = "succeeded", lambda: killed.append("done")
    assert not ae.cancel_job(done.id).cancel.is_set() and killed == ["live"]


def test_job_cancelled_while_waiting_for_the_gpu_slot_never_runs(store):
    sid = ae.create_session("speech", "", _upload(store), "a.wav")["id"]
    assert ae._RENDER_SLOT.acquire(timeout=1)
    try:
        job = ae._new_job(sid, "transcribe")
        job.cancel.set()
        ae._run(job, lambda j: pytest.fail("a cancelled job must not run"))
    finally:
        ae._RENDER_SLOT.release()
    assert job.status == "cancelled"
    loaded = ae.load_session(sid)
    assert loaded["stage"] == "uploaded" and loaded["error"] == "cancelled"
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `uv run pytest tests/test_audio_edit_service.py -q -k "transcri or separat or busy or cancel"`
Expected: FAIL with `AttributeError: module 'services.audio_edit' has no attribute '_free_vram'`.

- [ ] **Step 3: Implement**

In `backend/services/audio_edit.py`, replace the import line `from services.audio_edit_dsp import as_2d` with:

```python
from services import audio_edit_dsp as dsp
from services.audio_edit_dsp import as_2d
```

Append:

```python
# ── job runner ───────────────────────────────────────────────────────────

#: One Audio Edit job holds the GPU at a time; the rest wait their turn.
_RENDER_SLOT = threading.Semaphore(1)


class Cancelled(Exception):
    """The user cancelled the job."""


def _check_cancel(job: EditJob) -> None:
    if job.cancel.is_set():
        raise Cancelled()


def _free_vram() -> None:
    try:
        from services.model_manager import free_vram

        free_vram()
    except Exception:  # noqa: BLE001 — best effort after a job
        logger.debug("free_vram after an Audio Edit job failed", exc_info=True)


def _release_gpu() -> None:
    """Unload models other workspaces hold before an Audio Edit GPU stage."""
    try:
        from services.gpu_memory import release_gpu_holders

        release_gpu_holders(reason="audio_edit")
    except Exception:  # noqa: BLE001 — a failed release must not fail the edit
        logger.warning("Could not free the GPU before an Audio Edit stage", exc_info=True)


def _new_job(session_id: str, kind: str, render_id: Optional[str] = None) -> EditJob:
    job = EditJob(id=uuid.uuid4().hex, session_id=session_id, kind=kind, render_id=render_id)
    with _JOBS_LOCK:
        _JOBS[job.id] = job
    return job


def _launch(job: EditJob, work) -> EditJob:
    if RUN_INLINE:
        _run(job, work)
    else:
        threading.Thread(target=_run, args=(job, work), daemon=True,
                         name=f"audio-edit-{job.kind}-{job.id[:8]}").start()
    return job


def _run(job: EditJob, work) -> None:
    acquired = False
    try:
        job.stage = "waiting"
        while not (acquired := _RENDER_SLOT.acquire(timeout=0.25)):
            _check_cancel(job)
        _check_cancel(job)
        job.status = "running"
        if job.kind == "render":
            update_render(job.session_id, job.render_id, status="running")
        work(job)
        job.status, job.progress, job.stage = "succeeded", 1.0, "done"
    except Cancelled:
        job.status = "cancelled"
    except AudioEditError as exc:
        job.status, job.error, job.message = "failed", exc.code, exc.message
    except Exception as exc:  # noqa: BLE001 — surfaced to the UI as a failed job
        logger.exception("Audio Edit %s job %s failed", job.kind, job.id)
        job.status, job.error, job.message = "failed", "failed", f"{type(exc).__name__}: {exc}"[:500]
    finally:
        job.engine = None
        if acquired:
            _RENDER_SLOT.release()
        _free_vram()
        _finish(job)


def _finish(job: EditJob) -> None:
    """Persist a finished job's outcome into the session or render."""
    try:
        if job.kind == "render":
            changes: dict = {"status": job.status, "progress": round(job.progress, 3),
                             "finished_at": time.time()}
            if job.status != "succeeded":
                changes.update(error=job.error or job.status, message=job.message)
            update_render(job.session_id, job.render_id, **changes)
        elif job.status != "succeeded":
            session = _load_raw(job.session_id)
            back = "uploaded" if job.kind == "separate" or session["mode"] == "speech" else "separated"
            update_session(job.session_id, stage=back, error=job.error or job.status)
    except (AudioEditError, OSError, ValueError):
        logger.warning("Audio Edit job %s finished after its session was removed", job.id)


def cancel_job(job_id: str) -> EditJob:
    job = get_job(job_id)
    if job.status in LIVE_STATUSES:
        job.cancel.set()
        kill = job.engine
        if callable(kill):
            try:
                kill()
            except Exception:  # noqa: BLE001
                logger.warning("Could not stop the engine for job %s", job_id, exc_info=True)
    return job


# ── separation and transcription ─────────────────────────────────────────

def _start_stage_job(session_id: str, kind: str, allowed: dict, running_stage: str,
                     message: str) -> EditJob:
    with _STATE_LOCK:
        session = _load_raw(session_id)
        if _session_busy(session_id):
            raise AudioEditError("busy", "This session is already working.", 409)
        if session.get("stage") not in allowed[session["mode"]]:
            raise AudioEditError("not_ready", message, 409)
        job = _new_job(session_id, kind)
        session.update(stage=running_stage, error=None)
        _save_session(session)
    return job


def start_separation(session_id: str) -> EditJob:
    job = _start_stage_job(session_id, "separate", {"speech": (), "song": ("uploaded",)},
                           "separating", "Only a freshly uploaded song can be separated.")
    return _launch(job, _separate_work)


def start_transcription(session_id: str) -> EditJob:
    job = _start_stage_job(session_id, "transcribe",
                           {"speech": ("uploaded", "ready"), "song": ("separated", "ready")},
                           "transcribing", "Separate the song before transcribing it.")
    return _launch(job, _transcribe_work)


def separate_stems(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """HTDemucs: mono vocals and the summed stereo accompaniment, at SAMPLE_RATE."""
    status = separation_status()
    if not status["available"]:
        raise AudioEditError(
            "separation_unavailable",
            f"Song mode needs HTDemucs ({status['missing'] or 'model weights'}). Download it in Voice Lab first.",
            409)
    import torch
    from demucs.apply import apply_model

    from services.dataset_builder import load_htdemucs

    _release_gpu()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio, sr = read_audio(path)
    model = load_htdemucs(Path(status["weights"]), device)
    try:
        model_sr = int(model.samplerate)
        mix = dsp.resample(dsp.match_channels(audio, int(model.audio_channels)), sr, model_sr)
        wav = torch.from_numpy(np.ascontiguousarray(mix))
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            stems = apply_model(model, ((wav - mean) / std)[None], device=device,
                                split=True, overlap=0.25, progress=False)[0]
        stems = (stems * std + mean).cpu().numpy().astype(np.float32)
        v = list(model.sources).index("vocals")
        vocals = stems[v]
        instrumental = np.sum([stems[i] for i in range(len(model.sources)) if i != v], axis=0)
    finally:
        del model
        _free_vram()
    n = int(round(audio.shape[1] * SAMPLE_RATE / sr))
    vocals = dsp.fit_length(dsp.match_channels(dsp.resample(vocals, model_sr, SAMPLE_RATE), 1), n)
    instrumental = dsp.fit_length(dsp.match_channels(dsp.resample(instrumental, model_sr, SAMPLE_RATE), 2), n)
    return vocals.astype(np.float32), instrumental.astype(np.float32)


def _separate_work(job: EditJob) -> None:
    d = session_dir(job.session_id)
    job.stage, job.progress = "separate", 0.05
    vocals, instrumental = separate_stems(d / "source.wav")
    _check_cancel(job)
    write_audio(d / "vocals.wav", vocals, SAMPLE_RATE)
    write_audio(d / "instrumental.wav", instrumental, SAMPLE_RATE)
    update_session(job.session_id, stage="separated", error=None)


def _asr_transcribe(path: Path) -> dict:
    from services.asr_backend import get_active_asr_backend

    return get_active_asr_backend().transcribe(str(path), word_timestamps=True)


def _transcribe_work(job: EditJob) -> None:
    session = _load_raw(job.session_id)
    d = session_dir(job.session_id)
    source = d / ("vocals.wav" if session["mode"] == "song" else "source.wav")
    job.stage, job.progress = "transcribe", 0.05
    _release_gpu()
    result = _asr_transcribe(source)
    _check_cancel(job)
    words = words_from_asr(result if isinstance(result, dict) else {})
    transcript = {"language": result.get("language") if isinstance(result, dict) else None,
                  "text": " ".join(w["word"] for w in words), "words": words}
    _write_json(d / "transcript.json", transcript)
    update_session(job.session_id, stage="ready", error=None, transcript=transcript, spans=[])
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run pytest tests/test_audio_edit_service.py -q`
Expected: `35 passed`.

- [ ] **Step 5: Commit**

```powershell
git add backend/services/audio_edit.py tests/test_audio_edit_service.py
git commit -m "Audio Edit: job runner, stem separation and transcription"
```

## Task 9: Render requests and the speech render pipeline

**Files:**
- Modify: `backend/services/audio_edit.py` (append the render section)
- Test: `tests/test_audio_edit_render.py`

**Interfaces:**
- Consumes:
  - Task 2 DSP: `splice`/`apply_splices`, `match_loudness`, `context_around`, `estimate_duration`, `resample`, `fit_length`
  - Task 3 store, Task 4 `get_edit_engine` / `EngineError` / `engine_status`, Task 7 licences, Task 8 runner
  - `rvc.get_service().convert_file(...)`, `rvc.ConversionParams`
  - `rvc_registry.voice_files`
  - `watermark.mark_synthetic(waveform, sample_rate, *, context)`
- Produces, used by Tasks 10 and 11:
  - `CROSSFADE_RANGE_MS`, `RVC_PITCH_RANGE`, `VOCAL_GAIN_RANGE_DB`
  - `validate_render_request(mode, raw) -> params`
  - `start_render(session_id, raw) -> (EditJob, render)`
  - Seams: `_edit_engine(engine_id)`, `_engine_edit(engine, payload, job, on_progress)` (the single sidecar synthesis call site), `_rvc_voice_files(voice_id)`, `_run_rvc(src, dst, voice_id, pitch)`, `_mark(audio, sr)`
  - Helpers: `_with_margins(audio, r0, r1, x)`, `_apply_rvc(...)`, `_write_output(path, audio, sr, render)`, `_render_work(job)`, `_render_speech(job, session, render)`

Speech render, per the spec:
1. Deletions become `None` edits; they need no engine, ack or GPU.
2. For each replacement, cut the context window (Task 3 `span_window`, `SPEECH_CONTEXT_S`). F5-TTS regenerates only the span part (`parts` relative to the window, `fix_duration` from `estimate_duration`). The reply's `regions[0]` says where the new words landed in the 24 kHz output.
3. After resampling to 44.1 kHz, cut that region with `x` crossfade samples on each side. That is the insert `splice()` expects.
4. Optional RVC re-voices each insert. Then loudness matching against ±2 s of untouched context, and `mark_synthetic` on the insert only.
5. `apply_splices` writes the output. Every sample outside `[a − x, b + x)` of every span is the upload's own sample, so nothing outside the spans is re-encoded or re-watermarked.

`output.wav` is written as float WAV with `software` and `comment` metadata. The comment carries `commercial_use=true|false`, so an exported file carries its licence flag.

- [ ] **Step 1: Write the failing tests**

`tests/test_audio_edit_render.py`:

```python
"""Audio Edit render pipelines with fake engines (no GPU, no sidecars)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from services import audio_edit as ae
from services import audio_edit_dsp as dsp

SR = ae.SAMPLE_RATE

WORDS = [
    {"i": 0, "word": "the", "start": 0.10, "end": 0.30},
    {"i": 1, "word": "quick", "start": 0.35, "end": 0.70},
    {"i": 2, "word": "brown", "start": 0.80, "end": 1.20},
    {"i": 3, "word": "fox", "start": 1.30, "end": 1.60},
    {"i": 4, "word": "jumps", "start": 1.70, "end": 2.10},
    {"i": 5, "word": "high", "start": 2.20, "end": 2.60},
]


def _fake_decode(src: Path, dst: Path, channels: int) -> float:
    data, sr = sf.read(str(src), always_2d=True, dtype="float32")
    audio = dsp.match_channels(data.T, channels)
    sf.write(str(dst), audio.T, sr, subtype="FLOAT")
    return audio.shape[1] / sr


def _upload(tmp_path: Path, channels: int = 1) -> Path:
    rng = np.random.default_rng(1)
    data = (rng.standard_normal((4 * SR, channels)) * 0.1).astype(np.float32)
    path = tmp_path / f"take{channels}.wav"
    sf.write(str(path), data, SR, subtype="FLOAT")
    return path


def _span(sid="s1", start=0.35, end=0.70, original="quick", replacement="slow"):
    return {"id": sid, "start_s": start, "end_s": end, "original_text": original, "replacement_text": replacement}


class FakeEditor:
    """Stands in for an EditSidecar: writes a constant tone of the right length."""

    def __init__(self, sr: int = 24000):
        self.sr, self.fail, self.on_edit = sr, False, None
        self.payloads: list[dict] = []
        self.killed = self.stopped = 0

    def edit(self, payload, on_progress=None):
        from services.audio_edit_engines import EngineError

        self.payloads.append(payload)
        if self.on_edit:
            self.on_edit(payload)
        if self.fail:
            raise EngineError("CUDA out of memory")
        if on_progress:
            on_progress(0.5)
        frames = sf.info(payload["audio_path"]).frames
        seconds = frames / SR
        reply = {"op": "audio", "path": payload["out_path"], "sample_rate": self.sr}
        if "parts" in payload:
            (p0, p1), dur = payload["parts"][0], payload["fix_duration"][0]
            seconds = seconds - (p1 - p0) + dur
            reply["regions"] = [[p0, round(p0 + dur, 4)]]
        sf.write(payload["out_path"], np.full(int(round(seconds * self.sr)), 0.05, np.float32),
                 self.sr, subtype="FLOAT")
        return reply

    def kill(self):
        self.killed += 1

    def shutdown(self):
        self.stopped += 1


@pytest.fixture
def env(tmp_path, monkeypatch):
    from services import audio_edit_engines as aee
    from services import watermark

    monkeypatch.setenv("OMNIVOICE_AUDIO_EDIT_DIR", str(tmp_path / "audio_edit"))
    monkeypatch.setattr(ae, "decode_to_wav", _fake_decode)
    monkeypatch.setattr(ae, "RUN_INLINE", True)
    monkeypatch.setattr(ae, "_free_vram", lambda: None)
    acks: dict = {}
    monkeypatch.setattr(ae, "_read_acks", lambda: dict(acks))
    monkeypatch.setattr(ae, "_write_acks", lambda value: (acks.clear(), acks.update(value)))
    monkeypatch.setattr(ae, "_registry_records", lambda: {})
    events: list = []
    editor = FakeEditor()
    monkeypatch.setattr(ae, "_release_gpu", lambda: events.append("release"))
    monkeypatch.setattr(ae, "_edit_engine", lambda engine_id: events.append(f"engine:{engine_id}") or editor)
    monkeypatch.setattr(aee, "engine_status", lambda e: {
        "installed": True, "install_id": e, "needs_gpu": False, "available": True, "reason": None})
    marks: list = []

    def fake_mark(waveform, sample_rate, *, context, force=False):
        marks.append((tuple(waveform.shape), sample_rate, context))
        return waveform

    monkeypatch.setattr(watermark, "mark_synthetic", fake_mark)
    ae._JOBS.clear()
    yield SimpleNamespace(tmp=tmp_path, editor=editor, events=events, marks=marks, acks=acks)
    ae._JOBS.clear()


def _speech(env, spans) -> str:
    sid = ae.create_session("speech", "", _upload(env.tmp), "a.wav")["id"]
    ae.update_session(sid, stage="ready", transcript={
        "language": "en", "text": " ".join(w["word"] for w in WORDS), "words": WORDS})
    ae.set_spans(sid, spans)
    return sid


def _samples(seconds: float) -> int:
    return int(round(seconds * SR))


def test_render_request_checks(env, monkeypatch):
    from services import audio_edit_engines as aee

    sid = _speech(env, [_span()])
    with pytest.raises(ae.AudioEditError) as ack:
        ae.start_render(sid, {"engine": "f5tts"})
    assert ack.value.code == "licence_ack_required" and ack.value.status == 403
    ae.acknowledge_licence("f5tts")
    for raw, code in (({"engine": "ace_step"}, "invalid_engine"),
                      ({"engine": "f5tts", "crossfade_ms": 2}, "invalid_request"),
                      ({"engine": "f5tts", "rvc_pitch": 13}, "invalid_request"),
                      ({"engine": "f5tts", "seed": "x"}, "invalid_request"),
                      ("nope", "invalid_request")):
        with pytest.raises(ae.AudioEditError) as err:
            ae.start_render(sid, raw)
        assert (err.value.code, err.value.status) == (code, 422)
    monkeypatch.setattr(aee, "engine_status", lambda e: {
        "installed": False, "install_id": "f5tts", "needs_gpu": False, "available": False,
        "reason": "Install F5-TTS to use it."})
    with pytest.raises(ae.AudioEditError) as missing:
        ae.start_render(sid, {"engine": "f5tts"})
    assert missing.value.code == "engine_not_installed" and missing.value.status == 409
    params = ae.validate_render_request("song", {"engine": "ace_step"})
    assert params["crossfade_ms"] == 30 and params["vocal_gain_db"] == 0.0
    assert params["ace_step"] == {"strength": 0.7, "padding_s": 2.0, "caption": ""}
    assert 0 <= params["seed"] < 2 ** 31


def test_speech_render_splices_only_inside_the_spans(env):
    ae.acknowledge_licence("f5tts")
    sid = _speech(env, [_span(), _span("s2", 1.30, 1.60, "fox", "")])
    job, render = ae.start_render(sid, {"engine": "f5tts", "crossfade_ms": 20, "seed": 7})
    assert job.status == "succeeded" and render["status"] == "succeeded"
    assert render["commercial_use"] is False and render["engine"] == "f5tts"
    assert render["output_path"] == f"renders/{render['id']}/output.wav"
    assert [s["status"] for s in render["spans"]] == ["done", "done"]

    payload, = env.editor.payloads                      # the deletion needs no engine
    assert payload["raw_text"] == "the quick" and payload["target_text"] == "the slow"
    assert payload["parts"] == [[0.25, 0.6]] and payload["fix_duration"] == [0.28]
    assert payload["seed"] == 7
    assert env.events[:2] == ["release", "engine:f5tts"] and env.editor.stopped == 1
    assert len(env.marks) == 1 and env.marks[0][1:] == (SR, "audio_edit.render")

    path = ae.render_audio_path(sid, render["id"])
    out, sr = ae.read_audio(path)
    source, _ = ae.read_audio(ae.session_dir(sid) / "source.wav")
    x = dsp.ms_to_samples(20, SR)
    a1, b1, a2, b2 = _samples(0.35), _samples(0.70), _samples(1.30), _samples(1.60)
    body = _samples(0.53) - _samples(0.25)
    assert sr == SR
    assert out.shape[1] == source.shape[1] - (b1 - a1) + body - ((b2 - a2) + x)
    assert np.array_equal(out[:, :a1 - x], source[:, :a1 - x])
    tail = source.shape[1] - b2 - x
    assert np.array_equal(out[:, -tail:], source[:, -tail:])
    with sf.SoundFile(str(path)) as f:
        assert "commercial_use=false" in f.comment and f.software == "VoiceStudio Audio Edit"


def test_delete_only_render_needs_no_engine_or_ack(env):
    sid = _speech(env, [_span(replacement="")])
    job, render = ae.start_render(sid, {"engine": "f5tts"})
    assert job.status == "succeeded" and render["status"] == "succeeded"
    assert env.editor.payloads == [] and not any(e.startswith("engine:") for e in env.events)
    assert env.marks == []


def test_engine_failure_fails_the_render_and_keeps_partial_files(env):
    ae.acknowledge_licence("f5tts")
    env.editor.fail = True
    sid = _speech(env, [_span()])
    job, render = ae.start_render(sid, {"engine": "f5tts"})
    assert job.status == "failed" and render["status"] == "failed"
    assert render["error"] == "engine_failed" and "CUDA out of memory" in render["message"]
    assert (ae.render_dir(sid, render["id"]) / "span-0.window.wav").is_file()
    assert env.editor.stopped == 1
    with pytest.raises(ae.AudioEditError):
        ae.render_audio_path(sid, render["id"])


def test_cancel_during_an_edit_kills_the_sidecar(env):
    ae.acknowledge_licence("f5tts")

    def cancel_now(payload):
        job = next(j for j in ae._JOBS.values() if j.kind == "render")
        ae.cancel_job(job.id)

    env.editor.on_edit, env.editor.fail = cancel_now, True   # a killed sidecar errors out
    sid = _speech(env, [_span()])
    job, render = ae.start_render(sid, {"engine": "f5tts"})
    assert job.status == "cancelled" and render["status"] == "cancelled"
    assert env.editor.killed == 1


def test_rvc_pass_converts_each_insert(env, monkeypatch):
    ae.acknowledge_licence("f5tts")
    monkeypatch.setattr(ae, "_rvc_voice_files", lambda voice_id: ("model.pth", None))
    calls = []

    def fake_rvc(src, dst, voice_id, pitch):
        calls.append((Path(src).name, voice_id, pitch))
        audio, _ = ae.read_audio(src)
        ae.write_audio(dst, audio * 0.5, 40000)
        return dst

    monkeypatch.setattr(ae, "_run_rvc", fake_rvc)
    sid = _speech(env, [_span()])
    job, render = ae.start_render(sid, {"engine": "f5tts", "rvc_model_id": "alto", "rvc_pitch": 3})
    assert job.status == "succeeded" and calls == [("span-0.insert.wav", "alto", 3)]
    assert render["params"]["rvc_model_id"] == "alto"
    assert env.events.count("release") >= 3          # before F5-TTS, before and after RVC

    def unknown(voice_id):
        raise ae.AudioEditError("invalid_rvc_model", "That RVC voice is not installed.", 422)

    monkeypatch.setattr(ae, "_rvc_voice_files", unknown)
    with pytest.raises(ae.AudioEditError) as err:
        ae.start_render(sid, {"engine": "f5tts", "rvc_model_id": "gone"})
    assert err.value.code == "invalid_rvc_model"


def test_renders_list_newest_first_and_need_a_ready_session(env):
    ae.acknowledge_licence("f5tts")
    sid = _speech(env, [_span()])
    _, first = ae.start_render(sid, {"engine": "f5tts"})
    _, second = ae.start_render(sid, {"engine": "f5tts"})
    assert [r["id"] for r in ae.load_session(sid)["renders"]] == [second["id"], first["id"]]
    ae.update_session(sid, spans=[])
    with pytest.raises(ae.AudioEditError) as none:
        ae.start_render(sid, {"engine": "f5tts"})
    assert none.value.code == "no_spans" and none.value.status == 400
    ae.update_session(sid, stage="uploaded")
    with pytest.raises(ae.AudioEditError) as early:
        ae.start_render(sid, {"engine": "f5tts"})
    assert early.value.code == "not_ready"
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `uv run pytest tests/test_audio_edit_render.py -q`
Expected: FAIL with `AttributeError: module 'services.audio_edit' has no attribute 'start_render'`.

- [ ] **Step 3: Implement** (append to `backend/services/audio_edit.py`)

```python
# ── render requests ──────────────────────────────────────────────────────

CROSSFADE_RANGE_MS = (5, 200)
RVC_PITCH_RANGE = (-12, 12)
VOCAL_GAIN_RANGE_DB = (-12.0, 6.0)


def _bad_request(message: str) -> AudioEditError:
    return AudioEditError("invalid_request", message, 422)


def _number(raw: dict, key: str, default, lo, hi, kind=float):
    value = raw.get(key, default)
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _bad_request(f"{key} must be a number.")
    if kind is int and float(value) != int(value):
        raise _bad_request(f"{key} must be a whole number.")
    value = kind(value)
    if not lo <= value <= hi:
        raise _bad_request(f"{key} must be between {lo:g} and {hi:g}.")
    return value


def _rvc_voice_files(voice_id: str) -> tuple[str, Optional[str]]:
    from services import rvc_registry

    try:
        return rvc_registry.voice_files(voice_id)
    except (rvc_registry.VoiceNotFound, ValueError):
        raise AudioEditError("invalid_rvc_model", "That RVC voice is not installed.", 422) from None


def validate_render_request(mode: str, raw: object) -> dict:
    if not isinstance(raw, dict):
        raise _bad_request("The render request must be an object.")
    engine = raw.get("engine")
    if engine not in MODE_ENGINES[mode]:
        raise AudioEditError("invalid_engine",
                             f"{mode.capitalize()} sessions use: {', '.join(MODE_ENGINES[mode])}.", 422)
    rvc_id = raw.get("rvc_model_id") or None
    if rvc_id is not None:
        if not isinstance(rvc_id, str):
            raise _bad_request("rvc_model_id must be a string.")
        _rvc_voice_files(rvc_id)
    if raw.get("seed") is None:
        seed = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF
    else:
        seed = _number(raw, "seed", 0, 0, 2 ** 31 - 1, int)
    ace = raw.get("ace_step") or {}
    if not isinstance(ace, dict):
        raise _bad_request("ace_step must be an object.")
    return {
        "engine": engine,
        "crossfade_ms": _number(raw, "crossfade_ms", 30, *CROSSFADE_RANGE_MS, kind=int),
        "rvc_model_id": rvc_id,
        "rvc_pitch": _number(raw, "rvc_pitch", 0, *RVC_PITCH_RANGE, kind=int),
        "vocal_gain_db": _number(raw, "vocal_gain_db", 0.0, *VOCAL_GAIN_RANGE_DB),
        "seed": seed,
        "ace_step": {
            "strength": _number(ace, "strength", 0.7, 0.0, 1.0),
            "padding_s": _number(ace, "padding_s", 2.0, 0.0, 10.0),
            "caption": str(ace.get("caption") or "").strip()[:300],
        },
    }


def start_render(session_id: str, raw: object) -> tuple[EditJob, dict]:
    from services import audio_edit_engines as aee

    session = load_session(session_id)
    if session.get("stage") != "ready" or not session.get("transcript"):
        raise AudioEditError("not_ready", "Transcribe the audio before rendering.", 409)
    if not session.get("spans"):
        raise AudioEditError("no_spans", "Mark at least one span to edit.", 400)
    params = validate_render_request(session["mode"], raw)
    engine = params["engine"]
    regenerates = any(s["replacement_text"] for s in session["spans"])
    if regenerates:
        status = aee.engine_status(engine)
        if not status["available"]:
            raise AudioEditError("engine_not_installed",
                                 status["reason"] or f"{aee.DISPLAY_NAMES[engine]} is not available.", 409)
        require_licence(engine)
    rvc_used = bool(params["rvc_model_id"]) and regenerates
    render_id = uuid.uuid4().hex
    with _STATE_LOCK:
        if _job_alive(session_id, "separate") or _job_alive(session_id, "transcribe"):
            raise AudioEditError("busy", "Wait for the current step to finish.", 409)
        job = _new_job(session_id, "render", render_id)
        render = {
            "id": render_id, "session_id": session_id, "engine": engine, "status": "queued",
            "progress": 0.0, "spans": [{"span_id": s["id"], "status": "queued", "path": None}
                                       for s in session["spans"]],
            "output_path": None, "error": None, "message": None,
            "created_at": time.time(), "finished_at": None,
            "commercial_use": render_commercial_use(engine, rvc=rvc_used, demucs=session["mode"] == "song"),
            "params": params, "span_snapshot": [dict(s) for s in session["spans"]],
        }
        _save_render(render)
        current = _load_raw(session_id)
        current["render_ids"] = list(current.get("render_ids") or []) + [render_id]
        _save_session(current)
    _launch(job, _render_work)
    return job, load_render(session_id, render_id)


# ── render helpers ───────────────────────────────────────────────────────

def _edit_engine(engine_id: str):
    from services.audio_edit_engines import get_edit_engine

    return get_edit_engine(engine_id)


def _engine_edit(engine, payload: dict, job: EditJob, on_progress) -> dict:
    """The one place Audio Edit asks a sidecar for synthetic audio."""
    from services.audio_edit_engines import EngineError

    try:
        reply = engine.edit(payload, on_progress)
    except EngineError as exc:
        if job.cancel.is_set():
            raise Cancelled() from None
        raise AudioEditError("engine_failed", str(exc)[:500], 500) from None
    _check_cancel(job)
    return reply


def _mark(audio: np.ndarray, sr: int) -> np.ndarray:
    """Watermark one synthetic insert (never the untouched original)."""
    import torch

    from services import watermark

    a = as_2d(audio)
    marked = watermark.mark_synthetic(torch.from_numpy(np.ascontiguousarray(a)), sr,
                                      context="audio_edit.render")
    return np.asarray(marked.detach().cpu().numpy(), dtype=np.float32).reshape(a.shape)


def _with_margins(audio, r0: int, r1: int, x: int) -> np.ndarray:
    """``audio[r0 - x : r1 + x]``, zero-padded where it runs off either end."""
    a = as_2d(audio)
    lo, hi = r0 - x, r1 + x
    seg = a[:, max(0, lo):min(a.shape[1], hi)]
    return np.pad(seg, ((0, 0), (max(0, -lo), max(0, hi - a.shape[1])))).astype(np.float32)


def _set_span(job: EditJob, states: list[dict], idx: int, status: str, path: Optional[str] = None) -> None:
    states[idx]["status"] = status
    if path:
        states[idx]["path"] = path
    update_render(job.session_id, job.render_id, spans=states, progress=round(job.progress, 3))


def _progress(job: EditJob, n: int, total: int, lo: float = 0.05, width: float = 0.6):
    def report(fraction: float) -> None:
        job.progress = round(lo + width * (n + min(max(fraction, 0.0), 1.0)) / total, 3)
    return report


def _run_rvc(src: Path, dst: Path, voice_id: str, pitch: int) -> Path:
    from services import rvc

    model_path, index_path = _rvc_voice_files(voice_id)
    try:
        rvc.get_service().convert_file(str(src), str(dst), model_path, index_path,
                                       rvc.ConversionParams(pitch=pitch))
    except (rvc.RVCUnavailable, rvc.RVCConversionError, ValueError, OSError) as exc:
        raise AudioEditError("rvc_failed", f"RVC failed: {exc}"[:500], 500) from None
    return dst


def _apply_rvc(job: EditJob, inserts: dict[int, np.ndarray], rdir: Path, params: dict) -> dict[int, np.ndarray]:
    if not params.get("rvc_model_id") or not inserts:
        return inserts
    job.stage = "rvc"
    _release_gpu()
    converted: dict[int, np.ndarray] = {}
    try:
        for idx, audio in inserts.items():
            _check_cancel(job)
            src, dst = rdir / f"span-{idx}.insert.wav", rdir / f"span-{idx}.rvc.wav"
            write_audio(src, audio, SAMPLE_RATE)
            _run_rvc(src, dst, params["rvc_model_id"], params["rvc_pitch"])
            out, sr = read_audio(dst)
            out = dsp.resample(out, sr, SAMPLE_RATE)
            converted[idx] = dsp.fit_length(dsp.match_channels(out, audio.shape[0]), audio.shape[1])
    finally:
        _release_gpu()
    return converted


def _write_output(path: Path, audio, sr: int, render: dict) -> None:
    import soundfile as sf

    a = as_2d(audio)
    path.parent.mkdir(parents=True, exist_ok=True)
    flag = "true" if render.get("commercial_use") else "false"
    with sf.SoundFile(str(path), "w", samplerate=sr, channels=a.shape[0], format="WAV", subtype="FLOAT") as f:
        f.software = "VoiceStudio Audio Edit"
        f.comment = f"synthetic=audio_edit; engine={render['engine']}; commercial_use={flag}"
        f.write(a.T)


def _render_work(job: EditJob) -> None:
    session = _load_raw(job.session_id)
    render = load_render(job.session_id, job.render_id)
    if session["mode"] == "speech":
        audio = _render_speech(job, session, render)
    else:
        audio = _render_song(job, session, render)
    _check_cancel(job)
    job.stage, job.progress = "write", 0.97
    _write_output(render_dir(job.session_id, job.render_id) / "output.wav", audio, SAMPLE_RATE, render)
    update_render(job.session_id, job.render_id, output_path=f"renders/{job.render_id}/output.wav")


def _render_speech(job: EditJob, session: dict, render: dict) -> np.ndarray:
    sid, rid = job.session_id, job.render_id
    rdir = render_dir(sid, rid)
    params, spans = render["params"], render["span_snapshot"]
    words = session["transcript"]["words"]
    source, sr = read_audio(session_dir(sid) / "source.wav")
    duration = source.shape[1] / sr
    x = dsp.ms_to_samples(params["crossfade_ms"], sr)
    states = [dict(item) for item in render["spans"]]
    todo = [i for i, s in enumerate(spans) if s["replacement_text"]]
    inserts: dict[int, np.ndarray] = {}
    if todo:
        job.stage = "regenerate"
        _release_gpu()
        engine = _edit_engine("f5tts")
        job.engine = engine.kill
        try:
            for n, idx in enumerate(todo):
                _check_cancel(job)
                span = spans[idx]
                lo, hi = window_bounds(spans, idx, duration)
                win = span_window(span, words, duration, SPEECH_CONTEXT_S, lo, hi)
                ws, we = int(round(win.start_s * sr)), int(round(win.end_s * sr))
                window_path = rdir / f"span-{idx}.window.wav"
                write_audio(window_path, source[:, ws:we], sr)
                new_len = dsp.estimate_duration(span["original_text"], span["replacement_text"],
                                                span["end_s"] - span["start_s"])
                _set_span(job, states, idx, "running")
                reply = _engine_edit(engine, {
                    "audio_path": str(window_path), "out_path": str(rdir / f"span-{idx}.wav"),
                    "raw_text": win.raw_text, "target_text": win.target_text, "seed": params["seed"],
                    "parts": [[round(span["start_s"] - win.start_s, 4), round(span["end_s"] - win.start_s, 4)]],
                    "fix_duration": [round(new_len, 4)],
                }, job, _progress(job, n, len(todo)))
                regions = reply.get("regions") or []
                if not regions:
                    raise AudioEditError("engine_failed", "F5-TTS did not say where the new words are.", 500)
                out, out_sr = read_audio(Path(reply["path"]))
                out = dsp.resample(dsp.match_channels(out, 1), out_sr, sr)
                r0, r1 = (int(round(float(t) * sr)) for t in regions[0])
                inserts[idx] = _with_margins(out, r0, r1, x)
                _set_span(job, states, idx, "generated", f"renders/{rid}/span-{idx}.wav")
        finally:
            job.engine = None
            engine.shutdown()
    inserts = _apply_rvc(job, inserts, rdir, params)
    job.stage, job.progress = "splice", 0.85
    edits = []
    for idx, span in enumerate(spans):
        insert = None
        if idx in inserts:
            context = dsp.context_around(source, sr, span["start_s"], span["end_s"])
            insert = _mark(dsp.match_loudness(inserts[idx], context, sr), sr)
        edits.append((span["start_s"], span["end_s"], insert))
        states[idx]["status"] = "done"
    update_render(sid, rid, spans=states)
    return dsp.apply_splices(source, sr, edits, params["crossfade_ms"])
```

`_render_song` is added in Task 10. Until then, add this stub so `_render_work` imports cleanly. Task 10 replaces it:

```python
def _render_song(job: EditJob, session: dict, render: dict) -> np.ndarray:
    raise AudioEditError("not_ready", "Song renders arrive in the next task.", 409)
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run pytest tests/test_audio_edit_render.py tests/test_audio_edit_service.py -q`
Expected: `42 passed` (7 new + 35).

- [ ] **Step 5: Commit**

```powershell
git add backend/services/audio_edit.py tests/test_audio_edit_render.py
git commit -m "Audio Edit: render requests and speech pipeline"
```

## Task 10: Song render pipeline (singer engines, ACE-Step repaint, remix)

**Files:**
- Modify: `backend/services/audio_edit.py` (replace the `_render_song` stub; append helpers)
- Test: `tests/test_audio_edit_render.py` (append)

**Interfaces:**
- Consumes:
  - Task 5 `EDITORS["yingmusic"|"vevo2"]` through `_edit_engine`
  - Task 6 `AceStepClient`, `AceStepUnavailable`, `AceStepError`, `AceStepCancelled`
  - Task 8 `separate_stems`
  - Task 9 helpers
- Produces:
  - `song_mix(source, vocals, new_vocals, instrumental, vocal_gain_db) -> np.ndarray`
  - Seams `_ace_client(work_dir)` and `_ace_repaint(client, job, **kwargs) -> Path` (the single ACE-Step synthesis call site)
  - `_singer_inserts(...)`, `_ace_step_inserts(...)`, `_render_song(job, session, render)`

How a song render works:
- Song renders keep the song's length. Every insert body is exactly `b − a` samples. A deletion becomes a silent vocal insert, so the singer drops out while the backing track carries on.
- **YingMusic and Vevo2** regenerate the whole vocal window. The output is resampled and fitted to the window's length, then the span is cut out with crossfade margins.
- **ACE-Step** repaints a segment of the *full mix*: the window plus `padding_s` on each side. Each repainted segment goes through HTDemucs, and its vocal stem becomes the insert. The ACE-Step server is stopped before Demucs loads, so the two never share VRAM.
- Every insert then gets the same treatment: optional RVC (on by default in the UI when a model exists), loudness matching against untouched vocal context, and the watermark.
- `apply_splices` builds the edited vocal stem, which is also kept as `vocals.edited.wav`.
- Mixing:
  - At 0 dB the edit is added as a residual: `source + (new_vocals − vocals)`. Every sample outside the spans and their crossfades is then the upload's own sample.
  - At any other gain, the song is remixed from the stems with `dsp.remix`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_audio_edit_render.py`)

```python
# ── Task 10: song renders ───────────────────────────────────────────────

def _song(env, spans) -> str:
    sid = ae.create_session("song", "", _upload(env.tmp, channels=2), "s.wav")["id"]
    d = ae.session_dir(sid)
    source, _ = ae.read_audio(d / "source.wav")
    ae.write_audio(d / "vocals.wav", source.mean(axis=0, keepdims=True) * 0.6, SR)
    ae.write_audio(d / "instrumental.wav", source * 0.4, SR)
    ae.update_session(sid, stage="ready", transcript={
        "language": "en", "text": " ".join(w["word"] for w in WORDS), "words": WORDS})
    ae.set_spans(sid, spans)
    return sid


class FakeAce:
    def __init__(self):
        self.calls: list[dict] = []
        self.stopped = self.cancelled = 0
        self.error = None

    def repaint(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        audio, sr = ae.read_audio(kwargs["src_path"])
        ae.write_audio(kwargs["out_path"], audio * 0.9, sr)
        return Path(kwargs["out_path"])

    def cancel(self):
        self.cancelled += 1

    def stop(self):
        self.stopped += 1


def test_song_mix_residual_keeps_untouched_samples():
    rng = np.random.default_rng(3)
    source = (rng.standard_normal((2, 1000)) * 0.1).astype(np.float32)
    vocals = (rng.standard_normal((1, 1000)) * 0.1).astype(np.float32)
    instrumental = (source * 0.5).astype(np.float32)
    edited = vocals.copy()
    edited[:, 100:200] = 0.3
    mix = ae.song_mix(source, vocals, edited, instrumental, 0.0)
    assert np.array_equal(mix[:, :100], source[:, :100]) and np.array_equal(mix[:, 200:], source[:, 200:])
    assert np.allclose(mix[:, 100:200], source[:, 100:200] + (0.3 - vocals[:, 100:200]), atol=1e-6)
    assert np.array_equal(ae.song_mix(source, vocals, edited, instrumental, -6.0),
                          dsp.remix(edited, instrumental, -6.0))


def test_singer_render_keeps_length_and_outside_audio(env):
    ae.acknowledge_licence("yingmusic")
    env.editor.sr = 48000
    sid = _song(env, [_span(), _span("s2", 1.30, 1.60, "fox", "")])
    job, render = ae.start_render(sid, {"engine": "yingmusic", "crossfade_ms": 20})
    assert job.status == "succeeded" and render["status"] == "succeeded"
    payload, = env.editor.payloads
    assert "parts" not in payload and payload["target_text"] == "the slow"
    assert env.events[:2] == ["release", "engine:yingmusic"] and env.editor.stopped == 1
    assert len(env.marks) == 1                          # the silent deletion is not synthetic
    out, _ = ae.read_audio(ae.render_audio_path(sid, render["id"]))
    source, _ = ae.read_audio(ae.session_dir(sid) / "source.wav")
    x = dsp.ms_to_samples(20, SR)
    a1, b1, a2, b2 = _samples(0.35), _samples(0.70), _samples(1.30), _samples(1.60)
    assert out.shape == source.shape
    for lo, hi in ((0, a1 - x), (b1 + x, a2 - x), (b2 + x, source.shape[1])):
        assert np.array_equal(out[:, lo:hi], source[:, lo:hi])
    assert (ae.render_dir(sid, render["id"]) / "vocals.edited.wav").is_file()


def test_singer_engine_unavailable_is_409(env, monkeypatch):
    from services import audio_edit_engines as aee

    ae.acknowledge_licence("vevo2")
    monkeypatch.setattr(aee, "engine_status", lambda e: {
        "installed": True, "install_id": "vevo2", "needs_gpu": True, "available": False,
        "reason": "Vevo2 needs a CUDA or Apple GPU."})
    sid = _song(env, [_span()])
    with pytest.raises(ae.AudioEditError) as err:
        ae.start_render(sid, {"engine": "vevo2"})
    assert err.value.code == "engine_not_installed" and "GPU" in err.value.message


def test_ace_step_render_repaints_then_separates(env, monkeypatch):
    fake = FakeAce()
    monkeypatch.setattr(ae, "_ace_client", lambda work_dir: fake)
    separated = []

    def fake_separate(path):
        separated.append(Path(path).name)
        audio, sr = ae.read_audio(path)
        audio = dsp.resample(audio, sr, SR)
        return audio.mean(axis=0, keepdims=True), audio

    monkeypatch.setattr(ae, "separate_stems", fake_separate)
    sid = _song(env, [_span()])
    job, render = ae.start_render(sid, {"engine": "ace_step", "seed": 3, "vocal_gain_db": -3,
                                        "ace_step": {"strength": 0.4, "padding_s": 1.0, "caption": "pop"}})
    assert job.status == "succeeded" and render["status"] == "succeeded"
    call, = fake.calls
    assert (call["start_s"], call["end_s"]) == (0.1, 0.95)     # window inside a 0–1.95 s segment
    assert call["lyrics"] == "the slow" and call["caption"] == "pop" and call["strength"] == 0.4
    assert call["seed"] == 3 and call["language"] == "en"
    assert Path(call["src_path"]).name == "span-0.segment.wav"
    assert sf.info(str(call["src_path"])).frames == _samples(1.95)
    assert fake.stopped == 1 and separated == ["span-0.wav"] and len(env.marks) == 1
    out, _ = ae.read_audio(ae.render_audio_path(sid, render["id"]))
    assert out.shape[1] == ae.read_audio(ae.session_dir(sid) / "source.wav")[0].shape[1]


def test_ace_step_unavailable_fails_the_render(env, monkeypatch):
    from services.acestep_client import AceStepUnavailable

    fake = FakeAce()
    fake.error = AceStepUnavailable("ACE-Step did not answer within 60 s.")
    monkeypatch.setattr(ae, "_ace_client", lambda work_dir: fake)
    sid = _song(env, [_span()])
    job, render = ae.start_render(sid, {"engine": "ace_step"})
    assert job.status == "failed" and render["error"] == "ace_step_unavailable"
    assert "60 s" in render["message"] and fake.stopped == 1
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `uv run pytest tests/test_audio_edit_render.py -q -k "song or singer or ace"`
Expected: FAIL. `song_mix` is missing, and song renders fail with "Song renders arrive in the next task."

- [ ] **Step 3: Implement**

In `backend/services/audio_edit.py`, delete the Task 9 `_render_song` stub and append:

```python
# ── song renders ─────────────────────────────────────────────────────────

def song_mix(source, vocals, new_vocals, instrumental, vocal_gain_db: float) -> np.ndarray:
    """The edited song. At 0 dB the edit is added as a residual so every
    sample outside the edits is the upload's own; otherwise a stem remix."""
    if float(vocal_gain_db) == 0.0:
        delta = dsp.match_channels(new_vocals, 2) - dsp.match_channels(vocals, 2)
        return np.clip(dsp.match_channels(source, 2) + delta, -1.0, 1.0).astype(np.float32)
    return dsp.remix(new_vocals, instrumental, vocal_gain_db)


def _ace_client(work_dir: Path):
    from services.acestep_client import AceStepClient

    return AceStepClient(work_dir)


def _ace_repaint(client, job: EditJob, **kwargs) -> Path:
    """The one place Audio Edit asks ACE-Step for synthetic audio."""
    import httpx

    from services import acestep_client as ac

    try:
        out = client.repaint(**kwargs)
    except ac.AceStepCancelled:
        raise Cancelled() from None
    except ac.AceStepUnavailable as exc:
        raise AudioEditError("ace_step_unavailable", str(exc)[:500], 503) from None
    except (ac.AceStepError, httpx.HTTPError, OSError) as exc:
        if job.cancel.is_set():
            raise Cancelled() from None
        raise AudioEditError("engine_failed", f"ACE-Step failed: {exc}"[:500], 500) from None
    _check_cancel(job)
    return Path(out)


def _singer_inserts(job: EditJob, spans: list[dict], todo: list[int], words: list[dict],
                    vocals: np.ndarray, sr: int, rdir: Path, params: dict,
                    states: list[dict]) -> dict[int, np.ndarray]:
    inserts: dict[int, np.ndarray] = {}
    if not todo:
        return inserts
    x = dsp.ms_to_samples(params["crossfade_ms"], sr)
    duration = vocals.shape[1] / sr
    job.stage = "regenerate"
    _release_gpu()
    engine = _edit_engine(params["engine"])
    job.engine = engine.kill
    try:
        for n, idx in enumerate(todo):
            _check_cancel(job)
            span = spans[idx]
            lo, hi = window_bounds(spans, idx, duration)
            win = span_window(span, words, duration, SONG_CONTEXT_S, lo, hi)
            ws, we = int(round(win.start_s * sr)), int(round(win.end_s * sr))
            window_path = rdir / f"span-{idx}.window.wav"
            write_audio(window_path, vocals[:, ws:we], sr)
            _set_span(job, states, idx, "running")
            reply = _engine_edit(engine, {
                "audio_path": str(window_path), "out_path": str(rdir / f"span-{idx}.wav"),
                "raw_text": win.raw_text, "target_text": win.target_text, "seed": params["seed"],
            }, job, _progress(job, n, len(todo)))
            out, out_sr = read_audio(Path(reply["path"]))
            out = dsp.fit_length(dsp.resample(dsp.match_channels(out, 1), out_sr, sr), we - ws)
            a, b = int(round(span["start_s"] * sr)), int(round(span["end_s"] * sr))
            inserts[idx] = _with_margins(out, a - ws, b - ws, x)
            _set_span(job, states, idx, "generated", f"renders/{job.render_id}/span-{idx}.wav")
    finally:
        job.engine = None
        engine.shutdown()
    return inserts


def _ace_step_inserts(job: EditJob, spans: list[dict], todo: list[int], words: list[dict],
                      source: np.ndarray, sr: int, rdir: Path, params: dict,
                      states: list[dict], language: str) -> dict[int, np.ndarray]:
    inserts: dict[int, np.ndarray] = {}
    if not todo:
        return inserts
    x = dsp.ms_to_samples(params["crossfade_ms"], sr)
    duration = source.shape[1] / sr
    ace = params["ace_step"]
    job.stage = "regenerate"
    _release_gpu()
    client = _ace_client(rdir)
    job.engine = client.cancel
    repainted: dict[int, tuple[Path, int, int]] = {}
    try:
        for n, idx in enumerate(todo):
            _check_cancel(job)
            span = spans[idx]
            lo, hi = window_bounds(spans, idx, duration)
            win = span_window(span, words, duration, SONG_CONTEXT_S, lo, hi)
            seg_start = max(0.0, win.start_s - ace["padding_s"])
            seg_end = min(duration, win.end_s + ace["padding_s"])
            s0, s1 = int(round(seg_start * sr)), int(round(seg_end * sr))
            segment_path = rdir / f"span-{idx}.segment.wav"
            write_audio(segment_path, source[:, s0:s1], sr)
            _set_span(job, states, idx, "running")
            out = _ace_repaint(
                client, job, src_path=segment_path,
                start_s=round(win.start_s - seg_start, 3), end_s=round(win.end_s - seg_start, 3),
                lyrics=win.target_text, caption=ace["caption"], strength=ace["strength"],
                seed=params["seed"], out_path=rdir / f"span-{idx}.wav", language=language)
            repainted[idx] = (out, s0, s1)
            job.progress = round(0.05 + 0.5 * (n + 1) / len(todo), 3)
    finally:
        job.engine = None
        client.stop()
    job.stage = "separate"
    for idx, (path, s0, s1) in repainted.items():
        _check_cancel(job)
        vocal, _accompaniment = separate_stems(path)
        vocal = dsp.fit_length(vocal, s1 - s0)
        span = spans[idx]
        a, b = int(round(span["start_s"] * sr)), int(round(span["end_s"] * sr))
        inserts[idx] = _with_margins(vocal, a - s0, b - s0, x)
        _set_span(job, states, idx, "generated", f"renders/{job.render_id}/span-{idx}.wav")
    return inserts


def _render_song(job: EditJob, session: dict, render: dict) -> np.ndarray:
    sid, rid = job.session_id, job.render_id
    d, rdir = session_dir(sid), render_dir(sid, rid)
    params, spans = render["params"], render["span_snapshot"]
    words = session["transcript"]["words"]
    source, sr = read_audio(d / "source.wav")
    vocals, _ = read_audio(d / "vocals.wav")
    instrumental, _ = read_audio(d / "instrumental.wav")
    x = dsp.ms_to_samples(params["crossfade_ms"], sr)
    states = [dict(item) for item in render["spans"]]
    todo = [i for i, s in enumerate(spans) if s["replacement_text"]]
    if params["engine"] == "ace_step":
        language = session["transcript"].get("language") or "en"
        inserts = _ace_step_inserts(job, spans, todo, words, source, sr, rdir, params, states, language)
    else:
        inserts = _singer_inserts(job, spans, todo, words, vocals, sr, rdir, params, states)
    inserts = _apply_rvc(job, inserts, rdir, params)
    job.stage, job.progress = "remix", 0.9
    edits = []
    for idx, span in enumerate(spans):
        a, b = int(round(span["start_s"] * sr)), int(round(span["end_s"] * sr))
        if idx in inserts:
            context = dsp.context_around(vocals, sr, span["start_s"], span["end_s"])
            insert = _mark(dsp.match_loudness(inserts[idx], context, sr), sr)
        else:
            insert = np.zeros((1, (b - a) + 2 * x), np.float32)   # a deletion keeps the length
        edits.append((span["start_s"], span["end_s"], insert))
        states[idx]["status"] = "done"
    update_render(sid, rid, spans=states)
    new_vocals = dsp.apply_splices(vocals, sr, edits, params["crossfade_ms"])
    write_audio(rdir / "vocals.edited.wav", new_vocals, sr)
    return song_mix(source, vocals, new_vocals, instrumental, params["vocal_gain_db"])
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run pytest tests/test_audio_edit_render.py tests/test_audio_edit_service.py -q`
Expected: `47 passed`.

- [ ] **Step 5: Commit**

```powershell
git add backend/services/audio_edit.py tests/test_audio_edit_render.py
git commit -m "Audio Edit: song render pipeline and remix"
```

## Task 11: `/audio-edit` router, registration and router tests

**Files:**
- Create: `backend/api/routers/audio_edit.py`
- Modify:
  - `backend/services/audio_edit.py`: append `session_summary` and `list_session_summaries`
  - `backend/main.py`: import and register the router after `voice_lab`
- Test: `tests/test_audio_edit_router.py`

**Interfaces:**
- Consumes:
  - Every public function of `services.audio_edit` from Tasks 3 and 7 to 10
  - `api.dependencies.require_admin` and `require_desktop`
  - `core.path_security.upload_suffix`
- Produces 16 routes, all admin-only and desktop-only. Error bodies are `{"detail": {"code", "message"}}`.

| Method | Path | Body / query | Returns |
|---|---|---|---|
| GET | `/audio-edit/engines` | | `{engines[], separation, gpu}` |
| POST | `/audio-edit/licence/ack` | `{engine, accepted: true}` | `{engine, required, accepted_at}` |
| DELETE | `/audio-edit/licence/ack/{engine}` | | `{engine, required, accepted_at: null}` |
| POST | `/audio-edit/sessions` | multipart `file`, `mode`, `name?` | `{session, job}` (a song auto-starts separation) |
| GET | `/audio-edit/sessions` | `mode?`, `limit` 1..200 (default 50), `offset` | `{items[], total}` |
| GET | `/audio-edit/sessions/{session_id}` | | session with `renders` |
| DELETE | `/audio-edit/sessions/{session_id}` | | `{ok: true}` |
| POST | `/audio-edit/sessions/{session_id}/separate` | | `{job}` |
| POST | `/audio-edit/sessions/{session_id}/transcribe` | | `{job}` |
| PUT | `/audio-edit/sessions/{session_id}/spans` | `{spans: [...]}` | session |
| POST | `/audio-edit/sessions/{session_id}/render` | render request | `{job, render}` |
| GET | `/audio-edit/sessions/{session_id}/renders/{render_id}` | | render |
| GET | `/audio-edit/sessions/{session_id}/renders/{render_id}/audio` | | `audio/wav` attachment |
| GET | `/audio-edit/sessions/{session_id}/audio/{kind}` | `kind` ∈ source, vocals, instrumental | `audio/wav` |
| GET | `/audio-edit/jobs/{job_id}` | | job |
| POST | `/audio-edit/jobs/{job_id}/cancel` | | job |

`DELETE /sessions/{id}` returns `{ok: true}` rather than 204, because the renderer's `apiJson` always parses a body. This matches `DELETE /voice-conversion/voices/{id}`.

- [ ] **Step 1: Write the failing tests**

`tests/test_audio_edit_router.py`:

```python
"""HTTP surface of /audio-edit with fake decode, ASR, engines and Demucs."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services import audio_edit as ae
from services import audio_edit_dsp as dsp

SR = ae.SAMPLE_RATE
WORDS = [
    {"word": "the", "start": 0.10, "end": 0.30},
    {"word": "quick", "start": 0.35, "end": 0.70},
    {"word": "brown", "start": 0.80, "end": 1.20},
]


def _fake_decode(src: Path, dst: Path, channels: int) -> float:
    data, sr = sf.read(str(src), always_2d=True, dtype="float32")
    audio = dsp.match_channels(data.T, channels)
    sf.write(str(dst), audio.T, sr, subtype="FLOAT")
    return audio.shape[1] / sr


def _wav(seconds: float = 2.0, channels: int = 1) -> bytes:
    rng = np.random.default_rng(2)
    buf = io.BytesIO()
    sf.write(buf, (rng.standard_normal((int(seconds * SR), channels)) * 0.1).astype(np.float32),
             SR, format="WAV", subtype="FLOAT")
    return buf.getvalue()


class _Editor:
    def edit(self, payload, on_progress=None):
        seconds = sf.info(payload["audio_path"]).frames / SR
        (p0, p1), dur = payload["parts"][0], payload["fix_duration"][0]
        n = int(round((seconds - (p1 - p0) + dur) * 24000))
        sf.write(payload["out_path"], np.full(n, 0.05, np.float32), 24000, subtype="FLOAT")
        return {"path": payload["out_path"], "sample_rate": 24000, "regions": [[p0, p0 + dur]]}

    def kill(self):
        pass

    def shutdown(self):
        pass


@pytest.fixture
def client(tmp_path, monkeypatch):
    from api.dependencies import require_admin, require_desktop
    from api.routers import audio_edit as router_module
    from services import audio_edit_engines as aee
    from services import watermark

    monkeypatch.setenv("OMNIVOICE_AUDIO_EDIT_DIR", str(tmp_path / "audio_edit"))
    monkeypatch.setattr(ae, "decode_to_wav", _fake_decode)
    monkeypatch.setattr(ae, "RUN_INLINE", True)
    monkeypatch.setattr(ae, "_free_vram", lambda: None)
    monkeypatch.setattr(ae, "_release_gpu", lambda: None)
    monkeypatch.setattr(ae, "_registry_records", lambda: {})
    monkeypatch.setattr(ae, "_asr_transcribe", lambda path: {"language": "en", "words": WORDS})
    monkeypatch.setattr(ae, "_edit_engine", lambda engine_id: _Editor())
    monkeypatch.setattr(ae, "separation_status", lambda: {"available": True, "weights": "w", "missing": None})
    monkeypatch.setattr(aee, "engine_status", lambda e: {
        "installed": True, "install_id": e, "needs_gpu": False, "available": True, "reason": None})
    monkeypatch.setattr(aee, "has_gpu", lambda: True)
    monkeypatch.setattr(watermark, "mark_synthetic", lambda waveform, sample_rate, *, context, force=False: waveform)
    saved: dict = {}
    monkeypatch.setattr(ae, "_read_acks", lambda: dict(saved))
    monkeypatch.setattr(ae, "_write_acks", lambda value: (saved.clear(), saved.update(value)))
    app = FastAPI()
    app.include_router(router_module.router)
    app.dependency_overrides[require_admin] = lambda: None
    app.dependency_overrides[require_desktop] = lambda: None
    ae._JOBS.clear()
    yield TestClient(app)
    ae._JOBS.clear()


def _create(client, mode="speech", channels=1):
    res = client.post("/audio-edit/sessions", data={"mode": mode, "name": "Take"},
                      files={"file": ("take.wav", _wav(channels=channels), "audio/wav")})
    assert res.status_code == 200, res.text
    return res.json()


def _ready_speech(client) -> str:
    sid = _create(client)["session"]["id"]
    assert client.post(f"/audio-edit/sessions/{sid}/transcribe").json()["job"]["status"] == "succeeded"
    span = {"id": "s1", "start_s": 0.35, "end_s": 0.70, "original_text": "quick", "replacement_text": "slow"}
    assert client.put(f"/audio-edit/sessions/{sid}/spans", json={"spans": [span]}).status_code == 200
    return sid


def test_router_is_admin_and_desktop_only():
    from api.dependencies import require_admin, require_desktop
    from api.routers import audio_edit as router_module

    deps = {d.dependency for d in router_module.router.dependencies}
    assert {require_admin, require_desktop} <= deps
    assert len(router_module.router.routes) == 16


def test_engines_listing(client):
    body = client.get("/audio-edit/engines").json()
    assert [e["id"] for e in body["engines"]] == ["f5tts", "ace_step", "yingmusic", "vevo2"]
    assert body["separation"]["available"] is True and body["gpu"] is True


def test_speech_session_flow(client):
    created = _create(client)
    assert created["job"] is None and created["session"]["stage"] == "uploaded"
    sid = _ready_speech(client)
    session = client.get(f"/audio-edit/sessions/{sid}").json()
    assert session["stage"] == "ready" and session["spans"][0]["replacement_text"] == "slow"
    bad = client.put(f"/audio-edit/sessions/{sid}/spans", json={"spans": [
        {"id": "a", "start_s": 0.1, "end_s": 0.5, "original_text": "x", "replacement_text": "y"},
        {"id": "b", "start_s": 0.4, "end_s": 0.9, "original_text": "x", "replacement_text": "y"}]})
    assert bad.status_code == 400 and bad.json()["detail"]["code"] == "invalid_spans"
    audio = client.get(f"/audio-edit/sessions/{sid}/audio/source")
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
    missing = client.get(f"/audio-edit/sessions/{sid}/audio/vocals")
    assert missing.status_code == 404 and missing.json()["detail"]["code"] == "not_found"
    listing = client.get("/audio-edit/sessions", params={"mode": "speech"}).json()
    assert listing["total"] == 2 and listing["items"][0]["render_count"] == 0
    assert listing["items"][0]["size_bytes"] > 0
    assert client.get("/audio-edit/sessions", params={"mode": "song"}).json()["total"] == 0


def test_upload_rejects_bad_type_and_mode(client):
    exe = client.post("/audio-edit/sessions", data={"mode": "speech"},
                      files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert exe.status_code == 415 and exe.json()["detail"]["code"] == "invalid_audio"
    mode = client.post("/audio-edit/sessions", data={"mode": "podcast"},
                       files={"file": ("a.wav", _wav(), "audio/wav")})
    assert mode.status_code == 422 and mode.json()["detail"]["code"] == "invalid_mode"


def test_song_upload_starts_separation(client, monkeypatch):
    monkeypatch.setattr(ae, "separate_stems", lambda path: (
        ae.read_audio(path)[0].mean(axis=0, keepdims=True), ae.read_audio(path)[0]))
    body = _create(client, mode="song", channels=2)
    assert body["job"]["kind"] == "separate" and body["job"]["status"] == "succeeded"
    assert body["session"]["stage"] == "separated"


def test_render_flow_licence_and_jobs(client):
    sid = _ready_speech(client)
    denied = client.post(f"/audio-edit/sessions/{sid}/render", json={"engine": "f5tts"})
    assert denied.status_code == 403 and denied.json()["detail"]["code"] == "licence_ack_required"
    assert client.post("/audio-edit/licence/ack", json={"engine": "f5tts", "accepted": False}).status_code == 422
    ack = client.post("/audio-edit/licence/ack", json={"engine": "f5tts", "accepted": True}).json()
    assert ack["required"] is True and ack["accepted_at"]
    res = client.post(f"/audio-edit/sessions/{sid}/render", json={"engine": "f5tts", "seed": 1})
    assert res.status_code == 200, res.text
    job, render = res.json()["job"], res.json()["render"]
    assert job["status"] == "succeeded" and render["status"] == "succeeded"
    assert client.get(f"/audio-edit/sessions/{sid}/renders/{render['id']}").json()["id"] == render["id"]
    audio = client.get(f"/audio-edit/sessions/{sid}/renders/{render['id']}/audio")
    assert audio.status_code == 200 and "edited" in audio.headers["content-disposition"]
    assert client.get(f"/audio-edit/jobs/{job['id']}").json()["status"] == "succeeded"
    assert client.post(f"/audio-edit/jobs/{job['id']}/cancel").json()["status"] == "succeeded"
    assert client.get("/audio-edit/jobs/" + "0" * 32).status_code == 404
    assert client.delete("/audio-edit/licence/ack/f5tts").json()["accepted_at"] is None


def test_delete_session(client):
    sid = _create(client)["session"]["id"]
    assert client.delete(f"/audio-edit/sessions/{sid}").json() == {"ok": True}
    assert client.get(f"/audio-edit/sessions/{sid}").status_code == 404
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `uv run pytest tests/test_audio_edit_router.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.routers.audio_edit'`.

- [ ] **Step 3: Add the listing helpers** (append to `backend/services/audio_edit.py`)

```python
# ── listings ─────────────────────────────────────────────────────────────

def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def session_summary(session: dict) -> dict:
    renders = session.get("renders") or []
    latest = renders[0] if renders else None
    return {
        "id": session["id"], "mode": session["mode"], "name": session["name"],
        "stage": session["stage"], "error": session.get("error"),
        "duration_s": session["duration_s"], "created_at": session["created_at"],
        "render_count": len(renders),
        "latest_render": ({k: latest.get(k) for k in
                           ("id", "engine", "status", "commercial_use", "output_path", "created_at")}
                          if latest else None),
        "size_bytes": _dir_size(session_dir(session["id"])),
    }


def list_session_summaries(mode: Optional[str] = None, limit: int = 50, offset: int = 0) -> dict:
    if mode is not None and mode not in MODES:
        raise AudioEditError("invalid_mode", "Mode must be speech or song.", 422)
    sessions = [s for s in list_sessions() if mode is None or s["mode"] == mode]
    return {"items": [session_summary(s) for s in sessions[offset:offset + limit]], "total": len(sessions)}
```

- [ ] **Step 4: Create `backend/api/routers/audio_edit.py`**

```python
"""Audio Edit: regenerate chosen words in speech or sung vocals.

Admin-only and desktop-only. Sessions live under data/audio_edit/; the
heavy work runs on background jobs the UI polls through /jobs/{id}.
"""
from __future__ import annotations

import shutil
import uuid
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from api.dependencies import require_admin, require_desktop
from core.path_security import upload_suffix

router = APIRouter(
    prefix="/audio-edit",
    tags=["audio-edit"],
    dependencies=[Depends(require_admin), Depends(require_desktop)],
)

_MAX_UPLOAD_BYTES = 2 * 1024 ** 3
_CHUNK_BYTES = 1024 * 1024


def _svc():
    from services import audio_edit

    return audio_edit


def _call(fn, *args, **kwargs):
    ae = _svc()
    try:
        return fn(*args, **kwargs)
    except ae.AudioEditError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail()) from None


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


async def _save_upload(upload: UploadFile, dest) -> None:
    written = 0
    with open(dest, "wb") as fh:
        while chunk := await upload.read(_CHUNK_BYTES):
            written += len(chunk)
            if written > _MAX_UPLOAD_BYTES:
                raise _error(413, "too_large", "Uploads are limited to 2 GB.")
            fh.write(chunk)
    if written == 0:
        raise _error(400, "invalid_audio", "The uploaded file is empty.")


@router.get("/engines")
def list_engines() -> dict:
    return _call(_svc().engines_overview)


@router.post("/licence/ack")
def acknowledge_licence(body: dict = Body(...)) -> dict:
    if body.get("accepted") is not True:
        raise _error(422, "invalid_request", "Tick the acknowledgement first.")
    return _call(_svc().acknowledge_licence, body.get("engine"))


@router.delete("/licence/ack/{engine}")
def revoke_licence(engine: str) -> dict:
    return _call(_svc().revoke_licence, engine)


@router.post("/sessions")
async def create_session(
    file: UploadFile = File(...),
    mode: str = Form(...),
    name: str = Form("", max_length=120),
) -> dict:
    ae = _svc()
    ext = (upload_suffix(file.filename or "") or "").lower()
    if ext not in ae.ALLOWED_UPLOAD_EXTS:
        raise _error(415, "invalid_audio", "Upload a WAV, FLAC, MP3, M4A, OGG, Opus or video file.")
    if mode not in ae.MODES:
        raise _error(422, "invalid_mode", "Mode must be speech or song.")
    incoming = ae.root() / f".incoming-{uuid.uuid4().hex}"
    incoming.mkdir(parents=True, exist_ok=True)
    try:
        dest = incoming / f"upload{ext}"
        await _save_upload(file, dest)
        session = await run_in_threadpool(_call, ae.create_session, mode, name, dest, file.filename or "")
    finally:
        shutil.rmtree(incoming, ignore_errors=True)
    job = _call(ae.start_separation, session["id"]) if session["mode"] == "song" else None
    return {"session": _call(ae.load_session, session["id"]), "job": job.to_dict() if job else None}


@router.get("/sessions")
def list_sessions(
    mode: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    return _call(_svc().list_session_summaries, mode, limit, offset)


@router.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    return _call(_svc().load_session, session_id)


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    _call(_svc().delete_session, session_id)
    return {"ok": True}


@router.post("/sessions/{session_id}/separate")
def separate(session_id: str) -> dict:
    return {"job": _call(_svc().start_separation, session_id).to_dict()}


@router.post("/sessions/{session_id}/transcribe")
def transcribe(session_id: str) -> dict:
    return {"job": _call(_svc().start_transcription, session_id).to_dict()}


@router.put("/sessions/{session_id}/spans")
def put_spans(session_id: str, body: dict = Body(...)) -> dict:
    return _call(_svc().set_spans, session_id, body.get("spans"))


@router.post("/sessions/{session_id}/render")
def render(session_id: str, body: dict[str, Any] = Body(...)) -> dict:
    job, record = _call(_svc().start_render, session_id, body)
    return {"job": job.to_dict(), "render": record}


@router.get("/sessions/{session_id}/renders/{render_id}")
def get_render(session_id: str, render_id: str) -> dict:
    ae = _svc()
    _call(ae.session_dir, session_id)
    return _call(ae.load_render, session_id, _call(ae._check_id, render_id, "render"))


@router.get("/sessions/{session_id}/renders/{render_id}/audio")
def render_audio(session_id: str, render_id: str) -> FileResponse:
    ae = _svc()
    path = _call(ae.render_audio_path, session_id, _call(ae._check_id, render_id, "render"))
    name = _call(ae.load_session, session_id)["name"]
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in name).strip() or "audio"
    return FileResponse(path, media_type="audio/wav", filename=f"{safe} (edited).wav")


@router.get("/sessions/{session_id}/audio/{kind}")
def session_audio(session_id: str, kind: str) -> FileResponse:
    return FileResponse(_call(_svc().audio_path, session_id, kind), media_type="audio/wav")


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return _call(_svc().get_job, job_id).to_dict()


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    return _call(_svc().cancel_job, job_id).to_dict()
```

`_check_id(value, what)` is Task 3's id validator. It raises a 404 `AudioEditError` for anything that is not 32 lowercase hex characters, so a render id can never walk out of the session directory.

- [ ] **Step 5: Register the router in `backend/main.py`**

In the `from api.routers import (...)` block, add this directly after the `voice_lab,` line:

```python
    audio_edit,  # Audio Edit: regenerate words in speech and sung vocals (desktop only)
```

In the `_router_modules.extend([...])` list, add `audio_edit,` directly after `voice_lab,`.

- [ ] **Step 6: Run the tests to confirm they pass**

Run: `uv run pytest tests/test_audio_edit_router.py -q`
Expected: `7 passed`.

Run: `uv run pytest tests/test_admin_route_policy.py tests/test_no_orphan_service_modules.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add backend/api/routers/audio_edit.py backend/services/audio_edit.py backend/main.py tests/test_audio_edit_router.py
git commit -m "Audio Edit: /audio-edit router"
```

## Task 12: Watermark coverage, route inventory and OpenAPI contract

**Files:**
- Modify:
  - `tests/test_watermark_route_coverage.py`
  - `tests/fixtures/api_routes.txt` (regenerated)
  - `mobile/contract/openapi.json` (regenerated)

**Interfaces:**
- Consumes: Task 9 `_engine_edit(`, Task 10 `_ace_repaint(` and the Task 11 router.
- Produces: CI guards that fail if Audio Edit ever stops calling `mark_synthetic`, and an up-to-date route and contract snapshot.

- [ ] **Step 1: Teach the watermark guard about Audio Edit's synthesis calls**

In `tests/test_watermark_route_coverage.py`, extend `_SYNTH_CALL` by adding two alternatives after the `synthesize_chapter` line:

```python
    r"|\bsynthesize_chapter\("         # longform chapter assembly
    r"|\b_engine_edit\("               # Audio Edit sidecar inpainting (F5-TTS, YingMusic, Vevo2)
    r"|\b_ace_repaint\("               # Audio Edit ACE-Step repaint
)
```

Then append to `_PRODUCERS`, after `"services/persona_bundle.py",`:

```python
    "services/audio_edit.py",
```

- [ ] **Step 2: Run the guard**

Run: `uv run pytest tests/test_watermark_route_coverage.py -q`
Expected: PASS. `services/audio_edit.py` calls `_engine_edit(` and `_ace_repaint(` and references `mark_synthetic`, so it passes. As a check, temporarily rename `watermark.mark_synthetic` in `_mark` to `watermark.mark_synthetic_x`. `test_known_producer_still_marks[services/audio_edit.py]` must then fail. Revert the rename.

- [ ] **Step 3: Regenerate the route inventory**

Run (PowerShell):

```powershell
$env:OMNIVOICE_MODEL = "test"; uv run python scripts/dump_api_routes.py; Remove-Item Env:OMNIVOICE_MODEL
git diff --stat tests/fixtures/api_routes.txt
```

Expected: exactly 16 added lines, and nothing removed.

```
DELETE /audio-edit/licence/ack/{engine}
DELETE /audio-edit/sessions/{session_id}
GET /audio-edit/engines
GET /audio-edit/jobs/{job_id}
GET /audio-edit/sessions
GET /audio-edit/sessions/{session_id}
GET /audio-edit/sessions/{session_id}/audio/{kind}
GET /audio-edit/sessions/{session_id}/renders/{render_id}
GET /audio-edit/sessions/{session_id}/renders/{render_id}/audio
POST /audio-edit/jobs/{job_id}/cancel
POST /audio-edit/licence/ack
POST /audio-edit/sessions
POST /audio-edit/sessions/{session_id}/render
POST /audio-edit/sessions/{session_id}/separate
POST /audio-edit/sessions/{session_id}/transcribe
PUT /audio-edit/sessions/{session_id}/spans
```

- [ ] **Step 4: Regenerate the mobile OpenAPI contract**

Run: `uv run python scripts/export_openapi.py`
Then: `git diff --stat mobile/contract/openapi.json`
Expected: only `/audio-edit/...` paths and their schemas are added.

- [ ] **Step 5: Run the inventory and contract tests**

Run: `uv run pytest tests -q -k "api_route or openapi or contract or watermark or admin_route"`
Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add tests/test_watermark_route_coverage.py tests/fixtures/api_routes.txt mobile/contract/openapi.json
git commit -m "Audio Edit: watermark guard, route inventory and OpenAPI contract"
```

## Task 13: Renderer API client `lib/api/audio-edit.ts`

**Files:**
- Create: `electron/src/renderer/src/lib/api/audio-edit.ts`
- Test: `electron/src/renderer/src/lib/api/audio-edit.test.ts`

**Interfaces:**
- Consumes: `apiJson`, `apiPath` and `ApiError` from `./client`, and the Task 11 routes.
- Produces, used by Tasks 15 and 16:
  - **Types:** `AudioEditMode`, `AudioEditEngine`, `JobStatus`, `Word`, `Span`, `Session`, `SessionSummary`, `Render`, `EditJob`, `EngineStatus`, `EnginesOverview`, `RenderOptions`
  - **Constants:** `MODE_ENGINES`, `ENGINE_LABELS`, `DEFAULT_RENDER_OPTIONS`, `NUDGE_S`, `MAX_REPLACEMENT_CHARS`, `UPLOAD_ACCEPT`
  - **Pure helpers:** `isLive`, `spanFromWords`, `addSpan`, `nudgeSpan`, `wordInSpan`, `changedSpans`, `toRenderBody`, `toUploadForm`, `errorCode`, `formatSeconds`, `formatBytes`
  - **Endpoint functions:** `getEngines`, `acknowledgeLicence`, `revokeLicence`, `createSession`, `listSessions`, `getSession`, `deleteSession`, `separateSession`, `transcribeSession`, `saveSpans`, `startRender`, `getRender`, `getJob`, `cancelJob`, `installEngine`, `sourceAudioUrl`, `renderAudioUrl`

Spans are edited locally and saved as a list. A span whose replacement still equals its original text is a selection the user has not edited yet. The backend rejects those (`invalid_spans`), so `changedSpans` filters them out before every save and render.

- [ ] **Step 1: Write the failing test**

`electron/src/renderer/src/lib/api/audio-edit.test.ts`:

```ts
import { beforeEach, describe, expect, it, vi } from 'vitest';

const client = vi.hoisted(() => ({ apiJson: vi.fn(async () => ({})) }));
vi.mock('./client', async (importOriginal) => ({
  ...(await importOriginal<typeof import('./client')>()),
  apiJson: client.apiJson,
}));

import { ApiError } from './client';
import {
  addSpan,
  changedSpans,
  DEFAULT_RENDER_OPTIONS,
  errorCode,
  formatSeconds,
  nudgeSpan,
  saveSpans,
  spanFromWords,
  startRender,
  toRenderBody,
  toUploadForm,
  wordInSpan,
  type Span,
  type Word,
} from './audio-edit';

const WORDS: Word[] = [
  { i: 0, word: 'the', start: 0.1, end: 0.3 },
  { i: 1, word: 'quick', start: 0.35, end: 0.7 },
  { i: 2, word: 'brown', start: 0.8, end: 1.2 },
  { i: 3, word: 'fox', start: 1.3, end: 1.6 },
];

beforeEach(() => client.apiJson.mockClear());

describe('spans', () => {
  it('builds a span from a word range in either click order', () => {
    const span = spanFromWords(WORDS, 2, 1, 'a');
    expect(span).toEqual({
      id: 'a', start_s: 0.35, end_s: 1.2, original_text: 'quick brown', replacement_text: 'quick brown',
    });
    expect(wordInSpan(WORDS[1], [span])).toBe(span);
    expect(wordInSpan(WORDS[3], [span])).toBeUndefined();
  });

  it('refuses overlapping spans and keeps them sorted', () => {
    const late = spanFromWords(WORDS, 3, 3, 'late');
    const early = spanFromWords(WORDS, 0, 0, 'early');
    expect(addSpan([late], early)?.map((s) => s.id)).toEqual(['early', 'late']);
    expect(addSpan([late], spanFromWords(WORDS, 2, 3, 'x'))).toBeNull();
  });

  it('nudges edges by 10 ms without crossing neighbours or collapsing', () => {
    const spans: Span[] = [spanFromWords(WORDS, 1, 1, 'a'), spanFromWords(WORDS, 2, 2, 'b')];
    expect(nudgeSpan(spans, 'a', 'end', 0.01, 4)[0].end_s).toBe(0.71);
    expect(nudgeSpan(spans, 'a', 'end', 0.5, 4)[0].end_s).toBe(0.8);
    expect(nudgeSpan(spans, 'a', 'start', 1, 4)[0].start_s).toBe(0.69);
    expect(nudgeSpan(spans, 'a', 'start', -1, 4)[0].start_s).toBe(0);
    expect(nudgeSpan(spans, 'missing', 'start', 0.01, 4)).toBe(spans);
  });

  it('saves only edited spans', async () => {
    const untouched = spanFromWords(WORDS, 0, 0, 'u');
    const edited = { ...spanFromWords(WORDS, 1, 1, 'e'), replacement_text: 'slow' };
    const deleted = { ...spanFromWords(WORDS, 3, 3, 'd'), replacement_text: '' };
    expect(changedSpans([untouched, edited, deleted]).map((s) => s.id)).toEqual(['e', 'd']);
    await saveSpans('abc', changedSpans([untouched, edited]));
    expect(client.apiJson).toHaveBeenCalledWith('/audio-edit/sessions/abc/spans', {
      method: 'PUT',
      body: JSON.stringify({ spans: [edited] }),
    });
  });
});

describe('requests', () => {
  it('maps render options to the API body', async () => {
    const options = { ...DEFAULT_RENDER_OPTIONS.song, rvcModelId: 'alto', rvcPitch: 2, seed: 9 };
    expect(toRenderBody(options)).toEqual({
      engine: 'ace_step', crossfade_ms: 30, rvc_model_id: 'alto', rvc_pitch: 2, vocal_gain_db: 0,
      seed: 9, ace_step: { strength: 0.7, padding_s: 2, caption: '' },
    });
    expect(DEFAULT_RENDER_OPTIONS.speech.engine).toBe('f5tts');
    await startRender('abc', options);
    expect(client.apiJson).toHaveBeenCalledWith('/audio-edit/sessions/abc/render', {
      method: 'POST',
      body: JSON.stringify(toRenderBody(options)),
    });
  });

  it('builds the upload form', () => {
    const form = toUploadForm(new File(['x'], 'take.wav'), 'song', ' Demo ');
    expect(form.get('mode')).toBe('song');
    expect(form.get('name')).toBe('Demo');
    expect((form.get('file') as File).name).toBe('take.wav');
  });

  it('reads stable error codes from API errors', () => {
    const payload = { detail: { code: 'licence_ack_required', message: 'x' } };
    const err = new ApiError(403, 'denied', payload as ConstructorParameters<typeof ApiError>[2]);
    expect(errorCode(err)).toBe('licence_ack_required');
    expect(errorCode(new Error('x'))).toBeNull();
    expect(formatSeconds(75.5)).toBe('1:15.50');
  });
});
```

- [ ] **Step 2: Run the test to confirm it fails**

Run (in `electron/`): `bunx vp test --run src/renderer/src/lib/api/audio-edit.test.ts`
Expected: FAIL with "Failed to resolve import './audio-edit'".

- [ ] **Step 3: Implement `electron/src/renderer/src/lib/api/audio-edit.ts`**

```ts
import { ApiError, apiJson, apiPath } from './client';

const BASE = '/audio-edit';

export type AudioEditMode = 'speech' | 'song';
export type AudioEditEngine = 'f5tts' | 'ace_step' | 'yingmusic' | 'vevo2';
export type AudioKind = 'source' | 'vocals' | 'instrumental';
export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';
export type SessionStage = 'uploaded' | 'separating' | 'separated' | 'transcribing' | 'ready';

export const MODE_ENGINES: Record<AudioEditMode, AudioEditEngine[]> = {
  speech: ['f5tts'],
  song: ['ace_step', 'yingmusic', 'vevo2'],
};
export const ENGINE_LABELS: Record<AudioEditEngine, string> = {
  f5tts: 'F5-TTS',
  ace_step: 'ACE-Step 1.5',
  yingmusic: 'YingMusic-Singer-Plus',
  vevo2: 'Vevo2',
};
export const NUDGE_S = 0.01;
export const MAX_REPLACEMENT_CHARS = 500;
export const UPLOAD_ACCEPT = '.wav,.flac,.mp3,.m4a,.ogg,.opus,.aac,.mp4,.mov,.mkv,.webm';

export interface Word { i: number; word: string; start: number; end: number }
export interface Transcript { language: string | null; text: string; words: Word[] }
export interface Span {
  id: string;
  start_s: number;
  end_s: number;
  original_text: string;
  replacement_text: string;
}
export interface AceStepOptions { strength: number; padding_s: number; caption: string }
export interface RenderParams {
  engine: AudioEditEngine;
  crossfade_ms: number;
  rvc_model_id: string | null;
  rvc_pitch: number;
  vocal_gain_db: number;
  seed: number;
  ace_step: AceStepOptions;
}
export interface Render {
  id: string;
  session_id: string;
  engine: AudioEditEngine;
  status: JobStatus;
  progress: number;
  spans: { span_id: string; status: string; path: string | null }[];
  output_path: string | null;
  error: string | null;
  message: string | null;
  created_at: number;
  finished_at: number | null;
  commercial_use: boolean;
  params: RenderParams;
  span_snapshot: Span[];
}
export interface Session {
  id: string;
  mode: AudioEditMode;
  name: string;
  stage: SessionStage;
  error: string | null;
  duration_s: number;
  sample_rate: number;
  channels: number;
  created_at: number;
  transcript: Transcript | null;
  spans: Span[];
  render_ids: string[];
  renders: Render[];
}
export interface SessionSummary {
  id: string;
  mode: AudioEditMode;
  name: string;
  stage: SessionStage;
  error: string | null;
  duration_s: number;
  created_at: number;
  render_count: number;
  latest_render: Pick<Render, 'id' | 'engine' | 'status' | 'commercial_use' | 'output_path' | 'created_at'> | null;
  size_bytes: number;
}
export interface EditJob {
  id: string;
  session_id: string;
  kind: 'separate' | 'transcribe' | 'render';
  render_id: string | null;
  status: JobStatus;
  stage: string | null;
  progress: number;
  error: string | null;
  message: string | null;
  created_at: number;
}
export interface EngineStatus {
  id: AudioEditEngine;
  mode: AudioEditMode;
  display_name: string;
  installed: boolean;
  install_id: string;
  needs_gpu: boolean;
  available: boolean;
  reason: string | null;
  licence: string;
  needs_ack: boolean;
  acked_at: number | null;
  commercial_use: boolean;
}
export interface EnginesOverview {
  engines: EngineStatus[];
  separation: { available: boolean; weights: string | null; missing: string | null };
  gpu: boolean;
}
export interface RenderOptions {
  engine: AudioEditEngine;
  crossfadeMs: number;
  rvcModelId: string | null;
  rvcPitch: number;
  vocalGainDb: number;
  seed: number | null;
  aceStep: AceStepOptions;
}

const BASE_OPTIONS = {
  crossfadeMs: 30,
  rvcModelId: null,
  rvcPitch: 0,
  vocalGainDb: 0,
  seed: null,
  aceStep: { strength: 0.7, padding_s: 2, caption: '' },
};
export const DEFAULT_RENDER_OPTIONS: Record<AudioEditMode, RenderOptions> = {
  speech: { ...BASE_OPTIONS, engine: 'f5tts' },
  song: { ...BASE_OPTIONS, engine: 'ace_step' },
};

export const isLive = (status: JobStatus) => status === 'queued' || status === 'running';

let spanCounter = 0;
export function newSpanId(): string {
  spanCounter += 1;
  return `s${Date.now().toString(36)}${spanCounter}`;
}

export function spanFromWords(words: Word[], from: number, to: number, id = newSpanId()): Span {
  const [lo, hi] = from <= to ? [from, to] : [to, from];
  const picked = words.slice(lo, hi + 1);
  const text = picked.map((w) => w.word).join(' ');
  return {
    id,
    start_s: picked[0].start,
    end_s: picked[picked.length - 1].end,
    original_text: text,
    replacement_text: text,
  };
}

const overlaps = (a: Span, b: Span) => a.start_s < b.end_s && b.start_s < a.end_s;
const byStart = (a: Span, b: Span) => a.start_s - b.start_s;
const round3 = (v: number) => Math.round(v * 1000) / 1000;

/** The spans with *span* added, or null when it overlaps an existing one. */
export function addSpan(spans: Span[], span: Span): Span[] | null {
  if (spans.some((s) => overlaps(s, span))) return null;
  return [...spans, span].sort(byStart);
}

/** Move one edge, clamped to the neighbours and to a 10 ms minimum length. */
export function nudgeSpan(
  spans: Span[], id: string, edge: 'start' | 'end', deltaS: number, durationS: number,
): Span[] {
  const sorted = [...spans].sort(byStart);
  const i = sorted.findIndex((s) => s.id === id);
  if (i < 0) return spans;
  const span = sorted[i];
  const prevEnd = i > 0 ? sorted[i - 1].end_s : 0;
  const nextStart = i < sorted.length - 1 ? sorted[i + 1].start_s : durationS;
  const next =
    edge === 'start'
      ? { ...span, start_s: round3(Math.min(Math.max(span.start_s + deltaS, prevEnd), span.end_s - NUDGE_S)) }
      : { ...span, end_s: round3(Math.max(Math.min(span.end_s + deltaS, nextStart), span.start_s + NUDGE_S)) };
  return sorted.map((s) => (s.id === id ? next : s));
}

export const wordInSpan = (word: Word, spans: Span[]) =>
  spans.find((s) => word.start >= s.start_s - 1e-3 && word.end <= s.end_s + 1e-3);

/** Spans the user actually changed: the only ones the API accepts. */
export const changedSpans = (spans: Span[]) =>
  spans.filter((s) => s.replacement_text.trim() !== s.original_text.trim());

export function toRenderBody(o: RenderOptions) {
  return {
    engine: o.engine,
    crossfade_ms: Math.round(o.crossfadeMs),
    rvc_model_id: o.rvcModelId,
    rvc_pitch: o.rvcPitch,
    vocal_gain_db: o.vocalGainDb,
    seed: o.seed,
    ace_step: o.aceStep,
  };
}

export function toUploadForm(file: File, mode: AudioEditMode, name = ''): FormData {
  const form = new FormData();
  form.append('file', file);
  form.append('mode', mode);
  form.append('name', name.trim());
  return form;
}

export function errorCode(err: unknown): string | null {
  if (!(err instanceof ApiError)) return null;
  const payload = err.payload as { detail?: unknown; code?: unknown } | undefined;
  const detail = payload?.detail;
  if (detail && typeof detail === 'object' && typeof (detail as { code?: unknown }).code === 'string') {
    return (detail as { code: string }).code;
  }
  return typeof payload?.code === 'string' ? payload.code : null;
}

export function formatSeconds(s: number): string {
  const minutes = Math.floor(s / 60);
  return `${minutes}:${(s - minutes * 60).toFixed(2).padStart(5, '0')}`;
}

export function formatBytes(n: number): string {
  if (n < 1024 * 1024) return `${Math.max(1, Math.round(n / 1024))} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

const json = (method: string, body: unknown): RequestInit => ({ method, body: JSON.stringify(body) });
const sid = (id: string) => `${BASE}/sessions/${encodeURIComponent(id)}`;

export const getEngines = (signal?: AbortSignal) => apiJson<EnginesOverview>(`${BASE}/engines`, { signal });
export const acknowledgeLicence = (engine: AudioEditEngine) =>
  apiJson<{ engine: AudioEditEngine; required: boolean; accepted_at: number | null }>(
    `${BASE}/licence/ack`, json('POST', { engine, accepted: true }));
export const revokeLicence = (engine: AudioEditEngine) =>
  apiJson<{ engine: AudioEditEngine; required: boolean; accepted_at: null }>(
    `${BASE}/licence/ack/${encodeURIComponent(engine)}`, { method: 'DELETE' });
export const createSession = (file: File, mode: AudioEditMode, name = '') =>
  apiJson<{ session: Session; job: EditJob | null }>(`${BASE}/sessions`, {
    method: 'POST', body: toUploadForm(file, mode, name),
  });
export const listSessions = (signal?: AbortSignal) =>
  apiJson<{ items: SessionSummary[]; total: number }>(`${BASE}/sessions?limit=200`, { signal });
export const getSession = (id: string, signal?: AbortSignal) => apiJson<Session>(sid(id), { signal });
export const deleteSession = (id: string) => apiJson<{ ok: boolean }>(sid(id), { method: 'DELETE' });
export const separateSession = (id: string) => apiJson<{ job: EditJob }>(`${sid(id)}/separate`, { method: 'POST' });
export const transcribeSession = (id: string) => apiJson<{ job: EditJob }>(`${sid(id)}/transcribe`, { method: 'POST' });
export const saveSpans = (id: string, spans: Span[]) => apiJson<Session>(`${sid(id)}/spans`, json('PUT', { spans }));
export const startRender = (id: string, options: RenderOptions) =>
  apiJson<{ job: EditJob; render: Render }>(`${sid(id)}/render`, json('POST', toRenderBody(options)));
export const getRender = (id: string, renderId: string, signal?: AbortSignal) =>
  apiJson<Render>(`${sid(id)}/renders/${encodeURIComponent(renderId)}`, { signal });
export const getJob = (jobId: string, signal?: AbortSignal) =>
  apiJson<EditJob>(`${BASE}/jobs/${encodeURIComponent(jobId)}`, { signal });
export const cancelJob = (jobId: string) =>
  apiJson<EditJob>(`${BASE}/jobs/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' });
/** Starts the shared sidecar installer; progress shows in Settings → Models. */
export const installEngine = (installId: string) =>
  apiJson<Record<string, unknown>>(`/engines/sidecar/${encodeURIComponent(installId)}/install`, { method: 'POST' });
export const sourceAudioUrl = (id: string, kind: AudioKind) => apiPath(`${sid(id)}/audio/${kind}`);
export const renderAudioUrl = (id: string, renderId: string) =>
  apiPath(`${sid(id)}/renders/${encodeURIComponent(renderId)}/audio`);
```

- [ ] **Step 4: Run the test to confirm it passes**

Run (in `electron/`): `bunx vp test --run src/renderer/src/lib/api/audio-edit.test.ts`
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```powershell
git add electron/src/renderer/src/lib/api/audio-edit.ts electron/src/renderer/src/lib/api/audio-edit.test.ts
git commit -m "Audio Edit: renderer API client"
```

## Task 14: Audio Edit feature components

**Files** (all under `electron/src/renderer/src/features/audio-edit/`):
- Create:
  - `use-audio-edit-session.ts`
  - `licence-gate.tsx`
  - `transcript-spans.tsx`
  - `span-list.tsx`
  - `render-controls.tsx`
  - `edit-workspace.tsx`
  - `speech-tab.tsx`
  - `song-tab.tsx`
  - `history-tab.tsx`
  - `audio-edit-page.tsx`
- Test: `audio-edit.test.tsx`

**Interfaces:**
- Consumes:
  - Task 13 `@/lib/api/audio-edit`
  - `listVoices` from `@/lib/api/voice-conversion` (RVC models)
  - `SearchableSelect` from `@shared/components/SearchableSelect`
  - `Button`, `Input` and `Badge` from `@/components/ui/*`
  - `Tabs*` from `@/components/ui/tabs`
  - `WorkspaceHeader`, `FreeGpuButton`
  - `Link` from `@tanstack/react-router`
- Produces:
  - `AudioEditPage`, the route component Task 15 lazy-loads
  - The `audioEdit.*` keys Task 16 translates. Every `t('audioEdit.…')` call in this task appears in Task 16's English block.

UI behaviour, per the spec:
- **Upload and auto-start.** Drop or pick a file. Speech starts transcription right away. Songs separate first, then transcribe as soon as separation succeeds.
- **Selecting words.** The first click on a word sets an anchor, and the second click closes the span (the same word twice gives a one-word span). An overlapping selection shows `audioEdit.overlap`.
- **Editing spans.**
  - Each span has a replacement input; leaving it empty deletes the words.
  - ±10 ms nudges on each edge.
  - A remove button.
  - Spans save, debounced, on every change. Only edited spans are sent.
- **Before render.**
  - An engine that is not installed (or lacks a GPU) shows an install button and a link to Settings → Models instead of the render button.
  - An engine that needs a licence acknowledgement shows the gate.
  - An acknowledged engine shows a "Revoke acknowledgement" button.
- **Progress and results.**
  - A running job shows its stage, progress and a Cancel button.
  - The latest render shows the status, the Commercial/Non-commercial badge, original and edited players, and Export.
- **Song mode** adds:
  - Stem players for vocals and instrumental
  - An engine picker
  - Vocal gain
  - ACE-Step strength, padding and caption
  - RVC on by default when at least one RVC model exists

- [ ] **Step 1: Write the failing component tests**

`audio-edit.test.tsx`:

```tsx
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type { EngineStatus, Session, SessionSummary } from '@/lib/api/audio-edit';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, params?: Record<string, unknown>) => (params ? `${key} ${JSON.stringify(params)}` : key),
    i18n: { language: 'en' },
  }),
}));

const state = vi.hoisted(() => ({
  engines: undefined as unknown,
  session: undefined as unknown,
  sessions: undefined as unknown,
}));
vi.mock('@tanstack/react-query', () => ({
  useQueryClient: () => ({ invalidateQueries: vi.fn(async () => undefined) }),
  useQuery: ({ queryKey }: { queryKey: readonly string[] }) => {
    const data =
      queryKey[1] === 'engines' ? state.engines
        : queryKey[1] === 'session' ? state.session
          : queryKey[1] === 'sessions' ? state.sessions
            : queryKey[1] === 'rvc-voices' ? [] : undefined;
    return { data, isLoading: false, isError: false };
  },
}));
vi.mock('@tanstack/react-router', () => ({
  Link: ({ to, children }: { to: string; children: React.ReactNode }) => <a href={to}>{children}</a>,
}));
vi.mock('@/components/app-shell/workspace-header', () => ({
  WorkspaceHeader: ({ children }: { children: React.ReactNode }) => <header>{children}</header>,
}));
vi.mock('@/components/free-gpu-button', () => ({ FreeGpuButton: () => <button type="button">free</button> }));
vi.mock('@shared/components/SearchableSelect', () => ({
  default: ({ ariaLabel, value }: { ariaLabel: string; value: string }) => (
    <div role="combobox" aria-label={ariaLabel} data-value={value} />
  ),
}));
vi.mock('@/lib/api/audio-edit', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/lib/api/audio-edit')>()),
  acknowledgeLicence: vi.fn(async () => ({ engine: 'f5tts', required: true, accepted_at: 1 })),
  deleteSession: vi.fn(async () => ({ ok: true })),
  saveSpans: vi.fn(async () => ({})),
  installEngine: vi.fn(async () => ({})),
}));

import * as api from '@/lib/api/audio-edit';
import { AudioEditPage } from './audio-edit-page';
import { EditWorkspace } from './edit-workspace';
import { HistoryTab } from './history-tab';
import { SpanList } from './span-list';
import { TranscriptSpans } from './transcript-spans';

const WORDS = [
  { i: 0, word: 'the', start: 0.1, end: 0.3 },
  { i: 1, word: 'quick', start: 0.35, end: 0.7 },
  { i: 2, word: 'brown', start: 0.8, end: 1.2 },
];

function engine(patch: Partial<EngineStatus> = {}): EngineStatus {
  return {
    id: 'f5tts', mode: 'speech', display_name: 'F5-TTS', installed: true, install_id: 'f5tts',
    needs_gpu: false, available: true, reason: null, licence: 'CC-BY-NC-4.0 / MIT', needs_ack: true,
    acked_at: null, commercial_use: false, ...patch,
  };
}

function session(): Session {
  return {
    id: 'a'.repeat(32), mode: 'speech', name: 'Take', stage: 'ready', error: null, duration_s: 4,
    sample_rate: 44100, channels: 1, created_at: 1, render_ids: [], renders: [],
    transcript: { language: 'en', text: 'the quick brown', words: WORDS },
    spans: [{ id: 's1', start_s: 0.35, end_s: 0.7, original_text: 'quick', replacement_text: 'slow' }],
  };
}

beforeEach(() => {
  state.engines = { engines: [engine()], separation: { available: true, weights: 'w', missing: null }, gpu: true };
  state.session = session();
  state.sessions = { items: [], total: 0 };
});
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

it('renders the page with speech, song and history tabs', () => {
  render(<AudioEditPage />);
  expect(screen.getByTestId('audio-edit-page')).toBeInTheDocument();
  for (const tab of ['speech', 'song', 'history']) {
    expect(screen.getByRole('tab', { name: `audioEdit.tabs.${tab}` })).toBeInTheDocument();
  }
  expect(screen.getByRole('button', { name: 'audioEdit.choose_file' })).toBeInTheDocument();
});

it('creates a span from two word clicks', () => {
  const onWord = vi.fn();
  render(<TranscriptSpans words={WORDS} spans={[]} anchor={null} onWord={onWord} />);
  fireEvent.click(screen.getByRole('button', { name: 'quick' }));
  expect(onWord).toHaveBeenCalledWith(1);
});

it('nudges, edits and removes spans', () => {
  const onChange = vi.fn();
  const spans = session().spans;
  render(<SpanList spans={spans} durationS={4} onChange={onChange} />);
  fireEvent.click(screen.getAllByRole('button', { name: /audioEdit.nudge_later/ })[1]);
  expect(onChange).toHaveBeenLastCalledWith([{ ...spans[0], end_s: 0.71 }]);
  fireEvent.change(screen.getByRole('textbox'), { target: { value: '' } });
  expect(onChange).toHaveBeenLastCalledWith([{ ...spans[0], replacement_text: '' }]);
  fireEvent.click(screen.getByRole('button', { name: 'audioEdit.remove_span' }));
  expect(onChange).toHaveBeenLastCalledWith([]);
});

it('gates the render behind the licence acknowledgement', () => {
  render(<EditWorkspace mode="speech" sessionId={'a'.repeat(32)} onSessionChange={vi.fn()} />);
  expect(screen.getByTestId('audio-edit-licence-gate')).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'audioEdit.render' })).toBeNull();
  const accept = screen.getByRole('button', { name: 'audioEdit.licence_accept' });
  expect(accept).toBeDisabled();
  fireEvent.click(screen.getByRole('checkbox'));
  fireEvent.click(accept);
  expect(api.acknowledgeLicence).toHaveBeenCalledWith('f5tts');
});

it('shows the render button once acknowledged and an install prompt when missing', () => {
  state.engines = { engines: [engine({ acked_at: 5 })], separation: { available: true, weights: 'w', missing: null }, gpu: true };
  const { unmount } = render(<EditWorkspace mode="speech" sessionId={'a'.repeat(32)} onSessionChange={vi.fn()} />);
  expect(screen.getByRole('button', { name: 'audioEdit.render' })).toBeEnabled();
  expect(screen.getByRole('button', { name: 'audioEdit.licence_revoke' })).toBeInTheDocument();
  unmount();
  state.engines = {
    engines: [engine({ installed: false, available: false, reason: 'Install F5-TTS to use it.' })],
    separation: { available: true, weights: 'w', missing: null }, gpu: true,
  };
  render(<EditWorkspace mode="speech" sessionId={'a'.repeat(32)} onSessionChange={vi.fn()} />);
  expect(screen.getByTestId('audio-edit-engine-missing')).toBeInTheDocument();
  expect(screen.getByRole('link', { name: 'audioEdit.open_models' })).toHaveAttribute('href', '/settings/models');
  fireEvent.click(screen.getByRole('button', { name: 'audioEdit.install' }));
  expect(api.installEngine).toHaveBeenCalledWith('f5tts');
});

it('lists history with open and confirmed delete', () => {
  const item: SessionSummary = {
    id: 'b'.repeat(32), mode: 'song', name: 'Demo', stage: 'ready', error: null, duration_s: 30,
    created_at: 1_700_000_000, render_count: 1, size_bytes: 2_000_000,
    latest_render: { id: 'c'.repeat(32), engine: 'ace_step', status: 'succeeded', commercial_use: false,
      output_path: 'renders/x/output.wav', created_at: 1 },
  };
  state.sessions = { items: [item], total: 1 };
  const onOpen = vi.fn();
  render(<HistoryTab onOpen={onOpen} />);
  expect(screen.getByText('Demo')).toBeInTheDocument();
  expect(screen.getByTestId('audio-edit-badge')).toHaveTextContent('audioEdit.badge_noncommercial');
  fireEvent.click(screen.getByRole('button', { name: 'audioEdit.open' }));
  expect(onOpen).toHaveBeenCalledWith('song', item.id);
  fireEvent.click(screen.getByRole('button', { name: 'audioEdit.delete' }));
  fireEvent.click(screen.getByRole('button', { name: 'audioEdit.confirm_delete' }));
  expect(api.deleteSession).toHaveBeenCalledWith(item.id);
});
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run (in `electron/`): `bunx vp test --run src/renderer/src/features/audio-edit/audio-edit.test.tsx`
Expected: FAIL with "Failed to resolve import './audio-edit-page'".

- [ ] **Step 3: Write `use-audio-edit-session.ts`**

```ts
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import * as api from '@/lib/api/audio-edit';

export const audioEditKeys = {
  engines: ['audio-edit', 'engines'] as const,
  sessions: ['audio-edit', 'sessions'] as const,
  session: (id: string) => ['audio-edit', 'session', id] as const,
  job: (id: string) => ['audio-edit', 'job', id] as const,
  voices: ['audio-edit', 'rvc-voices'] as const,
};

export function useAudioEditEngines() {
  return useQuery({ queryKey: audioEditKeys.engines, queryFn: ({ signal }) => api.getEngines(signal) });
}

/** The open session plus the job currently working on it (polled every second). */
export function useAudioEditSession(sessionId: string | null) {
  const qc = useQueryClient();
  const [jobId, setJobId] = useState<string | null>(null);
  const session = useQuery({
    queryKey: audioEditKeys.session(sessionId ?? ''),
    queryFn: ({ signal }) => api.getSession(sessionId as string, signal),
    enabled: !!sessionId,
  });
  const job = useQuery({
    queryKey: audioEditKeys.job(jobId ?? ''),
    queryFn: ({ signal }) => api.getJob(jobId as string, signal),
    enabled: !!jobId,
    refetchInterval: (query) => (query.state.data && !api.isLive(query.state.data.status) ? false : 1000),
  });
  const status = job.data?.status;
  useEffect(() => {
    if (!sessionId || !status || api.isLive(status)) return;
    void qc.invalidateQueries({ queryKey: audioEditKeys.session(sessionId) });
    void qc.invalidateQueries({ queryKey: audioEditKeys.sessions });
  }, [status, job.data?.id, sessionId, qc]);
  useEffect(() => setJobId(null), [sessionId]);
  return { session: sessionId ? (session.data ?? null) : null, job: job.data ?? null, setJobId };
}
```

- [ ] **Step 4: Write `licence-gate.tsx`**

```tsx
import { useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { acknowledgeLicence, type EngineStatus } from '@/lib/api/audio-edit';
import { audioEditKeys } from './use-audio-edit-session';

export function CommercialBadge({ commercial }: { commercial: boolean }) {
  const { t } = useTranslation();
  return (
    <Badge variant={commercial ? 'secondary' : 'outline'} data-testid="audio-edit-badge">
      {t(commercial ? 'audioEdit.badge_commercial' : 'audioEdit.badge_noncommercial')}
    </Badge>
  );
}

/** One-time, revocable acknowledgement of a non-commercial engine licence. */
export function LicenceGate({ engine }: { engine: EngineStatus }) {
  const { t } = useTranslation();
  const qc = useQueryClient();
  const [checked, setChecked] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const accept = async () => {
    setError(null);
    try {
      await acknowledgeLicence(engine.id);
      await qc.invalidateQueries({ queryKey: audioEditKeys.engines });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };
  return (
    <div className="space-y-3 rounded-lg border border-amber-500/40 p-4" data-testid="audio-edit-licence-gate">
      <p className="text-sm font-medium">{t('audioEdit.licence_title', { engine: engine.display_name })}</p>
      <p className="text-sm text-muted-foreground">
        {t('audioEdit.licence_body', { engine: engine.display_name, licence: engine.licence })}
      </p>
      <label className="flex items-center gap-2 text-sm">
        <input type="checkbox" checked={checked} onChange={(e) => setChecked(e.target.checked)} />
        {t('audioEdit.licence_checkbox')}
      </label>
      {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
      <Button size="sm" disabled={!checked} onClick={() => void accept()}>
        {t('audioEdit.licence_accept')}
      </Button>
    </div>
  );
}
```

- [ ] **Step 5: Write `transcript-spans.tsx`**

```tsx
import { useTranslation } from 'react-i18next';
import { type Span, type Word, wordInSpan } from '@/lib/api/audio-edit';
import { cn } from '@/lib/utils';

interface Props {
  words: Word[];
  spans: Span[];
  anchor: number | null;
  onWord: (index: number) => void;
}

/** The transcript as clickable words; words inside a span are highlighted. */
export function TranscriptSpans({ words, spans, anchor, onWord }: Props) {
  const { t } = useTranslation();
  if (words.length === 0) return <p className="text-sm text-muted-foreground">{t('audioEdit.no_words')}</p>;
  return (
    <div className="flex flex-wrap gap-1 rounded-lg border p-3 leading-7" aria-label={t('audioEdit.transcript')}
      data-testid="audio-edit-transcript">
      {words.map((word) => (
        <button
          key={word.i}
          type="button"
          onClick={() => onWord(word.i)}
          aria-pressed={anchor === word.i}
          title={`${word.start.toFixed(2)}–${word.end.toFixed(2)} s`}
          className={cn(
            'rounded px-1 text-sm hover:bg-accent',
            wordInSpan(word, spans) && 'bg-primary/15 text-primary',
            anchor === word.i && 'ring-2 ring-primary',
          )}
        >
          {word.word}
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 6: Write `span-list.tsx`**

```tsx
import { Trash2Icon } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { formatSeconds, MAX_REPLACEMENT_CHARS, NUDGE_S, nudgeSpan, type Span } from '@/lib/api/audio-edit';

interface Props {
  spans: Span[];
  durationS: number;
  onChange: (spans: Span[]) => void;
}

export function SpanList({ spans, durationS, onChange }: Props) {
  const { t } = useTranslation();
  if (spans.length === 0) return <p className="text-sm text-muted-foreground">{t('audioEdit.no_spans')}</p>;
  const nudge = (id: string, edge: 'start' | 'end', delta: number) =>
    onChange(nudgeSpan(spans, id, edge, delta, durationS));
  return (
    <ul className="space-y-2" data-testid="audio-edit-spans">
      {spans.map((span) => (
        <li key={span.id} className="grid gap-2 rounded-lg border p-3 sm:grid-cols-[1fr_2fr_auto]">
          <div className="text-sm">
            <p className="font-medium">{span.original_text}</p>
            <p className="text-xs tabular-nums text-muted-foreground">
              {formatSeconds(span.start_s)} – {formatSeconds(span.end_s)}
            </p>
          </div>
          <Input
            value={span.replacement_text}
            maxLength={MAX_REPLACEMENT_CHARS}
            placeholder={t('audioEdit.replacement_placeholder')}
            aria-label={t('audioEdit.replacement_for', { words: span.original_text })}
            onChange={(e) =>
              onChange(spans.map((s) => (s.id === span.id ? { ...s, replacement_text: e.target.value } : s)))
            }
          />
          <div className="flex flex-wrap items-center gap-2">
            {(['start', 'end'] as const).map((edge) => (
              <span key={edge} className="flex items-center gap-0.5 text-xs">
                <span>{t(`audioEdit.${edge}`)}</span>
                <Button size="icon-xs" variant="outline" aria-label={t('audioEdit.nudge_earlier', { edge })}
                  onClick={() => nudge(span.id, edge, -NUDGE_S)}>−</Button>
                <Button size="icon-xs" variant="outline" aria-label={t('audioEdit.nudge_later', { edge })}
                  onClick={() => nudge(span.id, edge, NUDGE_S)}>+</Button>
              </span>
            ))}
            <Button size="icon-sm" variant="ghost" aria-label={t('audioEdit.remove_span')}
              onClick={() => onChange(spans.filter((s) => s.id !== span.id))}>
              <Trash2Icon />
            </Button>
          </div>
        </li>
      ))}
    </ul>
  );
}
```

- [ ] **Step 7: Write `render-controls.tsx`**

```tsx
import SearchableSelect from '@shared/components/SearchableSelect';
import { useQuery } from '@tanstack/react-query';
import { useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { Input } from '@/components/ui/input';
import {
  type AudioEditEngine, type AudioEditMode, ENGINE_LABELS, type EngineStatus, MODE_ENGINES, type RenderOptions,
} from '@/lib/api/audio-edit';
import { listVoices } from '@/lib/api/voice-conversion';
import { audioEditKeys } from './use-audio-edit-session';

interface Props {
  mode: AudioEditMode;
  options: RenderOptions;
  onChange: (options: RenderOptions) => void;
  engines: EngineStatus[];
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1">
      <span className="text-xs font-medium text-muted-foreground">{label}</span>
      {children}
    </div>
  );
}

function Range(props: { label: string; min: number; max: number; step: number; value: number;
  onChange: (value: number) => void }) {
  return (
    <label className="space-y-1 text-xs font-medium text-muted-foreground">
      <span>{props.label}</span>
      <input type="range" className="w-full" min={props.min} max={props.max} step={props.step}
        value={props.value} aria-label={props.label} onChange={(e) => props.onChange(Number(e.target.value))} />
    </label>
  );
}

export function RenderControls({ mode, options, onChange, engines }: Props) {
  const { t } = useTranslation();
  const voices = useQuery({ queryKey: audioEditKeys.voices, queryFn: ({ signal }) => listVoices(signal) });
  const touched = useRef(false);
  const set = (patch: Partial<RenderOptions>) => {
    touched.current = true;
    onChange({ ...options, ...patch });
  };
  const firstVoice = voices.data?.[0]?.id;
  useEffect(() => {
    // Song mode re-voices with RVC by default when a model exists.
    if (mode === 'song' && firstVoice && options.rvcModelId === null && !touched.current) {
      onChange({ ...options, rvcModelId: firstVoice });
    }
  }, [mode, firstVoice, options, onChange]);
  const engineOptions = MODE_ENGINES[mode].map((id) => {
    const status = engines.find((e) => e.id === id);
    return { value: id, label: status?.display_name ?? ENGINE_LABELS[id], disabled: status ? !status.available : false };
  });
  const voiceOptions = [
    { value: '', label: t('audioEdit.rvc_none') },
    ...(voices.data ?? []).map((v) => ({ value: v.id, label: v.name })),
  ];
  return (
    <div className="grid gap-4 sm:grid-cols-2" data-testid="audio-edit-controls">
      <Field label={t('audioEdit.engine')}>
        <SearchableSelect value={options.engine} options={engineOptions} ariaLabel={t('audioEdit.engine')}
          disabled={mode === 'speech'} onChange={(value: string) => set({ engine: value as AudioEditEngine })} />
      </Field>
      <Field label={t('audioEdit.rvc_model')}>
        <SearchableSelect value={options.rvcModelId ?? ''} options={voiceOptions} ariaLabel={t('audioEdit.rvc_model')}
          onChange={(value: string) => set({ rvcModelId: value || null })} />
      </Field>
      {options.rvcModelId && (
        <Range label={t('audioEdit.rvc_pitch', { value: options.rvcPitch })} min={-12} max={12} step={1}
          value={options.rvcPitch} onChange={(value) => set({ rvcPitch: value })} />
      )}
      <Range label={t('audioEdit.crossfade', { value: options.crossfadeMs })} min={5} max={200} step={5}
        value={options.crossfadeMs} onChange={(value) => set({ crossfadeMs: value })} />
      {mode === 'song' && (
        <Range label={t('audioEdit.vocal_gain', { value: options.vocalGainDb })} min={-12} max={6} step={0.5}
          value={options.vocalGainDb} onChange={(value) => set({ vocalGainDb: value })} />
      )}
      {mode === 'song' && options.engine === 'ace_step' && (
        <>
          <Range label={t('audioEdit.ace_strength', { value: options.aceStep.strength.toFixed(2) })} min={0} max={1}
            step={0.05} value={options.aceStep.strength}
            onChange={(value) => set({ aceStep: { ...options.aceStep, strength: value } })} />
          <Range label={t('audioEdit.ace_padding', { value: options.aceStep.padding_s })} min={0} max={10} step={0.5}
            value={options.aceStep.padding_s}
            onChange={(value) => set({ aceStep: { ...options.aceStep, padding_s: value } })} />
          <Field label={t('audioEdit.ace_caption')}>
            <Input value={options.aceStep.caption} maxLength={300} placeholder={t('audioEdit.ace_caption_placeholder')}
              onChange={(e) => set({ aceStep: { ...options.aceStep, caption: e.target.value } })} />
          </Field>
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 8: Write `edit-workspace.tsx`**

```tsx
import { useQueryClient } from '@tanstack/react-query';
import { Link } from '@tanstack/react-router';
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import * as api from '@/lib/api/audio-edit';
import { CommercialBadge, LicenceGate } from './licence-gate';
import { RenderControls } from './render-controls';
import { SpanList } from './span-list';
import { TranscriptSpans } from './transcript-spans';
import { audioEditKeys, useAudioEditEngines, useAudioEditSession } from './use-audio-edit-session';

const KNOWN_ERRORS = new Set([
  'invalid_audio', 'too_long', 'too_large', 'no_word_timestamps', 'invalid_spans', 'engine_not_installed',
  'licence_ack_required', 'ace_step_unavailable', 'separation_unavailable', 'engine_failed', 'busy',
]);

type T = ReturnType<typeof useTranslation>['t'];
function messageFor(err: unknown, t: T): string {
  const code = api.errorCode(err);
  if (code && KNOWN_ERRORS.has(code)) return t(`audioEdit.errors.${code}`);
  return err instanceof Error ? err.message : String(err);
}

interface Props {
  mode: api.AudioEditMode;
  sessionId: string | null;
  onSessionChange: (id: string | null) => void;
}

export function EditWorkspace({ mode, sessionId, onSessionChange }: Props) {
  const { t } = useTranslation();
  const qc = useQueryClient();
  const engines = useAudioEditEngines();
  const { session, job, setJobId } = useAudioEditSession(sessionId);
  const [spans, setSpans] = useState<api.Span[]>([]);
  const [options, setOptions] = useState<api.RenderOptions>(api.DEFAULT_RENDER_OPTIONS[mode]);
  const [anchor, setAnchor] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fail = (err: unknown) => setError(messageFor(err, t));

  const transcriptText = session?.transcript?.text;
  useEffect(() => {
    setSpans(session?.spans ?? []);
    setAnchor(null);
    // Reset local spans only when another session or a new transcript arrives.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session?.id, transcriptText]);

  const jobKind = job?.kind;
  const jobStatus = job?.status;
  const stage = session?.stage;
  useEffect(() => {
    // Songs: transcribe the separated vocals as soon as separation succeeds.
    if (!session || jobKind !== 'separate' || jobStatus !== 'succeeded' || stage !== 'separated') return;
    api.transcribeSession(session.id).then((r) => setJobId(r.job.id), fail);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobKind, jobStatus, stage]);

  const words = session?.transcript?.words ?? [];
  const edited = api.changedSpans(spans);
  const live = job !== null && api.isLive(job.status);
  const engine = engines.data?.engines.find((e) => e.id === options.engine) ?? null;
  const regenerates = edited.some((s) => s.replacement_text.trim() !== '');
  const needsAck = !!engine && engine.needs_ack && !engine.acked_at && regenerates;
  const latest = session?.renders[0] ?? null;

  const upload = async (file: File) => {
    setError(null);
    setBusy(true);
    try {
      const { session: created, job: first } = await api.createSession(file, mode);
      onSessionChange(created.id);
      setJobId(first ? first.id : (await api.transcribeSession(created.id)).job.id);
      void qc.invalidateQueries({ queryKey: audioEditKeys.sessions });
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  const updateSpans = (next: api.Span[]) => {
    setSpans(next);
    if (!session) return;
    const id = session.id;
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => {
      api.saveSpans(id, api.changedSpans(next)).catch(fail);
    }, 400);
  };

  const onWord = (index: number) => {
    if (anchor === null) {
      setAnchor(index);
      return;
    }
    const next = api.addSpan(spans, api.spanFromWords(words, anchor, index));
    setAnchor(null);
    if (!next) {
      setError(t('audioEdit.overlap'));
      return;
    }
    setError(null);
    updateSpans(next);
  };

  const retry = async () => {
    if (!session) return;
    setError(null);
    try {
      const r = session.mode === 'song' && session.stage === 'uploaded'
        ? await api.separateSession(session.id)
        : await api.transcribeSession(session.id);
      setJobId(r.job.id);
    } catch (err) {
      fail(err);
    }
  };

  const startRender = async () => {
    if (!session) return;
    setError(null);
    setBusy(true);
    try {
      if (saveTimer.current) clearTimeout(saveTimer.current);
      await api.saveSpans(session.id, edited);
      const { job: started } = await api.startRender(session.id, options);
      setJobId(started.id);
    } catch (err) {
      fail(err);
      if (api.errorCode(err) === 'licence_ack_required') void qc.invalidateQueries({ queryKey: audioEditKeys.engines });
    } finally {
      setBusy(false);
    }
  };

  const revoke = async () => {
    if (!engine) return;
    try {
      await api.revokeLicence(engine.id);
      await qc.invalidateQueries({ queryKey: audioEditKeys.engines });
    } catch (err) {
      fail(err);
    }
  };

  if (!session) {
    return (
      <div className="space-y-3" data-testid={`audio-edit-${mode}`}>
        {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
        <div
          className="rounded-lg border border-dashed p-8 text-center"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            const file = e.dataTransfer.files[0];
            if (file) void upload(file);
          }}
        >
          <p className="text-sm text-muted-foreground">{t(`audioEdit.upload_hint_${mode}`)}</p>
          <input ref={fileRef} type="file" accept={api.UPLOAD_ACCEPT} className="hidden" data-testid="audio-edit-file"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void upload(file);
              e.target.value = '';
            }} />
          <Button className="mt-3" disabled={busy} onClick={() => fileRef.current?.click()}>
            {t('audioEdit.choose_file')}
          </Button>
          {mode === 'song' && engines.data && !engines.data.separation.available && (
            <p className="mt-2 text-xs text-amber-600">
              {t('audioEdit.separation_missing', { missing: engines.data.separation.missing ?? 'HTDemucs' })}
            </p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-5" data-testid={`audio-edit-${mode}`}>
      <div className="flex flex-wrap items-center gap-3">
        <h3 className="font-medium">{session.name}</h3>
        <span className="text-xs tabular-nums text-muted-foreground">{api.formatSeconds(session.duration_s)}</span>
        <span className="text-xs text-muted-foreground" data-testid="audio-edit-stage">
          {t(`audioEdit.stage.${session.stage}`)}
        </span>
        <span className="flex-1" />
        <Button variant="outline" size="sm" onClick={() => onSessionChange(null)}>{t('audioEdit.new_session')}</Button>
      </div>
      {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
      {session.error && !live && (
        <div role="alert" className="flex items-center gap-2 text-sm text-destructive">
          <span>{t('audioEdit.stage_failed', { error: session.error })}</span>
          <Button size="xs" variant="outline" onClick={() => void retry()}>{t('audioEdit.retry')}</Button>
        </div>
      )}
      {live && job && (
        <div className="flex items-center gap-3 text-sm" data-testid="audio-edit-job">
          <span>{t(`audioEdit.job.${job.kind}`)}</span>
          <progress max={1} value={job.progress} className="h-2 flex-1" />
          <span className="tabular-nums">{Math.round(job.progress * 100)}%</span>
          <Button size="sm" variant="outline" onClick={() => api.cancelJob(job.id).catch(fail)}>
            {t('audioEdit.cancel')}
          </Button>
        </div>
      )}
      {mode === 'song' && (session.stage === 'separated' || session.stage === 'transcribing' || session.stage === 'ready') && (
        <div className="grid gap-2 sm:grid-cols-2">
          <Player label={t('audioEdit.stem_vocals')} src={api.sourceAudioUrl(session.id, 'vocals')} />
          <Player label={t('audioEdit.stem_instrumental')} src={api.sourceAudioUrl(session.id, 'instrumental')} />
        </div>
      )}
      {session.stage === 'ready' && session.transcript && (
        <>
          <p className="text-xs text-muted-foreground">{t('audioEdit.select_hint')} {t('audioEdit.insert_hint')}</p>
          <TranscriptSpans words={words} spans={spans} anchor={anchor} onWord={onWord} />
          <SpanList spans={spans} durationS={session.duration_s} onChange={updateSpans} />
          <RenderControls mode={mode} options={options} onChange={setOptions} engines={engines.data?.engines ?? []} />
          {engine && !engine.available && regenerates ? (
            <EngineMissing engine={engine} />
          ) : needsAck && engine ? (
            <LicenceGate engine={engine} />
          ) : (
            <div className="flex items-center gap-2">
              <Button disabled={edited.length === 0 || live || busy} onClick={() => void startRender()}>
                {t('audioEdit.render')}
              </Button>
              {engine?.needs_ack && engine.acked_at && (
                <Button size="xs" variant="ghost" onClick={() => void revoke()}>{t('audioEdit.licence_revoke')}</Button>
              )}
            </div>
          )}
        </>
      )}
      {latest && <RenderResult sessionId={session.id} render={latest} />}
    </div>
  );
}

function Player({ label, src }: { label: string; src: string }) {
  return (
    <label className="space-y-1 text-xs text-muted-foreground">
      <span>{label}</span>
      <audio controls preload="none" src={src} className="w-full" />
    </label>
  );
}

function EngineMissing({ engine }: { engine: api.EngineStatus }) {
  const { t } = useTranslation();
  const [state, setState] = useState<'idle' | 'started' | 'failed'>('idle');
  return (
    <div className="space-y-2 rounded-lg border p-4 text-sm" data-testid="audio-edit-engine-missing">
      <p>
        {engine.installed
          ? t('audioEdit.engine_needs_gpu', { engine: engine.display_name })
          : t('audioEdit.engine_missing', { engine: engine.display_name })}
      </p>
      {!engine.installed && (
        <div className="flex items-center gap-3">
          <Button size="sm" disabled={state === 'started'}
            onClick={() => api.installEngine(engine.install_id).then(() => setState('started'), () => setState('failed'))}>
            {t('audioEdit.install')}
          </Button>
          <Link to="/settings/models" className="text-sm underline">{t('audioEdit.open_models')}</Link>
        </div>
      )}
      {state === 'started' && <p className="text-xs text-muted-foreground">{t('audioEdit.install_started')}</p>}
      {state === 'failed' && <p className="text-xs text-destructive">{t('audioEdit.install_failed')}</p>}
    </div>
  );
}

function RenderResult({ sessionId, render }: { sessionId: string; render: api.Render }) {
  const { t } = useTranslation();
  const done = render.status === 'succeeded';
  return (
    <div className="space-y-3 rounded-lg border p-4" data-testid="audio-edit-result">
      <div className="flex items-center gap-2 text-sm">
        <span>{t(`audioEdit.render_status.${render.status}`)}</span>
        <CommercialBadge commercial={render.commercial_use} />
      </div>
      {render.status === 'failed' && <p className="text-sm text-destructive">{render.message ?? render.error}</p>}
      {done && (
        <>
          <div className="grid gap-2 sm:grid-cols-2">
            <Player label={t('audioEdit.original')} src={api.sourceAudioUrl(sessionId, 'source')} />
            <Player label={t('audioEdit.edited')} src={api.renderAudioUrl(sessionId, render.id)} />
          </div>
          <a className="text-sm underline" href={api.renderAudioUrl(sessionId, render.id)} download>
            {t('audioEdit.export')}
          </a>
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 9: Write the tab wrappers and history**

`speech-tab.tsx`:

```tsx
import { EditWorkspace } from './edit-workspace';

export function SpeechTab(props: { sessionId: string | null; onSessionChange: (id: string | null) => void }) {
  return <EditWorkspace mode="speech" {...props} />;
}
```

`song-tab.tsx`:

```tsx
import { EditWorkspace } from './edit-workspace';

export function SongTab(props: { sessionId: string | null; onSessionChange: (id: string | null) => void }) {
  return <EditWorkspace mode="song" {...props} />;
}
```

`history-tab.tsx`:

```tsx
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import {
  type AudioEditMode, deleteSession, ENGINE_LABELS, formatBytes, formatSeconds, listSessions, renderAudioUrl,
} from '@/lib/api/audio-edit';
import { CommercialBadge } from './licence-gate';
import { audioEditKeys } from './use-audio-edit-session';

export function HistoryTab({ onOpen }: { onOpen: (mode: AudioEditMode, id: string) => void }) {
  const { t, i18n } = useTranslation();
  const qc = useQueryClient();
  const sessions = useQuery({ queryKey: audioEditKeys.sessions, queryFn: ({ signal }) => listSessions(signal) });
  const [confirming, setConfirming] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  if (sessions.isLoading) return <p className="text-sm text-muted-foreground">{t('audioEdit.loading')}</p>;
  if (sessions.isError) return <p role="alert" className="text-sm text-destructive">{t('audioEdit.history_error')}</p>;
  const items = sessions.data?.items ?? [];
  if (items.length === 0) {
    return <p className="text-sm text-muted-foreground" data-testid="audio-edit-history-empty">{t('audioEdit.history_empty')}</p>;
  }
  const remove = async (id: string) => {
    setError(null);
    try {
      await deleteSession(id);
      setConfirming(null);
      await qc.invalidateQueries({ queryKey: audioEditKeys.sessions });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };
  return (
    <div className="space-y-2">
      {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
      <table className="w-full text-sm" data-testid="audio-edit-history">
        <thead className="text-left text-xs text-muted-foreground">
          <tr>
            <th className="py-2">{t('audioEdit.col_name')}</th>
            <th>{t('audioEdit.col_mode')}</th>
            <th>{t('audioEdit.col_engine')}</th>
            <th>{t('audioEdit.col_renders')}</th>
            <th>{t('audioEdit.col_created')}</th>
            <th>{t('audioEdit.col_size')}</th>
            <th>{t('audioEdit.col_actions')}</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item) => {
            const latest = item.latest_render;
            const playable = latest?.status === 'succeeded';
            return (
              <tr key={item.id} className="border-t align-middle">
                <td className="py-2">
                  <div className="font-medium">{item.name}</div>
                  <div className="text-xs tabular-nums text-muted-foreground">{formatSeconds(item.duration_s)}</div>
                </td>
                <td>{t(`audioEdit.tabs.${item.mode}`)}</td>
                <td>
                  {latest ? (
                    <span className="flex items-center gap-1">
                      {ENGINE_LABELS[latest.engine]}
                      <CommercialBadge commercial={latest.commercial_use} />
                    </span>
                  ) : '—'}
                </td>
                <td className="tabular-nums">{item.render_count}</td>
                <td>{new Date(item.created_at * 1000).toLocaleString(i18n.language)}</td>
                <td className="tabular-nums">{formatBytes(item.size_bytes)}</td>
                <td>
                  <div className="flex flex-wrap items-center gap-1">
                    <Button size="xs" variant="outline" onClick={() => onOpen(item.mode, item.id)}>
                      {t('audioEdit.open')}
                    </Button>
                    {playable && latest && (
                      <>
                        <audio controls preload="none" className="h-7 w-40" aria-label={t('audioEdit.play')}
                          src={renderAudioUrl(item.id, latest.id)} />
                        <a className="text-xs underline" href={renderAudioUrl(item.id, latest.id)} download>
                          {t('audioEdit.export')}
                        </a>
                      </>
                    )}
                    {confirming === item.id ? (
                      <>
                        <Button size="xs" variant="destructive" onClick={() => void remove(item.id)}>
                          {t('audioEdit.confirm_delete')}
                        </Button>
                        <Button size="xs" variant="ghost" onClick={() => setConfirming(null)}>
                          {t('audioEdit.cancel')}
                        </Button>
                      </>
                    ) : (
                      <Button size="xs" variant="ghost" onClick={() => setConfirming(item.id)}>
                        {t('audioEdit.delete')}
                      </Button>
                    )}
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 10: Write `audio-edit-page.tsx`**

```tsx
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { WorkspaceHeader } from '@/components/app-shell/workspace-header';
import { FreeGpuButton } from '@/components/free-gpu-button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import type { AudioEditMode } from '@/lib/api/audio-edit';
import { HistoryTab } from './history-tab';
import { SongTab } from './song-tab';
import { SpeechTab } from './speech-tab';

type Tab = 'speech' | 'song' | 'history';
const TABS: Tab[] = ['speech', 'song', 'history'];

/** Audio Edit: regenerate chosen words in speech or sung vocals. */
export function AudioEditPage() {
  const { t } = useTranslation();
  const [tab, setTab] = useState<Tab>('speech');
  const [open, setOpen] = useState<Record<AudioEditMode, string | null>>({ speech: null, song: null });
  const openSession = (mode: AudioEditMode, id: string | null) => setOpen((prev) => ({ ...prev, [mode]: id }));
  return (
    <div className="flex h-full min-h-0 flex-col" data-testid="audio-edit-page">
      <WorkspaceHeader>
        <h1 className="text-sm font-medium">{t('audioEdit.title')}</h1>
        <span className="min-w-0 flex-1" />
        <FreeGpuButton />
      </WorkspaceHeader>
      <section className="min-h-0 flex-1 overflow-y-auto px-6 py-6">
        <div className="mx-auto w-full max-w-6xl space-y-5">
          <div className="space-y-1">
            <h2 className="text-lg font-semibold">{t('audioEdit.title')}</h2>
            <p className="text-sm text-muted-foreground">{t('audioEdit.subtitle')}</p>
          </div>
          <Tabs value={tab} onValueChange={(value) => setTab(value as Tab)} className="gap-5">
            <TabsList aria-label={t('audioEdit.title')}>
              {TABS.map((value) => (
                <TabsTrigger key={value} value={value} className="px-3">
                  {t(`audioEdit.tabs.${value}`)}
                </TabsTrigger>
              ))}
            </TabsList>
            <TabsContent value="speech">
              <SpeechTab sessionId={open.speech} onSessionChange={(id) => openSession('speech', id)} />
            </TabsContent>
            <TabsContent value="song">
              <SongTab sessionId={open.song} onSessionChange={(id) => openSession('song', id)} />
            </TabsContent>
            <TabsContent value="history">
              <HistoryTab onOpen={(mode, id) => { openSession(mode, id); setTab(mode); }} />
            </TabsContent>
          </Tabs>
        </div>
      </section>
    </div>
  );
}
```

- [ ] **Step 11: Run the tests, type-check and lint**

Run (in `electron/`): `bunx vp test --run src/renderer/src/features/audio-edit/audio-edit.test.tsx`
Expected: `6 passed`.

Run (in `electron/`): `bun run typecheck` and `bun run lint`
Expected: no errors in `features/audio-edit/` or `lib/api/audio-edit.ts`.

- [ ] **Step 12: Commit**

```powershell
git add electron/src/renderer/src/features/audio-edit
git commit -m "Audio Edit: speech, song and history UI"
```

## Task 15: Route, sidebar entry and secondary-workspace layout

**Files** (all under `electron/src/renderer/src/`):
- Modify:
  - `routes/index.ts`: `audioEditRoute`, plus its entry in the route tree
  - `components/app-shell/workspace-menu.tsx`: the `/audio-edit` destination after `/voice-lab`, with `AudioWaveformIcon`
  - `components/app-shell/use-workspace-sidebar.ts`: `/audio-edit` in `SECONDARY_ROUTES`
- Test:
  - `components/app-shell/workspace-menu.test.tsx`
  - `components/app-shell/use-workspace-sidebar.test.tsx`

**Interfaces:**
- Consumes `AudioEditPage` from Task 14.
- Produces the `/audio-edit` route and the `nav.audio_edit` label key, which Task 16 translates.

- [ ] **Step 1: Write the failing tests**

Append to `components/app-shell/workspace-menu.test.tsx`:

```tsx
it('lists Audio Edit after Voice Lab in the voice group', () => {
  route.pathname = '/audio-edit';
  render(<WorkspaceNavigation />);
  const link = screen.getByRole('link', { name: 'nav.audio_edit' });
  expect(link).toHaveAttribute('href', '/audio-edit');
  const labels = screen.getAllByRole('link').map((a) => a.getAttribute('href'));
  expect(labels.indexOf('/audio-edit')).toBe(labels.indexOf('/voice-lab') + 1);
  route.pathname = '/gallery';
});
```

Append to `components/app-shell/use-workspace-sidebar.test.tsx`:

```tsx
it('treats Audio Edit as a secondary workspace that collapses the library', () => {
  state.path = '/audio-edit';
  const { result } = renderHook(useWorkspaceSidebarState);
  expect(result.current.secondaryWorkspace).toBe(true);
  expect(result.current.compact).toBe(true);
});
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run (in `electron/`): `bunx vp test --run src/renderer/src/components/app-shell/workspace-menu.test.tsx src/renderer/src/components/app-shell/use-workspace-sidebar.test.tsx`
Expected: FAIL. The menu test can't find the link `nav.audio_edit`, and the sidebar test's `secondaryWorkspace` is `false`.

- [ ] **Step 3: Add the route**

In `routes/index.ts`, directly after the `voiceLabRoute` definition:

```ts
export const audioEditRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/audio-edit',
  component: lazyRouteComponent(() => import('@/features/audio-edit/audio-edit-page'), 'AudioEditPage'),
});
```

In the `rootRoute.addChildren([...])` list, change:

```ts
  voiceLabRoute,
  transcriptionsRoute,
```

to:

```ts
  voiceLabRoute,
  audioEditRoute,
  transcriptionsRoute,
```

- [ ] **Step 4: Add the sidebar destination**

In `components/app-shell/workspace-menu.tsx`:
- **Icon import.** Add `AudioWaveformIcon,` to the `lucide-react` import list.
- **Destination union.** Change the end of the `Destination` path union from

  ```ts
      | '/voice-lab',
  ```

  to

  ```ts
      | '/voice-lab'
      | '/audio-edit',
  ```

- **Voice group.** In `voiceDestinations`, directly after the voice-lab entry, add:

  ```ts
    ['/audio-edit', 'nav.audio_edit', AudioWaveformIcon],
  ```

- **Icon motion.** In `iconMotion`, after `'/voice-lab': 'tilt',`, add:

  ```ts
    '/audio-edit': 'pulse',
  ```

- [ ] **Step 5: Mark it a secondary workspace**

In `components/app-shell/use-workspace-sidebar.ts`, add `'/audio-edit',` to the `SECONDARY_ROUTES` set, directly after `'/voice-lab',`.

- [ ] **Step 6: Run the tests to confirm they pass**

Run (in `electron/`): `bunx vp test --run src/renderer/src/components/app-shell`
Expected: all app-shell tests pass, including the two new ones.

Run (in `electron/`): `bun run typecheck`
Expected: no errors.

- [ ] **Step 7: Commit**

```powershell
git add electron/src/renderer/src/routes/index.ts electron/src/renderer/src/components/app-shell/workspace-menu.tsx electron/src/renderer/src/components/app-shell/workspace-menu.test.tsx electron/src/renderer/src/components/app-shell/use-workspace-sidebar.ts electron/src/renderer/src/components/app-shell/use-workspace-sidebar.test.tsx
git commit -m "Audio Edit: route and sidebar entry"
```

## Task 16: i18n for all 21 locales

**Files:**
- Modify: `electron/src/renderer/src/i18n/locales/{ar,de,en,es,fr,hi,id,it,ja,ko,nl,pl,pt,ru,sv,th,tr,uk,vi,zh-CN,zh-TW}.json`

**Interfaces:**
- Consumes every key used by Tasks 14 and 15.
- Produces `nav.audio_edit` and the `audioEdit` object in each locale.

Translation rules:
- Keep every `{{placeholder}}` exactly as written.
- Keep product and model names untranslated: VoiceStudio, F5-TTS, ACE-Step, YingMusic, Vevo2, RVC, HTDemucs.
- Keep key order identical to `en.json`.
- Write each file as UTF-8 without a BOM.

- [ ] **Step 1: Add the English strings**

In `en.json`, inside `"nav"`, directly after `"voice_lab": "Voice Lab",`, add:

```json
    "audio_edit": "Audio Edit",
```

In `en.json`, add this top-level object directly before `"voiceLab": {`:

```json
  "audioEdit": {
    "title": "Audio Edit",
    "subtitle": "Change, add or remove words in a recording. Everything outside your edits stays exactly as it was.",
    "tabs": { "speech": "Speech", "song": "Song", "history": "History" },
    "upload_hint_speech": "Drop a speech recording or video here (up to 10 minutes).",
    "upload_hint_song": "Drop a song here (up to 10 minutes). Vocals are separated from the music first.",
    "choose_file": "Choose file",
    "separation_missing": "Vocal separation needs {{missing}}. Install it in Settings → Models.",
    "new_session": "New file",
    "stage": {
      "uploaded": "Uploaded",
      "separating": "Separating vocals…",
      "separated": "Vocals separated",
      "transcribing": "Transcribing…",
      "ready": "Ready to edit"
    },
    "stage_failed": "That step failed: {{error}}",
    "retry": "Retry",
    "job": { "separate": "Separating vocals", "transcribe": "Transcribing", "render": "Rendering" },
    "cancel": "Cancel",
    "stem_vocals": "Vocals",
    "stem_instrumental": "Instrumental",
    "select_hint": "Click a word, then click another to select the words between them.",
    "insert_hint": "To add words, select the word next to the gap and retype it with the new words.",
    "overlap": "That selection overlaps an existing edit.",
    "no_words": "No words were found in this recording.",
    "transcript": "Transcript",
    "no_spans": "No edits yet. Select words in the transcript to start.",
    "replacement_placeholder": "New words (leave empty to delete)",
    "replacement_for": "Replacement for “{{words}}”",
    "start": "Start",
    "end": "End",
    "nudge_earlier": "Move {{edge}} 10 ms earlier",
    "nudge_later": "Move {{edge}} 10 ms later",
    "remove_span": "Remove edit",
    "engine": "Engine",
    "rvc_model": "RVC voice",
    "rvc_none": "None",
    "rvc_pitch": "RVC pitch: {{value}} semitones",
    "crossfade": "Crossfade: {{value}} ms",
    "vocal_gain": "Vocal level: {{value}} dB",
    "ace_strength": "Repaint strength: {{value}}",
    "ace_padding": "Context padding: {{value}} s",
    "ace_caption": "Style caption",
    "ace_caption_placeholder": "Optional, for example “soft female pop vocal”",
    "render": "Render edit",
    "engine_missing": "{{engine}} is not installed.",
    "engine_needs_gpu": "{{engine}} needs a supported GPU on this computer.",
    "install": "Install",
    "install_started": "Installing. Progress is shown in Settings → Models.",
    "install_failed": "The install could not start.",
    "open_models": "Open Settings → Models",
    "licence_title": "{{engine}} licence",
    "licence_body": "{{engine}} is licensed {{licence}}. Its output may only be used for non-commercial purposes.",
    "licence_checkbox": "I understand and will only use these renders non-commercially.",
    "licence_accept": "Accept and continue",
    "licence_revoke": "Revoke acknowledgement",
    "badge_commercial": "Commercial OK",
    "badge_noncommercial": "Non-commercial",
    "render_status": {
      "queued": "Queued",
      "running": "Rendering…",
      "succeeded": "Done",
      "failed": "Failed",
      "cancelled": "Cancelled"
    },
    "original": "Original",
    "edited": "Edited",
    "export": "Export WAV",
    "play": "Play",
    "open": "Open",
    "delete": "Delete",
    "confirm_delete": "Delete permanently",
    "loading": "Loading…",
    "history_error": "Could not load your edits.",
    "history_empty": "No edits yet.",
    "col_name": "File",
    "col_mode": "Type",
    "col_engine": "Engine",
    "col_renders": "Renders",
    "col_created": "Created",
    "col_size": "Size",
    "col_actions": "Actions",
    "errors": {
      "invalid_audio": "That file is not audio VoiceStudio can read.",
      "too_long": "Recordings longer than 10 minutes are not supported.",
      "too_large": "That file is too large.",
      "no_word_timestamps": "The transcriber returned no word timings for this recording.",
      "invalid_spans": "One of the edits is invalid. Check its start and end.",
      "engine_not_installed": "Install the selected engine first.",
      "licence_ack_required": "Accept the engine licence before rendering.",
      "ace_step_unavailable": "ACE-Step could not start. Check its install in Settings → Models.",
      "separation_unavailable": "Vocal separation is not installed.",
      "engine_failed": "The engine failed while rendering. Try again or pick another engine.",
      "busy": "Another Audio Edit job is running. Wait for it or cancel it."
    }
  },
```

- [ ] **Step 2: Add `nav.audio_edit` to the other 20 locales**

Put each value after that locale's `nav.voice_lab`:

| Locale | `nav.audio_edit` |
|---|---|
| ar | `تحرير الصوت` |
| de | `Audio bearbeiten` |
| es | `Editar audio` |
| fr | `Édition audio` |
| hi | `ऑडियो संपादन` |
| id | `Edit Audio` |
| it | `Modifica audio` |
| ja | `オーディオ編集` |
| ko | `오디오 편집` |
| nl | `Audio bewerken` |
| pl | `Edycja audio` |
| pt | `Editar áudio` |
| ru | `Редактор аудио` |
| sv | `Ljudredigering` |
| th | `แก้ไขเสียง` |
| tr | `Ses Düzenleme` |
| uk | `Редактор аудіо` |
| vi | `Chỉnh sửa âm thanh` |
| zh-CN | `音频编辑` |
| zh-TW | `音訊編輯` |

- [ ] **Step 3: Translate the `audioEdit` block into the other 20 locales**

For each of the 20 non-English files:
- Insert the whole `audioEdit` object, translated, directly before that file's `"voiceLab": {`.
- Use the same keys in the same order.
- Use `audioEdit.title` = that locale's `nav.audio_edit` value from Step 2.
- Follow the translation rules at the top of this task.

- [ ] **Step 4: Check key and placeholder parity**

Run from the repo root (PowerShell):

```powershell
@'
import json, pathlib, re, sys
root = pathlib.Path("electron/src/renderer/src/i18n/locales")
def flat(d, p=""):
    out = {}
    for k, v in d.items():
        key = f"{p}{k}"
        out.update(flat(v, key + ".") if isinstance(v, dict) else {key: v})
    return out
ph = lambda s: sorted(re.findall(r"\{\{\w+\}\}", s))
en = {k: v for k, v in flat(json.loads((root / "en.json").read_text("utf-8"))).items()
      if k.startswith("audioEdit.") or k == "nav.audio_edit"}
bad = 0
for f in sorted(root.glob("*.json")):
    raw = f.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        print(f.name, "has a BOM"); bad += 1
    loc = flat(json.loads(raw.decode("utf-8")))
    for k, v in en.items():
        if k not in loc:
            print(f.name, "missing", k); bad += 1
        elif ph(loc[k]) != ph(v):
            print(f.name, "placeholder mismatch", k); bad += 1
        elif f.name != "en.json" and loc[k] == v and k not in {"audioEdit.col_engine"}:
            print(f.name, "untranslated?", k)
print("bad:", bad); sys.exit(1 if bad else 0)
'@ | uv run python -
```

Expected:
- `bad: 0`.
- Any "untranslated?" lines are only legitimate cognates, such as "Audio" or "Engine" in some locales. Review each one by hand.

- [ ] **Step 5: Run the locale checks and the Audio Edit tests**

Run (in `electron/`): `bun run locale:check`
Expected: encoding, source-keys and coverage all pass.

Run (in `electron/`): `bunx vp test --run src/renderer/src/features/audio-edit src/renderer/src/components/app-shell`
Expected: all pass. The test i18n mock returns keys, so string changes cannot break them.

- [ ] **Step 6: Commit**

```powershell
git add electron/src/renderer/src/i18n/locales/ar.json electron/src/renderer/src/i18n/locales/de.json electron/src/renderer/src/i18n/locales/en.json electron/src/renderer/src/i18n/locales/es.json electron/src/renderer/src/i18n/locales/fr.json electron/src/renderer/src/i18n/locales/hi.json electron/src/renderer/src/i18n/locales/id.json electron/src/renderer/src/i18n/locales/it.json electron/src/renderer/src/i18n/locales/ja.json electron/src/renderer/src/i18n/locales/ko.json electron/src/renderer/src/i18n/locales/nl.json electron/src/renderer/src/i18n/locales/pl.json electron/src/renderer/src/i18n/locales/pt.json electron/src/renderer/src/i18n/locales/ru.json electron/src/renderer/src/i18n/locales/sv.json electron/src/renderer/src/i18n/locales/th.json electron/src/renderer/src/i18n/locales/tr.json electron/src/renderer/src/i18n/locales/uk.json electron/src/renderer/src/i18n/locales/vi.json electron/src/renderer/src/i18n/locales/zh-CN.json electron/src/renderer/src/i18n/locales/zh-TW.json
git commit -m "Audio Edit: strings for all 21 locales"
```

## Task 17: Bookkeeping and full verification

**Files:**
- Modify:
  - `docs/STRUCTURE.md`
  - `CHANGELOG.md`
  - `tests/test_changelog_style.py`: `_REF_ALLOWLIST`

**Interfaces:**
- Consumes everything from Tasks 1–16.
- Produces the final docs and a verified tree.

- [ ] **Step 1: Update `docs/STRUCTURE.md` counts**

Two lines change:
- `api/routers/             44 routers, auto-included; thin HTTP/WS surface` becomes `api/routers/             45 routers, auto-included; thin HTTP/WS surface`.
- `services/                111 modules of business logic - TTS, dubbing pipeline,` becomes `services/                115 modules of business logic - TTS, dubbing pipeline,`.

The four new modules are `audio_edit`, `audio_edit_dsp`, `audio_edit_engines` and `acestep_client`. Keep the tree-drawing characters on those lines unchanged: edit only the numbers.

Verify the new counts:

```powershell
(Get-ChildItem backend/api/routers/*.py | Where-Object Name -ne '__init__.py').Count
(Get-ChildItem backend/services/*.py | Where-Object Name -ne '__init__.py').Count
```

Expected: `44` and `114`, up from `43` and `110` before this plan.

The document counts one more in each folder than this command does, which is why it says 44 and 111 today. Apply the same deltas there: +1 router and +4 services.

- [ ] **Step 2: Add the CHANGELOG line**

In `CHANGELOG.md`, under `## [Unreleased]` → `### Added`, add this line after the existing bullets:

```markdown
- Audio Edit: change, add or remove words in a speech recording or a song's vocals and render only those words again, with F5-TTS for speech and ACE-Step, YingMusic or Vevo2 (optionally through RVC) for singing; untouched audio stays bit-identical and every render is watermarked
```

Added entries need a `(#N)` reference unless the entry is allowlisted. This feature is owner-authored work committed straight to `main`, with no issue or PR, so add it to `_REF_ALLOWLIST` in `tests/test_changelog_style.py`, after the last `# owner-authored entries without a public reference` item:

```python
    # owner-authored Audio Edit workspace, committed straight to main
    "Audio Edit: change, add or remove words",
```

Run: `uv run pytest tests/test_changelog_style.py -q`
Expected: pass. The line is under 400 characters, on one line, and has no emphasis.

- [ ] **Step 3: Run the full backend suite**

Run: `uv run pytest tests -q`
Expected: no failures. These are the Audio Edit-specific suites:

```powershell
uv run pytest tests/test_audio_edit_dsp.py tests/test_audio_edit_service.py tests/test_audio_edit_render.py tests/test_audio_edit_engines.py tests/test_audio_edit_song_engines.py tests/test_acestep_client.py tests/test_audio_edit_router.py tests/test_watermark_route_coverage.py tests/test_sidecar_install.py tests/test_sidecar_stdout_isolation_1428.py -q
```

Run: `uv run python scripts/check_model_licenses.py`
Expected: exit code 0.

- [ ] **Step 4: Run the full frontend checks**

Run (in `electron/`):

```powershell
bun run locale:check
bun run typecheck
bun run lint
bunx vp test --run
```

Expected: all pass, with no new lint warnings in `features/audio-edit/`, `lib/api/audio-edit.ts` or `components/app-shell/`.

- [ ] **Step 5: Manual acceptance on the MSI desktop**

Start the app with `bun run dev` and work through each check below.

**Speech, F5-TTS installed**
- [ ] Upload a 30 s English speech WAV. The transcript appears with clickable words.
- [ ] Select one word and replace it.
- [ ] The licence gate appears. Accept it, then render.
- [ ] The edited audio plays and the badge reads Non-commercial.
- [ ] Load the original and the output in Audacity and invert one against the other: everything more than one crossfade away from the edit cancels to silence.

**Song, ACE-Step installed**
- [ ] Upload a 30 s song. Separation runs, then transcription.
- [ ] The vocal and instrumental stem players both play.
- [ ] Replace one lyric line and render with ACE-Step.
- [ ] The edited song keeps the original backing track, and the vocals blend at both edges.

**Cancel and revoke**
- [ ] Start a render and cancel it. The job ends as Cancelled within a few seconds, and `nvidia-smi` shows the VRAM released.
- [ ] Revoke the F5-TTS acknowledgement. The gate returns on the next render.

**Missing engine**
- [ ] On a machine without Vevo2, choosing it shows the install prompt and the Settings → Models link instead of the render button.

**History**
- [ ] History lists both sessions with engine, render count, size and badge.
- [ ] Open switches to the right tab.
- [ ] Export downloads "<name> (edited).wav".
- [ ] Delete asks for confirmation, then removes the row and the session folder.

**Languages**
- [ ] Switch the UI language to Japanese and Arabic. Every Audio Edit label is translated and the Arabic layout is right-to-left.

- [ ] **Step 6: Commit the bookkeeping**

```powershell
git status --short
git add docs/STRUCTURE.md CHANGELOG.md tests/test_changelog_style.py
git commit -m "Docs: Audio Edit structure counts and changelog"
```

Expected from `git status --short`: only the three files above, plus the pre-existing unrelated ` M bun.lock`, which stays unstaged.

- [ ] **Step 7: Push**

```powershell
git log --oneline origin/main..main
git push origin main
```

Expected:
- The log lists only this plan's commits.
- The push goes to the `vasudev123t` fork. Never push to `debpalash`, never force-push, and never open a PR.

---

## Self-review

### Spec coverage

| Spec requirement | Task(s) |
|---|---|
| Goal 1: speech replace / insert / delete with F5-TTS | 3 (spans, windows), 4 (F5-TTS sidecar), 9 (speech render). Inserts use the "retype the neighbour" rule in the deviations table. |
| Goal 2: song with ACE-Step and YingMusic / Vevo2, both shipped | 5 (YingMusic, Vevo2 sidecars), 6 (ACE-Step client and install), 10 (song render) |
| Goal 3: one pipeline, one router, one session model | 3, 8, 11 |
| Goal 4: spans from ASR word timestamps with nudging | 3 (`spans_from_words`, validation), 8 (transcription job), 13 (`nudgeSpan`), 14 (`TranscriptSpans`, `SpanList`) |
| Goal 5: equal-power crossfades; untouched audio byte-identical | 2 (dsp splice and its byte-identity tests), 9 and 10 (per-span splice, residual song mix at 0 dB) |
| Goal 6: provenance via `mark_synthetic` | 9 (`_mark` on every synthetic insert), 12 (watermark route-coverage test) |
| Goal 7: licence honesty, gate and badge from the registry | 1 (registry records), 7 (acks, `render_commercial_use`), 14 (`LicenceGate`, `CommercialBadge`) |
| Goal 8: history on disk and listed | 3 (session store), 11 (`list_session_summaries`, list/delete routes), 14 (`HistoryTab`) |
| Upload limits, formats and errors (`invalid_audio`, `too_long`, `too_large`) | 3, 11, 14, 16 |
| Separation with HTDemucs for songs | 8 (`separate_stems`, `start_separation`) |
| Optional RVC pass | 9 (`_apply_rvc`), 10 (song path), 14 (`RenderControls` defaults RVC on for songs) |
| ACE-Step on loopback `127.0.0.1` only | 6 |
| GPU: one job at a time, free VRAM, kill sidecars on cancel | 8 (`_RENDER_SLOT`, `_free_vram`, `cancel_job`) |
| Desktop-only router | 11 (`require_admin`, `require_desktop`; desktop-guard test) |
| API surface and error envelope | 11 (16 routes), 12 (`api_routes.txt`, OpenAPI) |
| Sidebar after Voice Lab, secondary workspace | 15 |
| Strings in 21 locales | 16 |
| Docs and bookkeeping (engine docs, model inventory, STRUCTURE, CHANGELOG) | 1, 17 |
| Success criteria 1–7 | 17, Step 5 (manual acceptance) and Steps 3–4 (all suites) |

### Placeholder scan

The plan was searched for "TBD", "TODO", "FIXME", "fill in", "similar to Task" and "add appropriate".
- None is used as a placeholder.
- Every code step contains the code to write.
- Every run step names the command and the expected result.

Task 16 Step 3 asks the implementer to translate one English block into 20 locales. That is a concrete instruction with fixed keys and rules, and Step 4 machine-checks the result.

### Type consistency

Python:
- Render status values are `queued | running | succeeded | failed | cancelled` in the service (Task 8), the router (Task 11) and `JobStatus` (Task 13).
- Session stages are `uploaded | separating | separated | transcribing | ready` in Tasks 3, 8 and 13 and in the `audioEdit.stage.*` keys (Task 16).
- Engine ids are `f5tts | ace_step | yingmusic | vevo2` in `ENGINES` (Task 3), `EDITORS` (Tasks 4–5), the ack prefs (Task 7) and `AudioEditEngine` (Task 13).
- The install id for ACE-Step is `ace-step` (the `SidecarSpec` id, Task 6). The engine id is `ace_step`. The frontend always installs through `EngineStatus.install_id`.

Wire format:
- The render body is snake_case (`crossfade_ms`, `rvc_model_id`, `rvc_pitch`, `vocal_gain_db`, `seed`, `ace_step{strength,padding_s,caption}`), produced by `toRenderBody` (Task 13) and read by `validate_render_request` (Task 9).
- `DELETE /audio-edit/sessions/{id}` returns `{ok: true}` (Task 11). `deleteSession` types it as `{ ok: boolean }` (Task 13).
- `GET /audio-edit/engines` returns `{engines, separation, gpu}` (Task 7), matching `EnginesOverview` (Task 13).
- Error codes in `KNOWN_ERRORS` (Task 14) all have an `audioEdit.errors.*` string (Task 16).

### Test count checkpoints

| Task | Expected |
|---|---|
| 2 | dsp suite green |
| 3–8 | `tests/test_audio_edit_service.py`: 35 passed after Task 8 |
| 9 | 42 passed across the service and render suites |
| 10 | 47 passed |
| 11 | router suite: 7 passed |
| 13 | `audio-edit.test.ts`: 7 passed |
| 14 | `audio-edit.test.tsx`: 6 passed |
| 15 | app-shell suites green, including the 2 new tests |

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-11-02-audio-edit.md`. There are two ways to execute it:

1. **Subagent-driven (recommended).**
   - Use superpowers:subagent-driven-development.
   - Dispatch a fresh subagent per task, in order (1 → 17), and review between tasks.
   - Tasks 4, 5 and 6 each touch `sidecar_install.py`, so run them strictly in sequence.
   - Task 16 depends on the final key list from Tasks 14 and 15.
2. **Inline execution.**
   - Use superpowers:executing-plans in a single session, with a checkpoint after Tasks 6, 12 and 16.

Either way:
- Work on `main` in `D:\VoiceStudio`.
- Stage by explicit path and never stage `bun.lock`.
- Commit messages carry no agent credit.
- Finish with `git push origin main` to the `vasudev123t` fork: no force-push, no PR, never `debpalash`.
