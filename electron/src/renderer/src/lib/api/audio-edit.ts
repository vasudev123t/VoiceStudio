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
export interface Transcript { language: string | null; engine?: string; text: string; words: Word[] }
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
  asr?: TranscribeOptions | null;
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

// ── speech recognition (language + engine picked before transcribing) ──

export interface TranscribeOptions { engine: string; language: string }
export const DEFAULT_TRANSCRIBE_OPTIONS: TranscribeOptions = { engine: 'whisper', language: 'auto' };
export interface AsrDownload { engine: string; status: 'running' | 'succeeded' | 'failed'; progress: number;
  error: string | null; started_at: number }
export interface AsrEngine {
  id: string;
  kind: 'local' | 'cloud';
  label: string;
  description: string;
  languages: string[] | null;
  word_timestamps: boolean;
  provider: string | null;
  size_gb: number | null;
  gated: boolean;
  installed: boolean;
  configured: boolean;
  available: boolean;
  reason: string | null;
  download: AsrDownload | null;
}
export interface AsrEnginesOverview {
  default_engine: string;
  languages: { code: string; name: string }[];
  engines: AsrEngine[];
}
export interface AsrKeyField {
  id: string;
  label: string;
  secret: boolean;
  placeholder: string;
  has_value: boolean;
  source: 'settings' | 'env' | null;
  value: string | null;
  masked: string | null;
}
export interface AsrProvider { id: string; label: string; docs_url: string; fields: AsrKeyField[]; configured: boolean }

export const engineSupports = (engine: AsrEngine | undefined, language: string) =>
  !engine || !engine.languages || engine.languages.includes(language)
  || (language === 'auto' && engine.languages.length === 1);

/** Why the picked engine cannot run yet, or null when it can. */
export function transcribeBlocker(
  overview: AsrEnginesOverview | undefined, options: TranscribeOptions,
): 'key_missing' | 'not_installed' | 'missing_module' | 'language_unsupported' | null {
  const engine = overview?.engines.find((e) => e.id === options.engine);
  if (!engine) return null;
  if (!engineSupports(engine, options.language)) return 'language_unsupported';
  if (engine.available) return null;
  if (engine.reason === 'key_missing') return 'key_missing';
  if (engine.reason === 'not_installed') return 'not_installed';
  return 'missing_module';
}

const TRANSCRIBE_PREF_KEY = 'audioEdit.transcribeOptions';
export function loadTranscribeOptions(): TranscribeOptions {
  try {
    const raw = JSON.parse(globalThis.localStorage?.getItem(TRANSCRIBE_PREF_KEY) ?? 'null');
    if (raw && typeof raw.engine === 'string' && typeof raw.language === 'string') return raw;
  } catch {
    // ignore a corrupt preference
  }
  return DEFAULT_TRANSCRIBE_OPTIONS;
}
export function saveTranscribeOptions(options: TranscribeOptions): void {
  try {
    globalThis.localStorage?.setItem(TRANSCRIBE_PREF_KEY, JSON.stringify(options));
  } catch {
    // storage unavailable: the pick lasts for this view only
  }
}

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
export const transcribeSession = (id: string, options?: TranscribeOptions) =>
  apiJson<{ job: EditJob }>(`${sid(id)}/transcribe`, options ? json('POST', options) : { method: 'POST' });
export const getAsrEngines = (signal?: AbortSignal) => apiJson<AsrEnginesOverview>(`${BASE}/asr/engines`, { signal });
export const downloadAsrEngine = (engine: string) =>
  apiJson<AsrDownload>(`${BASE}/asr/engines/${encodeURIComponent(engine)}/download`, { method: 'POST' });
export const getAsrKeys = (signal?: AbortSignal) =>
  apiJson<{ providers: AsrProvider[] }>(`${BASE}/asr/keys`, { signal });
export const saveAsrKeys = (provider: string, fields: Record<string, string>) =>
  apiJson<AsrProvider>(`${BASE}/asr/keys/${encodeURIComponent(provider)}`, json('PUT', { fields }));
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
