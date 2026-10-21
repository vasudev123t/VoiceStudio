import { apiJson } from './client';

/** Voice conversion (RVC / Applio) — client for `/voice-conversion/*`. */

export type F0Method = 'rmvpe' | 'crepe' | 'fcpe' | 'harvest' | 'pm';
export type TrainF0Method = 'rmvpe' | 'crepe' | 'fcpe';
export type TrainSampleRate = 32000 | 40000 | 48000;

export interface VcStatus {
  available: boolean;
  backend: 'applio' | 'rvc-python' | null;
  device: string;
  device_override: string | null;
  applio: {
    configured_dir: string | null;
    dir: string | null;
    python: string | null;
    managed_dir: string;
    ready: boolean;
  };
  rvc_python: boolean;
  f0_methods: F0Method[];
  supported_params: string[];
  worker_running: boolean;
  active_job: VcJob | null;
  docs: string;
}

export interface RvcVoice {
  id: string;
  name: string;
  created_at: number | null;
  source: 'import' | 'trained' | string;
  sample_rate: number | null;
  version: string | null;
  f0: boolean | null;
  epochs: number | null;
  has_index: boolean;
  /** `bundled` (imported/trained with it), `rebuilt` (built in-app) or null. */
  index_source?: 'bundled' | 'rebuilt' | null;
  model_bytes: number;
  index_bytes: number;
}

export interface ConversionParams {
  pitch: number;
  f0Method: F0Method;
  indexRate: number;
  protect: number;
  filterRadius: number;
  rmsMixRate: number;
  cleanAudio: boolean;
}

/**
 * Defaults tuned for acted performances: protect 0.33 keeps breaths and
 * voiceless consonants (0.5 switches protection off) and a 0.1 volume envelope
 * keeps the performer's loudness contour (1.0 replaces it).
 */
export const DEFAULT_CONVERSION_PARAMS: ConversionParams = {
  pitch: 0,
  f0Method: 'rmvpe',
  indexRate: 0.75,
  protect: 0.33,
  filterRadius: 3,
  rmsMixRate: 0.1,
  cleanAudio: false,
};

/** "Expressive" preset from the review: rmvpe, protect 0.33, envelope 0.1, index 0.6. */
export const EXPRESSIVE_CONVERSION_PRESET: Partial<ConversionParams> = {
  f0Method: 'rmvpe',
  protect: 0.33,
  rmsMixRate: 0.1,
  indexRate: 0.6,
};

/** protect ≥ 0.5 disables consonant/breath protection entirely in RVC. */
export function protectionOff(params: Pick<ConversionParams, 'protect'>): boolean {
  return params.protect >= 0.5;
}

export interface ConversionEntry {
  id: string;
  created_at: number;
  voice_id: string;
  voice_name: string;
  filename: string;
  source_filename: string | null;
  source_name: string;
  sample_rate: number;
  duration_s: number;
  elapsed_s: number;
  backend: string;
  device: string;
  params: Record<string, unknown>;
  index_used?: boolean;
  audio_url: string;
  source_url: string | null;
}

export type VcJobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';

export interface VcJob {
  id: string;
  kind: 'train' | 'setup' | 'index';
  name: string;
  status: VcJobStatus;
  stage: string | null;
  progress: number;
  epoch: number;
  total_epochs: number;
  error: string | null;
  voice_id: string | null;
  created_at: number;
  finished_at: number | null;
  settings: Record<string, unknown>;
  log_tail: string[];
  log_lines: number;
}

export interface TrainInput {
  name: string;
  sampleRate: TrainSampleRate;
  epochs: number;
  batchSize: number;
  f0Method: TrainF0Method;
  saveEvery: number;
  checkpointing: boolean;
  files: File[];
  datasetAuthorization?: string | null;
}

/** 8 GB VRAM (RTX 3070 Laptop class) safe training defaults. */
export const DEFAULT_TRAIN_SETTINGS: Omit<TrainInput, 'name' | 'files'> = {
  sampleRate: 48000,
  epochs: 300,
  batchSize: 8,
  f0Method: 'rmvpe',
  saveEvery: 25,
  checkpointing: false,
};

export const PITCH_RANGE = [-24, 24] as const;

const BASE = '/voice-conversion';

export function isTerminal(status: VcJobStatus): boolean {
  return status === 'succeeded' || status === 'failed' || status === 'cancelled';
}

export function clampPitch(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(PITCH_RANGE[0], Math.min(PITCH_RANGE[1], Math.round(value)));
}

export function getStatus(signal?: AbortSignal) {
  return apiJson<VcStatus>(`${BASE}/status`, { signal });
}

export async function listVoices(signal?: AbortSignal): Promise<RvcVoice[]> {
  return (await apiJson<{ voices: RvcVoice[] }>(`${BASE}/voices`, { signal })).voices;
}

export function importVoice(name: string, model: File, index?: File | null) {
  const body = new FormData();
  body.append('name', name.trim());
  body.append('model', model, model.name);
  if (index) body.append('index', index, index.name);
  return apiJson<RvcVoice>(`${BASE}/voices/import`, { method: 'POST', body });
}

export function renameVoice(id: string, name: string) {
  return apiJson<RvcVoice>(`${BASE}/voices/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: name.trim() }),
  });
}

export function deleteVoice(id: string) {
  return apiJson<{ ok: boolean }>(`${BASE}/voices/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  });
}

/** Multipart body for `POST /voice-conversion/convert`. */
export function toConvertForm(audio: File, voiceId: string, params: ConversionParams): FormData {
  const body = new FormData();
  body.append('audio', audio, audio.name || 'recording.wav');
  body.append('voice_id', voiceId);
  body.append('pitch', String(clampPitch(params.pitch)));
  body.append('f0_method', params.f0Method);
  body.append('index_rate', String(params.indexRate));
  body.append('protect', String(params.protect));
  body.append('filter_radius', String(params.filterRadius));
  body.append('rms_mix_rate', String(params.rmsMixRate));
  body.append('clean_audio', params.cleanAudio ? 'true' : 'false');
  return body;
}

export function convertAudio(
  audio: File,
  voiceId: string,
  params: ConversionParams,
  signal?: AbortSignal,
) {
  return apiJson<ConversionEntry>(`${BASE}/convert`, {
    method: 'POST',
    body: toConvertForm(audio, voiceId, params),
    signal,
  });
}

export async function listHistory(signal?: AbortSignal): Promise<ConversionEntry[]> {
  return (await apiJson<{ items: ConversionEntry[] }>(`${BASE}/history`, { signal })).items;
}

export function deleteHistoryEntry(id: string) {
  return apiJson<{ ok: boolean }>(`${BASE}/history/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  });
}

export function toTrainForm(input: TrainInput): FormData {
  const body = new FormData();
  body.append('name', input.name.trim());
  body.append('sample_rate', String(input.sampleRate));
  body.append('epochs', String(input.epochs));
  body.append('batch_size', String(input.batchSize));
  body.append('f0_method', input.f0Method);
  body.append('save_every', String(Math.min(input.saveEvery, input.epochs)));
  body.append('checkpointing', input.checkpointing ? 'true' : 'false');
  if (input.datasetAuthorization) body.append('dataset_authorization', input.datasetAuthorization);
  else for (const file of input.files) body.append('files', file, file.name);
  return body;
}

export function startTraining(input: TrainInput) {
  return apiJson<VcJob>(`${BASE}/train`, { method: 'POST', body: toTrainForm(input) });
}

/** (Re)build a voice's FAISS index from recordings of its speaker. */
export function buildVoiceIndex(id: string, files: File[], datasetId?: string | null) {
  const body = new FormData();
  if (datasetId) body.append('dataset_id', datasetId);
  else for (const file of files) body.append('files', file, file.name);
  return apiJson<VcJob>(`${BASE}/voices/${encodeURIComponent(id)}/index`, {
    method: 'POST',
    body,
  });
}

export function getJob(id: string, tail = 200, signal?: AbortSignal) {
  return apiJson<VcJob>(`${BASE}/train/${encodeURIComponent(id)}?tail=${tail}`, { signal });
}

export async function listJobs(signal?: AbortSignal): Promise<VcJob[]> {
  return (await apiJson<{ jobs: VcJob[] }>(`${BASE}/jobs`, { signal })).jobs;
}

export function cancelJob(id: string) {
  return apiJson<VcJob>(`${BASE}/train/${encodeURIComponent(id)}/cancel`, { method: 'POST' });
}

export function startApplioSetup() {
  return apiJson<VcJob>(`${BASE}/setup`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ confirm: true }),
  });
}

export function updateSettings(body: {
  device?: 'auto' | 'cpu' | 'cuda' | 'mps';
  applio_dir_authorization?: string;
}) {
  return apiJson<VcStatus>(`${BASE}/settings`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export const vcQueryKeys = {
  all: ['voice-conversion'] as const,
  status: ['voice-conversion', 'status'] as const,
  voices: ['voice-conversion', 'voices'] as const,
  history: ['voice-conversion', 'history'] as const,
  jobs: ['voice-conversion', 'jobs'] as const,
  job: (id: string) => ['voice-conversion', 'job', id] as const,
};
