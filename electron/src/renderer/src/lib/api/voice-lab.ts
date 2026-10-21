import { apiJson, apiPath } from './client';

const BASE = '/voice-lab';

export interface StageCapability {
  available: boolean;
  model?: string;
  method?: string;
  missing?: string | null;
}

export interface VoiceLabCapabilities {
  separation: StageCapability;
  vad: StageCapability;
  speaker: StageCapability;
  transcription: StageCapability;
  quality: StageCapability;
  nonverbal: StageCapability;
}

export interface DatasetOptions {
  name: string;
  language: string;
  separate: boolean;
  transcribe: boolean;
  speakerFilter: boolean;
  tagNonverbal: boolean;
  minClipS: number;
  maxClipS: number;
  minQuality: number;
}

export const DEFAULT_DATASET_OPTIONS: DatasetOptions = {
  name: 'My voice',
  language: '',
  separate: true,
  transcribe: true,
  speakerFilter: true,
  tagNonverbal: true,
  minClipS: 1.5,
  maxClipS: 12,
  minQuality: 0.45,
};

export interface DatasetStats {
  segments: number;
  accepted: number;
  rejected: number;
  rejected_by_reason: Record<string, number>;
  accepted_seconds: number;
  laughs: number;
  breaths: number;
  mean_quality: number;
}

export interface DatasetSummary {
  id: string;
  name: string;
  created_at: number;
  stats: DatasetStats;
  stages: Record<string, string>;
  warnings: string[];
  sample_rate: number;
}

export interface DatasetSegment {
  source: string;
  start: number;
  end: number;
  duration: number;
  index: number;
  speaker_sim: number | null;
  speaker_ok: boolean;
  text: string;
  asr_conf: number | null;
  quality: number | null;
  tags: string[];
  accepted: boolean;
  reasons: string[];
  file: string | null;
}

export interface NonverbalEvent {
  type: 'laugh' | 'breath' | string;
  file: string;
  source: string;
  start: number;
  end: number;
}

export interface DatasetDetail extends DatasetSummary {
  segments: DatasetSegment[];
  nonverbal: NonverbalEvent[];
  path: string;
}

export interface LabJob {
  id: string;
  dataset_id: string;
  name: string;
  status: 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';
  stage: string | null;
  progress: number;
  error: string | null;
  log_tail: string;
  summary: DatasetSummary | null;
}

export const labQueryKeys = {
  capabilities: ['voice-lab', 'capabilities'] as const,
  datasets: ['voice-lab', 'datasets'] as const,
  dataset: (id: string) => ['voice-lab', 'dataset', id] as const,
  job: (id: string) => ['voice-lab', 'job', id] as const,
};

export const isLabJobDone = (status: LabJob['status'] | undefined) =>
  status === 'succeeded' || status === 'failed' || status === 'cancelled';

export const getCapabilities = (signal?: AbortSignal) =>
  apiJson<VoiceLabCapabilities>(`${BASE}/capabilities`, { signal });

export const listDatasets = async (signal?: AbortSignal) =>
  (await apiJson<{ datasets: DatasetSummary[] }>(`${BASE}/datasets`, { signal })).datasets;

export const getDataset = (id: string, signal?: AbortSignal) =>
  apiJson<DatasetDetail>(`${BASE}/datasets/${encodeURIComponent(id)}`, { signal });

export const getLabJob = (id: string, signal?: AbortSignal) =>
  apiJson<LabJob>(`${BASE}/jobs/${encodeURIComponent(id)}`, { signal });

export const cancelLabJob = (id: string) =>
  apiJson<LabJob>(`${BASE}/jobs/${encodeURIComponent(id)}/cancel`, { method: 'POST' });

export function toDatasetForm(files: File[], options: DatasetOptions, reference?: File | null): FormData {
  const body = new FormData();
  body.append('name', options.name.trim() || DEFAULT_DATASET_OPTIONS.name);
  if (options.language.trim()) body.append('language', options.language.trim());
  body.append('separate', String(options.separate));
  body.append('transcribe', String(options.transcribe));
  body.append('speaker_filter', String(options.speakerFilter));
  body.append('tag_nonverbal', String(options.tagNonverbal));
  body.append('min_clip_s', String(options.minClipS));
  body.append('max_clip_s', String(options.maxClipS));
  body.append('min_quality', String(options.minQuality));
  for (const file of files) body.append('files', file, file.name);
  if (reference) body.append('reference', reference, reference.name);
  return body;
}

export const buildDataset = (files: File[], options: DatasetOptions, reference?: File | null) =>
  apiJson<LabJob>(`${BASE}/datasets`, { method: 'POST', body: toDatasetForm(files, options, reference) });

export const datasetAudioUrl = (id: string, file: string) =>
  apiPath(`${BASE}/datasets/${encodeURIComponent(id)}/audio?file=${encodeURIComponent(file)}`);

/** Stages the next build will skip because their model is not on disk. */
export function missingModels(caps: VoiceLabCapabilities | undefined): string[] {
  if (!caps) return [];
  return [caps.separation, caps.speaker, caps.transcription]
    .map((stage) => stage.missing)
    .filter((m): m is string => Boolean(m));
}
