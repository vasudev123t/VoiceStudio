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

// ── A/B take scoring ─────────────────────────────────────────────────────────

export interface MetricScore {
  value: number;
  method: string;
  research_grade?: boolean;
  unit?: 'wer' | 'cer';
  transcript?: string;
}

export interface TakeScore {
  id: string;
  secs: MetricScore | null;
  wer: MetricScore | null;
  mos: MetricScore | null;
  overall: number;
  rank?: number;
  error: string | null;
}

export interface ScoreResponse {
  results: TakeScore[];
  reference_used: boolean;
}

export const scoreTakes = (historyIds: string[], referenceProfileId?: string | null) =>
  apiJson<ScoreResponse>(`${BASE}/score`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ history_ids: historyIds, reference_profile_id: referenceProfileId ?? null }),
  });

export const getScores = async (ids: string[], signal?: AbortSignal) =>
  ids.length === 0
    ? {}
    : (await apiJson<{ scores: Record<string, TakeScore> }>(
        `${BASE}/scores?ids=${encodeURIComponent(ids.join(','))}`,
        { signal },
      )).scores;

/** "SECS 0.71 · CER 4% · MOS 3.9" — compact per-take label. */
export function formatTakeScore(score: TakeScore): string {
  const parts: string[] = [];
  if (score.secs) parts.push(`SECS ${score.secs.value.toFixed(2)}`);
  if (score.wer) parts.push(`${(score.wer.unit ?? 'wer').toUpperCase()} ${Math.round(score.wer.value * 100)}%`);
  if (score.mos) parts.push(`MOS ${score.mos.value.toFixed(1)}${score.mos.research_grade ? '' : '*'}`);
  return parts.join(' · ');
}

// ── Fine-tuning (review item 6) ─────────────────────────────────────────────

export interface FinetuneRecipe {
  id: string;
  label: string;
  base: string;
  runnable: boolean;
  ready: boolean;
  missing_packages: string[];
  weights_on_disk: boolean;
  defaults: Record<string, number | boolean>;
  note?: string | null;
}

export interface FinetuneJob {
  id: string;
  recipe: string;
  name: string;
  dataset_id: string;
  status: 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';
  stage: string | null;
  progress: number;
  step: number;
  steps: number;
  loss: number | null;
  peak_vram_mb: number | null;
  error: string | null;
  adapter_id: string | null;
  log_tail?: string;
}

export interface FinetuneAdapter {
  id: string;
  name: string;
  recipe: string;
  steps?: number;
  clips?: number;
  language?: string | null;
  final_loss_ema?: number | null;
  active: boolean;
}

export interface FinetuneRequest {
  recipe: string;
  dataset_id: string;
  name: string;
  steps?: number;
  rank?: number;
}

export const getFinetuneRecipes = async (signal?: AbortSignal) =>
  (await apiJson<{ recipes: FinetuneRecipe[] }>(`${BASE}/finetune/recipes`, { signal })).recipes;

export const startFinetune = (body: FinetuneRequest) =>
  apiJson<FinetuneJob>(`${BASE}/finetune`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

export const getFinetuneJob = (id: string, signal?: AbortSignal) =>
  apiJson<FinetuneJob>(`${BASE}/finetune/jobs/${encodeURIComponent(id)}`, { signal });

export const cancelFinetuneJob = (id: string) =>
  apiJson<FinetuneJob>(`${BASE}/finetune/jobs/${encodeURIComponent(id)}/cancel`, { method: 'POST' });

export const listAdapters = (signal?: AbortSignal) =>
  apiJson<{ adapters: FinetuneAdapter[]; active: string | null }>(`${BASE}/finetune/adapters`, { signal });

export const setActiveAdapter = (adapterId: string | null) =>
  apiJson<{ active: string | null }>(`${BASE}/finetune/active`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ adapter_id: adapterId }),
  });

/** Why a recipe can't run yet, as a short list (package names / weights repo). */
export function recipeBlockers(recipe: FinetuneRecipe): string[] {
  const out = [...(recipe.missing_packages ?? [])].map((p) => `pip: ${p}`);
  if (!recipe.weights_on_disk) out.push(`weights: ${recipe.base}`);
  if (!recipe.runnable && out.length === 0) out.push(recipe.note ?? 'not wired yet');
  return out;
}

// ── Nonverbal library (review item 7) ───────────────────────────────────────

export type NonverbalKind = 'laugh' | 'breath' | 'sigh';

export interface NonverbalClip {
  file: string;
  kind: NonverbalKind;
  duration: number;
  rms_db: number;
  source?: string | null;
}

export interface NonverbalLibrary {
  id: string;
  name: string;
  created_at: number;
  source_dataset?: string | null;
  counts: Record<NonverbalKind, number>;
  active: boolean;
  clips?: NonverbalClip[];
}

export const listNonverbalLibraries = (signal?: AbortSignal) =>
  apiJson<{ libraries: NonverbalLibrary[]; active: string | null }>(`${BASE}/nonverbal`, { signal });

export const getNonverbalLibrary = (id: string, signal?: AbortSignal) =>
  apiJson<NonverbalLibrary>(`${BASE}/nonverbal/${encodeURIComponent(id)}`, { signal });

export const libraryFromDataset = (datasetId: string, name?: string) =>
  apiJson<NonverbalLibrary & { added: number; skipped: number }>(`${BASE}/nonverbal/from-dataset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_id: datasetId, name: name || null }),
  });

export const setActiveNonverbalLibrary = (id: string | null) =>
  apiJson<{ active: string | null }>(`${BASE}/nonverbal/active`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ library_id: id }),
  });

export function addNonverbalClip(id: string, kind: NonverbalKind, file: File) {
  const body = new FormData();
  body.append('kind', kind);
  body.append('file', file, file.name);
  return apiJson<NonverbalClip>(`${BASE}/nonverbal/${encodeURIComponent(id)}/clips`, { method: 'POST', body });
}

export const removeNonverbalClip = (id: string, file: string) =>
  apiJson<{ ok: boolean }>(`${BASE}/nonverbal/${encodeURIComponent(id)}/clips/${encodeURIComponent(file)}`, {
    method: 'DELETE',
  });

export const nonverbalAudioUrl = (id: string, file: string) =>
  apiPath(`${BASE}/nonverbal/${encodeURIComponent(id)}/audio?file=${encodeURIComponent(file)}`);
