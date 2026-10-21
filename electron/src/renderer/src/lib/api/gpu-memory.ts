import { apiJson } from './client';

export interface GpuDevice {
  index: number;
  name: string;
  total_mb: number | null;
  used_mb: number | null;
  free_mb: number | null;
}

export interface GpuProcess {
  pid: number;
  role: string;
  used_mb: number;
}

export interface GpuSnapshot {
  gpus: GpuDevice[];
  processes: GpuProcess[];
  voicestudio_mb: number;
  source: string | null;
  available: boolean;
}

export interface FreeGpuResult {
  released: string[];
  before: GpuSnapshot;
  after: GpuSnapshot;
}

export const getGpuMemory = (signal?: AbortSignal) =>
  apiJson<GpuSnapshot>('/system/gpu-memory?include_other=false', { signal });

export const freeGpu = () => apiJson<FreeGpuResult>('/system/free-gpu', { method: 'POST' });

/** "5.9 / 8.0 GB" for the first GPU, or null without an NVIDIA device. */
export function formatGpuUsage(snapshot: GpuSnapshot | undefined): string | null {
  const gpu = Array.isArray(snapshot?.gpus) ? snapshot.gpus[0] : undefined;
  if (!gpu || gpu.used_mb == null || gpu.total_mb == null) return null;
  return `${(gpu.used_mb / 1024).toFixed(1)} / ${(gpu.total_mb / 1024).toFixed(1)} GB`;
}

/** MB VoiceStudio's own processes freed between two snapshots (never negative). */
export function freedMb(result: FreeGpuResult): number {
  const before = Number(result.before?.voicestudio_mb) || 0;
  const after = Number(result.after?.voicestudio_mb) || 0;
  return Math.max(0, Math.round(before - after));
}
