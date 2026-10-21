import { describe, expect, it } from 'vitest';
import { formatGpuUsage, freedMb, type GpuSnapshot } from './gpu-memory';

const snap = (used: number, vs: number): GpuSnapshot => ({
  gpus: [{ index: 0, name: 'RTX 3070 Laptop', total_mb: 8192, used_mb: used, free_mb: 8192 - used }],
  processes: [],
  voicestudio_mb: vs,
  source: 'nvidia-smi',
  available: true,
});

describe('gpu memory helpers', () => {
  it('formats the first GPU', () => {
    expect(formatGpuUsage(snap(6090, 3592))).toBe('5.9 / 8.0 GB');
    expect(formatGpuUsage(undefined)).toBeNull();
    expect(formatGpuUsage({ ...snap(1, 1), gpus: [] })).toBeNull();
    expect(formatGpuUsage({} as GpuSnapshot)).toBeNull();
  });

  it('reports freed MB, never negative', () => {
    expect(freedMb({ released: ['rvc-worker'], before: snap(6090, 3592), after: snap(2600, 400) })).toBe(3192);
    expect(freedMb({ released: [], before: snap(1, 10), after: snap(1, 20) })).toBe(0);
  });
});
