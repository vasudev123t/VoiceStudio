import { useQuery } from '@tanstack/react-query';
import {
  getJob,
  getStatus,
  isTerminal,
  listHistory,
  listVoices,
  vcQueryKeys,
} from '@/lib/api/voice-conversion';

/** Backend/Applio readiness. Polls slowly while a setup/training job runs. */
export function useVcStatus() {
  return useQuery({
    queryKey: vcQueryKeys.status,
    queryFn: ({ signal }) => getStatus(signal),
    refetchInterval: (query) => (query.state.data?.active_job ? 5_000 : false),
  });
}

export function useRvcVoices() {
  return useQuery({
    queryKey: vcQueryKeys.voices,
    queryFn: ({ signal }) => listVoices(signal),
  });
}

export function useConversionHistory() {
  return useQuery({
    queryKey: vcQueryKeys.history,
    queryFn: ({ signal }) => listHistory(signal),
  });
}

/** Live job status + log tail; polls every 2 s until the job is terminal
 *  (the `voice_conversion` WebSocket event also invalidates it instantly). */
export function useVcJob(id: string | null) {
  return useQuery({
    queryKey: vcQueryKeys.job(id ?? ''),
    queryFn: ({ signal }) => getJob(id as string, 300, signal),
    enabled: Boolean(id),
    refetchInterval: (query) =>
      query.state.data && isTerminal(query.state.data.status) ? false : 2_000,
  });
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 MB';
  return `${(bytes / (1024 * 1024)).toFixed(bytes < 10 * 1024 * 1024 ? 1 : 0)} MB`;
}
