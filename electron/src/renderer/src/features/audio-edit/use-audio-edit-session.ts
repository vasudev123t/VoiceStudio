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
