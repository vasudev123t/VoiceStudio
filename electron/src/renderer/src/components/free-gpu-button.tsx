import { useQuery, useQueryClient } from '@tanstack/react-query';
import { CpuIcon } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { describeError } from '@/lib/api/client';
import { formatGpuUsage, freeGpu, freedMb, getGpuMemory } from '@/lib/api/gpu-memory';

/** Shows GPU usage and releases every model VoiceStudio holds on the GPU. */
export function FreeGpuButton({ disabled = false }: { disabled?: boolean }) {
  const { t } = useTranslation();
  const client = useQueryClient();
  const [busy, setBusy] = useState(false);
  const snapshot = useQuery({
    queryKey: ['gpu-memory'],
    queryFn: ({ signal }) => getGpuMemory(signal),
    refetchInterval: 15_000,
  });
  const usage = formatGpuUsage(snapshot.data);
  if (snapshot.data?.available === false) return null;
  const run = async () => {
    setBusy(true);
    try {
      const result = await freeGpu();
      toast.success(
        t('gpuMemory.freed', {
          mb: freedMb(result),
          released: (result.released ?? []).join(', ') || t('gpuMemory.nothing_loaded'),
        }),
      );
      await Promise.all([
        client.invalidateQueries({ queryKey: ['gpu-memory'] }),
        client.invalidateQueries({ queryKey: ['loaded-models'] }),
        client.invalidateQueries({ queryKey: ['sidebar-model-status'] }),
      ]);
    } catch (error) {
      toast.error(t('gpuMemory.free_failed', { message: describeError(error) }));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="flex items-center gap-2" data-testid="free-gpu">
      {usage && (
        <span className="text-xs text-muted-foreground tabular-nums" title={t('gpuMemory.usage_hint')}>
          {t('gpuMemory.usage', { usage, mb: Math.round(snapshot.data?.voicestudio_mb ?? 0) })}
        </span>
      )}
      <Button size="sm" variant="outline" disabled={disabled || busy} onClick={() => void run()}>
        <CpuIcon />
        {busy ? t('gpuMemory.freeing') : t('gpuMemory.free')}
      </Button>
    </div>
  );
}
