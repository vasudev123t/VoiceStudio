import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { GaugeIcon } from 'lucide-react';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { describeError } from '@/lib/api/client';
import type { HistoryItem } from '@/lib/api/types';
import { formatTakeScore, getScores, scoreTakes } from '@/lib/api/voice-lab';

/** Per-take SECS / WER / MOS, scored on demand and remembered. */
export function TakeScoreRow({ item }: { item: HistoryItem }) {
  const { t } = useTranslation();
  const client = useQueryClient();
  const [busy, setBusy] = useState(false);
  const key = ['voice-lab', 'scores', item.id];
  const score = useQuery({ queryKey: key, queryFn: ({ signal }) => getScores([item.id], signal) });
  const value = score.data?.[item.id];
  const run = async () => {
    setBusy(true);
    try {
      await scoreTakes([item.id]);
      await client.invalidateQueries({ queryKey: ['voice-lab', 'scores'] });
    } catch (error) {
      toast.error(describeError(error));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs" data-testid="take-score-row">
      <GaugeIcon className="size-3.5 text-muted-foreground" aria-hidden="true" />
      {value ? (
        <span className="tabular-nums" title={t('voiceLab.score_legend')}>
          {formatTakeScore(value)}
        </span>
      ) : (
        <span className="text-muted-foreground">{t('voiceLab.not_scored')}</span>
      )}
      <Button size="sm" variant="ghost" disabled={busy} onClick={() => void run()}>
        {busy ? t('voiceLab.scoring') : value ? t('voiceLab.rescore') : t('voiceLab.score_take')}
      </Button>
    </div>
  );
}
