import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { GaugeIcon, TrophyIcon } from 'lucide-react';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { audioUrl, describeError } from '@/lib/api/client';
import { useHistory } from '@/hooks/use-history';
import { formatTakeScore, getScores, scoreTakes, type TakeScore } from '@/lib/api/voice-lab';

const MAX = 6;

/** Pick 2–6 takes of the same line and rank them by SECS / WER / MOS. */
export function TakeCompare() {
  const { t } = useTranslation();
  const client = useQueryClient();
  const history = useHistory();
  const [picked, setPicked] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const items = (history.data ?? []).slice(0, 40);
  const ids = items.map((item) => item.id);
  const scores = useQuery({
    queryKey: ['voice-lab', 'scores', ids.join(',')],
    queryFn: ({ signal }) => getScores(ids, signal),
    enabled: ids.length > 0,
  });
  const toggle = (id: string) =>
    setPicked((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : prev.length >= MAX ? prev : [...prev, id]));
  const run = async () => {
    setBusy(true);
    try {
      const res = await scoreTakes(picked);
      if (!res.reference_used) toast.message(t('voiceLab.score_no_reference'));
      await client.invalidateQueries({ queryKey: ['voice-lab', 'scores'] });
    } catch (error) {
      toast.error(describeError(error));
    } finally {
      setBusy(false);
    }
  };
  const byId: Record<string, TakeScore> = scores.data ?? {};
  const best = picked
    .map((id) => byId[id])
    .filter(Boolean)
    .sort((a, b) => b.overall - a.overall)[0];
  return (
    <div className="space-y-3" data-testid="take-compare">
      <div className="flex flex-wrap items-center gap-2">
        <p className="text-xs text-muted-foreground">{t('voiceLab.compare_hint', { max: MAX })}</p>
        <span className="flex-1" />
        <Button size="sm" disabled={busy || picked.length === 0} onClick={() => void run()}>
          <GaugeIcon />
          {busy ? t('voiceLab.scoring') : t('voiceLab.score_selected', { count: picked.length })}
        </Button>
      </div>
      {items.length === 0 && <p className="text-xs text-muted-foreground">{t('voiceLab.no_takes')}</p>}
      <ul className="divide-y rounded-lg border">
        {items.map((item) => {
          const score = byId[item.id];
          return (
            <li key={item.id} className="flex items-center gap-3 px-3 py-2 text-sm">
              <input
                type="checkbox"
                aria-label={t('voiceLab.pick_take')}
                checked={picked.includes(item.id)}
                onChange={() => toggle(item.id)}
              />
              <span className="min-w-0 flex-1 truncate">{item.text}</span>
              {score && (
                <span className="flex items-center gap-1 text-xs tabular-nums text-muted-foreground" data-testid="take-score">
                  {best?.id === item.id && <TrophyIcon className="size-3 text-amber-500" />}
                  {formatTakeScore(score)}
                </span>
              )}
              <audio controls preload="none" className="h-7 w-40" src={audioUrl(item.audio_path)} />
            </li>
          );
        })}
      </ul>
      <p className="text-[11px] text-muted-foreground">{t('voiceLab.score_legend')}</p>
    </div>
  );
}
