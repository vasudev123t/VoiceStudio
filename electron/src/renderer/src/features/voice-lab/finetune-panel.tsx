import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { CheckIcon, PlayIcon, SquareIcon } from 'lucide-react';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { describeError } from '@/lib/api/client';
import {
  cancelFinetuneJob,
  getFinetuneJob,
  getFinetuneRecipes,
  isLabJobDone,
  labQueryKeys,
  listAdapters,
  listDatasets,
  recipeBlockers,
  setActiveAdapter,
  startFinetune,
} from '@/lib/api/voice-lab';

/** LoRA fine-tuning on a Voice Lab dataset; activate the resulting adapter. */
export function FinetunePanel() {
  const { t } = useTranslation();
  const client = useQueryClient();
  const recipes = useQuery({ queryKey: ['voice-lab', 'ft-recipes'], queryFn: ({ signal }) => getFinetuneRecipes(signal) });
  const datasets = useQuery({ queryKey: labQueryKeys.datasets, queryFn: ({ signal }) => listDatasets(signal) });
  const adapters = useQuery({ queryKey: ['voice-lab', 'adapters'], queryFn: ({ signal }) => listAdapters(signal) });
  const [recipeId, setRecipeId] = useState('omnivoice-lora');
  const [datasetId, setDatasetId] = useState('');
  const [name, setName] = useState('My voice');
  const [steps, setSteps] = useState(600);
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useQuery({
    queryKey: ['voice-lab', 'ft-job', jobId],
    queryFn: ({ signal }) => getFinetuneJob(jobId as string, signal),
    enabled: !!jobId,
    refetchInterval: (query) => (isLabJobDone(query.state.data?.status) ? false : 2000),
  });
  const recipe = (recipes.data ?? []).find((r) => r.id === recipeId);
  const blockers = recipe ? recipeBlockers(recipe) : [];
  const running = !!job.data && !isLabJobDone(job.data.status);
  if (job.data?.status === 'succeeded' && !adapters.isFetching && !(adapters.data?.adapters ?? []).some((a) => a.id === job.data?.adapter_id)) {
    void client.invalidateQueries({ queryKey: ['voice-lab', 'adapters'] });
  }
  const start = async () => {
    try {
      const res = await startFinetune({ recipe: recipeId, dataset_id: datasetId, name, steps });
      setJobId(res.id);
    } catch (error) {
      toast.error(describeError(error));
    }
  };
  const activate = async (id: string | null) => {
    try {
      await setActiveAdapter(id);
      toast.success(id ? t('voiceLab.ft_activated') : t('voiceLab.ft_deactivated'));
      await client.invalidateQueries({ queryKey: ['voice-lab', 'adapters'] });
    } catch (error) {
      toast.error(describeError(error));
    }
  };
  return (
    <div className="space-y-4" data-testid="finetune-panel">
      <p className="text-xs text-muted-foreground">{t('voiceLab.ft_hint')}</p>
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="space-y-1 text-xs">
          <span>{t('voiceLab.ft_recipe')}</span>
          <select className="h-8 w-full rounded-md border bg-background px-2 text-sm" value={recipeId} onChange={(e) => setRecipeId(e.target.value)}>
            {(recipes.data ?? []).map((r) => (
              <option key={r.id} value={r.id}>
                {r.label}
                {r.ready ? '' : ` — ${t('voiceLab.ft_blocked')}`}
              </option>
            ))}
          </select>
        </label>
        <label className="space-y-1 text-xs">
          <span>{t('voiceLab.ft_dataset')}</span>
          <select className="h-8 w-full rounded-md border bg-background px-2 text-sm" value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
            <option value="">{t('voiceLab.ft_pick_dataset')}</option>
            {(datasets.data ?? []).map((d) => (
              <option key={d.id} value={d.id}>
                {d.name} ({d.stats.accepted})
              </option>
            ))}
          </select>
        </label>
        <label className="space-y-1 text-xs">
          <span>{t('voiceLab.ft_name')}</span>
          <input className="h-8 w-full rounded-md border bg-background px-2 text-sm" value={name} maxLength={60} onChange={(e) => setName(e.target.value)} />
        </label>
        <label className="space-y-1 text-xs">
          <span>{t('voiceLab.ft_steps')}</span>
          <input type="number" min={10} max={20000} className="h-8 w-full rounded-md border bg-background px-2 text-sm" value={steps} onChange={(e) => setSteps(Number(e.target.value) || 600)} />
        </label>
      </div>
      {blockers.length > 0 && (
        <p className="rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs" data-testid="ft-blockers">
          {t('voiceLab.ft_missing')}: {blockers.join(', ')}
        </p>
      )}
      <div className="flex items-center gap-2">
        <Button size="sm" disabled={!recipe?.ready || !datasetId || !name.trim() || running} onClick={() => void start()}>
          <PlayIcon />
          {t('voiceLab.ft_start')}
        </Button>
        {running && jobId && (
          <Button size="sm" variant="outline" onClick={() => void cancelFinetuneJob(jobId)}>
            <SquareIcon />
            {t('common.cancel')}
          </Button>
        )}
      </div>
      {job.data && (
        <div className="space-y-1 rounded-md border px-3 py-2 text-xs" data-testid="ft-job">
          <div className="flex gap-3 tabular-nums">
            <span>{job.data.status}</span>
            <span>{job.data.stage}</span>
            <span>
              {job.data.step}/{job.data.steps}
            </span>
            {job.data.loss != null && <span>loss {job.data.loss.toFixed(3)}</span>}
            {job.data.peak_vram_mb != null && <span>{job.data.peak_vram_mb} MB</span>}
          </div>
          <div className="h-1.5 overflow-hidden rounded bg-muted">
            <div className="h-full bg-primary" style={{ width: `${Math.round(job.data.progress * 100)}%` }} />
          </div>
          {job.data.error && <p className="text-destructive">{job.data.error}</p>}
        </div>
      )}
      <div className="space-y-2">
        <h3 className="text-sm font-medium">{t('voiceLab.ft_adapters')}</h3>
        {(adapters.data?.adapters ?? []).length === 0 && <p className="text-xs text-muted-foreground">{t('voiceLab.ft_no_adapters')}</p>}
        <ul className="divide-y rounded-lg border">
          {(adapters.data?.adapters ?? []).map((a) => (
            <li key={a.id} className="flex items-center gap-3 px-3 py-2 text-sm">
              <span className="min-w-0 flex-1 truncate">{a.name}</span>
              <span className="text-xs text-muted-foreground">
                {a.recipe} · {a.steps ?? '?'} steps
              </span>
              <Button size="sm" variant={a.active ? 'default' : 'outline'} onClick={() => void activate(a.active ? null : a.id)}>
                {a.active && <CheckIcon />}
                {a.active ? t('voiceLab.ft_active') : t('voiceLab.ft_activate')}
              </Button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
