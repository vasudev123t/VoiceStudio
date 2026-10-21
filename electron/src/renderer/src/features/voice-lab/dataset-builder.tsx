import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { DatabaseIcon, FileAudioIcon, TriangleAlertIcon, XIcon } from 'lucide-react';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { Switch } from '@/components/ui/switch';
import { describeError } from '@/lib/api/client';
import {
  DEFAULT_DATASET_OPTIONS,
  buildDataset,
  cancelLabJob,
  datasetAudioUrl,
  getCapabilities,
  getDataset,
  getLabJob,
  isLabJobDone,
  labQueryKeys,
  listDatasets,
  missingModels,
  type DatasetOptions,
} from '@/lib/api/voice-lab';

function Toggle({
  label,
  hint,
  checked,
  disabled,
  onChange,
}: {
  label: string;
  hint?: string;
  checked: boolean;
  disabled?: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <div className="flex items-start justify-between gap-3 py-1.5">
      <span>
        <span className="text-sm font-medium">{label}</span>
        {hint && <span className="mt-0.5 block text-[11px] text-muted-foreground">{hint}</span>}
      </span>
      <Switch aria-label={label} checked={checked} disabled={disabled} onCheckedChange={onChange} />
    </div>
  );
}

export function DatasetBuilder() {
  const { t } = useTranslation();
  const client = useQueryClient();
  const [files, setFiles] = useState<File[]>([]);
  const [reference, setReference] = useState<File | null>(null);
  const [options, setOptions] = useState<DatasetOptions>(DEFAULT_DATASET_OPTIONS);
  const [jobId, setJobId] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const filesInput = useRef<HTMLInputElement>(null);
  const refInput = useRef<HTMLInputElement>(null);
  const caps = useQuery({ queryKey: labQueryKeys.capabilities, queryFn: ({ signal }) => getCapabilities(signal) });
  const datasets = useQuery({ queryKey: labQueryKeys.datasets, queryFn: ({ signal }) => listDatasets(signal) });
  const job = useQuery({
    queryKey: labQueryKeys.job(jobId ?? ''),
    queryFn: ({ signal }) => getLabJob(jobId as string, signal),
    enabled: Boolean(jobId),
    refetchInterval: (query) => (isLabJobDone(query.state.data?.status) ? false : 1_500),
  });
  const running = Boolean(jobId) && !isLabJobDone(job.data?.status);
  const status = job.data?.status;
  const doneId = status === 'succeeded' ? job.data?.dataset_id : null;
  useEffect(() => {
    if (!status || !isLabJobDone(status)) return;
    void client.invalidateQueries({ queryKey: labQueryKeys.datasets });
    if (doneId) setSelected(doneId);
  }, [client, status, doneId]);
  const set = <K extends keyof DatasetOptions>(key: K, value: DatasetOptions[K]) =>
    setOptions((prev) => ({ ...prev, [key]: value }));
  const missing = missingModels(caps.data);
  const start = async () => {
    try {
      const started = await buildDataset(files, options, reference);
      setJobId(started.id);
      toast.success(t('voiceLab.build_started'));
    } catch (error) {
      toast.error(describeError(error));
    }
  };
  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]" data-testid="dataset-builder">
      <div className="space-y-4">
        <div className="space-y-2 rounded-lg border p-4">
          <h3 className="text-sm font-semibold">{t('voiceLab.recordings')}</h3>
          <p className="text-xs text-muted-foreground">{t('voiceLab.recordings_hint')}</p>
          <input
            ref={filesInput}
            type="file"
            multiple
            accept="audio/*,video/*"
            className="hidden"
            onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
          />
          <div className="flex flex-wrap items-center gap-2">
            <Button size="sm" variant="outline" disabled={running} onClick={() => filesInput.current?.click()}>
              <FileAudioIcon />
              {t('voiceLab.choose_recordings')}
            </Button>
            <span className="text-xs text-muted-foreground">
              {files.length ? t('voiceLab.files_selected', { count: files.length }) : t('voiceLab.no_files')}
            </span>
          </div>
          <input
            ref={refInput}
            type="file"
            accept="audio/*"
            className="hidden"
            onChange={(e) => setReference(e.target.files?.[0] ?? null)}
          />
          <div className="flex flex-wrap items-center gap-2">
            <Button size="sm" variant="ghost" disabled={running} onClick={() => refInput.current?.click()}>
              {t('voiceLab.choose_reference')}
            </Button>
            {reference ? (
              <span className="flex items-center gap-1 text-xs">
                {reference.name}
                <button type="button" aria-label={t('common.remove', { term: reference.name })} onClick={() => setReference(null)}>
                  <XIcon className="size-3" />
                </button>
              </span>
            ) : (
              <span className="text-xs text-muted-foreground">{t('voiceLab.reference_hint')}</span>
            )}
          </div>
        </div>
        {missing.length > 0 && (
          <div role="note" data-testid="lab-missing-models" className="flex gap-2 rounded-lg border border-amber-500/40 p-3 text-xs">
            <TriangleAlertIcon className="size-4 shrink-0 text-amber-500" />
            <span>{t('voiceLab.missing_models', { models: missing.join(', ') })}</span>
          </div>
        )}
        {jobId && job.data && (
          <div className="space-y-2 rounded-lg border p-4" data-testid="lab-job">
            <div className="flex items-center justify-between text-sm">
              <span>
                {t(`voiceLab.status_${job.data.status}`)}
                {job.data.stage ? ` · ${t(`voiceLab.stage_${job.data.stage}`, { defaultValue: job.data.stage })}` : ''}
              </span>
              {running && (
                <Button size="sm" variant="ghost" onClick={() => void cancelLabJob(jobId)}>
                  {t('common.cancel')}
                </Button>
              )}
            </div>
            <Progress value={Math.round(job.data.progress * 100)} aria-label={t('voiceLab.building')} />
            {job.data.error && <p className="text-xs text-destructive">{job.data.error}</p>}
            <pre className="max-h-40 overflow-auto rounded bg-muted/40 p-2 text-[11px]">{job.data.log_tail}</pre>
          </div>
        )}
        <DatasetList selected={selected} onSelect={setSelected} datasets={datasets.data ?? []} />
        {selected && <DatasetView id={selected} />}
      </div>
      <aside className="space-y-3 rounded-lg border p-4">
        <label className="block space-y-1 text-sm">
          <span className="font-medium">{t('voiceLab.dataset_name')}</span>
          <Input value={options.name} disabled={running} onChange={(e) => set('name', e.target.value)} />
        </label>
        <label className="block space-y-1 text-sm">
          <span className="font-medium">{t('voiceLab.language')}</span>
          <Input
            value={options.language}
            placeholder="kn, en, hi…"
            disabled={running}
            onChange={(e) => set('language', e.target.value)}
          />
          <span className="block text-[11px] text-muted-foreground">{t('voiceLab.language_hint')}</span>
        </label>
        <Toggle label={t('voiceLab.separate')} hint={t('voiceLab.separate_hint')} checked={options.separate} disabled={running} onChange={(v) => set('separate', v)} />
        <Toggle label={t('voiceLab.speaker_filter')} hint={t('voiceLab.speaker_filter_hint')} checked={options.speakerFilter} disabled={running} onChange={(v) => set('speakerFilter', v)} />
        <Toggle label={t('voiceLab.transcribe')} checked={options.transcribe} disabled={running} onChange={(v) => set('transcribe', v)} />
        <Toggle label={t('voiceLab.tag_nonverbal')} hint={t('voiceLab.tag_nonverbal_hint')} checked={options.tagNonverbal} disabled={running} onChange={(v) => set('tagNonverbal', v)} />
        <div className="grid grid-cols-3 gap-2 text-xs">
          <label className="space-y-1">
            <span>{t('voiceLab.min_clip')}</span>
            <Input type="number" step="0.5" min={0.5} value={options.minClipS} disabled={running} onChange={(e) => set('minClipS', Number(e.target.value))} />
          </label>
          <label className="space-y-1">
            <span>{t('voiceLab.max_clip')}</span>
            <Input type="number" step="1" max={30} value={options.maxClipS} disabled={running} onChange={(e) => set('maxClipS', Number(e.target.value))} />
          </label>
          <label className="space-y-1">
            <span>{t('voiceLab.min_quality')}</span>
            <Input type="number" step="0.05" min={0} max={1} value={options.minQuality} disabled={running} onChange={(e) => set('minQuality', Number(e.target.value))} />
          </label>
        </div>
        <Button className="w-full" disabled={running || files.length === 0} onClick={() => void start()}>
          <DatabaseIcon />
          {running ? t('voiceLab.building') : t('voiceLab.build')}
        </Button>
      </aside>
    </div>
  );
}

function DatasetList({
  datasets,
  selected,
  onSelect,
}: {
  datasets: { id: string; name: string; stats: { accepted: number; accepted_seconds: number } }[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const { t } = useTranslation();
  if (datasets.length === 0) return <p className="text-xs text-muted-foreground">{t('voiceLab.no_datasets')}</p>;
  return (
    <ul className="divide-y rounded-lg border" aria-label={t('voiceLab.datasets')}>
      {datasets.map((d) => (
        <li key={d.id}>
          <button
            type="button"
            className={`flex w-full items-center justify-between px-3 py-2 text-left text-sm hover:bg-muted/40 ${selected === d.id ? 'bg-muted/50' : ''}`}
            onClick={() => onSelect(d.id)}
          >
            <span className="font-medium">{d.name}</span>
            <span className="text-xs text-muted-foreground">
              {t('voiceLab.dataset_stats', { clips: d.stats?.accepted ?? 0, seconds: Math.round(d.stats?.accepted_seconds ?? 0) })}
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}

function DatasetView({ id }: { id: string }) {
  const { t } = useTranslation();
  const detail = useQuery({ queryKey: labQueryKeys.dataset(id), queryFn: ({ signal }) => getDataset(id, signal) });
  const data = detail.data;
  if (!data) return null;
  return (
    <div className="space-y-3 rounded-lg border p-4" data-testid="dataset-view">
      <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
        {Object.entries(data.stages ?? {}).map(([stage, how]) => (
          <span key={stage}>
            {t(`voiceLab.stage_${stage}`, { defaultValue: stage })}: <span className="text-foreground">{how}</span>
          </span>
        ))}
      </div>
      {data.warnings?.map((w) => (
        <p key={w} className="text-xs text-amber-600">
          {w}
        </p>
      ))}
      <p className="text-[11px] text-muted-foreground">{data.path}</p>
      <div className="max-h-96 overflow-auto">
        <table className="w-full text-xs">
          <thead className="text-left text-muted-foreground">
            <tr>
              <th className="p-1">#</th>
              <th className="p-1">{t('voiceLab.col_text')}</th>
              <th className="p-1">{t('voiceLab.col_quality')}</th>
              <th className="p-1">{t('voiceLab.col_speaker')}</th>
              <th className="p-1">{t('voiceLab.col_tags')}</th>
              <th className="p-1" />
            </tr>
          </thead>
          <tbody>
            {data.segments.map((s, i) => (
              <tr key={`${s.source}-${s.start}`} className={s.accepted ? '' : 'opacity-50'}>
                <td className="p-1 tabular-nums">{i + 1}</td>
                <td className="p-1">
                  {s.text || <span className="text-muted-foreground">—</span>}
                  {!s.accepted && <span className="ml-1 text-destructive">({s.reasons.join(', ')})</span>}
                </td>
                <td className="p-1 tabular-nums">{s.quality?.toFixed(2) ?? '—'}</td>
                <td className="p-1 tabular-nums">{s.speaker_sim?.toFixed(2) ?? '—'}</td>
                <td className="p-1">{s.tags.join(', ')}</td>
                <td className="p-1">
                  {s.file && <audio controls preload="none" className="h-7" src={datasetAudioUrl(id, s.file)} />}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
