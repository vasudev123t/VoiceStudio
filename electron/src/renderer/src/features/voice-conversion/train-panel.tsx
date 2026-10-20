import { useEffect, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { FolderOpenIcon, PlayIcon, SquareIcon, UploadIcon } from 'lucide-react';
import { toast } from 'sonner';
// @ts-expect-error Shared JSX has no declaration file.
import SearchableSelect from '@shared/components/SearchableSelect';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { Switch } from '@/components/ui/switch';
import { getBridge } from '@/components/bridge';
import { describeError } from '@/lib/api/client';
import {
  cancelJob,
  DEFAULT_TRAIN_SETTINGS,
  isTerminal,
  startTraining,
  vcQueryKeys,
  type TrainF0Method,
  type TrainInput,
  type TrainSampleRate,
  type VcStatus,
} from '@/lib/api/voice-conversion';
import { useVcJob } from './queries';

type Settings = Omit<TrainInput, 'name' | 'files' | 'datasetAuthorization'>;
const RATES: TrainSampleRate[] = [32000, 40000, 48000];
const F0: TrainF0Method[] = ['rmvpe', 'crepe', 'fcpe'];

function NumberField({
  label,
  hint,
  value,
  min,
  max,
  onChange,
  disabled,
}: {
  label: string;
  hint?: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
  disabled?: boolean;
}) {
  return (
    <label className="space-y-1 text-xs">
      <span className="font-medium">{label}</span>
      <Input
        type="number"
        min={min}
        max={max}
        value={value}
        disabled={disabled}
        onChange={(event) => {
          const next = Number(event.target.value);
          if (Number.isFinite(next)) onChange(Math.max(min, Math.min(max, Math.round(next))));
        }}
      />
      {hint && <span className="block text-[11px] text-muted-foreground">{hint}</span>}
    </label>
  );
}

export function TrainPanel({ status }: { status: VcStatus }) {
  const { t } = useTranslation();
  const client = useQueryClient();
  const bridge = getBridge();
  const [name, setName] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [folder, setFolder] = useState<{ authorization: string; path: string } | null>(null);
  const [settings, setSettings] = useState<Settings>({ ...DEFAULT_TRAIN_SETTINGS });
  const [jobId, setJobId] = useState<string | null>(
    status.active_job?.kind === 'train' ? status.active_job.id : null,
  );
  const [starting, setStarting] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const log = useRef<HTMLPreElement>(null);
  const job = useVcJob(jobId).data;
  const running = Boolean(job && !isTerminal(job.status));
  const canTrain = status.applio.ready;
  const hasData = files.length > 0 || Boolean(folder);
  const set = <K extends keyof Settings>(key: K, value: Settings[K]) =>
    setSettings((current) => ({ ...current, [key]: value }));

  useEffect(() => {
    if (log.current) log.current.scrollTop = log.current.scrollHeight;
  }, [job?.log_tail.length, job?.log_lines]);
  useEffect(() => {
    if (job?.status === 'succeeded') {
      void client.invalidateQueries({ queryKey: vcQueryKeys.voices });
    }
  }, [client, job?.status]);

  const start = async () => {
    if (!name.trim() || !hasData || running || starting) return;
    setStarting(true);
    try {
      const started = await startTraining({
        name,
        files,
        datasetAuthorization: folder?.authorization ?? null,
        ...settings,
      });
      setJobId(started.id);
      setFolder(null); // capabilities are one-shot
    } catch (error) {
      toast.error(describeError(error));
    } finally {
      setStarting(false);
    }
  };
  const chooseFolder = async () => {
    const picker = bridge?.files.authorizeVoiceConversionDirectory;
    if (!picker) return;
    try {
      const picked = await picker('rvc_dataset_dir');
      if (picked) {
        setFolder(picked);
        setFiles([]);
      }
    } catch (error) {
      toast.error(describeError(error));
    }
  };
  const totalMb = files.reduce((sum, file) => sum + file.size, 0) / (1024 * 1024);

  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_360px]">
      <div className="min-w-0 space-y-6">
        {!canTrain && (
          <p role="status" className="rounded-lg bg-muted/50 p-3 text-sm text-muted-foreground">
            {t('voiceConversion.train_needs_applio')}
          </p>
        )}
        <section className="space-y-3" aria-label={t('voiceConversion.dataset')}>
          <h3 className="text-sm font-medium">{t('voiceConversion.dataset')}</h3>
          <p className="text-xs text-muted-foreground">{t('voiceConversion.dataset_tips')}</p>
          <input
            ref={input}
            type="file"
            multiple
            accept="audio/*,.wav,.flac,.mp3,.ogg,.m4a"
            className="sr-only"
            aria-label={t('voiceConversion.add_dataset_files')}
            onChange={(event) => {
              const chosen = Array.from(event.target.files ?? []);
              if (chosen.length) {
                setFiles((current) => [...current, ...chosen]);
                setFolder(null);
              }
              event.target.value = '';
            }}
          />
          <div className="flex flex-wrap gap-2">
            <Button
              variant="outline"
              disabled={running}
              onClick={() => input.current?.click()}
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault();
                const dropped = Array.from(event.dataTransfer.files);
                if (dropped.length && !running) {
                  setFiles((current) => [...current, ...dropped]);
                  setFolder(null);
                }
              }}
            >
              <UploadIcon />
              {t('voiceConversion.add_dataset_files')}
            </Button>
            {bridge?.files.authorizeVoiceConversionDirectory && (
              <Button variant="outline" disabled={running} onClick={() => void chooseFolder()}>
                <FolderOpenIcon />
                {t('voiceConversion.choose_dataset_folder')}
              </Button>
            )}
            {hasData && !running && (
              <Button
                variant="ghost"
                onClick={() => {
                  setFiles([]);
                  setFolder(null);
                }}
              >
                {t('clone.clear')}
              </Button>
            )}
          </div>
          <p className="text-xs text-muted-foreground" data-testid="vc-dataset-summary">
            {folder
              ? t('voiceConversion.dataset_folder', { path: folder.path })
              : files.length
                ? t('voiceConversion.dataset_files', {
                    files: files.length,
                    size: totalMb.toFixed(1),
                  })
                : t('voiceConversion.dataset_empty')}
          </p>
        </section>

        <section className="space-y-3" aria-label={t('voiceConversion.train_progress')}>
          <div className="flex flex-wrap items-center gap-2">
            <Input
              className="max-w-xs"
              aria-label={t('voiceConversion.voice_name')}
              placeholder={t('voiceConversion.voice_name')}
              value={name}
              maxLength={80}
              disabled={running}
              onChange={(event) => setName(event.target.value)}
            />
            <Button
              disabled={!canTrain || !name.trim() || !hasData || running || starting}
              onClick={() => void start()}
            >
              <PlayIcon />
              {t('voiceConversion.start_training')}
            </Button>
            {running && job && (
              <Button variant="ghost" onClick={() => void cancelJob(job.id)}>
                <SquareIcon />
                {t('voiceConversion.cancel_training')}
              </Button>
            )}
          </div>
          {job && (
            <div className="space-y-2" role="status">
              <div className="flex flex-wrap justify-between gap-2 text-xs text-muted-foreground">
                <span>
                  {t(`voiceConversion.job_status_${job.status}`)}
                  {job.stage ? ` · ${t(`voiceConversion.stage_${job.stage}`)}` : ''}
                </span>
                <span className="tabular-nums">
                  {t('voiceConversion.epoch_progress', {
                    epoch: job.epoch,
                    total: job.total_epochs,
                  })}
                </span>
              </div>
              <Progress
                value={Math.round(job.progress * 100)}
                aria-label={t('voiceConversion.train_progress')}
              />
              {job.error && <p className="text-xs text-destructive">{job.error}</p>}
              {job.status === 'succeeded' && (
                <p className="text-xs text-primary">
                  {t('voiceConversion.train_done', { name: job.name })}
                </p>
              )}
              <pre
                ref={log}
                aria-label={t('voiceConversion.train_log')}
                className="max-h-72 overflow-auto rounded-md bg-muted/40 p-3 font-mono text-[11px] leading-relaxed whitespace-pre-wrap"
              >
                {job.log_tail.join('\n')}
              </pre>
            </div>
          )}
        </section>
      </div>

      <aside className="space-y-4" aria-label={t('voiceConversion.train_settings')}>
        <h3 className="text-sm font-medium">{t('voiceConversion.train_settings')}</h3>
        <p className="text-[11px] text-muted-foreground">{t('voiceConversion.train_defaults_hint')}</p>
        <div className="space-y-1 text-xs">
          <span className="font-medium">{t('voiceConversion.sample_rate')}</span>
          <SearchableSelect
            value={String(settings.sampleRate)}
            options={RATES.map((rate) => ({ value: String(rate), label: `${rate / 1000} kHz` }))}
            ariaLabel={t('voiceConversion.sample_rate')}
            menuPortal
            disabled={running}
            onChange={(next: string) => set('sampleRate', Number(next) as TrainSampleRate)}
          />
        </div>
        <div className="space-y-1 text-xs">
          <span className="font-medium">{t('voiceConversion.f0_method')}</span>
          <SearchableSelect
            value={settings.f0Method}
            options={F0.map((value) => ({ value, label: t(`voiceConversion.f0_${value}`) }))}
            ariaLabel={t('voiceConversion.f0_method')}
            menuPortal
            disabled={running}
            onChange={(next: string) => set('f0Method', next as TrainF0Method)}
          />
        </div>
        <NumberField
          label={t('voiceConversion.epochs')}
          hint={t('voiceConversion.epochs_hint')}
          value={settings.epochs}
          min={1}
          max={10000}
          disabled={running}
          onChange={(value) => set('epochs', value)}
        />
        <NumberField
          label={t('voiceConversion.batch_size')}
          hint={t('voiceConversion.batch_size_hint')}
          value={settings.batchSize}
          min={1}
          max={50}
          disabled={running}
          onChange={(value) => set('batchSize', value)}
        />
        <NumberField
          label={t('voiceConversion.save_every')}
          value={settings.saveEvery}
          min={1}
          max={100}
          disabled={running}
          onChange={(value) => set('saveEvery', value)}
        />
        <div className="flex items-start justify-between gap-3 text-xs">
          <span>
            <span className="font-medium">{t('voiceConversion.checkpointing')}</span>
            <span className="mt-0.5 block text-[11px] text-muted-foreground">
              {t('voiceConversion.checkpointing_hint')}
            </span>
          </span>
          <Switch
            aria-label={t('voiceConversion.checkpointing')}
            checked={settings.checkpointing}
            disabled={running}
            onCheckedChange={(checked) => set('checkpointing', checked)}
          />
        </div>
      </aside>
    </div>
  );
}
