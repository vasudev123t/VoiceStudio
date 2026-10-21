import { useEffect, useMemo, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import {
  ArrowLeftRightIcon,
  DownloadIcon,
  MicIcon,
  SquareIcon,
  Trash2Icon,
  UploadIcon,
} from 'lucide-react';
import { toast } from 'sonner';
// @ts-expect-error Shared JSX has no declaration file.
import SearchableSelect from '@shared/components/SearchableSelect';
import { Button } from '@/components/ui/button';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import { PipelineFailure } from '@/components/pipeline-failure';
import { RecordingInputs } from '@/components/recording-inputs';
import { SaveAudioButton } from '@/components/save-audio-button';
import { WaveformPlayer } from '@/components/waveform-player';
import { getBridge } from '@/components/bridge';
import { useRecording } from '@/hooks/use-recording';
import { beginAppActivity } from '@/lib/app-activity';
import { audioUrl, describeError, isAbortError } from '@/lib/api/client';
import {
  convertAudio,
  DEFAULT_CONVERSION_PARAMS,
  deleteHistoryEntry,
  vcQueryKeys,
  type ConversionEntry,
  type ConversionParams,
  type VcStatus,
} from '@/lib/api/voice-conversion';
import { formatRelative } from '@/features/clone/format';
import { ConversionControls } from './conversion-controls';
import { useConversionHistory, useRvcVoices } from './queries';

const AUDIO_NAME = /\.(mp3|wav|m4a|flac|ogg|aac|webm|opus)$/i;
const PARAMS_KEY = 'voicestudio.voiceConversion.params.v2';
// v1 stored protect 0.5 / envelope 1.0 for many users (old defaults and the
// common "protect off" habit). v2 keeps every other saved knob but resets
// those two to the expressive defaults once.
const LEGACY_PARAMS_KEY = 'voicestudio.voiceConversion.params.v1';
const VOICE_KEY = 'voicestudio.voiceConversion.voice.v1';

function loadParams(): ConversionParams {
  try {
    const raw = localStorage.getItem(PARAMS_KEY);
    const legacy = raw ? null : localStorage.getItem(LEGACY_PARAMS_KEY);
    const parsed = raw
      ? (JSON.parse(raw) as Partial<ConversionParams>)
      : legacy
        ? {
            ...(JSON.parse(legacy) as Partial<ConversionParams>),
            protect: DEFAULT_CONVERSION_PARAMS.protect,
            rmsMixRate: DEFAULT_CONVERSION_PARAMS.rmsMixRate,
          }
        : {};
    const merged = { ...DEFAULT_CONVERSION_PARAMS };
    for (const key of Object.keys(merged) as (keyof ConversionParams)[]) {
      if (typeof parsed[key] === typeof merged[key]) Object.assign(merged, { [key]: parsed[key] });
    }
    return merged;
  } catch {
    return { ...DEFAULT_CONVERSION_PARAMS };
  }
}

export function ConvertPanel({ status }: { status: VcStatus }) {
  const { t, i18n } = useTranslation();
  const client = useQueryClient();
  const voices = useRvcVoices();
  const history = useConversionHistory();
  const [file, setFile] = useState<File | null>(null);
  const [sourceUrl, setSourceUrl] = useState('');
  const [voiceId, setVoiceId] = useState(() => localStorage.getItem(VOICE_KEY) ?? '');
  const [params, setParams] = useState<ConversionParams>(loadParams);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<ConversionEntry | null>(null);
  const [listen, setListen] = useState<'converted' | 'original'>('converted');
  const request = useRef<AbortController | null>(null);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    try {
      localStorage.setItem(PARAMS_KEY, JSON.stringify(params));
    } catch {
      // Private mode / quota: the in-memory state still applies.
    }
  }, [params]);
  useEffect(() => {
    const url = file ? URL.createObjectURL(file) : '';
    setSourceUrl(url);
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [file]);
  useEffect(() => () => request.current?.abort(), []);

  const voiceList = useMemo(() => voices.data ?? [], [voices.data]);
  const selected = voiceList.find((voice) => voice.id === voiceId) ?? null;
  useEffect(() => {
    if (!voiceList.length || selected) return;
    setVoiceId(voiceList[0].id);
  }, [selected, voiceList]);

  const ingest = (next: File) => {
    if (!next.size || !(next.type.startsWith('audio/') || AUDIO_NAME.test(next.name))) {
      setError(t('clone.unsupported_audio'));
      return;
    }
    request.current?.abort();
    setError('');
    setResult(null);
    setFile(next);
  };
  const recording = useRecording(ingest, false);
  const recordingBusy = recording.isStarting || recording.isRecording || recording.isCleaning;
  const ready = status.available && Boolean(file) && Boolean(selected) && !recordingBusy;

  const run = async () => {
    if (!ready || !file || !selected || request.current) return;
    const controller = new AbortController();
    request.current = controller;
    const finish = beginAppActivity('synthesis');
    setBusy(true);
    setError('');
    setResult(null);
    try {
      localStorage.setItem(VOICE_KEY, selected.id);
      const converted = await convertAudio(file, selected.id, params, controller.signal);
      if (!controller.signal.aborted) {
        setResult(converted);
        setListen('converted');
        void client.invalidateQueries({ queryKey: vcQueryKeys.history });
        toast.success(t('voiceConversion.converted_toast', { voice: converted.voice_name }));
      }
    } catch (failure) {
      if (!controller.signal.aborted && !isAbortError(failure)) setError(describeError(failure));
    } finally {
      finish();
      if (request.current === controller) {
        request.current = null;
        setBusy(false);
      }
    }
  };
  const cancel = () => {
    request.current?.abort();
    request.current = null;
    setBusy(false);
  };

  const voiceOptions = voiceList.map((voice) => ({
    value: voice.id,
    label: voice.sample_rate
      ? `${voice.name} · ${Math.round(voice.sample_rate / 1000)} kHz`
      : voice.name,
  }));

  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_320px]">
      <div className="min-w-0 space-y-6">
        <section className="space-y-3" aria-label={t('voiceConversion.input')}>
          <h3 className="text-sm font-medium">{t('voiceConversion.input')}</h3>
          <p className="text-xs text-muted-foreground">{t('voiceConversion.input_hint')}</p>
          <input
            ref={input}
            type="file"
            accept="audio/*,.mp3,.wav,.m4a,.flac,.ogg,.aac,.webm,.opus"
            className="sr-only"
            aria-label={t('voiceConversion.upload')}
            disabled={recordingBusy || busy}
            onChange={(event) => {
              const chosen = event.target.files?.[0];
              if (chosen) ingest(chosen);
              event.target.value = '';
            }}
          />
          <Button
            variant="outline"
            className="h-auto min-h-20 w-full whitespace-normal border-dashed p-4"
            disabled={recordingBusy || busy}
            onClick={() => input.current?.click()}
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault();
              const dropped = event.dataTransfer.files[0];
              if (!recordingBusy && !busy && dropped) ingest(dropped);
            }}
          >
            <UploadIcon />
            {file?.name ?? t('voiceConversion.drop_audio')}
          </Button>
          <RecordingInputs rec={recording} disabled={busy} />
          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="ghost"
              disabled={recording.isStarting || recording.isCleaning || busy}
              onClick={() => {
                if (recording.isRecording) recording.stop();
                else void recording.start();
              }}
            >
              {recording.isRecording ? <SquareIcon /> : <MicIcon />}
              {recording.isRecording
                ? t('voiceConversion.stop_recording', { seconds: recording.seconds })
                : t('voiceConversion.record')}
            </Button>
            {file && !busy && (
              <Button variant="ghost" onClick={() => setFile(null)}>
                {t('clone.clear')}
              </Button>
            )}
          </div>
          {sourceUrl && <WaveformPlayer src={sourceUrl} source="vc-source" height={40} />}
        </section>

        <section className="space-y-2" aria-label={t('voiceConversion.target_voice')}>
          <h3 className="text-sm font-medium">{t('voiceConversion.target_voice')}</h3>
          {voiceList.length ? (
            <SearchableSelect
              value={selected?.id ?? ''}
              options={voiceOptions}
              ariaLabel={t('voiceConversion.target_voice')}
              menuPortal
              disabled={busy}
              onChange={(next: string) => {
                setVoiceId(next);
                setResult(null);
              }}
            />
          ) : (
            <p className="text-sm text-muted-foreground">{t('voiceConversion.no_voices')}</p>
          )}
          {selected && !selected.has_index && (
            <p className="text-[11px] text-muted-foreground">{t('voiceConversion.no_index_note')}</p>
          )}
        </section>

        {error && <PipelineFailure fallback={error} onDismiss={() => setError('')} />}

        <div className="flex items-center gap-2">
          <Button className="min-w-48" disabled={!ready || busy} onClick={() => void run()}>
            <ArrowLeftRightIcon />
            {busy ? t('voiceConversion.converting') : t('voiceConversion.convert')}
          </Button>
          {busy && (
            <Button variant="ghost" onClick={cancel}>
              {t('common.cancel')}
            </Button>
          )}
        </div>

        {result && (
          <ResultCard
            entry={result}
            listen={listen}
            onListen={setListen}
            locale={i18n.language}
          />
        )}
      </div>

      <aside className="space-y-8">
        <ConversionControls
          params={params}
          onChange={setParams}
          f0Methods={status.f0_methods}
          supported={status.supported_params}
          disabled={busy}
          hasIndex={selected ? selected.has_index : true}
        />
        <HistoryList
          items={history.data ?? []}
          onOpen={(entry) => {
            setResult(entry);
            setListen('converted');
          }}
          onDelete={async (entry) => {
            try {
              await deleteHistoryEntry(entry.id);
              if (result?.id === entry.id) setResult(null);
              void client.invalidateQueries({ queryKey: vcQueryKeys.history });
            } catch (failure) {
              toast.error(describeError(failure));
            }
          }}
        />
      </aside>
    </div>
  );
}

function ResultCard({
  entry,
  listen,
  onListen,
  locale,
}: {
  entry: ConversionEntry;
  listen: 'converted' | 'original';
  onListen: (value: 'converted' | 'original') => void;
  locale: string;
}) {
  const { t } = useTranslation();
  const converted = audioUrl(entry.filename);
  const original = entry.source_filename ? audioUrl(entry.source_filename) : null;
  const suggested = `${entry.voice_name.replace(/[^\w.-]+/g, '_')}-${entry.id}.wav`;
  return (
    <section className="space-y-3 border-t pt-4" aria-label={t('voiceConversion.result')}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-medium">{t('voiceConversion.result')}</h3>
        <div className="flex items-center gap-1">
          {original && (
            <ToggleGroup
              value={[listen]}
              size="sm"
              onValueChange={(value) => {
                const next = value[0];
                if (next === 'converted' || next === 'original') onListen(next);
              }}
              aria-label={t('voiceConversion.ab_compare')}
            >
              <ToggleGroupItem value="original">
                {t('voiceConversion.ab_original')}
              </ToggleGroupItem>
              <ToggleGroupItem value="converted">
                {t('voiceConversion.ab_converted')}
              </ToggleGroupItem>
            </ToggleGroup>
          )}
          {getBridge() ? (
            <SaveAudioButton url={converted} suggestedName={suggested} />
          ) : (
            <a
              href={converted}
              download={suggested}
              className="inline-flex h-7 items-center gap-1.5 px-2 text-sm hover:underline"
            >
              <DownloadIcon className="size-3.5" aria-hidden="true" />
              {t('clone.download')}
            </a>
          )}
        </div>
      </div>
      <WaveformPlayer
        key={listen}
        src={listen === 'original' && original ? original : converted}
        source={`vc-result-${listen}`}
        height={48}
      />
      <p className="text-xs text-muted-foreground">
        {t('voiceConversion.result_meta', {
          voice: entry.voice_name,
          rate: Math.round(entry.sample_rate / 1000),
          seconds: entry.duration_s.toLocaleString(locale),
          elapsed: entry.elapsed_s.toLocaleString(locale),
          device: entry.device,
        })}
      </p>
    </section>
  );
}

function HistoryList({
  items,
  onOpen,
  onDelete,
}: {
  items: ConversionEntry[];
  onOpen: (entry: ConversionEntry) => void;
  onDelete: (entry: ConversionEntry) => void;
}) {
  const { t, i18n } = useTranslation();
  return (
    <section className="space-y-2" aria-label={t('voiceConversion.history')}>
      <h3 className="text-sm font-medium">{t('voiceConversion.history')}</h3>
      {!items.length && (
        <p className="text-xs text-muted-foreground">{t('voiceConversion.history_empty')}</p>
      )}
      <ul className="max-h-80 space-y-1 overflow-y-auto">
        {items.map((entry) => (
          <li key={entry.id} className="group flex items-center gap-1">
            <Button
              variant="ghost"
              className="h-auto min-w-0 flex-1 flex-col items-start gap-0 py-1.5 text-left"
              onClick={() => onOpen(entry)}
            >
              <span className="w-full truncate text-xs font-medium">{entry.voice_name}</span>
              <span className="w-full truncate text-[11px] text-muted-foreground">
                {entry.source_name} · {formatRelative(entry.created_at, i18n.language)}
              </span>
            </Button>
            <Button
              variant="ghost"
              size="icon-xs"
              aria-label={t('voiceConversion.delete_conversion')}
              title={t('voiceConversion.delete_conversion')}
              onClick={() => onDelete(entry)}
            >
              <Trash2Icon />
            </Button>
          </li>
        ))}
      </ul>
    </section>
  );
}
