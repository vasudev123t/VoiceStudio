// @ts-expect-error Shared JSX has no declaration file.
import SearchableSelect from '@shared/components/SearchableSelect';
import { useQueryClient } from '@tanstack/react-query';
import { Link } from '@tanstack/react-router';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import * as api from '@/lib/api/audio-edit';
import { audioEditKeys } from './use-audio-edit-session';

interface Props {
  overview: api.AsrEnginesOverview | undefined;
  value: api.TranscribeOptions;
  onChange: (value: api.TranscribeOptions) => void;
  disabled?: boolean;
}

/** Language and speech recogniser, picked before a file is transcribed. */
export function TranscribeOptions({ overview, value, onChange, disabled }: Props) {
  const { t } = useTranslation();
  const qc = useQueryClient();
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const engines = overview?.engines ?? [];
  const engine = engines.find((e) => e.id === value.engine);
  const blocker = api.transcribeBlocker(overview, value);

  const languageOptions = (overview?.languages ?? [{ code: 'auto', name: 'Auto-detect' }]).map((l) => ({
    value: l.code,
    label: l.code === 'auto' ? t('audioEdit.asr.language_auto') : `${l.name} (${l.code})`,
  }));
  const tag = (e: api.AsrEngine) => {
    if (e.kind === 'cloud') return e.available ? t('audioEdit.asr.tag_cloud') : t('audioEdit.asr.tag_needs_key');
    return e.available ? t('audioEdit.asr.tag_local') : t('audioEdit.asr.tag_not_installed');
  };
  const engineOptions = engines.length
    ? engines.map((e) => ({
      value: e.id,
      label: `${e.label} · ${tag(e)}`,
      disabled: !api.engineSupports(e, value.language),
    }))
    : [{ value: 'whisper', label: t('audioEdit.asr.whisper_default') }];

  const pickLanguage = (language: string) => {
    const current = engines.find((e) => e.id === value.engine);
    // Keep the engine when it handles the language, else fall back to Whisper.
    onChange({ language, engine: api.engineSupports(current, language) ? value.engine : 'whisper' });
  };

  const download = async () => {
    if (!engine) return;
    setDownloadError(null);
    try {
      await api.downloadAsrEngine(engine.id);
    } catch (err) {
      setDownloadError(err instanceof Error ? err.message : String(err));
    }
    await qc.invalidateQueries({ queryKey: audioEditKeys.asrEngines });
  };

  const dl = engine?.download;
  return (
    <div className="space-y-2 rounded-lg border p-3 text-left" data-testid="audio-edit-transcribe-options">
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="space-y-1">
          <span className="text-xs font-medium text-muted-foreground">{t('audioEdit.asr.language')}</span>
          <SearchableSelect value={value.language} options={languageOptions} ariaLabel={t('audioEdit.asr.language')}
            disabled={disabled} onChange={(v: string) => pickLanguage(v)} />
        </div>
        <div className="space-y-1">
          <span className="text-xs font-medium text-muted-foreground">{t('audioEdit.asr.engine')}</span>
          <SearchableSelect value={value.engine} options={engineOptions} ariaLabel={t('audioEdit.asr.engine')}
            disabled={disabled} onChange={(v: string) => onChange({ ...value, engine: v })} />
        </div>
      </div>
      {engine?.description && <p className="text-xs text-muted-foreground">{engine.description}</p>}
      {value.language === 'auto' && value.engine === 'whisper' && (
        <p className="text-xs text-muted-foreground">{t('audioEdit.asr.auto_hint')}</p>
      )}
      {engine && !engine.word_timestamps && (
        <p className="text-xs text-muted-foreground">{t('audioEdit.asr.phrase_timing')}</p>
      )}
      {blocker === 'key_missing' && (
        <p className="text-xs text-amber-600" data-testid="audio-edit-asr-blocker">
          {t('audioEdit.asr.key_missing', { engine: engine?.label })}{' '}
          <Link to="/settings/models/$family" params={{ family: 'asr' }} className="underline">
            {t('audioEdit.asr.open_settings')}
          </Link>
        </p>
      )}
      {blocker === 'not_installed' && engine && (
        <div className="flex flex-wrap items-center gap-2 text-xs text-amber-600" data-testid="audio-edit-asr-blocker">
          <span>{t('audioEdit.asr.model_missing', { engine: engine.label, size: engine.size_gb ?? '?' })}</span>
          {dl?.status === 'running' ? (
            <span className="tabular-nums">{t('audioEdit.asr.downloading', { pct: Math.round(dl.progress * 100) })}</span>
          ) : (
            <Button size="xs" variant="outline" disabled={disabled} onClick={() => void download()}>
              {t('audioEdit.asr.download')}
            </Button>
          )}
        </div>
      )}
      {dl?.status === 'failed' && <p className="text-xs text-destructive">{dl.error}</p>}
      {downloadError && <p className="text-xs text-destructive">{downloadError}</p>}
      {blocker === 'missing_module' && engine && (
        <p className="text-xs text-amber-600" data-testid="audio-edit-asr-blocker">
          {t('audioEdit.asr.module_missing', { engine: engine.label, module: engine.reason?.split(':')[1] ?? '' })}
        </p>
      )}
      {blocker === 'language_unsupported' && (
        <p className="text-xs text-amber-600" data-testid="audio-edit-asr-blocker">
          {t('audioEdit.asr.language_unsupported')}
        </p>
      )}
    </div>
  );
}
