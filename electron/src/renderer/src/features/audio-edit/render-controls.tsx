// @ts-expect-error Shared JSX has no declaration file.
import SearchableSelect from '@shared/components/SearchableSelect';
import { useQuery } from '@tanstack/react-query';
import { useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { Input } from '@/components/ui/input';
import {
  type AudioEditEngine, type AudioEditMode, ENGINE_LABELS, type EngineStatus, MODE_ENGINES, type RenderOptions,
} from '@/lib/api/audio-edit';
import { listVoices } from '@/lib/api/voice-conversion';
import { audioEditKeys } from './use-audio-edit-session';

interface Props {
  mode: AudioEditMode;
  options: RenderOptions;
  onChange: (options: RenderOptions) => void;
  engines: EngineStatus[];
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1">
      <span className="text-xs font-medium text-muted-foreground">{label}</span>
      {children}
    </div>
  );
}

function Range(props: { label: string; min: number; max: number; step: number; value: number;
  onChange: (value: number) => void }) {
  return (
    <label className="space-y-1 text-xs font-medium text-muted-foreground">
      <span>{props.label}</span>
      <input type="range" className="w-full" min={props.min} max={props.max} step={props.step}
        value={props.value} aria-label={props.label} onChange={(e) => props.onChange(Number(e.target.value))} />
    </label>
  );
}

export function RenderControls({ mode, options, onChange, engines }: Props) {
  const { t } = useTranslation();
  const voices = useQuery({ queryKey: audioEditKeys.voices, queryFn: ({ signal }) => listVoices(signal) });
  const touched = useRef(false);
  const set = (patch: Partial<RenderOptions>) => {
    touched.current = true;
    onChange({ ...options, ...patch });
  };
  const firstVoice = voices.data?.[0]?.id;
  useEffect(() => {
    // Song mode re-voices with RVC by default when a model exists.
    if (mode === 'song' && firstVoice && options.rvcModelId === null && !touched.current) {
      onChange({ ...options, rvcModelId: firstVoice });
    }
  }, [mode, firstVoice, options, onChange]);
  const engineOptions = MODE_ENGINES[mode].map((id) => {
    const status = engines.find((e) => e.id === id);
    return { value: id, label: status?.display_name ?? ENGINE_LABELS[id], disabled: status ? !status.available : false };
  });
  const voiceOptions = [
    { value: '', label: t('audioEdit.rvc_none') },
    ...(voices.data ?? []).map((v) => ({ value: v.id, label: v.name })),
  ];
  return (
    <div className="grid gap-4 sm:grid-cols-2" data-testid="audio-edit-controls">
      <Field label={t('audioEdit.engine')}>
        <SearchableSelect value={options.engine} options={engineOptions} ariaLabel={t('audioEdit.engine')}
          disabled={mode === 'speech'} onChange={(value: string) => set({ engine: value as AudioEditEngine })} />
      </Field>
      <Field label={t('audioEdit.rvc_model')}>
        <SearchableSelect value={options.rvcModelId ?? ''} options={voiceOptions} ariaLabel={t('audioEdit.rvc_model')}
          onChange={(value: string) => set({ rvcModelId: value || null })} />
      </Field>
      {options.rvcModelId && (
        <Range label={t('audioEdit.rvc_pitch', { value: options.rvcPitch })} min={-12} max={12} step={1}
          value={options.rvcPitch} onChange={(value) => set({ rvcPitch: value })} />
      )}
      <Range label={t('audioEdit.crossfade', { value: options.crossfadeMs })} min={5} max={200} step={5}
        value={options.crossfadeMs} onChange={(value) => set({ crossfadeMs: value })} />
      {mode === 'song' && (
        <Range label={t('audioEdit.vocal_gain', { value: options.vocalGainDb })} min={-12} max={6} step={0.5}
          value={options.vocalGainDb} onChange={(value) => set({ vocalGainDb: value })} />
      )}
      {mode === 'song' && options.engine === 'ace_step' && (
        <>
          <Range label={t('audioEdit.ace_strength', { value: options.aceStep.strength.toFixed(2) })} min={0} max={1}
            step={0.05} value={options.aceStep.strength}
            onChange={(value) => set({ aceStep: { ...options.aceStep, strength: value } })} />
          <Range label={t('audioEdit.ace_padding', { value: options.aceStep.padding_s })} min={0} max={10} step={0.5}
            value={options.aceStep.padding_s}
            onChange={(value) => set({ aceStep: { ...options.aceStep, padding_s: value } })} />
          <Field label={t('audioEdit.ace_caption')}>
            <Input value={options.aceStep.caption} maxLength={300} placeholder={t('audioEdit.ace_caption_placeholder')}
              onChange={(e) => set({ aceStep: { ...options.aceStep, caption: e.target.value } })} />
          </Field>
        </>
      )}
    </div>
  );
}
