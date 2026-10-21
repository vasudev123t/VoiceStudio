import { useId } from 'react';
import { useTranslation } from 'react-i18next';
import { InfoIcon, RotateCcwIcon, SparklesIcon, TriangleAlertIcon } from 'lucide-react';
// @ts-expect-error Shared JSX has no declaration file.
import SearchableSelect from '@shared/components/SearchableSelect';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import {
  DEFAULT_CONVERSION_PARAMS,
  EXPRESSIVE_CONVERSION_PRESET,
  PITCH_RANGE,
  protectionOff,
  type ConversionParams,
  type F0Method,
} from '@/lib/api/voice-conversion';

const F0_LABEL_KEYS: Record<F0Method, string> = {
  rmvpe: 'voiceConversion.f0_rmvpe',
  crepe: 'voiceConversion.f0_crepe',
  fcpe: 'voiceConversion.f0_fcpe',
  harvest: 'voiceConversion.f0_harvest',
  pm: 'voiceConversion.f0_pm',
};

interface SliderRowProps {
  label: string;
  hint: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format?: (value: number) => string;
  disabled?: boolean;
  onChange: (value: number) => void;
}

function SliderRow({ label, hint, value, min, max, step, format, disabled, onChange }: SliderRowProps) {
  const id = useId();
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between gap-2 text-xs">
        <span className="flex items-center gap-1 font-medium" id={id}>
          {label}
          <Tooltip>
            <TooltipTrigger
              render={
                <button
                  type="button"
                  className="text-muted-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  aria-label={hint}
                />
              }
            >
              <InfoIcon className="size-3" aria-hidden="true" />
            </TooltipTrigger>
            <TooltipContent className="max-w-64">{hint}</TooltipContent>
          </Tooltip>
        </span>
        <output className="tabular-nums text-primary">{format ? format(value) : value}</output>
      </div>
      <Slider
        thumbProps={{ 'aria-labelledby': id }}
        min={min}
        max={max}
        step={step}
        value={[value]}
        disabled={disabled}
        onValueChange={(next) => onChange(Array.isArray(next) ? next[0] : next)}
      />
    </div>
  );
}

/** RVC inference controls; unsupported knobs for the active backend are hidden. */
export function ConversionControls({
  params,
  onChange,
  f0Methods,
  supported,
  disabled = false,
  hasIndex = true,
}: {
  params: ConversionParams;
  onChange: (next: ConversionParams) => void;
  f0Methods: F0Method[];
  supported: string[];
  disabled?: boolean;
  /** False when the selected voice has no `.index`: the index rate does nothing. */
  hasIndex?: boolean;
}) {
  const { t } = useTranslation();
  const set = <K extends keyof ConversionParams>(key: K, value: ConversionParams[K]) =>
    onChange({ ...params, [key]: value });
  const allows = (name: string) => supported.length === 0 || supported.includes(name);
  const methods = f0Methods.length ? f0Methods : (['rmvpe'] as F0Method[]);
  const options = methods.map((value) => ({ value, label: t(F0_LABEL_KEYS[value]) }));
  return (
    <section className="space-y-4" aria-label={t('voiceConversion.controls')}>
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium">{t('voiceConversion.controls')}</h3>
        <div className="flex items-center gap-1">
        <Button
          variant="ghost"
          size="xs"
          title={t('voiceConversion.expressive_preset_hint')}
          disabled={disabled}
          onClick={() =>
            onChange({
              ...params,
              ...EXPRESSIVE_CONVERSION_PRESET,
              f0Method: methods.includes('rmvpe') ? 'rmvpe' : methods[0],
            })
          }
        >
          <SparklesIcon />
          {t('voiceConversion.expressive_preset')}
        </Button>
        <Button
          variant="ghost"
          size="icon-xs"
          aria-label={t('voiceConversion.reset_controls')}
          title={t('voiceConversion.reset_controls')}
          disabled={disabled}
          onClick={() => onChange({ ...DEFAULT_CONVERSION_PARAMS, f0Method: methods[0] })}
        >
          <RotateCcwIcon />
        </Button>
        </div>
      </div>
      <SliderRow
        label={t('voiceConversion.pitch')}
        hint={t('voiceConversion.pitch_hint')}
        value={params.pitch}
        min={PITCH_RANGE[0]}
        max={PITCH_RANGE[1]}
        step={1}
        format={(v) => (v > 0 ? `+${v}` : String(v))}
        disabled={disabled}
        onChange={(v) => set('pitch', v)}
      />
      <div className="space-y-1.5">
        <span className="text-xs font-medium">{t('voiceConversion.f0_method')}</span>
        <SearchableSelect
          value={methods.includes(params.f0Method) ? params.f0Method : methods[0]}
          options={options}
          disabled={disabled}
          ariaLabel={t('voiceConversion.f0_method')}
          menuPortal
          onChange={(next: string) => set('f0Method', next as F0Method)}
        />
        <p className="text-[11px] text-muted-foreground">{t('voiceConversion.f0_hint')}</p>
      </div>
      <SliderRow
        label={t('voiceConversion.index_rate')}
        hint={t('voiceConversion.index_rate_hint')}
        value={params.indexRate}
        min={0}
        max={1}
        step={0.05}
        format={(v) => (hasIndex ? v.toFixed(2) : t('voiceConversion.index_rate_off'))}
        disabled={disabled || !hasIndex}
        onChange={(v) => set('indexRate', v)}
      />
      {!hasIndex && (
        <p
          role="status"
          data-testid="vc-no-index"
          className="flex items-start gap-1.5 text-[11px] text-warning"
        >
          <TriangleAlertIcon className="mt-0.5 size-3 shrink-0" aria-hidden="true" />
          {t('voiceConversion.no_index_warning')}
        </p>
      )}
      <SliderRow
        label={t('voiceConversion.protect')}
        hint={t('voiceConversion.protect_hint')}
        value={params.protect}
        min={0}
        max={0.5}
        step={0.01}
        format={(v) => v.toFixed(2)}
        disabled={disabled}
        onChange={(v) => set('protect', v)}
      />
      {protectionOff(params) && (
        <p
          role="status"
          data-testid="vc-protect-off"
          className="flex items-start gap-1.5 text-[11px] text-warning"
        >
          <TriangleAlertIcon className="mt-0.5 size-3 shrink-0" aria-hidden="true" />
          {t('voiceConversion.protect_off_warning')}
        </p>
      )}
      <SliderRow
        label={t('voiceConversion.rms_mix')}
        hint={t('voiceConversion.rms_mix_hint')}
        value={params.rmsMixRate}
        min={0}
        max={1}
        step={0.05}
        format={(v) => v.toFixed(2)}
        disabled={disabled}
        onChange={(v) => set('rmsMixRate', v)}
      />
      {allows('filter_radius') && (
        <SliderRow
          label={t('voiceConversion.filter_radius')}
          hint={t('voiceConversion.filter_radius_hint')}
          value={params.filterRadius}
          min={0}
          max={7}
          step={1}
          disabled={disabled}
          onChange={(v) => set('filterRadius', v)}
        />
      )}
      {allows('clean_audio') && (
        <div className="flex items-start justify-between gap-3 text-xs">
          <span>
            <span className="font-medium">{t('voiceConversion.clean_audio')}</span>
            <span className="mt-0.5 block text-[11px] text-muted-foreground">
              {t('voiceConversion.clean_audio_hint')}
            </span>
          </span>
          <Switch
            aria-label={t('voiceConversion.clean_audio')}
            checked={params.cleanAudio}
            disabled={disabled}
            onCheckedChange={(checked) => set('cleanAudio', checked)}
          />
        </div>
      )}
    </section>
  );
}
