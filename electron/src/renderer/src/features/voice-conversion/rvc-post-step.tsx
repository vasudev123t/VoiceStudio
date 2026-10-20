import { useEffect, useId } from 'react';
import { useTranslation } from 'react-i18next';
import { ArrowLeftRightIcon, ChevronDownIcon } from 'lucide-react';
import { Link } from '@tanstack/react-router';
// @ts-expect-error Shared JSX has no declaration file.
import SearchableSelect from '@shared/components/SearchableSelect';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/popover';
import { buttonVariants } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { setCloneSetting, useCloneSettings } from '@/lib/store/clone-settings';
import { cn } from '@/lib/utils';
import { useRvcVoices, useVcStatus } from './queries';

/**
 * "Convert with RVC voice" composer pill for the clone workspace. Hidden until
 * at least one RVC voice exists, so the default composer is unchanged.
 */
export function RvcPostStepControl({ disabled = false }: { disabled?: boolean }) {
  const { t } = useTranslation();
  const settings = useCloneSettings();
  const voices = useRvcVoices();
  const status = useVcStatus();
  const id = useId();
  const list = voices.data ?? [];
  const selected = list.find((voice) => voice.id === settings.rvcVoiceId) ?? null;

  // A deleted voice must not silently keep failing every take.
  useEffect(() => {
    if (voices.isSuccess && settings.rvcVoiceId && !selected) setCloneSetting('rvcVoiceId', '');
  }, [selected, settings.rvcVoiceId, voices.isSuccess]);

  if (!list.length) return null;
  const enabled = Boolean(selected);
  const unavailable = status.data ? !status.data.available : false;
  return (
    <Popover>
      <PopoverTrigger
        aria-label={
          enabled
            ? t('voiceConversion.post_step_on', { name: selected?.name ?? '' })
            : t('voiceConversion.post_step_toggle')
        }
        className={cn(
          buttonVariants({ variant: 'ghost', size: 'lg' }),
          'gap-1.5 font-normal text-muted-foreground hover:text-foreground data-popup-open:bg-secondary data-popup-open:text-foreground',
          enabled && 'text-foreground',
        )}
      >
        <ArrowLeftRightIcon aria-hidden="true" />
        <span className="max-w-28 truncate">
          {selected ? selected.name : t('voiceConversion.post_step_short')}
        </span>
        {enabled && <span aria-hidden="true" className="size-1.5 shrink-0 rounded-full bg-primary" />}
        <ChevronDownIcon aria-hidden="true" className="size-3 opacity-70" />
      </PopoverTrigger>
      <PopoverContent
        side="top"
        align="start"
        sideOffset={10}
        aria-labelledby={id}
        className="w-[min(340px,calc(100vw-32px))] space-y-3 p-4"
      >
        <div className="flex items-start justify-between gap-3 text-xs">
          <span>
            <span id={id} className="text-sm font-semibold">
              {t('voiceConversion.post_step_toggle')}
            </span>
            <span className="mt-0.5 block text-[11px] text-muted-foreground">
              {t('voiceConversion.post_step_hint')}
            </span>
          </span>
          <Switch
            aria-label={t('voiceConversion.post_step_toggle')}
            checked={enabled}
            disabled={disabled}
            onCheckedChange={(checked) =>
              setCloneSetting('rvcVoiceId', checked ? (selected?.id ?? list[0].id) : '')
            }
          />
        </div>
        {enabled && (
          <SearchableSelect
            value={selected?.id ?? ''}
            options={list.map((voice) => ({ value: voice.id, label: voice.name }))}
            ariaLabel={t('voiceConversion.target_voice')}
            menuPortal
            disabled={disabled}
            onChange={(next: string) => setCloneSetting('rvcVoiceId', next)}
          />
        )}
        {enabled && unavailable && (
          <p role="status" className="text-[11px] text-warning">
            {t('voiceConversion.post_step_unavailable')}
          </p>
        )}
        <Link to="/voice-conversion" className="block text-[11px] text-primary hover:underline">
          {t('voiceConversion.open_workspace')}
        </Link>
      </PopoverContent>
    </Popover>
  );
}
