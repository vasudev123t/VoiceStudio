import { CheckCircle2Icon, KeyRoundIcon, LoaderCircleIcon } from 'lucide-react';
import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { describeError } from '@/lib/api/client';
import { type AsrProvider, getAsrKeys, saveAsrKeys } from '@/lib/api/audio-edit';
import { SettingsActionError } from './settings-action-error';
import { SettingsRow, SettingsSection } from './settings-layout';

const KEYS_QUERY = ['audio-edit', 'asr-keys'] as const;
const ENGINES_QUERY = ['audio-edit', 'asr-engines'] as const;

/** Settings → Models → ASR: masked API keys for the Audio Edit cloud recognisers. */
export function AudioEditAsrKeysSettings() {
  const { t } = useTranslation();
  const keys = useQuery({ queryKey: KEYS_QUERY, queryFn: ({ signal }) => getAsrKeys(signal) });
  return (
    <SettingsSection icon={KeyRoundIcon} title={t('settings.asr_cloud_keys_title')}>
      <SettingsRow id="asr-cloud-keys-overview" title={t('settings.asr_cloud_keys_desc')}
        description={t('settings.asr_cloud_keys_hint')}>
        {keys.isPending && <LoaderCircleIcon className="size-4 animate-spin text-muted-foreground" />}
      </SettingsRow>
      {keys.isError && (
        <div className="p-4">
          <SettingsActionError title={t('modelSettings.failed')} detail={describeError(keys.error)} />
        </div>
      )}
      {(keys.data?.providers ?? []).map((p) => <ProviderRow key={p.id} provider={p} />)}
    </SettingsSection>
  );
}

function ProviderRow({ provider }: { provider: AsrProvider }) {
  const { t } = useTranslation();
  const qc = useQueryClient();
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState('');
  const dirty = Object.keys(draft).length > 0;

  const persist = async (fields: Record<string, string>) => {
    setSaving(true);
    setError('');
    try {
      const next = await saveAsrKeys(provider.id, fields);
      qc.setQueryData(KEYS_QUERY, (old: { providers: AsrProvider[] } | undefined) =>
        old ? { providers: old.providers.map((p) => (p.id === next.id ? next : p)) } : old);
      await qc.invalidateQueries({ queryKey: ENGINES_QUERY });
      setDraft({});
      setSaved(true);
    } catch (reason) {
      setError(describeError(reason));
    } finally {
      setSaving(false);
    }
  };
  const clear = () => persist(Object.fromEntries(provider.fields.map((f) => [f.id, ''])));
  const stored = provider.fields.some((f) => f.has_value && f.source === 'settings');

  return (
    <SettingsRow
      id={`asr-cloud-key-${provider.id}`}
      title={provider.label}
      description={provider.configured
        ? t('settings.asr_cloud_key_ready')
        : t('settings.asr_cloud_key_missing')}
    >
      <div className="flex w-full flex-wrap items-center gap-2" data-testid={`asr-key-${provider.id}`}>
        {provider.fields.map((f) => (
          <Input
            key={f.id}
            aria-label={`${provider.label} ${f.label}`}
            type={f.secret ? 'password' : 'text'}
            autoComplete="off"
            className="w-full font-mono @2xl:w-64"
            value={draft[f.id] ?? (f.secret ? '' : (f.value ?? ''))}
            placeholder={f.secret
              ? (f.has_value ? `${f.masked ?? '••••'}${f.source === 'env' ? ' (env)' : ''}` : f.label)
              : (f.placeholder || f.label)}
            disabled={saving}
            onChange={(e) => {
              setSaved(false);
              setDraft({ ...draft, [f.id]: e.target.value });
            }}
          />
        ))}
        <Button disabled={!dirty || saving} onClick={() => void persist(draft)}>
          {saving && <LoaderCircleIcon className="animate-spin" />}
          {saving ? t('common.saving') : t('common.save')}
        </Button>
        {stored && (
          <Button variant="ghost" disabled={saving} onClick={() => void clear()}>
            {t('settings.asr_cloud_key_clear')}
          </Button>
        )}
        <a href={provider.docs_url} target="_blank" rel="noreferrer" className="text-xs underline text-muted-foreground">
          {t('settings.asr_cloud_key_get')}
        </a>
        {saved && !dirty && (
          <span role="status" className="inline-flex items-center gap-1 text-xs text-emerald-500">
            <CheckCircle2Icon className="size-3.5" />
            {t('settings.asr_compat_saved')}
          </span>
        )}
        {error && <span role="alert" className="text-xs text-destructive">{error}</span>}
      </div>
    </SettingsRow>
  );
}
