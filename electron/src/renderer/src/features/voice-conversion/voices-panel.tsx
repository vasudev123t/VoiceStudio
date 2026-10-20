import { useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { CheckIcon, FileAudioIcon, PencilIcon, PlusIcon, Trash2Icon, XIcon } from 'lucide-react';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ConfirmDialog } from '@/features/clone/confirm-dialog';
import { describeError } from '@/lib/api/client';
import {
  deleteVoice,
  importVoice,
  renameVoice,
  vcQueryKeys,
  type RvcVoice,
} from '@/lib/api/voice-conversion';
import { formatBytes, useRvcVoices } from './queries';

/** Strip Applio's `_<epoch>e_<step>s` suffix so "Narrator_300e_9000s.pth" → "Narrator". */
export function voiceNameFromFile(filename: string): string {
  return filename
    .replace(/\.pth$/i, '')
    .replace(/_\d+e_\d+s$/i, '')
    .replace(/[_-]+/g, ' ')
    .trim();
}

export function ImportVoiceForm({ onDone }: { onDone?: (voice: RvcVoice) => void }) {
  const { t } = useTranslation();
  const client = useQueryClient();
  const [name, setName] = useState('');
  const [model, setModel] = useState<File | null>(null);
  const [index, setIndex] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const modelInput = useRef<HTMLInputElement>(null);
  const indexInput = useRef<HTMLInputElement>(null);
  const valid = Boolean(name.trim() && model) && !busy;
  const submit = async () => {
    if (!valid || !model) return;
    setBusy(true);
    try {
      const voice = await importVoice(name, model, index);
      toast.success(t('voiceConversion.imported', { name: voice.name }));
      setName('');
      setModel(null);
      setIndex(null);
      await client.invalidateQueries({ queryKey: vcQueryKeys.voices });
      onDone?.(voice);
    } catch (error) {
      toast.error(describeError(error));
    } finally {
      setBusy(false);
    }
  };
  return (
    <form
      className="space-y-3 rounded-lg border border-border/60 p-4"
      aria-label={t('voiceConversion.import_title')}
      onSubmit={(event) => {
        event.preventDefault();
        void submit();
      }}
    >
      <div>
        <h3 className="text-sm font-medium">{t('voiceConversion.import_title')}</h3>
        <p className="mt-0.5 text-xs text-muted-foreground">{t('voiceConversion.import_hint')}</p>
      </div>
      <input
        ref={modelInput}
        type="file"
        accept=".pth"
        className="sr-only"
        aria-label={t('voiceConversion.model_file')}
        onChange={(event) => {
          const chosen = event.target.files?.[0] ?? null;
          setModel(chosen);
          if (chosen && !name.trim()) setName(voiceNameFromFile(chosen.name));
          event.target.value = '';
        }}
      />
      <input
        ref={indexInput}
        type="file"
        accept=".index"
        className="sr-only"
        aria-label={t('voiceConversion.index_file')}
        onChange={(event) => {
          setIndex(event.target.files?.[0] ?? null);
          event.target.value = '';
        }}
      />
      <div className="grid gap-2 sm:grid-cols-2">
        <Button type="button" variant="outline" onClick={() => modelInput.current?.click()}>
          <FileAudioIcon />
          <span className="truncate">{model?.name ?? t('voiceConversion.choose_model')}</span>
        </Button>
        <Button type="button" variant="outline" onClick={() => indexInput.current?.click()}>
          <FileAudioIcon />
          <span className="truncate">{index?.name ?? t('voiceConversion.choose_index')}</span>
        </Button>
      </div>
      <Input
        aria-label={t('voiceConversion.voice_name')}
        placeholder={t('voiceConversion.voice_name')}
        value={name}
        maxLength={80}
        onChange={(event) => setName(event.target.value)}
      />
      <Button type="submit" disabled={!valid}>
        <PlusIcon />
        {busy ? t('voiceConversion.importing') : t('voiceConversion.import')}
      </Button>
    </form>
  );
}

function VoiceRow({ voice }: { voice: RvcVoice }) {
  const { t } = useTranslation();
  const client = useQueryClient();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(voice.name);
  const [confirming, setConfirming] = useState(false);
  const refresh = () => client.invalidateQueries({ queryKey: vcQueryKeys.voices });
  const save = async () => {
    if (!draft.trim() || draft.trim() === voice.name) return setEditing(false);
    try {
      await renameVoice(voice.id, draft);
      setEditing(false);
      void refresh();
    } catch (error) {
      toast.error(describeError(error));
    }
  };
  return (
    <li className="flex items-center gap-3 rounded-lg px-3 py-2 hover:bg-muted/40">
      <div className="min-w-0 flex-1">
        {editing ? (
          <Input
            autoFocus
            aria-label={t('voiceConversion.voice_name')}
            value={draft}
            maxLength={80}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') void save();
              if (event.key === 'Escape') setEditing(false);
            }}
          />
        ) : (
          <p className="truncate text-sm font-medium">{voice.name}</p>
        )}
        <p className="truncate text-[11px] text-muted-foreground">
          {[
            voice.source === 'trained'
              ? t('voiceConversion.source_trained')
              : t('voiceConversion.source_imported'),
            voice.sample_rate ? `${Math.round(voice.sample_rate / 1000)} kHz` : null,
            voice.version,
            voice.has_index ? t('voiceConversion.has_index') : t('voiceConversion.no_index'),
            formatBytes(voice.model_bytes + voice.index_bytes),
          ]
            .filter(Boolean)
            .join(' · ')}
        </p>
      </div>
      {editing ? (
        <>
          <Button size="icon-sm" variant="ghost" aria-label={t('common.save')} onClick={() => void save()}>
            <CheckIcon />
          </Button>
          <Button size="icon-sm" variant="ghost" aria-label={t('common.cancel')} onClick={() => setEditing(false)}>
            <XIcon />
          </Button>
        </>
      ) : (
        <Button
          size="icon-sm"
          variant="ghost"
          aria-label={t('voiceConversion.rename')}
          title={t('voiceConversion.rename')}
          onClick={() => {
            setDraft(voice.name);
            setEditing(true);
          }}
        >
          <PencilIcon />
        </Button>
      )}
      <Button
        size="icon-sm"
        variant="ghost"
        aria-label={t('common.delete')}
        title={t('common.delete')}
        onClick={() => setConfirming(true)}
      >
        <Trash2Icon />
      </Button>
      <ConfirmDialog
        open={confirming}
        onOpenChange={setConfirming}
        title={t('voiceConversion.delete_title', { name: voice.name })}
        description={t('voiceConversion.delete_body')}
        confirmLabel={t('common.delete')}
        onConfirm={async () => {
          try {
            await deleteVoice(voice.id);
            void refresh();
          } catch (error) {
            toast.error(describeError(error));
          }
        }}
      />
    </li>
  );
}

export function VoicesPanel() {
  const { t } = useTranslation();
  const voices = useRvcVoices();
  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_360px]">
      <section className="space-y-2" aria-label={t('voiceConversion.voices')}>
        <h3 className="text-sm font-medium">{t('voiceConversion.voices')}</h3>
        {voices.isError && <p className="text-sm text-destructive">{describeError(voices.error)}</p>}
        {voices.data && !voices.data.length && (
          <p className="text-sm text-muted-foreground">{t('voiceConversion.no_voices')}</p>
        )}
        <ul className="space-y-1">
          {(voices.data ?? []).map((voice) => (
            <VoiceRow key={voice.id} voice={voice} />
          ))}
        </ul>
      </section>
      <ImportVoiceForm />
    </div>
  );
}
