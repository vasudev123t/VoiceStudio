import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { CheckIcon, LibraryIcon, Trash2Icon, UploadIcon } from 'lucide-react';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { describeError } from '@/lib/api/client';
import {
  addNonverbalClip,
  createNonverbalLibrary,
  getNonverbalLibrary,
  labQueryKeys,
  libraryFromDataset,
  listDatasets,
  listNonverbalLibraries,
  nonverbalAudioUrl,
  removeNonverbalClip,
  setActiveNonverbalLibrary,
  type NonverbalKind,
} from '@/lib/api/voice-lab';

const KEY = ['voice-lab', 'nonverbal'] as const;
const KINDS: NonverbalKind[] = ['laugh', 'breath', 'sigh'];

/** Real laughs/breaths/sighs of the target, spliced at [laughter]/[breath]/[sigh] tags. */
export function NonverbalLibraryPanel() {
  const { t } = useTranslation();
  const client = useQueryClient();
  const libs = useQuery({ queryKey: KEY, queryFn: ({ signal }) => listNonverbalLibraries(signal) });
  const datasets = useQuery({ queryKey: labQueryKeys.datasets, queryFn: ({ signal }) => listDatasets(signal) });
  const [datasetId, setDatasetId] = useState('');
  const [openId, setOpenId] = useState<string | null>(null);
  const [kind, setKind] = useState<NonverbalKind>('laugh');
  const detail = useQuery({
    queryKey: [...KEY, openId],
    queryFn: ({ signal }) => getNonverbalLibrary(openId as string, signal),
    enabled: !!openId,
  });
  const refresh = () => client.invalidateQueries({ queryKey: KEY });
  const act = async (work: () => Promise<unknown>, ok?: string) => {
    try {
      await work();
      if (ok) toast.success(ok);
      await refresh();
    } catch (error) {
      toast.error(describeError(error));
    }
  };
  const libraries = libs.data?.libraries ?? [];
  return (
    <div className="space-y-4" data-testid="nonverbal-library">
      <p className="text-xs text-muted-foreground">{t('voiceLab.nv_hint')}</p>
      <div className="flex flex-wrap items-end gap-2">
        <label className="space-y-1 text-xs">
          <span>{t('voiceLab.ft_dataset')}</span>
          <select className="h-8 min-w-56 rounded-md border bg-background px-2 text-sm" value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
            <option value="">{t('voiceLab.ft_pick_dataset')}</option>
            {(datasets.data ?? []).map((d) => (
              <option key={d.id} value={d.id}>
                {d.name} ({d.stats.laughs}/{d.stats.breaths})
              </option>
            ))}
          </select>
        </label>
        <Button size="sm" disabled={!datasetId} onClick={() => void act(() => libraryFromDataset(datasetId), t('voiceLab.nv_created'))}>
          <LibraryIcon />
          {t('voiceLab.nv_from_dataset')}
        </Button>
        <Button size="sm" variant="outline" onClick={() => void act(() => createNonverbalLibrary(t('voiceLab.nv_default_name')), t('voiceLab.nv_created'))}>
          {t('voiceLab.nv_new')}
        </Button>
      </div>
      {libraries.length === 0 && <p className="text-xs text-muted-foreground">{t('voiceLab.nv_none')}</p>}
      <ul className="divide-y rounded-lg border">
        {libraries.map((lib) => (
          <li key={lib.id} className="space-y-2 px-3 py-2 text-sm">
            <div className="flex items-center gap-3">
              <button type="button" className="min-w-0 flex-1 truncate text-left" onClick={() => setOpenId(openId === lib.id ? null : lib.id)}>
                {lib.name}
              </button>
              <span className="text-xs tabular-nums text-muted-foreground">
                {KINDS.map((k) => `${t(`voiceLab.nv_kind_${k}`)} ${lib.counts[k] ?? 0}`).join(' · ')}
              </span>
              <Button
                size="sm"
                variant={lib.active ? 'default' : 'outline'}
                onClick={() => void act(() => setActiveNonverbalLibrary(lib.active ? null : lib.id))}
              >
                {lib.active && <CheckIcon />}
                {lib.active ? t('voiceLab.ft_active') : t('voiceLab.ft_activate')}
              </Button>
            </div>
            {openId === lib.id && (
              <div className="space-y-2 pl-2">
                <div className="flex items-center gap-2 text-xs">
                  <select className="h-7 rounded-md border bg-background px-2" value={kind} onChange={(e) => setKind(e.target.value as NonverbalKind)}>
                    {KINDS.map((k) => (
                      <option key={k} value={k}>
                        {t(`voiceLab.nv_kind_${k}`)}
                      </option>
                    ))}
                  </select>
                  <label className="inline-flex cursor-pointer items-center gap-1 rounded-md border px-2 py-1">
                    <UploadIcon className="size-3" />
                    {t('voiceLab.nv_add_clip')}
                    <input
                      type="file"
                      accept="audio/*"
                      className="hidden"
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        e.target.value = '';
                        if (file) void act(() => addNonverbalClip(lib.id, kind, file).then(() => detail.refetch()));
                      }}
                    />
                  </label>
                </div>
                <ul className="space-y-1">
                  {(detail.data?.clips ?? []).map((clip) => (
                    <li key={clip.file} className="flex items-center gap-2 text-xs">
                      <span className="w-14">{t(`voiceLab.nv_kind_${clip.kind}`)}</span>
                      <span className="w-12 tabular-nums">{clip.duration.toFixed(1)} s</span>
                      <audio controls preload="none" className="h-7 w-48" src={nonverbalAudioUrl(lib.id, clip.file)} />
                      <Button
                        size="icon"
                        variant="ghost"
                        aria-label={t('voiceLab.nv_remove')}
                        onClick={() => void act(() => removeNonverbalClip(lib.id, clip.file).then(() => detail.refetch()))}
                      >
                        <Trash2Icon />
                      </Button>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
