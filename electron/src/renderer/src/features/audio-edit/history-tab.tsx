import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import {
  type AudioEditMode, deleteSession, ENGINE_LABELS, formatBytes, formatSeconds, listSessions, renderAudioUrl,
} from '@/lib/api/audio-edit';
import { CommercialBadge } from './licence-gate';
import { audioEditKeys } from './use-audio-edit-session';

export function HistoryTab({ onOpen }: { onOpen: (mode: AudioEditMode, id: string) => void }) {
  const { t, i18n } = useTranslation();
  const qc = useQueryClient();
  const sessions = useQuery({ queryKey: audioEditKeys.sessions, queryFn: ({ signal }) => listSessions(signal) });
  const [confirming, setConfirming] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  if (sessions.isLoading) return <p className="text-sm text-muted-foreground">{t('audioEdit.loading')}</p>;
  if (sessions.isError) return <p role="alert" className="text-sm text-destructive">{t('audioEdit.history_error')}</p>;
  const items = sessions.data?.items ?? [];
  if (items.length === 0) {
    return <p className="text-sm text-muted-foreground" data-testid="audio-edit-history-empty">{t('audioEdit.history_empty')}</p>;
  }
  const remove = async (id: string) => {
    setError(null);
    try {
      await deleteSession(id);
      setConfirming(null);
      await qc.invalidateQueries({ queryKey: audioEditKeys.sessions });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };
  return (
    <div className="space-y-2">
      {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
      <table className="w-full text-sm" data-testid="audio-edit-history">
        <thead className="text-left text-xs text-muted-foreground">
          <tr>
            <th className="py-2">{t('audioEdit.col_name')}</th>
            <th>{t('audioEdit.col_mode')}</th>
            <th>{t('audioEdit.col_engine')}</th>
            <th>{t('audioEdit.col_renders')}</th>
            <th>{t('audioEdit.col_created')}</th>
            <th>{t('audioEdit.col_size')}</th>
            <th>{t('audioEdit.col_actions')}</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item) => {
            const latest = item.latest_render;
            const playable = latest?.status === 'succeeded';
            return (
              <tr key={item.id} className="border-t align-middle">
                <td className="py-2">
                  <div className="font-medium">{item.name}</div>
                  <div className="text-xs tabular-nums text-muted-foreground">{formatSeconds(item.duration_s)}</div>
                </td>
                <td>{t(`audioEdit.tabs.${item.mode}`)}</td>
                <td>
                  {latest ? (
                    <span className="flex items-center gap-1">
                      {ENGINE_LABELS[latest.engine]}
                      <CommercialBadge commercial={latest.commercial_use} />
                    </span>
                  ) : '—'}
                </td>
                <td className="tabular-nums">{item.render_count}</td>
                <td>{new Date(item.created_at * 1000).toLocaleString(i18n.language)}</td>
                <td className="tabular-nums">{formatBytes(item.size_bytes)}</td>
                <td>
                  <div className="flex flex-wrap items-center gap-1">
                    <Button size="xs" variant="outline" onClick={() => onOpen(item.mode, item.id)}>
                      {t('audioEdit.open')}
                    </Button>
                    {playable && latest && (
                      <>
                        <audio controls preload="none" className="h-7 w-40" aria-label={t('audioEdit.play')}
                          src={renderAudioUrl(item.id, latest.id)} />
                        <a className="text-xs underline" href={renderAudioUrl(item.id, latest.id)} download>
                          {t('audioEdit.export')}
                        </a>
                      </>
                    )}
                    {confirming === item.id ? (
                      <>
                        <Button size="xs" variant="destructive" onClick={() => void remove(item.id)}>
                          {t('audioEdit.confirm_delete')}
                        </Button>
                        <Button size="xs" variant="ghost" onClick={() => setConfirming(null)}>
                          {t('audioEdit.cancel')}
                        </Button>
                      </>
                    ) : (
                      <Button size="xs" variant="ghost" onClick={() => setConfirming(item.id)}>
                        {t('audioEdit.delete')}
                      </Button>
                    )}
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
