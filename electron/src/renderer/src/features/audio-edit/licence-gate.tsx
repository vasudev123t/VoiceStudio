import { useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { acknowledgeLicence, type EngineStatus } from '@/lib/api/audio-edit';
import { audioEditKeys } from './use-audio-edit-session';

export function CommercialBadge({ commercial }: { commercial: boolean }) {
  const { t } = useTranslation();
  return (
    <Badge variant={commercial ? 'secondary' : 'outline'} data-testid="audio-edit-badge">
      {t(commercial ? 'audioEdit.badge_commercial' : 'audioEdit.badge_noncommercial')}
    </Badge>
  );
}

/** One-time, revocable acknowledgement of a non-commercial engine licence. */
export function LicenceGate({ engine }: { engine: EngineStatus }) {
  const { t } = useTranslation();
  const qc = useQueryClient();
  const [checked, setChecked] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const accept = async () => {
    setError(null);
    try {
      await acknowledgeLicence(engine.id);
      await qc.invalidateQueries({ queryKey: audioEditKeys.engines });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };
  return (
    <div className="space-y-3 rounded-lg border border-amber-500/40 p-4" data-testid="audio-edit-licence-gate">
      <p className="text-sm font-medium">{t('audioEdit.licence_title', { engine: engine.display_name })}</p>
      <p className="text-sm text-muted-foreground">
        {t('audioEdit.licence_body', { engine: engine.display_name, licence: engine.licence })}
      </p>
      <label className="flex items-center gap-2 text-sm">
        <input type="checkbox" checked={checked} onChange={(e) => setChecked(e.target.checked)} />
        {t('audioEdit.licence_checkbox')}
      </label>
      {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
      <Button size="sm" disabled={!checked} onClick={() => void accept()}>
        {t('audioEdit.licence_accept')}
      </Button>
    </div>
  );
}
