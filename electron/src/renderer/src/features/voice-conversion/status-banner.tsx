import { useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { CpuIcon, DownloadIcon, FolderOpenIcon, TriangleAlertIcon, XIcon } from 'lucide-react';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { getBridge } from '@/components/bridge';
import { ConfirmDialog } from '@/features/clone/confirm-dialog';
import { describeError } from '@/lib/api/client';
import {
  cancelJob,
  isTerminal,
  startApplioSetup,
  updateSettings,
  vcQueryKeys,
  type VcStatus,
} from '@/lib/api/voice-conversion';
import { useVcJob } from './queries';

/** Readiness banner: missing backend → setup steps; ready → backend · device. */
export function StatusBanner({ status }: { status: VcStatus }) {
  const { t } = useTranslation();
  const client = useQueryClient();
  const bridge = getBridge();
  const [confirming, setConfirming] = useState(false);
  const [setupJobId, setSetupJobId] = useState<string | null>(
    status.active_job?.kind === 'setup' ? status.active_job.id : null,
  );
  const setupJob = useVcJob(setupJobId);
  const job = setupJob.data;
  const running = Boolean(job && !isTerminal(job.status));
  const refresh = () => client.invalidateQueries({ queryKey: vcQueryKeys.all });

  const chooseFolder = async () => {
    const picker = bridge?.files.authorizeVoiceConversionDirectory;
    if (!picker) return;
    try {
      const picked = await picker('applio_dir');
      if (!picked) return;
      await updateSettings({ applio_dir_authorization: picked.authorization });
      toast.success(t('voiceConversion.applio_folder_set', { path: picked.path }));
      void refresh();
    } catch (error) {
      toast.error(describeError(error));
    }
  };
  const install = async () => {
    try {
      const started = await startApplioSetup();
      setSetupJobId(started.id);
    } catch (error) {
      toast.error(describeError(error));
    }
  };

  if (status.available && !running) {
    return (
      <p
        className="flex items-center gap-2 text-xs text-muted-foreground"
        data-testid="vc-ready"
      >
        <CpuIcon className="size-3.5" aria-hidden="true" />
        {t('voiceConversion.ready_on', {
          backend: status.backend === 'applio' ? 'Applio' : 'rvc-python',
          device: status.device,
        })}
      </p>
    );
  }

  return (
    <section
      role="alert"
      className="space-y-3 rounded-lg border border-warning/40 bg-warning/5 p-4 text-sm"
    >
      <div className="flex items-start gap-2">
        <TriangleAlertIcon className="mt-0.5 size-4 shrink-0 text-warning" aria-hidden="true" />
        <div className="space-y-1">
          <p className="font-medium">{t('voiceConversion.missing_title')}</p>
          <p className="text-muted-foreground">{t('voiceConversion.missing_body')}</p>
          <ol className="list-decimal space-y-0.5 pl-5 text-xs text-muted-foreground">
            <li>{t('voiceConversion.setup_step_install')}</li>
            <li>{t('voiceConversion.setup_step_folder')}</li>
            <li>{t('voiceConversion.setup_step_env')}</li>
          </ol>
          <p className="text-xs text-muted-foreground">
            {t('voiceConversion.setup_docs', { path: status.docs })}
          </p>
        </div>
      </div>
      {job && (
        <div className="space-y-1.5" role="status">
          <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
            <span>
              {t(`voiceConversion.job_status_${job.status}`)}
              {job.stage ? ` · ${t(`voiceConversion.stage_${job.stage}`)}` : ''}
            </span>
            {running && (
              <Button size="xs" variant="ghost" onClick={() => void cancelJob(job.id)}>
                <XIcon />
                {t('common.cancel')}
              </Button>
            )}
          </div>
          <Progress value={Math.round(job.progress * 100)} aria-label={t('voiceConversion.setup_progress')} />
          {job.error && <p className="text-xs text-destructive">{job.error}</p>}
          {job.status === 'succeeded' && (
            <Button size="sm" variant="outline" onClick={() => void refresh()}>
              {t('common.refresh')}
            </Button>
          )}
        </div>
      )}
      <div className="flex flex-wrap gap-2">
        <Button size="sm" disabled={running} onClick={() => setConfirming(true)}>
          <DownloadIcon />
          {t('voiceConversion.install_applio')}
        </Button>
        {bridge?.files.authorizeVoiceConversionDirectory && (
          <Button size="sm" variant="outline" disabled={running} onClick={() => void chooseFolder()}>
            <FolderOpenIcon />
            {t('voiceConversion.choose_applio_folder')}
          </Button>
        )}
        <Button size="sm" variant="ghost" onClick={() => void refresh()}>
          {t('voiceConversion.check_again')}
        </Button>
      </div>
      <ConfirmDialog
        open={confirming}
        onOpenChange={setConfirming}
        destructive={false}
        title={t('voiceConversion.install_confirm_title')}
        description={t('voiceConversion.install_confirm_body', { path: status.applio.managed_dir })}
        confirmLabel={t('voiceConversion.install_applio')}
        onConfirm={install}
      />
    </section>
  );
}
