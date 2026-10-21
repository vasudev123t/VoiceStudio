import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { WorkspaceHeader } from '@/components/app-shell/workspace-header';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { FreeGpuButton } from '@/components/free-gpu-button';
import { describeError } from '@/lib/api/client';
import { ConvertPanel } from './convert-panel';
import { useVcStatus } from './queries';
import { StatusBanner } from './status-banner';
import { TrainPanel } from './train-panel';
import { VoicesPanel } from './voices-panel';

type Tab = 'convert' | 'voices' | 'train';
const TABS: Tab[] = ['convert', 'voices', 'train'];

/** Voice conversion (RVC / Applio): perform a line, re-voice it, keep the performance. */
export function VoiceConversionPage() {
  const { t } = useTranslation();
  const status = useVcStatus();
  const [tab, setTab] = useState<Tab>('convert');
  return (
    <div className="flex h-full min-h-0 flex-col">
      <WorkspaceHeader>
        <h1 className="text-sm font-medium">{t('voiceConversion.title')}</h1>
        <span className="min-w-0 flex-1" />
        <FreeGpuButton />
      </WorkspaceHeader>
      <section className="min-h-0 flex-1 overflow-y-auto px-6 py-6">
        <div className="mx-auto w-full max-w-6xl space-y-5">
          <div className="space-y-1">
            <h2 className="text-lg font-semibold">{t('voiceConversion.title')}</h2>
            <p className="text-sm text-muted-foreground">{t('voiceConversion.subtitle')}</p>
            <p className="text-[11px] text-muted-foreground">{t('voiceConversion.consent_note')}</p>
          </div>
          {status.isError && (
            <div role="alert" className="flex items-center gap-2 text-sm text-destructive">
              <span>{describeError(status.error)}</span>
              <Button size="sm" variant="outline" onClick={() => void status.refetch()}>
                {t('common.retry')}
              </Button>
            </div>
          )}
          {status.data && <StatusBanner status={status.data} />}
          <Tabs value={tab} onValueChange={(value) => setTab(value as Tab)} className="gap-5">
            <TabsList aria-label={t('voiceConversion.title')}>
              {TABS.map((value) => (
                <TabsTrigger key={value} value={value} className="px-3">
                  {t(`voiceConversion.tab_${value}`)}
                </TabsTrigger>
              ))}
            </TabsList>
            {status.data && (
              <>
                <TabsContent value="convert">
                  <ConvertPanel status={status.data} />
                </TabsContent>
                <TabsContent value="voices">
                  <VoicesPanel />
                </TabsContent>
                <TabsContent value="train">
                  <TrainPanel status={status.data} />
                </TabsContent>
              </>
            )}
          </Tabs>
        </div>
      </section>
    </div>
  );
}
