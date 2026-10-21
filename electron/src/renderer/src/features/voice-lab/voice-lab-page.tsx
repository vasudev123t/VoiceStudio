import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { WorkspaceHeader } from '@/components/app-shell/workspace-header';
import { FreeGpuButton } from '@/components/free-gpu-button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { DatasetBuilder } from './dataset-builder';
import { TakeCompare } from './take-compare';

type Tab = 'dataset' | 'compare';
const TABS: Tab[] = ['dataset', 'compare'];

/** Voice Lab: build datasets from real recordings, score takes, fine-tune. */
export function VoiceLabPage() {
  const { t } = useTranslation();
  const [tab, setTab] = useState<Tab>('dataset');
  return (
    <div className="flex h-full min-h-0 flex-col" data-testid="voice-lab-page">
      <WorkspaceHeader>
        <h1 className="text-sm font-medium">{t('voiceLab.title')}</h1>
        <span className="min-w-0 flex-1" />
        <FreeGpuButton />
      </WorkspaceHeader>
      <section className="min-h-0 flex-1 overflow-y-auto px-6 py-6">
        <div className="mx-auto w-full max-w-6xl space-y-5">
          <div className="space-y-1">
            <h2 className="text-lg font-semibold">{t('voiceLab.title')}</h2>
            <p className="text-sm text-muted-foreground">{t('voiceLab.subtitle')}</p>
          </div>
          <Tabs value={tab} onValueChange={(value) => setTab(value as Tab)} className="gap-5">
            <TabsList aria-label={t('voiceLab.title')}>
              {TABS.map((value) => (
                <TabsTrigger key={value} value={value} className="px-3">
                  {t(`voiceLab.tab_${value}`)}
                </TabsTrigger>
              ))}
            </TabsList>
            <TabsContent value="dataset">
              <DatasetBuilder />
            </TabsContent>
            <TabsContent value="compare">
              <TakeCompare />
            </TabsContent>
          </Tabs>
        </div>
      </section>
    </div>
  );
}
