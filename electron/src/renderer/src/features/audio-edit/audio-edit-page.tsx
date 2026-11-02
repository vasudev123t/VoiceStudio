import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { WorkspaceHeader } from '@/components/app-shell/workspace-header';
import { FreeGpuButton } from '@/components/free-gpu-button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import type { AudioEditMode } from '@/lib/api/audio-edit';
import { HistoryTab } from './history-tab';
import { SongTab } from './song-tab';
import { SpeechTab } from './speech-tab';

type Tab = 'speech' | 'song' | 'history';
const TABS: Tab[] = ['speech', 'song', 'history'];

/** Audio Edit: regenerate chosen words in speech or sung vocals. */
export function AudioEditPage() {
  const { t } = useTranslation();
  const [tab, setTab] = useState<Tab>('speech');
  const [open, setOpen] = useState<Record<AudioEditMode, string | null>>({ speech: null, song: null });
  const openSession = (mode: AudioEditMode, id: string | null) => setOpen((prev) => ({ ...prev, [mode]: id }));
  return (
    <div className="flex h-full min-h-0 flex-col" data-testid="audio-edit-page">
      <WorkspaceHeader>
        <h1 className="text-sm font-medium">{t('audioEdit.title')}</h1>
        <span className="min-w-0 flex-1" />
        <FreeGpuButton />
      </WorkspaceHeader>
      <section className="min-h-0 flex-1 overflow-y-auto px-6 py-6">
        <div className="mx-auto w-full max-w-6xl space-y-5">
          <div className="space-y-1">
            <h2 className="text-lg font-semibold">{t('audioEdit.title')}</h2>
            <p className="text-sm text-muted-foreground">{t('audioEdit.subtitle')}</p>
          </div>
          <Tabs value={tab} onValueChange={(value) => setTab(value as Tab)} className="gap-5">
            <TabsList aria-label={t('audioEdit.title')}>
              {TABS.map((value) => (
                <TabsTrigger key={value} value={value} className="px-3">
                  {t(`audioEdit.tabs.${value}`)}
                </TabsTrigger>
              ))}
            </TabsList>
            <TabsContent value="speech">
              <SpeechTab sessionId={open.speech} onSessionChange={(id) => openSession('speech', id)} />
            </TabsContent>
            <TabsContent value="song">
              <SongTab sessionId={open.song} onSessionChange={(id) => openSession('song', id)} />
            </TabsContent>
            <TabsContent value="history">
              <HistoryTab onOpen={(mode, id) => { openSession(mode, id); setTab(mode); }} />
            </TabsContent>
          </Tabs>
        </div>
      </section>
    </div>
  );
}
