import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, params?: Record<string, unknown>) => (params ? `${key} ${JSON.stringify(params)}` : key),
    i18n: { language: 'en' },
  }),
}));
const caps = {
  separation: { available: true, model: 'htdemucs', missing: null },
  vad: { available: true },
  speaker: { available: true, method: 'mfcc-stats', missing: 'speechbrain/spkrec-ecapa-voxceleb (~83 MB)' },
  transcription: { available: true, missing: null },
  quality: { available: true },
  nonverbal: { available: true },
};
vi.mock('@tanstack/react-query', () => ({
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
  useQuery: ({ queryKey }: { queryKey: readonly string[] }) => ({
    data: queryKey[1] === 'capabilities' ? caps : queryKey[1] === 'datasets' ? [] : undefined,
  }),
}));
vi.mock('@/components/app-shell/workspace-header', () => ({
  WorkspaceHeader: ({ children }: { children: React.ReactNode }) => <header>{children}</header>,
}));
vi.mock('@/hooks/use-history', () => ({ useHistory: () => ({ data: [] }) }));
vi.mock('@/components/free-gpu-button', () => ({ FreeGpuButton: () => <button type="button">free</button> }));

import { VoiceLabPage } from './voice-lab-page';

afterEach(cleanup);

it('renders the dataset builder tab with model warnings and a disabled build button', () => {
  render(<VoiceLabPage />);
  expect(screen.getByTestId('voice-lab-page')).toBeInTheDocument();
  expect(screen.getByTestId('dataset-builder')).toBeInTheDocument();
  expect(screen.getByTestId('lab-missing-models')).toHaveTextContent('spkrec-ecapa');
  expect(screen.getByRole('button', { name: /voiceLab.build$/ })).toBeDisabled();
  expect(screen.getByText('voiceLab.no_datasets')).toBeInTheDocument();
});

it('shows the fine-tune tab with recipe blockers', async () => {
  const { FinetunePanel } = await import('./finetune-panel');
  render(<FinetunePanel />);
  expect(screen.getByTestId('finetune-panel')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /voiceLab.ft_start/ })).toBeDisabled();
  expect(screen.getByText('voiceLab.ft_no_adapters')).toBeInTheDocument();
});

it('shows the nonverbal library tab empty state', async () => {
  const { NonverbalLibraryPanel } = await import('./nonverbal-library');
  render(<NonverbalLibraryPanel />);
  expect(screen.getByTestId('nonverbal-library')).toBeInTheDocument();
  expect(screen.getByText('voiceLab.nv_none')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /voiceLab.nv_from_dataset/ })).toBeDisabled();
});
