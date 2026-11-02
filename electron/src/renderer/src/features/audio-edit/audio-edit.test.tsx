import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type { EngineStatus, Session, SessionSummary } from '@/lib/api/audio-edit';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, params?: Record<string, unknown>) => (params ? `${key} ${JSON.stringify(params)}` : key),
    i18n: { language: 'en' },
  }),
}));

const state = vi.hoisted(() => ({
  engines: undefined as unknown,
  session: undefined as unknown,
  sessions: undefined as unknown,
}));
vi.mock('@tanstack/react-query', () => ({
  useQueryClient: () => ({ invalidateQueries: vi.fn(async () => undefined) }),
  useQuery: ({ queryKey }: { queryKey: readonly string[] }) => {
    const data =
      queryKey[1] === 'engines' ? state.engines
        : queryKey[1] === 'session' ? state.session
          : queryKey[1] === 'sessions' ? state.sessions
            : queryKey[1] === 'rvc-voices' ? [] : undefined;
    return { data, isLoading: false, isError: false };
  },
}));
vi.mock('@tanstack/react-router', () => ({
  Link: ({ to, children }: { to: string; children: React.ReactNode }) => <a href={to}>{children}</a>,
}));
vi.mock('@/components/app-shell/workspace-header', () => ({
  WorkspaceHeader: ({ children }: { children: React.ReactNode }) => <header>{children}</header>,
}));
vi.mock('@/components/free-gpu-button', () => ({ FreeGpuButton: () => <button type="button">free</button> }));
vi.mock('@shared/components/SearchableSelect', () => ({
  default: ({ ariaLabel, value }: { ariaLabel: string; value: string }) => (
    <div role="combobox" aria-label={ariaLabel} data-value={value} />
  ),
}));
vi.mock('@/lib/api/audio-edit', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/lib/api/audio-edit')>()),
  acknowledgeLicence: vi.fn(async () => ({ engine: 'f5tts', required: true, accepted_at: 1 })),
  deleteSession: vi.fn(async () => ({ ok: true })),
  saveSpans: vi.fn(async () => ({})),
  installEngine: vi.fn(async () => ({})),
}));

import * as api from '@/lib/api/audio-edit';
import { AudioEditPage } from './audio-edit-page';
import { EditWorkspace } from './edit-workspace';
import { HistoryTab } from './history-tab';
import { SpanList } from './span-list';
import { TranscriptSpans } from './transcript-spans';

const WORDS = [
  { i: 0, word: 'the', start: 0.1, end: 0.3 },
  { i: 1, word: 'quick', start: 0.35, end: 0.7 },
  { i: 2, word: 'brown', start: 0.8, end: 1.2 },
];

function engine(patch: Partial<EngineStatus> = {}): EngineStatus {
  return {
    id: 'f5tts', mode: 'speech', display_name: 'F5-TTS', installed: true, install_id: 'f5tts',
    needs_gpu: false, available: true, reason: null, licence: 'CC-BY-NC-4.0 / MIT', needs_ack: true,
    acked_at: null, commercial_use: false, ...patch,
  };
}

function session(): Session {
  return {
    id: 'a'.repeat(32), mode: 'speech', name: 'Take', stage: 'ready', error: null, duration_s: 4,
    sample_rate: 44100, channels: 1, created_at: 1, render_ids: [], renders: [],
    transcript: { language: 'en', text: 'the quick brown', words: WORDS },
    spans: [{ id: 's1', start_s: 0.35, end_s: 0.7, original_text: 'quick', replacement_text: 'slow' }],
  };
}

beforeEach(() => {
  state.engines = { engines: [engine()], separation: { available: true, weights: 'w', missing: null }, gpu: true };
  state.session = session();
  state.sessions = { items: [], total: 0 };
});
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

it('renders the page with speech, song and history tabs', () => {
  render(<AudioEditPage />);
  expect(screen.getByTestId('audio-edit-page')).toBeInTheDocument();
  for (const tab of ['speech', 'song', 'history']) {
    expect(screen.getByRole('tab', { name: `audioEdit.tabs.${tab}` })).toBeInTheDocument();
  }
  expect(screen.getByRole('button', { name: 'audioEdit.choose_file' })).toBeInTheDocument();
});

it('creates a span from two word clicks', () => {
  const onWord = vi.fn();
  render(<TranscriptSpans words={WORDS} spans={[]} anchor={null} onWord={onWord} />);
  fireEvent.click(screen.getByRole('button', { name: 'quick' }));
  expect(onWord).toHaveBeenCalledWith(1);
});

it('nudges, edits and removes spans', () => {
  const onChange = vi.fn();
  const spans = session().spans;
  render(<SpanList spans={spans} durationS={4} onChange={onChange} />);
  fireEvent.click(screen.getAllByRole('button', { name: /audioEdit.nudge_later/ })[1]);
  expect(onChange).toHaveBeenLastCalledWith([{ ...spans[0], end_s: 0.71 }]);
  fireEvent.change(screen.getByRole('textbox'), { target: { value: '' } });
  expect(onChange).toHaveBeenLastCalledWith([{ ...spans[0], replacement_text: '' }]);
  fireEvent.click(screen.getByRole('button', { name: 'audioEdit.remove_span' }));
  expect(onChange).toHaveBeenLastCalledWith([]);
});

it('gates the render behind the licence acknowledgement', () => {
  render(<EditWorkspace mode="speech" sessionId={'a'.repeat(32)} onSessionChange={vi.fn()} />);
  expect(screen.getByTestId('audio-edit-licence-gate')).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'audioEdit.render' })).toBeNull();
  const accept = screen.getByRole('button', { name: 'audioEdit.licence_accept' });
  expect(accept).toBeDisabled();
  fireEvent.click(screen.getByRole('checkbox'));
  fireEvent.click(accept);
  expect(api.acknowledgeLicence).toHaveBeenCalledWith('f5tts');
});

it('shows the render button once acknowledged and an install prompt when missing', () => {
  state.engines = { engines: [engine({ acked_at: 5 })], separation: { available: true, weights: 'w', missing: null }, gpu: true };
  const { unmount } = render(<EditWorkspace mode="speech" sessionId={'a'.repeat(32)} onSessionChange={vi.fn()} />);
  expect(screen.getByRole('button', { name: 'audioEdit.render' })).toBeEnabled();
  expect(screen.getByRole('button', { name: 'audioEdit.licence_revoke' })).toBeInTheDocument();
  unmount();
  state.engines = {
    engines: [engine({ installed: false, available: false, reason: 'Install F5-TTS to use it.' })],
    separation: { available: true, weights: 'w', missing: null }, gpu: true,
  };
  render(<EditWorkspace mode="speech" sessionId={'a'.repeat(32)} onSessionChange={vi.fn()} />);
  expect(screen.getByTestId('audio-edit-engine-missing')).toBeInTheDocument();
  expect(screen.getByRole('link', { name: 'audioEdit.open_models' })).toHaveAttribute('href', '/settings/models');
  fireEvent.click(screen.getByRole('button', { name: 'audioEdit.install' }));
  expect(api.installEngine).toHaveBeenCalledWith('f5tts');
});

it('lists history with open and confirmed delete', () => {
  const item: SessionSummary = {
    id: 'b'.repeat(32), mode: 'song', name: 'Demo', stage: 'ready', error: null, duration_s: 30,
    created_at: 1_700_000_000, render_count: 1, size_bytes: 2_000_000,
    latest_render: { id: 'c'.repeat(32), engine: 'ace_step', status: 'succeeded', commercial_use: false,
      output_path: 'renders/x/output.wav', created_at: 1 },
  };
  state.sessions = { items: [item], total: 1 };
  const onOpen = vi.fn();
  render(<HistoryTab onOpen={onOpen} />);
  expect(screen.getByText('Demo')).toBeInTheDocument();
  expect(screen.getByTestId('audio-edit-badge')).toHaveTextContent('audioEdit.badge_noncommercial');
  fireEvent.click(screen.getByRole('button', { name: 'audioEdit.open' }));
  expect(onOpen).toHaveBeenCalledWith('song', item.id);
  fireEvent.click(screen.getByRole('button', { name: 'audioEdit.delete' }));
  fireEvent.click(screen.getByRole('button', { name: 'audioEdit.confirm_delete' }));
  expect(api.deleteSession).toHaveBeenCalledWith(item.id);
});
