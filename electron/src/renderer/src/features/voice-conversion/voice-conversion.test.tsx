import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import type { VcStatus } from '@/lib/api/voice-conversion';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, params?: Record<string, unknown>) =>
      params ? `${key} ${JSON.stringify(params)}` : key,
    i18n: { language: 'en' },
  }),
}));
const api = vi.hoisted(() => ({ startApplioSetup: vi.fn(), cancelJob: vi.fn() }));
vi.mock('@/lib/api/voice-conversion', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/lib/api/voice-conversion')>()),
  startApplioSetup: api.startApplioSetup,
  cancelJob: api.cancelJob,
}));
vi.mock('@tanstack/react-query', () => ({
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
}));
vi.mock('./queries', () => ({ useVcJob: () => ({ data: undefined }) }));
vi.mock('@/components/bridge', () => ({ getBridge: () => null }));

import { ConversionControls } from './conversion-controls';
import { StatusBanner } from './status-banner';
import { voiceNameFromFile } from './voices-panel';
import { DEFAULT_CONVERSION_PARAMS } from '@/lib/api/voice-conversion';

afterEach(cleanup);

const status = (over: Partial<VcStatus> = {}): VcStatus => ({
  available: false,
  backend: null,
  device: 'cuda:0',
  device_override: null,
  applio: { configured_dir: null, dir: null, python: null, managed_dir: '/data/tools/applio', ready: false },
  rvc_python: false,
  f0_methods: [],
  supported_params: [],
  worker_running: false,
  active_job: null,
  docs: 'docs/voice-conversion.md',
  ...over,
});

it('shows setup steps and asks before any heavy install', () => {
  render(<StatusBanner status={status()} />);
  expect(screen.getByRole('alert')).toHaveTextContent('voiceConversion.missing_title');
  fireEvent.click(screen.getByRole('button', { name: 'voiceConversion.install_applio' }));
  expect(api.startApplioSetup).not.toHaveBeenCalled();
  expect(screen.getByRole('dialog')).toHaveTextContent('voiceConversion.install_confirm_title');
});

it('collapses to a one-line readiness note once a backend is available', () => {
  render(
    <StatusBanner
      status={status({ available: true, backend: 'applio', applio: { ...status().applio, ready: true } })}
    />,
  );
  expect(screen.queryByRole('alert')).toBeNull();
  expect(screen.getByTestId('vc-ready')).toHaveTextContent('"backend":"Applio"');
  expect(screen.getByTestId('vc-ready')).toHaveTextContent('"device":"cuda:0"');
});

it('hides knobs the active backend ignores and explains protect', () => {
  const onChange = vi.fn();
  const view = render(
    <ConversionControls
      params={DEFAULT_CONVERSION_PARAMS}
      onChange={onChange}
      f0Methods={['rmvpe', 'crepe', 'fcpe']}
      supported={['pitch', 'f0_method', 'index_rate', 'protect', 'rms_mix_rate', 'clean_audio']}
    />,
  );
  expect(screen.queryByText('voiceConversion.filter_radius')).toBeNull();
  expect(screen.getByText('voiceConversion.protect')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'voiceConversion.protect_hint' })).toBeInTheDocument();
  fireEvent.click(screen.getByRole('switch', { name: 'voiceConversion.clean_audio' }));
  expect(onChange).toHaveBeenCalledWith({ ...DEFAULT_CONVERSION_PARAMS, cleanAudio: true });

  view.rerender(
    <ConversionControls
      params={DEFAULT_CONVERSION_PARAMS}
      onChange={onChange}
      f0Methods={['rmvpe', 'harvest']}
      supported={['filter_radius']}
    />,
  );
  expect(screen.getByText('voiceConversion.filter_radius')).toBeInTheDocument();
  expect(screen.queryByRole('switch', { name: 'voiceConversion.clean_audio' })).toBeNull();
});

it('derives a voice name from Applio weight files', () => {
  expect(voiceNameFromFile('Narrator_300e_9000s.pth')).toBe('Narrator');
  expect(voiceNameFromFile('my-voice.PTH')).toBe('my voice');
});
