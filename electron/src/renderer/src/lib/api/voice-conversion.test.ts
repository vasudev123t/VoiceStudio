import { describe, expect, it } from 'vitest';
import {
  clampPitch,
  DEFAULT_CONVERSION_PARAMS,
  DEFAULT_TRAIN_SETTINGS,
  EXPRESSIVE_CONVERSION_PRESET,
  isTerminal,
  protectionOff,
  toConvertForm,
  toTrainForm,
  vcQueryKeys,
} from './voice-conversion';

const clip = () => new File([new Uint8Array([1, 2, 3])], 'line.wav', { type: 'audio/wav' });

describe('toConvertForm', () => {
  it('sends the exact multipart fields the backend validates', () => {
    const form = toConvertForm(clip(), 'narrator', {
      ...DEFAULT_CONVERSION_PARAMS,
      pitch: 12,
      f0Method: 'crepe',
      indexRate: 0.5,
      protect: 0.2,
      filterRadius: 0,
      rmsMixRate: 0,
      cleanAudio: true,
    });
    expect((form.get('audio') as File).name).toBe('line.wav');
    expect(Object.fromEntries([...form.entries()].filter(([key]) => key !== 'audio'))).toEqual({
      voice_id: 'narrator',
      pitch: '12',
      f0_method: 'crepe',
      index_rate: '0.5',
      protect: '0.2',
      filter_radius: '0',
      rms_mix_rate: '0',
      clean_audio: 'true',
    });
  });

  it('clamps pitch into the RVC range before sending', () => {
    expect(toConvertForm(clip(), 'v', { ...DEFAULT_CONVERSION_PARAMS, pitch: 99 }).get('pitch')).toBe('24');
    expect(clampPitch(-40)).toBe(-24);
    expect(clampPitch(3.6)).toBe(4);
    expect(clampPitch(Number.NaN)).toBe(0);
  });

  it('defaults keep the performance: protect 0.33 and the expressive volume envelope option', () => {
    expect(DEFAULT_CONVERSION_PARAMS.protect).toBeCloseTo(0.33);
    expect(DEFAULT_CONVERSION_PARAMS.f0Method).toBe('rmvpe');
    expect(DEFAULT_CONVERSION_PARAMS.rmsMixRate).toBeCloseTo(0.1);
    expect(toConvertForm(clip(), 'v', DEFAULT_CONVERSION_PARAMS).get('rms_mix_rate')).toBe('0.1');
  });

  it('flags protect 0.5 as protection off and offers the expressive preset', () => {
    expect(protectionOff({ protect: 0.5 })).toBe(true);
    expect(protectionOff({ protect: 0.33 })).toBe(false);
    expect(EXPRESSIVE_CONVERSION_PRESET).toEqual({
      f0Method: 'rmvpe',
      protect: 0.33,
      rmsMixRate: 0.1,
      indexRate: 0.6,
    });
  });
});

describe('toTrainForm', () => {
  it('uploads dataset files when no desktop folder capability is present', () => {
    const form = toTrainForm({
      ...DEFAULT_TRAIN_SETTINGS,
      name: '  Narrator ',
      files: [clip(), new File(['x'], 'b.flac')],
    });
    expect(form.get('name')).toBe('Narrator');
    expect(form.get('sample_rate')).toBe('48000');
    expect(form.get('batch_size')).toBe('8');
    expect(form.get('checkpointing')).toBe('false');
    expect(form.getAll('files')).toHaveLength(2);
    expect(form.has('dataset_authorization')).toBe(false);
  });

  it('prefers the one-shot folder capability and never sends a raw path', () => {
    const form = toTrainForm({
      ...DEFAULT_TRAIN_SETTINGS,
      name: 'n',
      files: [clip()],
      datasetAuthorization: 'token',
      epochs: 10,
      saveEvery: 25,
    });
    expect(form.get('dataset_authorization')).toBe('token');
    expect(form.has('files')).toBe(false);
    expect(form.get('save_every')).toBe('10');
  });

  it('ships 8 GB-safe defaults', () => {
    expect(DEFAULT_TRAIN_SETTINGS).toMatchObject({ batchSize: 8, sampleRate: 48000, checkpointing: false });
  });
});

it('classifies terminal job states and namespaces query keys', () => {
  expect(isTerminal('running')).toBe(false);
  expect(['succeeded', 'failed', 'cancelled'].every((s) => isTerminal(s as never))).toBe(true);
  expect(vcQueryKeys.job('a')[0]).toBe(vcQueryKeys.all[0]);
});
