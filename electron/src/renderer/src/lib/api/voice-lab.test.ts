import { describe, expect, it } from 'vitest';
import { DEFAULT_DATASET_OPTIONS, datasetAudioUrl, missingModels, toDatasetForm, type VoiceLabCapabilities } from './voice-lab';

describe('voice lab api helpers', () => {
  it('builds the multipart form the backend expects', () => {
    const file = new File(['x'], 'take.mp3', { type: 'audio/mpeg' });
    const ref = new File(['y'], 'ref.wav', { type: 'audio/wav' });
    const form = toDatasetForm([file], { ...DEFAULT_DATASET_OPTIONS, language: ' kn ', separate: false }, ref);
    expect(form.get('language')).toBe('kn');
    expect(form.get('separate')).toBe('false');
    expect(form.get('speaker_filter')).toBe('true');
    expect(form.get('min_clip_s')).toBe('1.5');
    expect((form.get('files') as File).name).toBe('take.mp3');
    expect((form.get('reference') as File).name).toBe('ref.wav');
    const noLang = toDatasetForm([file], DEFAULT_DATASET_OPTIONS);
    expect(noLang.get('language')).toBeNull();
    expect(noLang.get('reference')).toBeNull();
  });

  it('lists the models a build would skip', () => {
    const caps = {
      separation: { available: false, missing: 'adefossez/HTDemucs (~84 MB)' },
      vad: { available: true },
      speaker: { available: true, method: 'mfcc-stats', missing: 'speechbrain/spkrec-ecapa-voxceleb (~83 MB)' },
      transcription: { available: true, missing: null },
      quality: { available: true },
      nonverbal: { available: true },
    } as VoiceLabCapabilities;
    expect(missingModels(caps)).toEqual(['adefossez/HTDemucs (~84 MB)', 'speechbrain/spkrec-ecapa-voxceleb (~83 MB)']);
    expect(missingModels(undefined)).toEqual([]);
  });

  it('encodes clip urls', () => {
    expect(datasetAudioUrl('me-abc123', 'clips/me_0001.wav')).toContain(
      '/voice-lab/datasets/me-abc123/audio?file=clips%2Fme_0001.wav',
    );
  });
});
