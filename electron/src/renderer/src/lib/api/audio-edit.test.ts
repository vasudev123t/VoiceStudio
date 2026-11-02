import { beforeEach, describe, expect, it, vi } from 'vitest';

const client = vi.hoisted(() => ({ apiJson: vi.fn(async () => ({})) }));
vi.mock('./client', async (importOriginal) => ({
  ...(await importOriginal<typeof import('./client')>()),
  apiJson: client.apiJson,
}));

import { ApiError } from './client';
import {
  addSpan,
  changedSpans,
  DEFAULT_RENDER_OPTIONS,
  errorCode,
  formatSeconds,
  nudgeSpan,
  saveSpans,
  spanFromWords,
  startRender,
  toRenderBody,
  toUploadForm,
  wordInSpan,
  type Span,
  type Word,
} from './audio-edit';

const WORDS: Word[] = [
  { i: 0, word: 'the', start: 0.1, end: 0.3 },
  { i: 1, word: 'quick', start: 0.35, end: 0.7 },
  { i: 2, word: 'brown', start: 0.8, end: 1.2 },
  { i: 3, word: 'fox', start: 1.3, end: 1.6 },
];

beforeEach(() => client.apiJson.mockClear());

describe('spans', () => {
  it('builds a span from a word range in either click order', () => {
    const span = spanFromWords(WORDS, 2, 1, 'a');
    expect(span).toEqual({
      id: 'a', start_s: 0.35, end_s: 1.2, original_text: 'quick brown', replacement_text: 'quick brown',
    });
    expect(wordInSpan(WORDS[1], [span])).toBe(span);
    expect(wordInSpan(WORDS[3], [span])).toBeUndefined();
  });

  it('refuses overlapping spans and keeps them sorted', () => {
    const late = spanFromWords(WORDS, 3, 3, 'late');
    const early = spanFromWords(WORDS, 0, 0, 'early');
    expect(addSpan([late], early)?.map((s) => s.id)).toEqual(['early', 'late']);
    expect(addSpan([late], spanFromWords(WORDS, 2, 3, 'x'))).toBeNull();
  });

  it('nudges edges by 10 ms without crossing neighbours or collapsing', () => {
    const spans: Span[] = [spanFromWords(WORDS, 1, 1, 'a'), spanFromWords(WORDS, 2, 2, 'b')];
    expect(nudgeSpan(spans, 'a', 'end', 0.01, 4)[0].end_s).toBe(0.71);
    expect(nudgeSpan(spans, 'a', 'end', 0.5, 4)[0].end_s).toBe(0.8);
    expect(nudgeSpan(spans, 'a', 'start', 1, 4)[0].start_s).toBe(0.69);
    expect(nudgeSpan(spans, 'a', 'start', -1, 4)[0].start_s).toBe(0);
    expect(nudgeSpan(spans, 'missing', 'start', 0.01, 4)).toBe(spans);
  });

  it('saves only edited spans', async () => {
    const untouched = spanFromWords(WORDS, 0, 0, 'u');
    const edited = { ...spanFromWords(WORDS, 1, 1, 'e'), replacement_text: 'slow' };
    const deleted = { ...spanFromWords(WORDS, 3, 3, 'd'), replacement_text: '' };
    expect(changedSpans([untouched, edited, deleted]).map((s) => s.id)).toEqual(['e', 'd']);
    await saveSpans('abc', changedSpans([untouched, edited]));
    expect(client.apiJson).toHaveBeenCalledWith('/audio-edit/sessions/abc/spans', {
      method: 'PUT',
      body: JSON.stringify({ spans: [edited] }),
    });
  });
});

describe('requests', () => {
  it('maps render options to the API body', async () => {
    const options = { ...DEFAULT_RENDER_OPTIONS.song, rvcModelId: 'alto', rvcPitch: 2, seed: 9 };
    expect(toRenderBody(options)).toEqual({
      engine: 'ace_step', crossfade_ms: 30, rvc_model_id: 'alto', rvc_pitch: 2, vocal_gain_db: 0,
      seed: 9, ace_step: { strength: 0.7, padding_s: 2, caption: '' },
    });
    expect(DEFAULT_RENDER_OPTIONS.speech.engine).toBe('f5tts');
    await startRender('abc', options);
    expect(client.apiJson).toHaveBeenCalledWith('/audio-edit/sessions/abc/render', {
      method: 'POST',
      body: JSON.stringify(toRenderBody(options)),
    });
  });

  it('builds the upload form', () => {
    const form = toUploadForm(new File(['x'], 'take.wav'), 'song', ' Demo ');
    expect(form.get('mode')).toBe('song');
    expect(form.get('name')).toBe('Demo');
    expect((form.get('file') as File).name).toBe('take.wav');
  });

  it('reads stable error codes from API errors', () => {
    const payload = { detail: { code: 'licence_ack_required', message: 'x' } };
    const err = new ApiError(403, 'denied', payload as ConstructorParameters<typeof ApiError>[2]);
    expect(errorCode(err)).toBe('licence_ack_required');
    expect(errorCode(new Error('x'))).toBeNull();
    expect(formatSeconds(75.5)).toBe('1:15.50');
  });
});
