import { useQueryClient } from '@tanstack/react-query';
import { Link } from '@tanstack/react-router';
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import * as api from '@/lib/api/audio-edit';
import { CommercialBadge, LicenceGate } from './licence-gate';
import { RenderControls } from './render-controls';
import { SpanList } from './span-list';
import { TranscribeOptions } from './transcribe-options';
import { TranscriptSpans } from './transcript-spans';
import { audioEditKeys, useAsrEngines, useAudioEditEngines, useAudioEditSession } from './use-audio-edit-session';

const KNOWN_ERRORS = new Set([
  'invalid_audio', 'too_long', 'too_large', 'no_word_timestamps', 'invalid_spans', 'engine_not_installed',
  'licence_ack_required', 'ace_step_unavailable', 'separation_unavailable', 'engine_failed', 'busy',
  'asr_key_missing', 'asr_model_missing', 'asr_language_unsupported', 'asr_auth_failed', 'asr_failed',
]);

type T = ReturnType<typeof useTranslation>['t'];
function messageFor(err: unknown, t: T): string {
  const code = api.errorCode(err);
  if (code && KNOWN_ERRORS.has(code)) return t(`audioEdit.errors.${code}`);
  return err instanceof Error ? err.message : String(err);
}

interface Props {
  mode: api.AudioEditMode;
  sessionId: string | null;
  onSessionChange: (id: string | null) => void;
}

export function EditWorkspace({ mode, sessionId, onSessionChange }: Props) {
  const { t } = useTranslation();
  const qc = useQueryClient();
  const engines = useAudioEditEngines();
  const { session, job, setJobId } = useAudioEditSession(sessionId);
  const asrEngines = useAsrEngines();
  const [asr, setAsrState] = useState<api.TranscribeOptions>(api.loadTranscribeOptions);
  const setAsr = (next: api.TranscribeOptions) => {
    setAsrState(next);
    api.saveTranscribeOptions(next);
  };
  const asrBlocker = api.transcribeBlocker(asrEngines.data, asr);
  const [spans, setSpans] = useState<api.Span[]>([]);
  const [options, setOptions] = useState<api.RenderOptions>(api.DEFAULT_RENDER_OPTIONS[mode]);
  const [anchor, setAnchor] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fail = (err: unknown) => setError(messageFor(err, t));

  const transcriptText = session?.transcript?.text;
  useEffect(() => {
    setSpans(session?.spans ?? []);
    setAnchor(null);
    // Reset local spans only when another session or a new transcript arrives.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session?.id, transcriptText]);

  const jobKind = job?.kind;
  const jobStatus = job?.status;
  const stage = session?.stage;
  useEffect(() => {
    // Songs: transcribe the separated vocals as soon as separation succeeds.
    if (!session || jobKind !== 'separate' || jobStatus !== 'succeeded' || stage !== 'separated') return;
    api.transcribeSession(session.id, asr).then((r) => setJobId(r.job.id), fail);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobKind, jobStatus, stage]);

  const words = session?.transcript?.words ?? [];
  const edited = api.changedSpans(spans);
  const live = job !== null && api.isLive(job.status);
  const engine = engines.data?.engines.find((e) => e.id === options.engine) ?? null;
  const regenerates = edited.some((s) => s.replacement_text.trim() !== '');
  const needsAck = !!engine && engine.needs_ack && !engine.acked_at && regenerates;
  const latest = session?.renders[0] ?? null;

  const upload = async (file: File) => {
    setError(null);
    setBusy(true);
    try {
      const { session: created, job: first } = await api.createSession(file, mode);
      onSessionChange(created.id);
      setJobId(first ? first.id : (await api.transcribeSession(created.id, asr)).job.id);
      void qc.invalidateQueries({ queryKey: audioEditKeys.sessions });
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  const updateSpans = (next: api.Span[]) => {
    setSpans(next);
    if (!session) return;
    const id = session.id;
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => {
      api.saveSpans(id, api.changedSpans(next)).catch(fail);
    }, 400);
  };

  const onWord = (index: number) => {
    if (anchor === null) {
      setAnchor(index);
      return;
    }
    const next = api.addSpan(spans, api.spanFromWords(words, anchor, index));
    setAnchor(null);
    if (!next) {
      setError(t('audioEdit.overlap'));
      return;
    }
    setError(null);
    updateSpans(next);
  };

  const retry = async () => {
    if (!session) return;
    setError(null);
    try {
      const r = session.mode === 'song' && session.stage === 'uploaded'
        ? await api.separateSession(session.id)
        : await api.transcribeSession(session.id, asr);
      setJobId(r.job.id);
    } catch (err) {
      fail(err);
    }
  };

  const retranscribe = async () => {
    if (!session) return;
    setError(null);
    try {
      setJobId((await api.transcribeSession(session.id, asr)).job.id);
    } catch (err) {
      fail(err);
    }
  };

  const startRender = async () => {
    if (!session) return;
    setError(null);
    setBusy(true);
    try {
      if (saveTimer.current) clearTimeout(saveTimer.current);
      await api.saveSpans(session.id, edited);
      const { job: started } = await api.startRender(session.id, options);
      setJobId(started.id);
    } catch (err) {
      fail(err);
      if (api.errorCode(err) === 'licence_ack_required') void qc.invalidateQueries({ queryKey: audioEditKeys.engines });
    } finally {
      setBusy(false);
    }
  };

  const revoke = async () => {
    if (!engine) return;
    try {
      await api.revokeLicence(engine.id);
      await qc.invalidateQueries({ queryKey: audioEditKeys.engines });
    } catch (err) {
      fail(err);
    }
  };

  if (!session) {
    return (
      <div className="space-y-3" data-testid={`audio-edit-${mode}`}>
        {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
        <div
          className="rounded-lg border border-dashed p-8 text-center"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            const file = e.dataTransfer.files[0];
            if (file && !asrBlocker) void upload(file);
          }}
        >
          <TranscribeOptions overview={asrEngines.data} value={asr} onChange={setAsr} disabled={busy} />
          <p className="mt-3 text-sm text-muted-foreground">{t(`audioEdit.upload_hint_${mode}`)}</p>
          <input ref={fileRef} type="file" accept={api.UPLOAD_ACCEPT} className="hidden" data-testid="audio-edit-file"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file && !asrBlocker) void upload(file);
              e.target.value = '';
            }} />
          <Button className="mt-3" disabled={busy || !!asrBlocker} onClick={() => fileRef.current?.click()}>
            {t('audioEdit.choose_file')}
          </Button>
          {mode === 'song' && engines.data && !engines.data.separation.available && (
            <p className="mt-2 text-xs text-amber-600">
              {t('audioEdit.separation_missing', { missing: engines.data.separation.missing ?? 'HTDemucs' })}
            </p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-5" data-testid={`audio-edit-${mode}`}>
      <div className="flex flex-wrap items-center gap-3">
        <h3 className="font-medium">{session.name}</h3>
        <span className="text-xs tabular-nums text-muted-foreground">{api.formatSeconds(session.duration_s)}</span>
        <span className="text-xs text-muted-foreground" data-testid="audio-edit-stage">
          {t(`audioEdit.stage.${session.stage}`)}
        </span>
        <span className="flex-1" />
        <Button variant="outline" size="sm" onClick={() => onSessionChange(null)}>{t('audioEdit.new_session')}</Button>
      </div>
      {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
      {session.error && !live && (
        <div role="alert" className="flex items-center gap-2 text-sm text-destructive">
          <span>{t('audioEdit.stage_failed', { error: session.error })}</span>
          <Button size="xs" variant="outline" onClick={() => void retry()}>{t('audioEdit.retry')}</Button>
        </div>
      )}
      {live && job && (
        <div className="flex items-center gap-3 text-sm" data-testid="audio-edit-job">
          <span>{t(`audioEdit.job.${job.kind}`)}</span>
          <progress max={1} value={job.progress} className="h-2 flex-1" />
          <span className="tabular-nums">{Math.round(job.progress * 100)}%</span>
          <Button size="sm" variant="outline" onClick={() => api.cancelJob(job.id).catch(fail)}>
            {t('audioEdit.cancel')}
          </Button>
        </div>
      )}
      {!live && (session.stage === 'ready' || session.stage === 'uploaded' || session.stage === 'separated') && (
        <details className="rounded-lg border p-3 text-sm" data-testid="audio-edit-retranscribe">
          <summary className="cursor-pointer text-xs text-muted-foreground">
            {t('audioEdit.asr.retranscribe_summary', {
              engine: asrEngines.data?.engines.find((e) => e.id === (session.asr?.engine ?? 'whisper'))?.label
                ?? session.asr?.engine ?? 'Whisper',
              language: session.transcript?.language ?? session.asr?.language ?? 'auto',
            })}
          </summary>
          <div className="mt-2 space-y-2">
            <TranscribeOptions overview={asrEngines.data} value={asr} onChange={setAsr} disabled={busy} />
            <Button size="sm" variant="outline" disabled={busy || !!asrBlocker || (mode === 'song' && session.stage === 'uploaded')}
              onClick={() => void retranscribe()}>
              {t('audioEdit.asr.retranscribe')}
            </Button>
            {session.stage === 'ready' && session.spans.length > 0 && (
              <p className="text-xs text-muted-foreground">{t('audioEdit.asr.retranscribe_clears')}</p>
            )}
          </div>
        </details>
      )}
      {mode === 'song' && (session.stage === 'separated' || session.stage === 'transcribing' || session.stage === 'ready') && (
        <div className="grid gap-2 sm:grid-cols-2">
          <Player label={t('audioEdit.stem_vocals')} src={api.sourceAudioUrl(session.id, 'vocals')} />
          <Player label={t('audioEdit.stem_instrumental')} src={api.sourceAudioUrl(session.id, 'instrumental')} />
        </div>
      )}
      {session.stage === 'ready' && session.transcript && (
        <>
          <p className="text-xs text-muted-foreground">{t('audioEdit.select_hint')} {t('audioEdit.insert_hint')}</p>
          <TranscriptSpans words={words} spans={spans} anchor={anchor} onWord={onWord} />
          <SpanList spans={spans} durationS={session.duration_s} onChange={updateSpans} />
          <RenderControls mode={mode} options={options} onChange={setOptions} engines={engines.data?.engines ?? []} />
          {engine && !engine.available && regenerates ? (
            <EngineMissing engine={engine} />
          ) : needsAck && engine ? (
            <LicenceGate engine={engine} />
          ) : (
            <div className="flex items-center gap-2">
              <Button disabled={edited.length === 0 || live || busy} onClick={() => void startRender()}>
                {t('audioEdit.render')}
              </Button>
              {engine?.needs_ack && engine.acked_at && (
                <Button size="xs" variant="ghost" onClick={() => void revoke()}>{t('audioEdit.licence_revoke')}</Button>
              )}
            </div>
          )}
        </>
      )}
      {latest && <RenderResult sessionId={session.id} render={latest} />}
    </div>
  );
}

function Player({ label, src }: { label: string; src: string }) {
  return (
    <label className="space-y-1 text-xs text-muted-foreground">
      <span>{label}</span>
      <audio controls preload="none" src={src} className="w-full" />
    </label>
  );
}

function EngineMissing({ engine }: { engine: api.EngineStatus }) {
  const { t } = useTranslation();
  const [state, setState] = useState<'idle' | 'started' | 'failed'>('idle');
  return (
    <div className="space-y-2 rounded-lg border p-4 text-sm" data-testid="audio-edit-engine-missing">
      <p>
        {engine.installed
          ? t('audioEdit.engine_needs_gpu', { engine: engine.display_name })
          : t('audioEdit.engine_missing', { engine: engine.display_name })}
      </p>
      {!engine.installed && (
        <div className="flex items-center gap-3">
          <Button size="sm" disabled={state === 'started'}
            onClick={() => api.installEngine(engine.install_id).then(() => setState('started'), () => setState('failed'))}>
            {t('audioEdit.install')}
          </Button>
          <Link to="/settings/models" className="text-sm underline">{t('audioEdit.open_models')}</Link>
        </div>
      )}
      {state === 'started' && <p className="text-xs text-muted-foreground">{t('audioEdit.install_started')}</p>}
      {state === 'failed' && <p className="text-xs text-destructive">{t('audioEdit.install_failed')}</p>}
    </div>
  );
}

function RenderResult({ sessionId, render }: { sessionId: string; render: api.Render }) {
  const { t } = useTranslation();
  const done = render.status === 'succeeded';
  return (
    <div className="space-y-3 rounded-lg border p-4" data-testid="audio-edit-result">
      <div className="flex items-center gap-2 text-sm">
        <span>{t(`audioEdit.render_status.${render.status}`)}</span>
        <CommercialBadge commercial={render.commercial_use} />
      </div>
      {render.status === 'failed' && <p className="text-sm text-destructive">{render.message ?? render.error}</p>}
      {done && (
        <>
          <div className="grid gap-2 sm:grid-cols-2">
            <Player label={t('audioEdit.original')} src={api.sourceAudioUrl(sessionId, 'source')} />
            <Player label={t('audioEdit.edited')} src={api.renderAudioUrl(sessionId, render.id)} />
          </div>
          <a className="text-sm underline" href={api.renderAudioUrl(sessionId, render.id)} download>
            {t('audioEdit.export')}
          </a>
        </>
      )}
    </div>
  );
}
