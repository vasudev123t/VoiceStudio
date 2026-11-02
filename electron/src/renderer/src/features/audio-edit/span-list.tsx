import { Trash2Icon } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { formatSeconds, MAX_REPLACEMENT_CHARS, NUDGE_S, nudgeSpan, type Span } from '@/lib/api/audio-edit';

interface Props {
  spans: Span[];
  durationS: number;
  onChange: (spans: Span[]) => void;
}

export function SpanList({ spans, durationS, onChange }: Props) {
  const { t } = useTranslation();
  if (spans.length === 0) return <p className="text-sm text-muted-foreground">{t('audioEdit.no_spans')}</p>;
  const nudge = (id: string, edge: 'start' | 'end', delta: number) =>
    onChange(nudgeSpan(spans, id, edge, delta, durationS));
  return (
    <ul className="space-y-2" data-testid="audio-edit-spans">
      {spans.map((span) => (
        <li key={span.id} className="grid gap-2 rounded-lg border p-3 sm:grid-cols-[1fr_2fr_auto]">
          <div className="text-sm">
            <p className="font-medium">{span.original_text}</p>
            <p className="text-xs tabular-nums text-muted-foreground">
              {formatSeconds(span.start_s)} – {formatSeconds(span.end_s)}
            </p>
          </div>
          <Input
            value={span.replacement_text}
            maxLength={MAX_REPLACEMENT_CHARS}
            placeholder={t('audioEdit.replacement_placeholder')}
            aria-label={t('audioEdit.replacement_for', { words: span.original_text })}
            onChange={(e) =>
              onChange(spans.map((s) => (s.id === span.id ? { ...s, replacement_text: e.target.value } : s)))
            }
          />
          <div className="flex flex-wrap items-center gap-2">
            {(['start', 'end'] as const).map((edge) => (
              <span key={edge} className="flex items-center gap-0.5 text-xs">
                <span>{t(`audioEdit.${edge}`)}</span>
                <Button size="icon-xs" variant="outline" aria-label={t('audioEdit.nudge_earlier', { edge })}
                  onClick={() => nudge(span.id, edge, -NUDGE_S)}>−</Button>
                <Button size="icon-xs" variant="outline" aria-label={t('audioEdit.nudge_later', { edge })}
                  onClick={() => nudge(span.id, edge, NUDGE_S)}>+</Button>
              </span>
            ))}
            <Button size="icon-sm" variant="ghost" aria-label={t('audioEdit.remove_span')}
              onClick={() => onChange(spans.filter((s) => s.id !== span.id))}>
              <Trash2Icon />
            </Button>
          </div>
        </li>
      ))}
    </ul>
  );
}
