import { useTranslation } from 'react-i18next';
import { type Span, type Word, wordInSpan } from '@/lib/api/audio-edit';
import { cn } from '@/lib/utils';

interface Props {
  words: Word[];
  spans: Span[];
  anchor: number | null;
  onWord: (index: number) => void;
}

/** The transcript as clickable words; words inside a span are highlighted. */
export function TranscriptSpans({ words, spans, anchor, onWord }: Props) {
  const { t } = useTranslation();
  if (words.length === 0) return <p className="text-sm text-muted-foreground">{t('audioEdit.no_words')}</p>;
  return (
    <div className="flex flex-wrap gap-1 rounded-lg border p-3 leading-7" aria-label={t('audioEdit.transcript')}
      data-testid="audio-edit-transcript">
      {words.map((word) => (
        <button
          key={word.i}
          type="button"
          onClick={() => onWord(word.i)}
          aria-pressed={anchor === word.i}
          title={`${word.start.toFixed(2)}–${word.end.toFixed(2)} s`}
          className={cn(
            'rounded px-1 text-sm hover:bg-accent',
            wordInSpan(word, spans) && 'bg-primary/15 text-primary',
            anchor === word.i && 'ring-2 ring-primary',
          )}
        >
          {word.word}
        </button>
      ))}
    </div>
  );
}
