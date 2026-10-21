"""Kannada (kn) number verbalization — a native backend for text_normalization.

``num2words`` ships a ``kn`` locale, but it is not usable for speech: it emits
the bare virama stem before units (21 → "ಇಪ್ಪತ್ತ್ ಒಂದು" instead of
ಇಪ್ಪತ್ತೊಂದು) and doubles the multiplier for lakhs (100000 → "ಒಂದು ಒಂದು
ಲಕ್ಷ"). So ``kn`` deliberately stays out of ``_NUM2WORDS_LANGS`` and is
routed here instead, the same way Malayalam is.

Rules (standard spoken/written Kannada):

* tens + unit is one word with vowel sandhi on the ``-ತ್ತು`` stem:
  ಇಪ್ಪತ್ತು + ಒಂದು → ಇಪ್ಪತ್ತೊಂದು, + ಐದು → ಇಪ್ಪತ್ತೈದು, + ಆರು → ಇಪ್ಪತ್ತಾರು;
  before a consonant the stem keeps its inherent vowel: ಇಪ್ಪತ್ತಮೂರು,
  ಇಪ್ಪತ್ತನಾಲ್ಕು;
* hundreds with a remainder take the ``-ನೂರ`` form as a separate word:
  150 → ನೂರ ಐವತ್ತು, 250 → ಇನ್ನೂರ ಐವತ್ತು;
* thousands and lakhs with a remainder take the genitive linking form:
  2026 → ಎರಡು ಸಾವಿರದ ಇಪ್ಪತ್ತಾರು, 150000 → ಒಂದು ಲಕ್ಷದ ಐವತ್ತು ಸಾವಿರ;
  Indian grouping (10^5 = ಲಕ್ಷ, 10^7 = ಕೋಟಿ);
* decimals read "<int> ದಶಮಾಂಶ <digit> <digit>…"; percentages read
  "ಶೇಕಡಾ <number>" (the percent word precedes the number in Kannada).

Years are plain cardinals (no "nineteen eighty-four" form). Pure Python,
output contains no digits, ZWJ or ZWNJ.
"""

from __future__ import annotations

_UNITS = ["ಸೊನ್ನೆ", "ಒಂದು", "ಎರಡು", "ಮೂರು", "ನಾಲ್ಕು", "ಐದು", "ಆರು", "ಏಳು", "ಎಂಟು", "ಒಂಬತ್ತು"]
_TEENS = [
    "ಹತ್ತು", "ಹನ್ನೊಂದು", "ಹನ್ನೆರಡು", "ಹದಿಮೂರು", "ಹದಿನಾಲ್ಕು", "ಹದಿನೈದು",
    "ಹದಿನಾರು", "ಹದಿನೇಳು", "ಹದಿನೆಂಟು", "ಹತ್ತೊಂಬತ್ತು",
]
_TENS = [None, None, "ಇಪ್ಪತ್ತು", "ಮೂವತ್ತು", "ನಲವತ್ತು", "ಐವತ್ತು", "ಅರವತ್ತು", "ಎಪ್ಪತ್ತು",
         "ಎಂಬತ್ತು", "ತೊಂಬತ್ತು"]
_HUNDREDS = [None, "ನೂರು", "ಇನ್ನೂರು", "ಮುನ್ನೂರು", "ನಾನೂರು", "ಐನೂರು", "ಆರುನೂರು",
             "ಏಳುನೂರು", "ಎಂಟುನೂರು", "ಒಂಬೈನೂರು"]

THOUSAND, THOUSAND_LINK = "ಸಾವಿರ", "ಸಾವಿರದ"
LAKH, LAKH_LINK = "ಲಕ್ಷ", "ಲಕ್ಷದ"
CRORE = "ಕೋಟಿ"
DECIMAL_WORD = "ದಶಮಾಂಶ"
PERCENT_WORD = "ಶೇಕಡಾ"

_U = "\u0cc1"  # ು
# Independent vowel → dependent vowel sign ("" = inherent a).
_VOWEL_SIGN = {
    "ಅ": "", "ಆ": "\u0cbe", "ಇ": "\u0cbf", "ಈ": "\u0cc0", "ಉ": "\u0cc1", "ಊ": "\u0cc2",
    "ಎ": "\u0cc6", "ಏ": "\u0cc7", "ಐ": "\u0cc8", "ಒ": "\u0cca", "ಓ": "\u0ccb", "ಔ": "\u0ccc",
}


def _join_tens(tens_word: str, unit_word: str) -> str:
    """ಇಪ್ಪತ್ತು + ಒಂದು → ಇಪ್ಪತ್ತೊಂದು; ಇಪ್ಪತ್ತು + ಮೂರು → ಇಪ್ಪತ್ತಮೂರು."""
    stem = tens_word[:-1] if tens_word.endswith(_U) else tens_word  # ಇಪ್ಪತ್ತ
    first = unit_word[0]
    if first in _VOWEL_SIGN:
        return stem + _VOWEL_SIGN[first] + unit_word[1:]
    return stem + unit_word


def _below_hundred(n: int) -> str:
    if n < 10:
        return _UNITS[n]
    if n < 20:
        return _TEENS[n - 10]
    tens, unit = divmod(n, 10)
    if unit == 0:
        return _TENS[tens]
    return _join_tens(_TENS[tens], _UNITS[unit])


def _below_thousand(n: int) -> str:
    if n < 100:
        return _below_hundred(n)
    hundreds, rest = divmod(n, 100)
    word = _HUNDREDS[hundreds]
    if rest == 0:
        return word
    return f"{word[:-1]} {_below_hundred(rest)}"  # ನೂರು → ನೂರ


def _below_crore(n: int) -> str:
    """0 ≤ n < 10^7 with Indian grouping and linking forms."""
    if n < 1000:
        return _below_thousand(n)
    lakhs, rest = divmod(n, 100_000)
    thousands, units = divmod(rest, 1000)
    parts: list[str] = []
    if lakhs:
        parts += [_below_hundred(lakhs), LAKH_LINK if rest else LAKH]
    if thousands:
        parts += [_below_hundred(thousands), THOUSAND_LINK if units else THOUSAND]
    if units:
        parts.append(_below_thousand(units))
    return " ".join(parts)


def cardinal(n: int) -> str:
    """Render an integer 0 ≤ n < 10^12 as Kannada words."""
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("cardinal() expects an int")
    if n < 0 or n >= 10**12:
        raise ValueError("cardinal() supports 0..999999999999")
    if n < 10**7:
        return _below_crore(n)
    crores, rest = divmod(n, 10**7)
    head = f"{cardinal(crores)} {CRORE}"
    return f"{head} {_below_crore(rest)}" if rest else head


def decimal(int_part: int, frac_digits: str) -> str:
    """3.14 → ಮೂರು ದಶಮಾಂಶ ಒಂದು ನಾಲ್ಕು (fraction digits read one by one)."""
    digits = " ".join(_UNITS[int(d)] for d in frac_digits)
    return f"{cardinal(int_part)} {DECIMAL_WORD} {digits}"


def percent(raw: str) -> str:
    """'50' → ಶೇಕಡಾ ಐವತ್ತು; '2.5' → ಶೇಕಡಾ ಎರಡು ದಶಮಾಂಶ ಐದು."""
    if "." in raw:
        a, b = raw.split(".", 1)
        body = decimal(int(a), b)
    else:
        body = cardinal(int(raw))
    return f"{PERCENT_WORD} {body}"
