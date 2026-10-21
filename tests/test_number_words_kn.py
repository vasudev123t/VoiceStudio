"""Kannada number verbalization (review item 3)."""
import re

import pytest

from services import number_words_kn as kn
from services.text_normalization import _native_lang, _num2words_lang, normalize_text

CASES = {
    0: "ಸೊನ್ನೆ", 7: "ಏಳು", 10: "ಹತ್ತು", 11: "ಹನ್ನೊಂದು", 19: "ಹತ್ತೊಂಬತ್ತು",
    20: "ಇಪ್ಪತ್ತು", 21: "ಇಪ್ಪತ್ತೊಂದು", 22: "ಇಪ್ಪತ್ತೆರಡು", 23: "ಇಪ್ಪತ್ತಮೂರು",
    25: "ಇಪ್ಪತ್ತೈದು", 26: "ಇಪ್ಪತ್ತಾರು", 27: "ಇಪ್ಪತ್ತೇಳು", 28: "ಇಪ್ಪತ್ತೆಂಟು",
    31: "ಮೂವತ್ತೊಂದು", 45: "ನಲವತ್ತೈದು", 50: "ಐವತ್ತು", 67: "ಅರವತ್ತೇಳು",
    99: "ತೊಂಬತ್ತೊಂಬತ್ತು", 100: "ನೂರು", 101: "ನೂರ ಒಂದು", 150: "ನೂರ ಐವತ್ತು",
    250: "ಇನ್ನೂರ ಐವತ್ತು", 500: "ಐನೂರು", 900: "ಒಂಬೈನೂರು",
    1000: "ಒಂದು ಸಾವಿರ", 1500: "ಒಂದು ಸಾವಿರದ ಐನೂರು",
    2026: "ಎರಡು ಸಾವಿರದ ಇಪ್ಪತ್ತಾರು", 25000: "ಇಪ್ಪತ್ತೈದು ಸಾವಿರ",
    100000: "ಒಂದು ಲಕ್ಷ", 150000: "ಒಂದು ಲಕ್ಷದ ಐವತ್ತು ಸಾವಿರ",
    205000: "ಎರಡು ಲಕ್ಷದ ಐದು ಸಾವಿರ",
    1234567: "ಹನ್ನೆರಡು ಲಕ್ಷದ ಮೂವತ್ತನಾಲ್ಕು ಸಾವಿರದ ಐನೂರ ಅರವತ್ತೇಳು",
    10000000: "ಒಂದು ಕೋಟಿ", 12500000: "ಒಂದು ಕೋಟಿ ಇಪ್ಪತ್ತೈದು ಲಕ್ಷ",
}


@pytest.mark.parametrize("n,words", CASES.items())
def test_cardinals(n, words):
    assert kn.cardinal(n) == words


def test_fixes_the_num2words_defects():
    # The two concrete num2words kn bugs that keep kn out of _NUM2WORDS_LANGS.
    assert kn.cardinal(21) != "ಇಪ್ಪತ್ತ್ ಒಂದು"
    assert kn.cardinal(100000).count("ಒಂದು") == 1


def test_every_number_renders_clean():
    bad = re.compile(r"[0-9\u0ce6-\u0cef\u200c\u200d]|\u0ccd\s|\u0ccd$")
    for n in list(range(0, 2100)) + list(range(2100, 1_000_000, 997)):
        word = kn.cardinal(n)
        assert word and not bad.search(word), (n, word)


def test_decimal_and_percent():
    assert kn.decimal(3, "14") == "ಮೂರು ದಶಮಾಂಶ ಒಂದು ನಾಲ್ಕು"
    assert kn.percent("50") == "ಶೇಕಡಾ ಐವತ್ತು"
    assert kn.percent("2.5") == "ಶೇಕಡಾ ಎರಡು ದಶಮಾಂಶ ಐದು"


def test_rejects_bad_input():
    with pytest.raises(TypeError):
        kn.cardinal(True)
    with pytest.raises(ValueError):
        kn.cardinal(-1)


def test_routing_uses_native_not_num2words():
    assert _num2words_lang("kn") is None
    assert _num2words_lang("Kannada") is None
    assert _native_lang("Kannada") == "kn"
    assert _native_lang("kn-IN") == "kn"


@pytest.mark.parametrize("lang,text,expected", [
    ("Kannada", "ನನಗೆ 2 ಬೆಕ್ಕುಗಳಿವೆ", "ನನಗೆ ಎರಡು ಬೆಕ್ಕುಗಳಿವೆ"),
    ("kn", "ಬೆಲೆ 25000 ರೂಪಾಯಿ", "ಬೆಲೆ ಇಪ್ಪತ್ತೈದು ಸಾವಿರ ರೂಪಾಯಿ"),
    ("kn-IN", "2026 ವರ್ಷ", "ಎರಡು ಸಾವಿರದ ಇಪ್ಪತ್ತಾರು ವರ್ಷ"),
    ("kn", "50% ಮತಗಳು", "ಶೇಕಡಾ ಐವತ್ತು ಮತಗಳು"),
    ("kn", "3.5 ಕಿಲೋ", "ಮೂರು ದಶಮಾಂಶ ಐದು ಕಿಲೋ"),
    ("kn", "೨೧ ಜನ", "ಇಪ್ಪತ್ತೊಂದು ಜನ"),  # native Kannada digits
    ("kn", "10ನೇ ತರಗತಿ", "10ನೇ ತರಗತಿ"),  # glued suffix: conservative, unchanged
])
def test_normalize_text_kannada(lang, text, expected):
    assert normalize_text(text, lang) == expected
