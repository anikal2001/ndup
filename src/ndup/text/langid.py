"""Per-chunk language identification.

Uses fastText ``lid.176.ftz`` when the model file and the ``fasttext`` package
are both available. Otherwise a small offline heuristic covers the fixtures:
English, Spanish, French, German, Chinese, and Cyrillic text.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

_WORD = re.compile(r"[a-zA-Záéíóúüñàèìòùâêîôûäëïöüç']+")

_VOCAB: dict[str, set[str]] = {
    "en": {
        "the",
        "and",
        "of",
        "to",
        "in",
        "is",
        "that",
        "for",
        "it",
        "with",
        "as",
        "on",
        "be",
        "this",
        "are",
        "or",
        "from",
        "at",
        "by",
        "an",
        "was",
        "were",
        "not",
        "we",
        "our",
        "their",
        "page",
        "notes",
    },
    "es": {
        "el",
        "la",
        "de",
        "que",
        "y",
        "en",
        "los",
        "del",
        "se",
        "las",
        "un",
        "por",
        "con",
        "una",
        "su",
        "para",
        "es",
        "al",
        "lo",
        "como",
        "más",
        "pero",
        "sus",
        "le",
        "ya",
        "fue",
        "esta",
        "este",
    },
    "fr": {
        "le",
        "la",
        "les",
        "des",
        "une",
        "et",
        "est",
        "que",
        "dans",
        "pour",
        "qui",
        "sur",
        "pas",
        "plus",
        "avec",
        "par",
        "nous",
        "vous",
        "du",
    },
    "de": {
        "der",
        "die",
        "und",
        "den",
        "von",
        "zu",
        "das",
        "mit",
        "sich",
        "des",
        "auf",
        "für",
        "ist",
        "im",
        "dem",
        "nicht",
        "ein",
        "eine",
        "auch",
    },
}


def detect_language(text: str, model_path: Path | None = None) -> str:
    if model_path is not None and model_path.is_file():
        detected = _fasttext(text, model_path)
        if detected is not None:
            return detected
    return detect_language_heuristic(text)


def detect_language_heuristic(text: str) -> str:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return "und"
    cjk = sum(1 for char in letters if "\u4e00" <= char <= "\u9fff")
    if cjk / len(letters) >= 0.2:
        return "zh"
    cyr = sum(1 for char in letters if "\u0400" <= char <= "\u04ff")
    if cyr / len(letters) >= 0.2:
        return "ru"
    words = [match.group(0).lower() for match in _WORD.finditer(text)]
    if not words:
        return "und"
    scores = {lang: sum(1 for word in words if word in vocab) for lang, vocab in _VOCAB.items()}
    best = max(scores, key=lambda lang: (scores[lang], lang))
    if scores[best] >= 2:
        return best
    return "en"


def majority_language(weighted: list[tuple[str, int]]) -> str:
    """Token-weighted majority. Ties break toward English, then the code."""
    totals: Counter[str] = Counter()
    for lang, tokens in weighted:
        if tokens > 0 and lang and lang != "und":
            totals[lang] += tokens
    if not totals:
        return "und"
    ranked = sorted(totals.items(), key=lambda item: (-item[1], item[0] != "en", item[0]))
    return ranked[0][0]


def _fasttext(text: str, model_path: Path) -> str | None:
    try:
        import fasttext
    except ImportError:
        return None
    model = fasttext.load_model(str(model_path))
    labels, _probs = model.predict(text.replace("\n", " ")[:2000] or " ")
    if not labels:
        return None
    label = str(labels[0])
    if label.startswith("__label__"):
        return label.split("__label__", 1)[1]
    return label
