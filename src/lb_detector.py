"""
Dehyphenation for historical Latin OCR using the Flair line-break detector.

Model: mschonhardt/latin-contextual-lb-detector
  Input:  Latin text with <lb/> as whitespace-separated tokens at line breaks
  Output: WB (word boundary → keep space) or NB (no boundary → merge word)

Reference notebook:
  https://github.com/michaelscho/ml-notebooks/blob/main/latin_lb_detector.ipynb
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path

log = logging.getLogger(__name__)

_SOFT_HYPHEN = "¬"  # ¬  — OCR representation of a print line-hyphen
_LB = "<lb/>"

_tagger = None  # lazy-loaded singleton
_freq = None    # corpus word frequencies (src/codebooks/word_freq.json, from the edition text), for the guards on merges without ¬
_PUNCT_END = re.compile(r"[.,:;?!]$")


def _word_key(w: str) -> str:
    w = unicodedata.normalize("NFD", w.lower())
    w = "".join(c for c in w if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", w.replace("v", "u").replace("j", "i"))


def _get_freq() -> dict:
    global _freq
    if _freq is None:
        f = Path(__file__).resolve().parent / "codebooks" / "word_freq.json"
        _freq = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    return _freq


def _merge_without_hyphen(left: str, right: str) -> bool:
    """guards for a merge the model proposes where the line has no ¬: never after punctuation; the joined
    word must occur at least three times in the corpus, or the second piece must not be a word itself"""
    a, b = (left.split() or [""])[-1], (right.split() or [""])[0]
    if not a or not b or _PUNCT_END.search(a):
        return False
    freq = _get_freq()
    if not freq:
        return True
    return freq.get(_word_key(a + b), 0) >= 3 or freq.get(_word_key(b), 0) < 2


def _get_tagger():
    global _tagger
    if _tagger is None:
        from flair.models import SequenceTagger  # type: ignore
        log.info("Loading Flair line-break model (first call, may take a moment)…")
        _tagger = SequenceTagger.load("mschonhardt/latin-contextual-lb-detector")
    return _tagger


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def join_lines(lines: list[str], use_model: bool = True) -> str:
    """
    Join a list of raw OCR lines into a single dehyphenated string.

    Strips trailing ¬ from each line before joining.  When use_model=True,
    the Flair model decides whether each line break is a word-internal split
    (NB → merge) or a word boundary (WB → space). The model also rejects a ¬ printed after a whole word.
    Where a line has no ¬, a merge the model proposes is kept only if the line does not end in
    punctuation and the joined word is a corpus word or the second piece is not a word on its own
    (the model alone merged "ibi, | dum" and "et | videtur").  Falls back to a heuristic
    (¬ present → merge, otherwise space) if the model is unavailable.
    """
    if not lines:
        return ""
    cleaned = [line.rstrip().rstrip(_SOFT_HYPHEN).rstrip() for line in lines]
    if len(cleaned) == 1:
        return cleaned[0]

    if use_model:
        try:
            return _join_with_model(cleaned, lines)
        except Exception as exc:
            log.warning("LB model failed (%s); using heuristic fallback.", exc)

    return _join_heuristic(cleaned, lines)


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


def _join_heuristic(cleaned: list[str], raw: list[str]) -> str:
    """Merge where original line ended with ¬; otherwise space-join."""
    result = cleaned[0]
    for i in range(1, len(cleaned)):
        if raw[i - 1].rstrip().endswith(_SOFT_HYPHEN):
            result += cleaned[i]
        else:
            result += " " + cleaned[i]
    return result


def _join_with_model(cleaned: list[str], raw: list[str]) -> str:
    """Use the Flair model to classify each line break.

    Builds one sentence per break: "line_i <lb/> line_i+1".
    All sentences are predicted in a single batched call so the model
    sees exactly the two-line context it was trained on.
    """
    from flair.data import Sentence  # type: ignore

    tagger = _get_tagger()

    # One sentence per adjacent line pair
    sentences = [
        Sentence(f"{cleaned[i]} {_LB} {cleaned[i + 1]}", use_tokenizer=False)
        for i in range(len(cleaned) - 1)
    ]
    tagger.predict(sentences)  # batched — one forward pass per mini-batch

    result = cleaned[0]
    for i, sent in enumerate(sentences):
        pred = "NB"  # safe default: insert space
        for token in sent:
            if token.text == _LB:
                label = token.get_label()
                pred = label.value if label else "WB"
                break
        hyphen = raw[i].rstrip().endswith(_SOFT_HYPHEN)
        if pred == "WB" and not hyphen and not _merge_without_hyphen(cleaned[i], cleaned[i + 1]):
            pred = "NB"
        if pred == "WB":
            result += cleaned[i + 1]   # no boundary: merge (hyphenated word)
        else:
            result += " " + cleaned[i + 1]  # word boundary: keep space
    return result
