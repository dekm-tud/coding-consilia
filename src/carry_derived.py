"""
Carry derived layers (LLM translations, keywords) across a re-segmentation.

`build.py` / `pipeline.py` regenerate every consilium from PageXML, which contains
only the transcription. Translations and keywords were merged in later (merge_llm_output,
apply_keyword_*). Without this step a rebuild would silently discard them.

For the casus correction (casus.py) the translations of the two neighbours change:
  * owner (lost the trailing casus): body translations are cut at the signature,
  * receiver (gained the casus): `translations.{lang}.casus` holds the casus translation.

Translations are aligned with the Latin body by section index. The section that contains
the cut is split at the sentence boundary after the translated signature, chosen by
proportion. If that cannot be done unambiguously the piece is left EMPTY (never guessed)
and listed in output/debug/casus_retranslate.json for a GPU re-translation.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

DERIVED_KEYS = ("translations", "keywords_raw", "keywords", "keywords_final", "keywords_source")
LANGS = ("en", "de")

# sentence boundary: . : ; ? ! followed by whitespace
_RE_BOUNDARY = re.compile(r"(?<=[.:;?!])\s+")
# Words of a translated closing-signature sentence. Deliberately NOT "iuris"/"Rechte"
# (they occur in citations such as "D. de regulis iuris"); "utriusque iuris doctor" is
# caught by "doctor"/"Doktor".
_RE_SIGNATURE_SENTENCE = re.compile(
    r"Bald|Bal\.|Bartol|subscri|sealed|seal of|signed|unterschr|unterzeichn|Siegel|besiegel|"
    r"doctor|Doktor|"
    # the LLM sometimes expands "Ego Bal." to another name ("I, Bartolus"): accept any "I, Name"
    r"(?-i:\bI,\s+(?!Paragraph|Part|Section|Book|Chapter|Title)[A-Z]|\bIch,\s+(?!Teil|Paragraph|Buch|Kapitel|Titel)[A-Z])",
    re.IGNORECASE,
)
# the name-bearing part of the signature; when several sentences qualify the LAST one that
# lies close after the nearest candidate is the real closing signature
_RE_SIGNATURE_NAME = re.compile(
    r"Bald|Bal\.|Bartol|"
    r"(?-i:\bI,\s+(?!Paragraph|Part|Section|Book|Chapter|Title)[A-Z]|\bIch,\s+(?!Teil|Paragraph|Buch|Kapitel|Titel)[A-Z])",
    re.IGNORECASE,
)
ADVANCE_CHARS = 160
# short sentences that merely finish the signature ("de Perusio, iu. utr.", "usw.", "and colleagues.")
_RE_SIGNATURE_TAIL = re.compile(
    r"^(?:[\w.,*\s]{0,60})(?:Peru|Per\.|doc|Doct|iu\.|utr|etc|usw|und c|and c|colleagues|Kollegen|legum|"
    r"V\.\s?[Il1]\.\s?D|laws|Rechte|Recht)\S*\s*$",
    re.IGNORECASE,
)
# the label the print sets in front of a casus, as the LLM renders it (CASE / FALL / OASV …, upper case)
_RE_CASUS_LABEL = re.compile(
    r"(?<![A-Za-z*])(?:CASE|CASUS|CASVS|CASU|FALL|FACTUM|C\s?A\s?S\s?V?\s?S?|C\sA(?=\s)|OASV\w*|A\s?S\s?V\s?S)(?![A-Za-z])"
)
PROPORTION_TOLERANCE = 0.15


def split_translated_section(text: str, p_expected: float) -> tuple[str, str] | None:
    """Split one translated section into (owner part, casus part).

    1. If the translation carries the casus label (upper case CASE / FALL / …), the casus
       starts at the label nearest to the Latin proportion `p_expected`.
    2. Otherwise the cut is the sentence boundary after the closing-signature sentence
       (plus the short sentences that only finish the signature) nearest to `p_expected`.
    The cut must lie within PROPORTION_TOLERANCE of `p_expected`; otherwise None
    (the caller leaves the piece empty and lists it for re-translation)."""
    if not text or not text.strip():
        return None
    total = len(text)

    labels = [(abs(m.start() / total - p_expected), m.start()) for m in _RE_CASUS_LABEL.finditer(text)
              if 0 < m.start() < total]
    labels = [c for c in labels if c[0] <= PROPORTION_TOLERANCE]
    if labels:
        cut = min(labels)[1]
        return text[:cut].rstrip(), text[cut:].lstrip()

    starts = [0] + [m.end() for m in _RE_BOUNDARY.finditer(text)] + [total]
    sentences = [(starts[i], starts[i + 1]) for i in range(len(starts) - 1)]
    cands: list[tuple[float, int]] = []
    for k, (a, b) in enumerate(sentences):
        if b >= total or not _RE_SIGNATURE_SENTENCE.search(text[a:b]):
            continue
        cut = b
        # absorb the short sentences that finish the signature
        j = k + 1
        while j < len(sentences) and sentences[j][1] < total and _RE_SIGNATURE_TAIL.match(text[sentences[j][0]:sentences[j][1]]):
            cut = sentences[j][1]
            j += 1
        cands.append((abs(cut / total - p_expected), cut, k))
    if not cands:
        return None
    diff, cut, k = min(cands)
    if diff > PROPORTION_TOLERANCE:
        return None
    # "... Baldus and Franciscus advised. I, Baldus." : advance to the closing signature
    later = [c for c in cands if c[1] > cut and c[1] - cut <= ADVANCE_CHARS
             and _RE_SIGNATURE_NAME.search(text[sentences[c[2]][0]:sentences[c[2]][1]])]
    if later:
        cut = max(later, key=lambda c: c[1])[1]
    return text[:cut].rstrip(), text[cut:].lstrip()


def _split_lang(tr_body: list[str], move: dict) -> tuple[list[str], list[str], bool]:
    """-> (owner_body_tr, casus_tr, needs_retranslation) for one language."""
    n_old, n_own = move["old_sections"], move["owner_sections"]
    if len(tr_body) != n_old:
        # alignment cannot be trusted: keep what precedes the split section, blank the rest
        keep = tr_body[: max(n_own - 1, 0)] if move["mid_section"] else tr_body[:n_own]
        pad = [""] if move["mid_section"] else []
        return keep + pad, [""] * max(move["casus_sections"], 1), True
    if not move["mid_section"]:
        return tr_body[:n_own], tr_body[n_own:], False
    si = n_own - 1
    p = move["mixed_owner_chars"] / max(move["mixed_owner_chars"] + move["mixed_casus_chars"], 1)
    parts = split_translated_section(tr_body[si], p)
    if parts is None:
        return tr_body[:si] + [""], [""] + tr_body[si + 1:], True
    return tr_body[:si] + [parts[0]], [parts[1]] + tr_body[si + 1:], False


def carry_over(new: dict, old: dict, moves: list[dict]) -> dict:
    """Copy derived fields from `old` into `new` (both dicts keyed by consilium id) and
    apply the translation split for `moves`. Returns a report:
      {"retranslate": [...], "stale_keywords": [...], "carried": n, "missing": n}
    """
    report = {"retranslate": [], "stale_keywords": [], "carried": 0, "missing": 0, "misaligned": []}
    for cid, c in new.items():
        o = old.get(cid)
        if not o:
            report["missing"] += 1
            continue
        for k in DERIVED_KEYS:
            if k in o and not c.get(k):      # a prebuilt print already carries its translations
                c[k] = o[k]
        report["carried"] += 1

    for mv in moves:
        owner, recv = new.get(mv["owner_id"]), new.get(mv["receiver_id"])
        old_recv = old.get(mv["receiver_id"]) or {}
        if not owner or not recv:
            continue
        report["stale_keywords"] += [mv["owner_id"], mv["receiver_id"]]
        if old_recv.get("casus"):
            continue  # `old` was already corrected: translations are split, nothing to do
        old_owner_tr = (old.get(mv["owner_id"]) or {}).get("translations")
        if not old_owner_tr:
            continue  # no translations yet: nothing to split
        for lang in LANGS:
            tr = old_owner_tr.get(lang) or {}
            body = list(tr.get("body") or [])
            owner_body, casus_tr, retr = _split_lang(body, mv)
            owner["translations"][lang]["body"] = owner_body
            recv.setdefault("translations", {}).setdefault(lang, {"argument": "", "body": []})
            recv["translations"][lang]["casus"] = casus_tr
            if retr:
                report["retranslate"].append(mv["owner_id"])
    report["stale_keywords"] = sorted(set(report["stale_keywords"]))
    # translations must stay index-aligned with the Latin sections
    report["misaligned"] = [
        cid for cid, c in new.items()
        if c.get("translations") and any(
            len((c["translations"].get(l) or {}).get("body") or []) != len(c.get("body") or [])
            or (c.get("casus") and len((c["translations"].get(l) or {}).get("casus") or []) != len(c["casus"]))
            for l in LANGS)
    ]
    if report["misaligned"]:
        log.warning("%d consilia have translations that are not aligned with the Latin: %s",
                    len(report["misaligned"]), report["misaligned"][:8])
    return report


MIN_CHUNK_CHARS = 50   # same threshold as embed_consilia.py / prepare_units.py


def find_pending_translations(consilia: dict) -> list[dict]:
    """Latin pieces (body sections and casus sections longer than MIN_CHUNK_CHARS) of
    consilia that HAVE translations but whose translation piece is missing or empty.
    Stateless, so it stays correct across repeated builds."""
    pending = []
    for cid, c in consilia.items():
        tr = c.get("translations")
        if not tr:
            continue
        for lang in LANGS:
            t = tr.get(lang) or {}
            for kind in ("body", "casus"):
                la = c.get(kind) or []
                tt = t.get(kind) or []
                for i, piece in enumerate(la):
                    if len(piece.strip()) <= MIN_CHUNK_CHARS:
                        continue
                    if i >= len(tt) or not (tt[i] or "").strip():
                        pending.append({"id": cid, "lang": lang, "kind": kind, "index": i,
                                        "chars": len(piece)})
    return pending
