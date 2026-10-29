"""
Misplaced-casus correction.

In the print, the statement of facts (CASVS / Punctus / Factum tale est) of a
consilium is set *before* its heading, i.e. directly after the closing signature
("Ego Baldus.") of the preceding consilium. Transkribus tags those lines
`cons_body`, so the annotation-driven segmenter attaches them to the previous
consilium. This module

  1. finds the signature + casus marker inside a consilium's body lines and the
     exact split point (line index + character offset),
  2. splits the lines there,
  3. decides by lexical overlap with the two neighbouring *argumenta* whether the
     block belongs to the next consilium (direction check),
  4. applies manual decisions from src/casus_overrides.json,
  5. splits existing EN/DE translations at the matching place (carry_translations).

Nothing here touches PageXML. All decisions are reproducible from the annotations
plus casus_overrides.json.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
OVERRIDES_PATH = Path(__file__).parent / "casus_overrides.json"

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

# Closing signature. Optional tail: "de Perusio", "legum doctor", "etc.".
# A line-end soft hyphen inside the name ("Bal¬ dus", "Pe¬ rusio") is tolerated: the
# print's ¬ is not always a word break (see lb_detector), so lines are joined with a space.
_SH = r"(?:¬\s*)?"
_SIG = (
    r"(?:\bEgo\s+Bal" + _SH + r"(?:dus|do|d)?\b\.?"
    r"|\bBal" + _SH + r"\.?\s+de\s+Pe" + _SH + r"r" + _SH + r"u\w*\.?)"
    r"(?:\s*de\s+Pe" + _SH + r"r" + _SH + r"u\w*\.?)?"
    r"(?:[\s,]*legum\s+doctor\.?)?"
    r"(?:[\s,]*etc\.?)?"
)
# a line-end ¬ may also follow a whole word ("consulo ego¬ | Baldus")
_SIG = _SIG.replace(r"\s+", r"¬?\s+").replace(r"\s*", r"¬?\s*")
RE_SIGNATURE = re.compile(_SIG, re.IGNORECASE)

# Casus marker, incl. OCR variants (C A S VS, CNSVS, OASVS, "AS VS" with lost C).
_MARK = (
    r"(?:C\s*A\s*S(?:\s*V\s*S|\s*V|\s*S)?(?![A-Za-z])|C\s*N\s*S\s*V\s*S|O\s*A\s*S\s*V\s*S?|(?<![A-Za-z])A\s*S\s+V\s*S"
    r"|Casus\s+(?:talis|siper|super|est)|Alias\s+est\s+casus"
    r"|Punctus[.,]?\s+(?:talis|ralis|quonis|quaestionis|quionis|seu|est|super|facti|sic)"
    r"|Factum\s+(?:tale|proponitur|super|sic|quod)|Factu\s+proponitur|Factum,?\s+super|Proponitur)"
)
_MARK = _MARK.replace(r"\s+", r"¬?\s+").replace(r"\s*", r"¬?\s*")
RE_MARK = re.compile(_MARK, re.IGNORECASE)
# Opening formula that may sit between the signature and the marker and belongs to the casus.
RE_OPENER_ONLY = re.compile(
    r"(?:in\s*(?:(?:christi|dei|domini)\s+)?nomine(?:\s+(?:christi|dei|domini))?\s*(?:amen)?\.?)?",
    re.IGNORECASE,
)
RE_HEADING = re.compile(r"CONSILI[VU]M\s+[IVXLCDM]+", re.IGNORECASE)

TAIL_SIGNATURE_SLACK = 40   # chars allowed after a signature that closes the casus block itself
MARK_WINDOW = 200   # max distance (chars) between signature end and marker


def _strip_number(text: str) -> str:
    # same rule as segment_consilia._strip_leading_number (kept local: no import cycle)
    return re.sub(r"^[1-9]\d?(?:\s+|$)", "", text, count=1)


@dataclasses.dataclass
class CasusCut:
    line_index: int       # index into body_lines of the line that contains the cut
    char: int             # offset within that line's number-stripped text
    marker: str           # matched marker text
    n_candidates: int     # signature+marker pairs found in this body
    heading_in_gap: bool  # a "CONSILIVM <roman>" heading sits between signature and marker
    trailing: bool = True  # no further closing signature follows: the block runs to the end of the body


def find_casus_cut(body_lines: list) -> Optional[CasusCut]:
    """Locate the last signature+marker pair in the body lines. The cut is placed at
    the end of the signature, so the signature stays with the consilium it closes."""
    flat_parts: list[str] = []
    starts: list[int] = []
    pos = 0
    for lr in body_lines:
        t = _strip_number(lr.text).strip()
        starts.append(pos)
        flat_parts.append(t)
        pos += len(t) + 1
    flat = " ".join(flat_parts)

    found = []
    for m in RE_SIGNATURE.finditer(flat):
        window = flat[m.end(): m.end() + MARK_WINDOW]
        mk = RE_MARK.search(window)
        if mk:
            found.append((m, mk, window[: mk.start()]))
    if not found:
        return None
    m, mk, gap = found[-1]
    # Cut right after the signature when only an opener formula follows it; otherwise the
    # text between signature and marker is the rest of the signature clause ("V. I. D. et ad
    # fidem me subscripsi …") and stays with the owner: cut at the marker.
    cut = m.end() if RE_OPENER_ONLY.fullmatch(gap.strip()) else m.end() + mk.start()
    # line containing `cut` (cut == start of a line -> boundary belongs to that line, char 0)
    k = 0
    for i, s in enumerate(starts):
        if s <= cut:
            k = i
        else:
            break
    char = cut - starts[k]
    # cut may sit on the joining space after line k-1 / at end of line k
    if char >= len(flat_parts[k]) and k + 1 < len(body_lines):
        k, char = k + 1, 0
    # A casus set before the next heading may itself end in "Ego Bal." (the requester's
    # question is signed); a signature followed by more text means the block is not the tail.
    trailing = all(len(flat) - m2.end() <= TAIL_SIGNATURE_SLACK
                   for m2 in RE_SIGNATURE.finditer(flat, m.end() + mk.end()))
    return CasusCut(k, char, mk.group(0), len(found), bool(RE_HEADING.search(gap)), trailing)


def split_lines_at(body_lines: list, cut: CasusCut) -> tuple[list, list]:
    """(prefix_lines, casus_lines). A line containing the cut is split in two copies."""
    k, c = cut.line_index, cut.char
    lr = body_lines[k]
    t = _strip_number(lr.text).strip()
    if c <= 0:
        return list(body_lines[:k]), list(body_lines[k:])
    pre, post = t[:c].rstrip(), t[c:].lstrip()
    prefix = list(body_lines[:k]) + ([dataclasses.replace(lr, text=pre)] if pre else [])
    casus = ([dataclasses.replace(lr, text=post)] if post else []) + list(body_lines[k + 1:])
    return prefix, casus


# ---------------------------------------------------------------------------
# Direction check: lexical overlap between casus and the two *argumenta*
# ---------------------------------------------------------------------------

_STEM_LEN = 5
_MIN_TOKEN = 4


def _norm(text: str) -> str:
    text = text.lower()
    for a, b in (("æ", "e"), ("œ", "e"), ("ę", "e"), ("j", "i"), ("v", "u"), ("¬", "")):
        text = text.replace(a, b)
    return text


def stems(text: str) -> set[str]:
    return {w[:_STEM_LEN] for w in re.findall(r"[a-z]+", _norm(text)) if len(w) >= _MIN_TOKEN}


class Idf:
    """Inverse document frequency of stems over the consilia of one print."""

    def __init__(self, doc_texts: list[str]):
        df: Counter = Counter()
        for t in doc_texts:
            df.update(stems(t))
        self.n = max(len(doc_texts), 1)
        self.df = df

    def weight(self, stem: str) -> float:
        return math.log((1 + self.n) / (1 + self.df.get(stem, 0))) + 1.0

    def vec(self, text: str) -> dict[str, float]:
        tf = Counter(w[:_STEM_LEN] for w in re.findall(r"[a-z]+", _norm(text)) if len(w) >= _MIN_TOKEN)
        return {s: (1 + math.log(c)) * self.weight(s) for s, c in tf.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(w * b.get(s, 0.0) for s, w in a.items())
    na = math.sqrt(sum(w * w for w in a.values()))
    nb = math.sqrt(sum(w * w for w in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def top_shared(a: dict[str, float], b: dict[str, float], k: int = 8) -> list[str]:
    sh = sorted(((a[s] * b[s], s) for s in a.keys() & b.keys()), reverse=True)
    return [s for _, s in sh[:k]]


def load_overrides() -> dict:
    if OVERRIDES_PATH.exists():
        return json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    return {}


MIN_COS_NEXT = 0.05     # below this the casus is not clearly about the next consilium
MIN_RATIO = 1.15        # next must beat the owner by this factor


def decide(owner_id: str, cos_owner: float, cos_next: float, has_next: bool,
           has_argument: bool, heading_in_gap: bool, overrides: dict,
           trailing: bool = True) -> tuple[str, str]:
    """-> (status, reason). status: 'move' | 'keep' | 'review'."""
    ov = overrides.get(owner_id)
    if ov and ov.get("decision") in ("move", "keep"):
        return ov["decision"], f"override: {ov.get('reason', '')}".strip()
    if not trailing:
        return "keep", "not_trailing: a later signature follows, so this is not the casus of the next consilium"
    if heading_in_gap:
        return "review", "heading_in_gap"
    if not has_next:
        return "review", "no_next_consilium"
    if not has_argument:
        return "review", "next_has_no_argumentum"
    if cos_next >= MIN_COS_NEXT and cos_next > MIN_RATIO * cos_owner:
        return "move", f"auto: next {cos_next:.3f} vs owner {cos_owner:.3f}"
    return "review", f"direction unclear: next {cos_next:.3f}, owner {cos_owner:.3f}"


# ---------------------------------------------------------------------------
# Review CSV
# ---------------------------------------------------------------------------

REVIEW_FIELDS = [
    "volume", "owner_id", "receiver_id", "status", "reason", "score_owner", "score_next",
    "shared_next", "shared_owner", "n_candidates", "marker", "casus_start", "owner_argument",
    "receiver_argument",
]


def write_review_csv(rows: list[dict], path: Path, volumes: set[str]) -> None:
    """Replace the rows of `volumes` in the CSV (other prints' rows are kept)."""
    keep: list[dict] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as fh:
            keep = [r for r in csv.DictReader(fh) if r["volume"] not in volumes]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=REVIEW_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in keep + rows:
            w.writerow(r)
