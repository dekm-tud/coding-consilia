"""
Detect consilium boundaries and extract title / summary / body from a stream of
PageXML-derived text lines.

Detection strategy (in priority order):
  1. Regex: line matches CONSILIVM / CONSILIUM + Roman numeral pattern
  2. Geometric fallback: line is significantly narrower than its column AND
     appears to be all-caps (likely a corrupted heading missed by regex)

Body-opener detection (within a consilium's summary section):
  1. Regex: line contains "In Christi nomine" or Latin variants
  2. Dagger prefix: line starts with † (paragraph marker in the body)
  3. Geometric fallback: first narrow, uppercase line after summary items
     that doesn't look like a numbered-item continuation
"""

from __future__ import annotations

import logging
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from parse_pagexml import PageData, TextLine, TextRegion
from lb_detector import join_lines
import casus as _casus

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Roman-numeral helpers
# ---------------------------------------------------------------------------

_WORD_FORMS: dict[str, int] = {
    "PRIMVM": 1, "SECVNDVM": 2, "TERTIVM": 3, "QVARTVM": 4,
    "QVINTVM": 5, "SEXTVM": 6, "SEPTIMVM": 7, "OCTAVVM": 8,
    "NONVM": 9, "DECIMVM": 10,
}

_ROMAN_PAIRS = [
    ("M", 1000), ("CM", 900), ("D", 500), ("CD", 400),
    ("C", 100), ("XC", 90), ("L", 50), ("XL", 40),
    ("X", 10), ("IX", 9), ("V", 5), ("IV", 4), ("I", 1),
]


def roman_to_int(s: str) -> Optional[int]:
    s = s.upper().strip().rstrip(".").replace(" ", "")  # strip spaces (handles "CC CCXXX")
    s = s.replace("S", "C")  # S is a common OCR misread of C (handles CCSCXXXVIII → CCCCXXXVIII)
    s = s.replace("E", "C")  # e is a common OCR misread of C (handles CCceV → CCCCV=405)
    if s in _WORD_FORMS:
        return _WORD_FORMS[s]
    result, i = 0, 0
    while i < len(s):
        for numeral, value in _ROMAN_PAIRS:
            if s[i: i + len(numeral)] == numeral:
                result += value
                i += len(numeral)
                break
        else:
            return None  # unrecognised character
    return result or None


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_RE_HEADING = re.compile(
    # Optional "Ad idem." / "Ad idem " prefix (period may be absent in OCR)
    r"^(?:(?:A[Dd]\s*idem|Adidem|Super\s+eodem|Se\s+summat|Ad\s+eadem)\s*\.?\s+)?"
    # Optional leading paragraph number
    r"(?:\d+\s+)?"
    r"(?:"
    # Standard variants: 0-2 leading OCR garbage chars + CONSILIVM stem + VM/UM ending
    #   CONSILI / CONSILII  – standard and double-I-at-end (CCONSILIIVM)
    #   CONSIILI            – double-I-in-middle (CONSIILIVM)
    #   COSILI              – missing N
    #   CNSILI              – missing O
    #   CONILI              – missing S
    #   CONSLLI             – LL instead of LI (CONSLLIVM)
    #   CONSILR             – R instead of final I (CONSILRVM)
    #   ONSILI              – missing leading C
    #   NSILI               – missing leading CO
    #   CONISLI             – transposed IS (CONISLIVM, four cases in vol. 4 and 5)
    #   CONSILIM            – missing V (CONSILIM)
    r"[A-Z]{0,2}(?:CONSILI{1,2}|CONSIILI|CONISLI|COSILI|CNSILI|CONILI|CONSLLI|CONSILR|ONSILI|NSILI)(?:[OUV][MN]?|M)"
    r"|"
    # Severely truncated suffix variants where the CONSIL- prefix is entirely missing;
    # up to 4 leading garbage chars then the tail LI[UV]M / I[UV]M / [UV]M
    # (handles LIVM CCLIII. → n=253, ILIVM CCCC.  . → n=400, VM CCCCXLIIII. → n=444)
    r"[A-Z]{0,4}(?:LI[UV]M|I[UV]M|[UV]M)"
    r")"
    r"\s*\.?\s+"                               # separator (optional period + space)
    r"(PRIMVM|SECVNDVM|TERTIVM|QVARTVM|QVINTVM|SEXTVM|SEPTIMVM"
    r"|OCTAVVM|NONVM|DECIMVM"
    r"|[IVXLCDMS]+(?:\s[IVXLCDMS]{2,})*)"          # roman numeral; S→C OCR noise
                                               # (handles CCSCXXXVIII → CCCCXXXVIII=438)
    r"\s*(?:[.?]{1,3}(?:[A-Za-z\s.?]{0,40})?)?\s*$",  # trailing: 0-3 dots/? + optional short text (a repeated heading line fits)
    re.IGNORECASE,
)

# Normalise OCR misread 'e'→'C' only when sandwiched between Roman-numeral chars
# (handles CCceV → CCcCV → roman_to_int → CCCCV = 405; avoids matching body text)
_RE_NUMERAL_E = re.compile(r"(?<=[IVXLCDMSivxlcdms])[eE](?=[IVXLCDMSivxlcdms.])")

# Marginal numbers at line starts (1-99; the edition text puts them as "n " before the line). A number
# followed by a period ("2. l. cum pater") is a citation and stays.
_RE_LEADING_NUMBER = re.compile(r"^([1-9]\d?)(?:\s+|$)")

# A soft hyphen left inside a joined text ("co¬ lonello", "suprascri¬. ptos") before a lower-case letter.
_RE_INNER_HYPHEN = re.compile(r"¬\.?\s*(?=[a-zęæœ])")

# Matches lines that consist *solely* of a marginal number (no content after it).
_RE_STANDALONE_NUMBER = re.compile(r"^[1-9]\d?\s*$")


def _normalize_ocr_e(text: str) -> str:
    return _RE_NUMERAL_E.sub("C", text)


def _strip_leading_number(text: str) -> str:
    """Remove the marginal number (1-99) from the start of a line."""
    return _RE_LEADING_NUMBER.sub("", text, count=1)


# Alternative heading format: Arabic numerals, e.g. "Consilium 473. et 414."
# The second number is an OCR-variant of the true number (healed by _heal_sequence).
_RE_ARABIC_HEADING = re.compile(
    r"^Consilium\s+(\d+)\.(?:\s*et\s+(\d+)\.)?\s*$",
    re.IGNORECASE,
)

# Pure Roman numeral / word-fragment patterns to reject from geometric fallback
_RE_PURE_NUMERAL = re.compile(r"^[IVXLCDM\s.]+$", re.IGNORECASE)
_RE_WORD_FRAGMENT = re.compile(
    r"^(?:VM|LVM|IVM|LIVM|LIVM?|CONSIL\.?)\b",  # tail fragments + truncated CONSILIVM
    re.IGNORECASE,
)

# Section labels that are NOT consilium headings — they belong to surrounding content.
#   CASVS / CNSVS / OASVS  — OCR variants of "casus" (legal case description)
#   ADDITIO BAL. / ADDIIIO RAL. — OCR variants of "Additio Bal[di]" (author addendum)
_RE_SECTION_LABEL = re.compile(
    r"^(?:"
    r"[A-Z]{0,2}[AC][ANO]?SI?V[SUI§]*"   # CASVS, CASIVS, CASV, CNSVS, CASVI§ …
    r"|ADDI[A-Z]{1,3}O(?:\s+[A-Z]{2,4})?"  # ADDITIO BAL., ADDITIO. (standalone) …
    r"|AD\s*IDEM"                         # ADIDEM / AD IDEM — citation sub-section
    r"|PACTV?M(?:\s+\w+)?"               # PACTVM TALE — contract title, not a heading
    r"|(?:FI|PI)NIS"                      # FINIS / PINIS — end-of-volume colophon
    r")\s*[.§]*\s*$",
    re.IGNORECASE,
)

_RE_SUMMARY_ITEM = re.compile(r"^\d+\s+[A-Z]")  # "1 Feudum an …"

# Body openers: "In Christi nomine", "In nomine Christi / Dei / Domini", …
# "NOMI" (not "NOMINE") catches soft-hyphenated "N Christi Nomi¬" fragments.
# Extended with formulae common in v4/v5 prints.
_RE_BODY_OPENER = re.compile(
    r"(?:CHRIST[A-Z]\s+NOMI"        # "Christi Nomine", "Christr Nomi¬" (OCR variants)
    r"|NOMINE\s+CHRISTI"
    r"|IN\s+NOMINE\s+(?:CHRISTI|DEI|DOMINI|SANCTI)"
    r"|IN\s+CHRIST[A-Z]\s+NOMI"     # "In Christi Nomine" incl. OCR typos
    r"|IN\s+[lI]HRISTI\s+NOMI"     # "In lhristi nomine" — OCR l/C confusion
    r"|IN\s+DEI\s+NOMI"             # "In Dei Nomine" or "In Dei Nomi¬"
    r"|AD\s+EU?IDENT"               # "Ad evidentiam / Ad euidentiam"
    r"|PR[AĘ]EMITTEND"              # "praemittendum / praemittendi est"
    r"|PRIMO\s+PUNCT[OI]"           # "primo puncto dicendum est"
    r"|CONSIDER[AAIO]T[OAIS]"       # "considerato / considerata / consideratis"
    r"|PUNCT[VU][MS]\s*,?\s*(?:QUAES|VTRUM|UTRUM)"  # "Punctus quaestionis / Punctus vtrum"
    r")",
    re.IGNORECASE,
)

# Ratio above the running median height of summary lines that triggers a
# geometric body-opener detection.
_TALL_LINE_RATIO = 1.3
# Minimum number of summary lines needed before attempting geometric detection.
_GEO_OPENER_MIN_CONTEXT = 3

# Dagger paragraph marker — reliable body indicator when leading a line
_RE_DAGGER_LEAD = re.compile(r"^[††]")

# ---------------------------------------------------------------------------
# Geometric threshold
# ---------------------------------------------------------------------------

_SHORT_LINE_RATIO = 0.58  # lines narrower than this fraction of column width
                           # are candidates for headings or body openers


# ---------------------------------------------------------------------------
# Split-heading pre-processing
# ---------------------------------------------------------------------------


def _merge_colinear_splits(lines: list["LineRef"]) -> list["LineRef"]:
    """
    Preflight pass: merge consecutive lines from the *same region* that sit at
    the same vertical position (side-by-side layout splits) when doing so
    produces a valid consilium heading that neither line individually matches.

    Transkribus occasionally segments a single printed heading line into two
    TextLines — e.g. 'CONSILIVM' (left) + 'CCCLIII.' (right) — because the
    two halves are typeset slightly apart.  Joining them by x-order (no space
    when touching, space when there is a small gap) produces 'CONSILIVM
    CCCLIII.' which the regex then catches correctly.

    Guard: if either line already matches _RE_HEADING we leave it alone so
    we never disturb a working detection.
    """
    result: list["LineRef"] = []
    i = 0
    while i < len(lines):
        lr = lines[i]
        if i + 1 >= len(lines):
            result.append(lr)
            i += 1
            continue

        nxt = lines[i + 1]

        # Must be same page and same region
        if lr.page_filename != nxt.page_filename or lr.region_id != nxt.region_id:
            result.append(lr)
            i += 1
            continue

        # Don't disturb lines that already match the heading regex
        if _RE_HEADING.match(lr.text) or _RE_HEADING.match(nxt.text):
            result.append(lr)
            i += 1
            continue

        # Compute y-midpoints and heights from line_coords
        a_ys = [p[1] for p in lr.line_coords]
        b_ys = [p[1] for p in nxt.line_coords]
        ya_mid = (min(a_ys) + max(a_ys)) / 2
        yb_mid = (min(b_ys) + max(b_ys)) / 2
        avg_h = max(max(a_ys) - min(a_ys), max(b_ys) - min(b_ys), 1)

        if abs(ya_mid - yb_mid) >= avg_h * 0.4:
            result.append(lr)
            i += 1
            continue

        # Determine left / right order by x-min
        a_xs = [p[0] for p in lr.line_coords]
        b_xs = [p[0] for p in nxt.line_coords]
        if min(a_xs) <= min(b_xs):
            left, right = lr, nxt
            x_gap = min(b_xs) - max(a_xs)
        else:
            left, right = nxt, lr
            x_gap = min(a_xs) - max(b_xs)

        # Reject if lines are too far apart (different columns)
        if x_gap > avg_h * 1.5:
            result.append(lr)
            i += 1
            continue

        sep = "" if x_gap < avg_h * 0.5 else " "
        merged_text = (left.text + sep + right.text).strip()

        if not _RE_HEADING.match(merged_text):
            result.append(lr)
            i += 1
            continue

        merged = _make_merged_lineref(left, right, merged_text)
        log.info(
            "Colinear merge on %s: %r + %r → %r",
            lr.page_filename, lr.text, nxt.text, merged_text,
        )
        result.append(merged)
        i += 2

    return result


def _is_heading_fragment(lr: "LineRef") -> bool:
    """Short all-caps line that could be part of a multi-line heading."""
    text = lr.text
    if text.rstrip().endswith("¬"):
        return False
    stripped = text.replace(".", "").replace(" ", "")
    return lr.width_ratio < _SHORT_LINE_RATIO and stripped.isupper() and len(stripped) >= 2


def _make_merged_lineref(base: "LineRef", other: "LineRef", combined_text: str) -> "LineRef":
    """Return a new LineRef merging two fragments, using the first as anchor."""
    all_pts = base.line_coords + other.line_coords
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    merged_coords = [
        [min(xs), min(ys)], [max(xs), min(ys)],
        [max(xs), max(ys)], [min(xs), max(ys)],
    ]
    return LineRef(  # type: ignore[call-arg]  # forward ref resolved at runtime
        page_filename=base.page_filename,
        region_id=base.region_id,
        region_type=base.region_type,
        region_coords=base.region_coords,
        line_id=base.line_id,
        line_coords=merged_coords,
        text=combined_text.strip(),
        has_soft_hyphen=False,
    )


def _try_merge(texts: list[str]) -> str | None:
    """
    Try all orderings and concatenation styles of the given text fragments
    to find one that matches _RE_HEADING. Returns the matching combined text
    or None if no combination works.
    """
    from itertools import permutations
    stripped = [t.rstrip(".").strip() for t in texts]
    # Only try forward (in order) and, for pairs, reverse
    candidates = [" ".join(stripped)]
    if len(stripped) == 2:
        candidates.append(" ".join(reversed(stripped)))
        # Also try direct concatenation (for char-level splits like CONSILI + VM)
        candidates.append(stripped[0] + stripped[1])
        candidates.append(stripped[1] + stripped[0])
    for candidate in candidates:
        if _RE_HEADING.match(candidate + "."):  # add period so regex's \.? matches
            return candidate
    return None


def merge_split_headings(lines: list["LineRef"]) -> list["LineRef"]:
    """
    Pre-processing pass: merge consecutive narrow all-caps lines when together
    they form a valid consilium heading (handles 2- and 3-part splits).
    """
    result: list[LineRef] = []
    i = 0
    while i < len(lines):
        lr = lines[i]
        # Fast path: not a fragment candidate
        if not _is_heading_fragment(lr) or _RE_HEADING.match(lr.text):
            result.append(lr)
            i += 1
            continue

        # Try to extend with 1 or 2 more fragment lines
        merged = None
        for window in range(1, 3):
            if i + window >= len(lines):
                break
            group = lines[i: i + window + 1]
            if not all(_is_heading_fragment(g) for g in group):
                break
            combined = _try_merge([g.text for g in group])
            if combined:
                merged = _make_merged_lineref(group[0], group[-1], combined)
                log.info(
                    "Merged split heading on %s: %s → '%s'",
                    lr.page_filename,
                    [g.text for g in group],
                    combined,
                )
                result.append(merged)
                i += window + 1
                break

        if merged is None:
            result.append(lr)
            i += 1
    return result


def _width_ratio(line_coords: list[list[int]], region_coords: list[list[int]]) -> float:
    rx = [p[0] for p in region_coords]
    lx = [p[0] for p in line_coords]
    rw = max(rx) - min(rx)
    return (max(lx) - min(lx)) / rw if rw else 1.0


# ---------------------------------------------------------------------------
# Line reference (annotated text stream entry)
# ---------------------------------------------------------------------------


@dataclass
class LineRef:
    page_filename: str
    region_id: str
    region_type: str
    region_coords: list[list[int]]
    line_id: str
    line_coords: list[list[int]]
    text: str
    has_soft_hyphen: bool = False
    line_type: str = ""  # structural annotation, e.g. "cons_title", "cons_argument", "cons_body"

    @property
    def width_ratio(self) -> float:
        return _width_ratio(self.line_coords, self.region_coords)

    @property
    def height(self) -> int:
        if not self.line_coords:
            return 0
        ys = [p[1] for p in self.line_coords]
        return max(ys) - min(ys)


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------


@dataclass
class RegionSpan:
    page_filename: str
    region_id: str
    region_type: str
    region_coords: list[list[int]]
    first_line_id: str
    last_line_id: str


@dataclass
class Consilium:
    id: str
    n: int
    roman: str
    volume: str
    author_viaf: str
    title: str
    summary: str
    body: list[str]   # sections split at leading † markers
    argument: list[dict] = field(default_factory=list)  # [{"n": int, "text": str}, …]
    sources: list[dict] = field(default_factory=list)
    casus: list[str] = field(default_factory=list)  # statement of facts, set before the heading in the print
    body_meta: list[dict] = field(default_factory=list)  # per body section: {"marginal": n or None}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "n": self.n,
            "roman": self.roman,
            "volume": self.volume,
            "author_viaf": self.author_viaf,
            "title": self.title,
            "summary": self.summary,
            "argument": self.argument,
            "casus": self.casus,
            "body": self.body,
            "body_meta": self.body_meta,
            "sources": self.sources,
        }


# ---------------------------------------------------------------------------
# Page-stream iterator
# ---------------------------------------------------------------------------

_SKIP_TYPES = {"header"}


def iter_lines(pages: list[PageData]) -> Iterator[LineRef]:
    """Yield all content TextLines across pages in reading order, skipping headers."""
    for page in pages:
        for region in page.regions:
            if region.region_type in _SKIP_TYPES:
                continue
            for line in region.lines:
                if not line.text:
                    continue
                yield LineRef(
                    page_filename=page.filename,
                    region_id=region.region_id,
                    region_type=region.region_type,
                    region_coords=region.coords.to_list(),
                    line_id=line.line_id,
                    line_coords=line.coords.to_list(),
                    text=line.text,
                    has_soft_hyphen=line.has_soft_hyphen,
                    line_type=line.line_type,
                )


# ---------------------------------------------------------------------------
# Shared helpers (used by both ConsiliumSegmenter and AnnotatedConsiliumSegmenter)
# ---------------------------------------------------------------------------


def _sections_from_body_lines(body_lines: list[LineRef], use_lb_model: bool, marginal: list | None = None) -> list[str]:
    """Split body lines into sections at every † character.

    Groups lines by dagger boundaries first, then calls join_lines on each
    group separately (small batches keep the LB model accurate). Any
    mid-line daggers within a joined group are split afterwards.

    The marginal number at a line start is taken out of the text (it would stand in the running text
    and break hyphenated words). If ``marginal`` is a list it receives one entry per section: the number
    of the line on which the section's † stands; a number on a line without † goes to the † of the line
    above when that one has no number of its own, else to the † of the next line; otherwise None.
    """
    if not body_lines:
        return []

    # Group lines: a line that starts with † opens a new section
    groups: list[list[str]] = [[]]
    line_numbers: list[tuple[int | None, int]] = []      # (number, count of † on the line) per kept line
    for lr in body_lines:
        m = _RE_LEADING_NUMBER.match(lr.text)
        text = _strip_leading_number(lr.text)
        if not text.strip():
            continue
        line_numbers.append((int(m.group(1)) if m else None, text.count("†")))
        if _RE_DAGGER_LEAD.match(text) and groups[-1]:
            groups.append([])
        groups[-1].append(text)

    sections: list[str] = []
    for group in groups:
        if not group:
            continue
        joined = join_lines(group, use_model=use_lb_model)
        # a ¬ inside a line (one PageXML line holding two printed lines) joins the word as well
        joined = _RE_INNER_HYPHEN.sub("", joined)
        # Handle mid-line daggers within an already-joined group
        parts = re.split(r"(?=[††])", joined)
        sections.extend(p.strip() for p in parts if p.strip())

    if marginal is not None:
        marginal.extend([None] * len(sections))
        lead = 0 if sections and sections[0].startswith("†") else 1     # a first section without † comes first
        k = 0                                                            # † seen so far
        pending = None                                                   # number on a line without †
        for j, (num, crosses) in enumerate(line_numbers):
            if crosses:
                idx = k + lead
                if idx < len(marginal) and marginal[idx] is None:
                    marginal[idx] = num if num is not None else pending
                pending = None
                k += crosses
            elif num is not None:
                prev = line_numbers[j - 1][1] if j else 0
                last = k - 1 + lead
                if prev and 0 <= last < len(marginal) and marginal[last] is None:
                    marginal[last] = num                                 # the number stands one line below its †
                else:
                    pending = num                                        # ... or, more often, one line above
    return sections


def _build_sources(spans: list["RegionSpan"]) -> list[dict]:
    by_page: dict[str, list[RegionSpan]] = {}
    for span in spans:
        by_page.setdefault(span.page_filename, []).append(span)
    return [
        {
            "page": page_fn,
            "regions": [
                {
                    "region_id": s.region_id,
                    "type": s.region_type,
                    "coords": s.region_coords,
                    "first_line": s.first_line_id,
                    "last_line": s.last_line_id,
                }
                for s in spans
            ],
        }
        for page_fn, spans in by_page.items()
    ]


# ---------------------------------------------------------------------------
# Sequence healing
# ---------------------------------------------------------------------------


def _heal_sequence(consilia: list[Consilium], volume: str) -> list[Consilium]:
    """
    Post-processing: reclassify out-of-sequence consilia to the missing number
    they most likely represent (OCR garbled the numeral to a different valid one).

    For example CXIII (113) appearing between XCI (92) and XCIIII (94) in page
    order is clearly a misread of XCIII (93) → reclassified to n=93.

    Condition for reclassification:
      - The consilium's n is outside the range (prev_n, next_n) in page order
      - There is exactly one missing number in (prev_n, next_n)

    After reclassification the IDs of all consilia are rebuilt so that
    duplicates generated by the original false detection are cleaned up.
    """
    result = list(consilia)
    detected: set[int] = {c.n for c in result if c.n > 0}

    for idx in range(len(result)):
        c = result[idx]

        # Neighbours in page order (skip n=0 entries when looking for neighbours)
        prev_n = next(
            (result[j].n for j in range(idx - 1, -1, -1) if result[j].n > 0),
            0,
        )
        next_n = next(
            (result[j].n for j in range(idx + 1, len(result)) if result[j].n > 0),
            501,
        )

        if c.n == 0:
            # Geometric heading with unknown number: assign if only one number is
            # missing between the two numbered neighbours (e.g. CCSCXXXVIII → n=438)
            if next_n == prev_n + 2 and (prev_n + 1) not in detected:
                correct_n = prev_n + 1
                log.info(
                    "Sequence heal (geo n=0): assign n=%d at %s (neighbours: %d, %d)",
                    correct_n,
                    c.sources[0]["page"] if c.sources else "?",
                    prev_n, next_n,
                )
                detected.add(correct_n)
                c.n = correct_n
            continue

        if prev_n < c.n < next_n:
            continue  # in sequence, nothing to heal

        # Out of sequence: exactly one missing number in (prev_n, next_n)?
        missing_in_range = [
            m for m in range(prev_n + 1, next_n) if m not in detected
        ]
        if len(missing_in_range) != 1:
            continue

        correct_n = missing_in_range[0]
        log.info(
            "Sequence heal: reclassify n=%d → n=%d at %s (neighbours: %d, %d)",
            c.n, correct_n,
            c.sources[0]["page"] if c.sources else "?",
            prev_n, next_n,
        )
        detected.discard(c.n)
        detected.add(correct_n)
        c.n = correct_n

    # Rebuild IDs: the original false-duplicate suffixes (-2, -3) are now stale
    used_ids: dict[str, int] = {}
    unknown_count = 0
    for c in result:
        if c.n > 0:
            base_id = f"consilium-{volume.lower()}-{c.n}"
        else:
            unknown_count += 1
            base_id = f"consilium-{volume.lower()}-unknown-{unknown_count}"
        count = used_ids.get(base_id, 0) + 1
        used_ids[base_id] = count
        c.id = base_id if count == 1 else f"{base_id}-{count}"

    return result


# ---------------------------------------------------------------------------
# Segmenter
# ---------------------------------------------------------------------------


class ConsiliumSegmenter:
    """
    Stateful segmenter that consumes a stream of LineRef objects and
    emits Consilium objects.

    State machine:
      BEFORE → (regex/geometric heading) → SUMMARY
      SUMMARY → (body opener) → BODY
      SUMMARY → (next heading) → SUMMARY  [with warning: empty body]
      BODY → (next heading) → SUMMARY     [flush previous consilium]
    """

    def __init__(self, volume: str, author_viaf: str = "", use_lb_model: bool = True):
        self.volume = volume
        self.author_viaf = author_viaf
        self.use_lb_model = use_lb_model
        self._unknown_count = 0
        self._used_ids: dict[str, int] = {}  # base_id → count of times seen
        self._reset_state()

    # ------------------------------------------------------------------
    # Internal state helpers
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        self._state: str = "BEFORE"
        self._title: str = ""
        self._roman: str = ""
        self._n: int = 0
        self._n2: int = 0   # secondary number from arabic headings (e.g. "Consilium 473. et 414.")
        self._summary_lines: list[LineRef] = []
        self._body_lines: list[LineRef] = []
        self._spans: list[RegionSpan] = []
        self._current_span: Optional[RegionSpan] = None
        self._section_heights: list[int] = []  # heights of lines seen in current summary

    def _start_heading(self, text: str, roman: str, n: int, n2: int = 0) -> None:
        self._title = text
        self._roman = roman
        self._n = n
        self._n2 = n2
        self._summary_lines = []
        self._body_lines = []
        self._spans = []
        self._current_span = None
        self._section_heights = []
        self._state = "SUMMARY"

    def _track_span(self, lr: LineRef) -> None:
        """Extend or start a RegionSpan for the current consilium."""
        if (
            self._current_span is None
            or self._current_span.region_id != lr.region_id
            or self._current_span.page_filename != lr.page_filename
        ):
            self._current_span = RegionSpan(
                page_filename=lr.page_filename,
                region_id=lr.region_id,
                region_type=lr.region_type,
                region_coords=lr.region_coords,
                first_line_id=lr.line_id,
                last_line_id=lr.line_id,
            )
            self._spans.append(self._current_span)
        else:
            self._current_span.last_line_id = lr.line_id

    def _assemble(self, line_refs: list[LineRef]) -> str:
        """Join lines into a dehyphenated string for summary.

        Preserves leading numbers that are part of numbered summary items
        (e.g. '1 Executor fideicommissa…') but drops lines that consist
        solely of a marginal number with no content.
        """
        texts = [lr.text for lr in line_refs if not _RE_STANDALONE_NUMBER.match(lr.text.strip())]
        return join_lines(texts, use_model=self.use_lb_model)

    def _build_body_sections(self) -> list[str]:
        """Split body into sections at every † character. See _sections_from_body_lines."""
        self._marginal = []
        return _sections_from_body_lines(self._body_lines, self.use_lb_model, self._marginal)

    def _build_sources(self) -> list[dict]:
        return _build_sources(self._spans)

    def _flush(self) -> list[Consilium]:
        if not self._title:
            return []
        if not self._body_lines and not self._summary_lines:
            log.warning("Empty consilium '%s' — skipping.", self._title)
            return []

        if not self._body_lines:
            # Recovery: scan summary lines for a deferred body-opener that was
            # missed during streaming (OCR errors, novel phrases, page artifacts).
            split_at: int | None = None
            _n_sum = len(self._summary_lines)

            # Pass 1: text-based opener or leading dagger (most reliable)
            for i, lr in enumerate(self._summary_lines):
                if self._is_body_opener(lr.text):
                    split_at = i
                    break
                if _RE_DAGGER_LEAD.match(lr.text):
                    split_at = i
                    break

            # Pass 2: page artifact (running header like "Consilia.") followed by
            # body text that starts with a lowercase letter.
            _RE_PAGE_HDR = re.compile(r"^(?:Consilia|Consiliorum)\.?\s*$", re.IGNORECASE)
            if split_at is None:
                for i in range(_n_sum - 1):
                    if _RE_PAGE_HDR.match(self._summary_lines[i].text.strip()):
                        j = i + 1
                        while j < _n_sum:
                            nxt = self._summary_lines[j].text.strip()
                            if not _RE_PAGE_HDR.match(nxt) and not re.match(r"^\d+$", nxt):
                                if nxt and nxt[0].islower():
                                    split_at = j
                                break
                            j += 1
                        if split_at is not None:
                            break

            # Pass 3: embedded dagger (†) within the first 40 characters of any
            # line in the first 10 summary lines — handles cases like
            # "v † primo puncto" or "sacrilegi † robatores" where the big initial
            # is OCR-separated and the dagger marks the real body opener.
            _RE_DAGGER_ANY = re.compile(r"[††]")
            if split_at is None:
                for i, lr in enumerate(self._summary_lines):
                    if i == 0 or i >= 10:
                        continue
                    if _RE_DAGGER_ANY.search(lr.text[:40]):
                        split_at = i
                        break

            # Pass 4: last-resort heuristic — after a sequence of summary
            # propositions (short lines ending with '.'), the body starts.
            # Look within the first 10 summary lines for the last such proposition
            # and split at the next line, provided ≥ 5 lines remain.
            if split_at is None:
                _RE_ARTIFACT = re.compile(
                    r"^(?:Consilia|Consiliorum)\.?\s*$|\d+$", re.IGNORECASE
                )
                last_prop_idx: int | None = None
                for i, lr in enumerate(self._summary_lines[:10]):
                    text = lr.text.strip()
                    if (text.endswith(".")
                            and not text.endswith("¬")
                            and len(text) < 65
                            and not _RE_ARTIFACT.match(text)):
                        last_prop_idx = i
                if last_prop_idx is not None:
                    candidate = last_prop_idx + 1
                    if _n_sum - candidate >= 5:
                        split_at = candidate

            if split_at is not None:
                self._body_lines = self._summary_lines[split_at:]
                self._summary_lines = self._summary_lines[:split_at]
                log.info(
                    "Body recovered for '%s' (n=%d): split at line %d (%r)",
                    self._title, self._n, split_at,
                    self._body_lines[0].text[:60],
                )
            else:
                log.warning(
                    "No body found for '%s' (n=%d); storing all content as summary.",
                    self._title, self._n,
                )

        base_id = (
            f"consilium-{self.volume.lower()}-{self._n}"
            if self._n > 0
            else self._next_unknown_id()
        )
        cid = self._unique_id(base_id)
        summary = self._assemble(self._summary_lines)
        body = self._build_body_sections()
        sources = self._build_sources()

        c = Consilium(
            id=cid,
            n=self._n,
            roman=self._roman,
            volume=self.volume,
            author_viaf=self.author_viaf,
            title=self._title,
            summary=summary,
            body=body,
            body_meta=[{"marginal": m} for m in self._marginal] if any(m is not None for m in self._marginal) else [],
            sources=sources,
        )

        n2 = self._n2
        self._reset_state()

        result: list[Consilium] = [c]
        if n2 > 0:
            # Emit a second entry for the secondary number in arabic-style headings
            # (e.g. "Consilium 473. et 414." → also create n=414 which heals to n=474)
            clone_base_id = f"consilium-{self.volume.lower()}-{n2}"
            clone = Consilium(
                id=self._unique_id(clone_base_id),
                n=n2,
                roman=str(n2),
                volume=self.volume,
                author_viaf=self.author_viaf,
                title=c.title,
                summary=summary,
                body=body[:],
                sources=sources[:],
            )
            result.append(clone)

        return result

    def _next_unknown_id(self) -> str:
        self._unknown_count += 1
        return f"consilium-{self.volume.lower()}-unknown-{self._unknown_count}"

    def _unique_id(self, base_id: str) -> str:
        """Return base_id on first use; append -2, -3, … on duplicates."""
        count = self._used_ids.get(base_id, 0) + 1
        self._used_ids[base_id] = count
        if count == 1:
            return base_id
        log.warning("Duplicate consilium ID '%s' (occurrence %d).", base_id, count)
        return f"{base_id}-{count}"

    # ------------------------------------------------------------------
    # Line classification helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_heading(text: str) -> tuple[bool, str, int, int]:
        """Return (matched, roman_or_num_str, primary_n, secondary_n).

        secondary_n is non-zero only for arabic-numeral headings that name two
        consilia (e.g. 'Consilium 473. et 414.').
        """
        m = _RE_HEADING.match(text) or _RE_HEADING.match(_normalize_ocr_e(text))
        if m:
            roman = m.group(1).upper()
            n = roman_to_int(roman) or 0
            return True, roman, n, 0
        m2 = _RE_ARABIC_HEADING.match(text)
        if m2:
            n1 = int(m2.group(1))
            n2 = int(m2.group(2)) if m2.group(2) else 0
            return True, str(n1), n1, n2
        return False, "", 0, 0

    @staticmethod
    def _is_body_opener(text: str) -> bool:
        if _RE_BODY_OPENER.search(text):
            return True
        if _RE_DAGGER_LEAD.match(text):
            return True
        return False

    @staticmethod
    def _is_body_opener_geometric(lr: LineRef, section_heights: list[int]) -> bool:
        """Tall, full-width line after summary items — body opener even without
        a recognisable opener formula (handles diverse v4/v5 openers).

        Only fires when we have enough context lines to compute a reliable median,
        and only for wide lines so we don't confuse body openers with headings.
        """
        if len(section_heights) < _GEO_OPENER_MIN_CONTEXT:
            return False
        median_h = statistics.median(section_heights)
        if median_h <= 0 or lr.height < median_h * _TALL_LINE_RATIO:
            return False
        # Wide line: spans a meaningful fraction of the column
        if lr.width_ratio < _SHORT_LINE_RATIO:
            return False
        # Not a numbered summary item
        if _RE_SUMMARY_ITEM.match(lr.text):
            return False
        return True

    @staticmethod
    def _looks_like_heading_geometrically(lr: LineRef) -> bool:
        """Narrow, all-caps line not matching a numbered summary item.

        Rejects obvious fragments: lines ending with ¬ (word split mid-line),
        very short strings, or lines that are clearly partial (fewer than 4
        distinct alpha characters).
        """
        text = lr.text
        if text.rstrip().endswith("¬"):
            return False  # fragment of a word split at line end
        stripped = text.replace(".", "").replace(" ", "").replace("¬", "")
        alpha_chars = sum(1 for c in stripped if c.isalpha())
        if alpha_chars < 4:
            return False
        if _RE_PURE_NUMERAL.match(text.strip()):
            return False  # bare numeral fragment (e.g. "CCLXVI.")
        if _RE_WORD_FRAGMENT.match(text.strip()):
            return False  # tail of a split word (e.g. "LIVM CCLIII.")
        # Strip spaces, periods, and OCR noise for label matching
        # "CA S VS." → "CASVS"   "CAS. VS." → "CASVS"   "CASV§." → "CASV"
        _norm = re.sub(r"[\s.§*]", "", text.strip())
        if _RE_SECTION_LABEL.match(text.strip()) or _RE_SECTION_LABEL.match(_norm):
            return False  # section label (casus, additio) — belongs to adjacent consilium
        return (
            lr.width_ratio < _SHORT_LINE_RATIO
            and stripped.isupper()
            and len(text) >= 6
            and not _RE_SUMMARY_ITEM.match(text)
        )

    @staticmethod
    def _is_summary_continuation(text: str) -> bool:
        """Line that continues a previous summary item (starts lowercase)."""
        return bool(text) and text[0].islower()

    # ------------------------------------------------------------------
    # Main processing loop
    # ------------------------------------------------------------------

    def process(self, lines: Iterator[LineRef]) -> list[Consilium]:
        # Pre-process: geometry-based colinear merge first, then text-based split-heading merge
        all_lines = _merge_colinear_splits(list(lines))
        all_lines = merge_split_headings(all_lines)
        results: list[Consilium] = []
        # (results populated by the loop below; healed and ID-rebuilt at end)

        for lr in all_lines:
            text = lr.text

            # ── 1. Check for a new consilium heading ──────────────────
            matched, roman, n, n2 = self._is_heading(text)

            if not matched and self._state in ("BODY", "BEFORE"):
                # Geometric fallback only when we expect text (BODY) or
                # haven't found the first heading yet (BEFORE)
                if self._looks_like_heading_geometrically(lr):
                    log.warning(
                        "Geometric heading detected on page %s: '%s'",
                        lr.page_filename, text,
                    )
                    matched, roman, n, n2 = True, "?", 0, 0

            if matched:
                results.extend(self._flush())
                self._start_heading(text, roman, n, n2)
                self._track_span(lr)
                continue

            # ── 2. State machine ──────────────────────────────────────
            if self._state == "BEFORE":
                continue  # skip preamble / ToC lines before first heading

            self._track_span(lr)

            if self._state == "SUMMARY":
                self._section_heights.append(lr.height)
                if self._is_body_opener(text):
                    self._state = "BODY"
                    self._body_lines.append(lr)
                elif self._is_body_opener_geometric(lr, self._section_heights[:-1]):
                    # Geometric fallback: tall, wide line signals body start even
                    # without a recognised opener formula (diverse v4/v5 openers).
                    log.debug(
                        "Geometric body opener on %s: %r  h=%d median=%.0f",
                        lr.page_filename, text[:60], lr.height,
                        statistics.median(self._section_heights[:-1]),
                    )
                    self._state = "BODY"
                    self._body_lines.append(lr)
                elif self._is_summary_continuation(text):
                    # Continuation of a multi-line summary item → append to last item
                    if self._summary_lines:
                        self._summary_lines.append(lr)
                    else:
                        self._summary_lines.append(lr)
                else:
                    self._summary_lines.append(lr)

            elif self._state == "BODY":
                self._body_lines.append(lr)

        # Flush the last consilium
        results.extend(self._flush())

        results = _heal_sequence(results, self.volume)
        return results


# ---------------------------------------------------------------------------
# Annotation-based segmenter (for PageXML with cons_title/cons_argument/cons_body
# line-level structure tags — see AnnotatedConsiliumSegmenter below)
# ---------------------------------------------------------------------------

# Argument-map item numbering: "1 Clausula, praesentes …" opens item 1; unlike
# _RE_LEADING_NUMBER (marginal body glosses, capped at 1-10), argument items are
# not capped — a consilium can have well over 10 numbered points.
_RE_ARGUMENT_ITEM = re.compile(r"^(\d{1,3})[.]?\s+(.*)$")     # "1 Clausula …" or "1. Clausula …"
_RE_ARGUMENT_ITEM_BARE = re.compile(r"^(\d{1,3})\s*$")


def _group_argument_items(argument_lines: list[LineRef], use_lb_model: bool) -> list[dict]:
    """Group cons_argument lines into numbered items.

    A line starting with a leading integer opens a new item; subsequent lines
    without a leading integer are continuations of the current item (wrapped
    text), joined via join_lines (handles dehyphenation exactly like the body).
    """
    items: list[tuple[int, list[str]]] = []
    for lr in argument_lines:
        text = lr.text.strip()
        if not text:
            continue
        m = _RE_ARGUMENT_ITEM.match(text)
        if m:
            n = int(m.group(1))
            # the same number again with a lower-case start is a wrapped line whose marginal
            # number was read twice, not a new item
            if items and items[-1][0] == n and m.group(2)[:1].islower():
                items[-1][1].append(m.group(2))
                continue
            items.append((n, [m.group(2)] if m.group(2) else []))
            continue
        m_bare = _RE_ARGUMENT_ITEM_BARE.match(text)
        if m_bare:
            items.append((int(m_bare.group(1)), []))
            continue
        if items:
            items[-1][1].append(text)
        else:
            # No item number seen yet (rare OCR/annotation anomaly) — keep the
            # text rather than silently dropping it.
            items.append((0, [text]))

    result: list[dict] = []
    for n, fragments in items:
        text = join_lines(fragments, use_model=use_lb_model) if fragments else ""
        result.append({"n": n, "text": text})
    return result


def _spans_from_lines(lines: list[LineRef]) -> list[RegionSpan]:
    """Region spans for an ordered line list (same grouping rule as the segmenters' track_span)."""
    spans: list[RegionSpan] = []
    for lr in lines:
        cur = spans[-1] if spans else None
        if cur is None or cur.region_id != lr.region_id or cur.page_filename != lr.page_filename:
            spans.append(RegionSpan(
                page_filename=lr.page_filename, region_id=lr.region_id,
                region_type=lr.region_type, region_coords=lr.region_coords,
                first_line_id=lr.line_id, last_line_id=lr.line_id,
            ))
        else:
            cur.last_line_id = lr.line_id
    return spans


class AnnotatedConsiliumSegmenter:
    """
    Segmenter for PageXML where every TextLine already carries a structural
    type annotation (cons_title / cons_argument / cons_body / header /
    page_number). Boundaries are a direct line-type lookup — no regex/geometry
    heuristics needed.

    State machine: a "cons_title" line following a non-title line starts a new
    consilium (flushing the previous one first). Consecutive cons_title lines
    accumulate (multi-line titles); cons_argument and cons_body lines
    accumulate into their own buffers until the next title.
    """

    _CONTENT_TYPES = {"cons_title", "cons_argument", "cons_body"}

    def __init__(self, volume: str, author_viaf: str = "", use_lb_model: bool = True,
                 split_casus: bool = True):
        self.volume = volume
        self.author_viaf = author_viaf
        self.use_lb_model = use_lb_model
        self.split_casus = split_casus   # False reproduces the pre-correction segmentation
        self._used_ids: dict[str, int] = {}
        self._unknown_count = 0
        self.casus_report: list[dict] = []   # one row per signature+casus candidate
        self.casus_moves: list[dict] = []    # accepted moves (drives the translation split)

    def _next_unknown_id(self) -> str:
        self._unknown_count += 1
        return f"consilium-{self.volume.lower()}-unknown-{self._unknown_count}"

    def _unique_id(self, base_id: str) -> str:
        count = self._used_ids.get(base_id, 0) + 1
        self._used_ids[base_id] = count
        if count == 1:
            return base_id
        log.warning("Duplicate consilium ID '%s' (occurrence %d).", base_id, count)
        return f"{base_id}-{count}"

    def _apply_casus(self, results: list[Consilium], records: list[dict]) -> None:
        """Move the casus that follows a closing signature to the next consilium.

        See casus.py. Decision per candidate: manual override > direction check by
        lexical overlap with the two argumenta > otherwise left in place and listed
        in the review report.
        """
        overrides = _casus.load_overrides()
        doc_text = [c.summary + " " + " ".join(c.body) for c in results]
        idf = _casus.Idf(doc_text)
        n_rec = len(records)
        casus_for: dict[int, list[LineRef]] = {}   # receiver index -> casus lines (in print order)

        for i, rec in enumerate(records):
            owner = results[i]
            body = rec["body"]
            cut = _casus.find_casus_cut(body) if body else None
            if cut is None:
                continue
            prefix, casus_lines = _casus.split_lines_at(body, cut)
            if not casus_lines:
                continue
            casus_text = " ".join(_strip_leading_number(l.text) for l in casus_lines)
            nxt = results[i + 1] if i + 1 < n_rec else None
            old_secs = owner.body
            pre_marg: list = []
            pre_secs = _sections_from_body_lines(prefix, self.use_lb_model, pre_marg)
            v_cas = idf.vec(casus_text)
            v_own = idf.vec(owner.summary + " " + " ".join(pre_secs))
            v_nxt = idf.vec(nxt.summary + " " + " ".join(nxt.body)) if nxt else {}
            sc_o, sc_n = _casus.cosine(v_cas, v_own), _casus.cosine(v_cas, v_nxt)
            sh_o, sh_n = _casus.top_shared(v_cas, v_own), _casus.top_shared(v_cas, v_nxt)
            status, reason = _casus.decide(
                owner.id, sc_o, sc_n, nxt is not None, bool(nxt and nxt.summary.strip()),
                cut.heading_in_gap, overrides, cut.trailing,
            )
            cas_secs = _sections_from_body_lines(casus_lines, self.use_lb_model)
            first = _strip_leading_number(casus_lines[0].text).lstrip()
            mid = not _RE_DAGGER_LEAD.match(first)
            expect = len(pre_secs) + len(cas_secs) - (1 if (mid and pre_secs) else 0)
            if status == "move" and len(old_secs) != expect:
                status, reason = "review", f"section_count_mismatch: old {len(old_secs)} != {expect}"

            self.casus_report.append({
                "volume": self.volume, "owner_id": owner.id,
                "receiver_id": nxt.id if nxt else "", "status": status, "reason": reason,
                "score_owner": round(sc_o, 4), "score_next": round(sc_n, 4),
                "shared_next": " ".join(sh_n), "shared_owner": " ".join(sh_o),
                "n_candidates": cut.n_candidates, "marker": cut.marker,
                "casus_start": casus_text[:160],
                "owner_argument": owner.summary[:200],
                "receiver_argument": nxt.summary[:200] if nxt else "",
            })
            if status != "move":
                continue

            # ---- apply the move -------------------------------------------------
            owner.body = pre_secs
            owner.body_meta = [{"marginal": m} for m in pre_marg] if any(m is not None for m in pre_marg) else []
            split_line = body[cut.line_index]
            later = {id(l) for l in body[cut.line_index + 1:]}
            owner_stream: list[LineRef] = []
            for l in rec["stream"]:
                if l is split_line:
                    if cut.char > 0 and prefix and prefix[-1].line_id == l.line_id:
                        owner_stream.append(prefix[-1])   # signature part of the split line
                elif id(l) not in later:
                    owner_stream.append(l)
            casus_stream = list(casus_lines)
            owner.sources = _build_sources(_spans_from_lines(owner_stream))
            rec["stream"] = owner_stream
            casus_for[i + 1] = casus_stream
            self.casus_moves.append({
                "volume": self.volume, "owner_id": owner.id, "receiver_id": nxt.id,
                "old_sections": len(old_secs), "owner_sections": len(pre_secs),
                "casus_sections": len(cas_secs),
                "mid_section": bool(mid and pre_secs),
                "mixed_owner_chars": len(pre_secs[-1]) if pre_secs else 0,
                "mixed_casus_chars": len(cas_secs[0]) if cas_secs else 0,
            })
            log.info("  casus moved: %s -> %s (%s)", owner.id, nxt.id, reason)

        for j, lines in casus_for.items():
            recv = results[j]
            recv.casus = _sections_from_body_lines(lines, self.use_lb_model)
            recv.sources = _build_sources(_spans_from_lines(lines + records[j]["stream"]))

    def process(self, lines: Iterator[LineRef]) -> list[Consilium]:
        results: list[Consilium] = []
        records: list[dict] = []          # per consilium: raw body lines + stream, for the casus pass
        stream: list[LineRef] = []

        title_lines: list[LineRef] = []
        argument_lines: list[LineRef] = []
        body_lines: list[LineRef] = []
        spans: list[RegionSpan] = []
        current_span: Optional[RegionSpan] = None
        prev_type = ""

        def track_span(lr: LineRef) -> None:
            nonlocal current_span
            if (
                current_span is None
                or current_span.region_id != lr.region_id
                or current_span.page_filename != lr.page_filename
            ):
                current_span = RegionSpan(
                    page_filename=lr.page_filename,
                    region_id=lr.region_id,
                    region_type=lr.region_type,
                    region_coords=lr.region_coords,
                    first_line_id=lr.line_id,
                    last_line_id=lr.line_id,
                )
                spans.append(current_span)
            else:
                current_span.last_line_id = lr.line_id

        def flush() -> None:
            nonlocal title_lines, argument_lines, body_lines, spans, current_span, stream
            if not title_lines:
                # Preamble before the first heading (or a stray reset) — drop,
                # matching the old segmenter's BEFORE-state behaviour.
                title_lines, argument_lines, body_lines, spans, current_span = [], [], [], [], None
                stream = []
                return

            title_text = join_lines([lr.text for lr in title_lines], use_model=self.use_lb_model)
            matched, roman, n, _n2 = ConsiliumSegmenter._is_heading(title_text)
            if not matched:
                log.warning("Annotated title did not match heading pattern: %r", title_text)

            argument_items = _group_argument_items(argument_lines, self.use_lb_model)
            summary = " ".join(
                f"{item['n']} {item['text']}".strip() for item in argument_items
            ).strip()
            marginal: list = []
            body = _sections_from_body_lines(body_lines, self.use_lb_model, marginal)

            base_id = (
                f"consilium-{self.volume.lower()}-{n}" if n > 0 else self._next_unknown_id()
            )
            cid = self._unique_id(base_id)

            results.append(Consilium(
                id=cid,
                n=n,
                roman=roman,
                volume=self.volume,
                author_viaf=self.author_viaf,
                title=title_text,
                summary=summary,
                argument=argument_items,
                body=body,
                body_meta=[{"marginal": m} for m in marginal] if any(m is not None for m in marginal) else [],
                sources=_build_sources(spans),
            ))
            records.append({"body": body_lines, "stream": stream})

            title_lines, argument_lines, body_lines, spans, current_span = [], [], [], [], None
            stream = []

        for lr in lines:
            t = lr.line_type
            if t not in self._CONTENT_TYPES:
                continue  # skip header / page_number / unannotated lines

            if t == "cons_title" and prev_type != "cons_title":
                flush()

            track_span(lr)
            stream.append(lr)
            if t == "cons_title":
                title_lines.append(lr)
            elif t == "cons_argument":
                argument_lines.append(lr)
            else:  # cons_body
                body_lines.append(lr)
            prev_type = t

        flush()  # last consilium
        if self.split_casus:
            self._apply_casus(results, records)
        return results
