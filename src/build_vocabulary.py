"""
Stage 8a: induce a closed, controlled keyword vocabulary from the canonical
per-consilium keywords (src/apply_keyword_mapping.py's output, stage 7c).

Why: stage 7 (BGE-M3 clustering + per-cluster LLM naming) consolidates
*phrasing* but does not bound vocabulary size — for the 500-consilium
v1 print it produced 1938 canonical labels, more than half used in
only 1-2 consilia. It also let citation/reference noise through — names of
jurists cited as authorities (e.g. "Baldus", "Bartolus"), place names, and
bibliographic source titles (e.g. "Digest", "Justinian Code") got extracted
as if they were topics, because the per-consilium extraction prompt asked
the model to "index the passage" without restricting to conceptual content.

An earlier version of this script gated inclusion on a doc-frequency
threshold (a label needed >= N distinct consilia to make the cut). That
was wrong: frequency measures how often something happened to recur in
THIS 500-consilium sample, not whether it's a meaningful, generalizable
topic. "impotence" and "consummation" (canon-law marriage-nullity grounds)
appeared in only 1 and 9 consilia respectively and would have been cut
entirely — along with the rest of the sexuality/reproduction/marital-
relations area (rape, concubinage, adultery, incest, bastardy, pregnancy...
38 consilia touch this area in total, but no single label individually
cleared a frequency bar). A rare keyword can be exactly the kind of thing
worth keeping. So vocabulary membership is now decided by hand, not by a
metric:

**Method — one curated exclusion list, reviewed against ALL candidate
labels by legal-historical judgment, not frequency:**

`src/vocabulary_exclusions.json` — every canonical label was read and
individually classified (for v1: all 1908 labels remaining after
removing exact entity/citation matches). Categories excluded:
  - entity/citation noise: jurist names cited as authorities, place names,
    bibliographic source titles, named religious orders/families/factions
  - vague or generic non-topics the extraction model grabbed without them
    naming real content (e.g. "comparison", "context", "purpose", "half")
  - untranslated Latin formula fragments, grammatical artifacts (stray
    "infinitive", "pronoun" — textual-analysis noise, not subject matter)
  - case-incidental physical objects unlikely to recur as a topic (a
    specific cart, a specific wagon) as opposed to a real regulated
    category (kept: "clothing" — sumptuary law is a real recurring genre)
  - redundant near-duplicates of a broader kept term

Everything NOT excluded is in the vocabulary, regardless of how rarely it
occurred in this one volume — including many frequency-1 labels that are
real, important legal-historical categories.

`--min-doc-freq` still exists as an optional secondary floor (default 1,
i.e. off) if you want to layer a frequency cut on top of the curated list
for some other purpose — it is deliberately NOT the primary mechanism.

When new volumes are annotated and run through the stage-7 pipeline, re-run
this script over the combined corpus:
    python src/build_vocabulary.py --list-candidates   # find new capitalized entity noise
    # hand-classify new candidates into vocabulary_exclusions.json, then:
    python src/build_vocabulary.py                     # regenerate output/vocabulary.json
`--list-candidates` only surfaces capitalized-word labels (the pattern that
catches entity/citation noise) — it will NOT surface new vague/generic or
redundant labels the way the original full manual read-through did. If the
new volume introduces a lot of new vocabulary, budget time for another full
read of the new candidate labels, the same way this one was built — that
manual judgment pass is the actual mechanism, not something the script
automates away.

Usage:
    python src/build_vocabulary.py
    python src/build_vocabulary.py --volume Baldo_Cons_Print_Venice_1575_v1
    python src/build_vocabulary.py --list-candidates   # print un-reviewed capitalized labels
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output"
DEFAULT_EXCLUSIONS = ROOT / "src" / "vocabulary_exclusions.json"
DEFAULT_MIN_DOC_FREQ = 1  # off by default — curation (vocabulary_exclusions.json) is the primary gate


def load_exclusions(path: Path) -> set[str]:
    """Real vocabulary exclusions — every non-underscore-prefixed key's labels."""
    data = json.loads(path.read_text(encoding="utf-8"))
    excluded: set[str] = set()
    for key, labels in data.items():
        if key.startswith("_"):
            continue
        excluded.update(labels)
    return excluded


def load_reviewed_keep(path: Path) -> set[str]:
    """Labels manually reviewed and deliberately NOT excluded — suppressed from --list-candidates."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data.get("_reviewed_keep_as_topics", []))


def _is_entity_candidate(label: str) -> bool:
    """Heuristic used to surface NEW capitalized labels for manual review.

    Not used to filter the vocabulary directly — only vocabulary_exclusions.json
    (a hand-reviewed list) does that. This just flags labels worth reviewing.
    """
    words = label.split()
    for i, w in enumerate(words):
        if w[:1].isupper() and w.lower() not in ("i", "ii", "iii", "iv", "v"):
            return True
    return False


def compute_doc_freq(consilia: dict, volume: Optional[str]) -> tuple[Counter, int]:
    """Returns (doc_freq, n_consilia_with_keyword_data) — the latter excludes
    consilia from not-yet-processed volumes (no `keywords` field) so reporting
    doesn't imply the whole corpus contributed when only some volumes have."""
    doc_freq: Counter = Counter()
    n_with_data = 0
    for c in consilia.values():
        if volume and c.get("volume") != volume:
            continue
        kw = c.get("keywords") or {}
        labels = set(kw.get("argument") or []) | set(kw.get("body") or [])
        if not labels:
            continue
        n_with_data += 1
        for label in labels:
            doc_freq[label] += 1
    return doc_freq, n_with_data


def build_vocabulary(doc_freq: Counter, exclusions: set[str], min_doc_freq: int) -> list[dict]:
    terms = [
        {"label": label, "doc_freq": freq}
        for label, freq in doc_freq.items()
        if label not in exclusions and freq >= min_doc_freq
    ]
    terms.sort(key=lambda t: (-t["doc_freq"], t["label"]))
    return terms


def list_candidates(doc_freq: Counter, exclusions: set[str], reviewed_keep: set[str]) -> None:
    candidates = [
        (label, freq) for label, freq in doc_freq.items()
        if _is_entity_candidate(label) and label not in exclusions and label not in reviewed_keep
    ]
    candidates.sort(key=lambda t: -t[1])
    if not candidates:
        print("No un-reviewed capitalized-word labels found.")
        return
    print(f"{len(candidates)} capitalized-word label(s) not yet in vocabulary_exclusions.json "
          f"— review and classify into the right category, or leave out if it's a genuine topic:\n")
    for label, freq in candidates:
        print(f"  {freq:4d}  {label}")


def run(consilia_path: Path, exclusions_path: Path, min_doc_freq: int,
        volume: Optional[str], out_path: Path, do_list_candidates: bool) -> None:
    data = json.loads(consilia_path.read_text(encoding="utf-8"))
    consilia = data.get("consilia", {})
    if not consilia:
        log.error("No consilia found in %s", consilia_path)
        sys.exit(1)

    exclusions = load_exclusions(exclusions_path) if exclusions_path.exists() else set()
    reviewed_keep = load_reviewed_keep(exclusions_path) if exclusions_path.exists() else set()
    doc_freq, n_docs = compute_doc_freq(consilia, volume)

    if do_list_candidates:
        list_candidates(doc_freq, exclusions, reviewed_keep)
        return

    terms = build_vocabulary(doc_freq, exclusions, min_doc_freq)
    log.info("%d consilia have keyword data, %d distinct canonical labels seen, %d excluded as entity/citation noise",
              n_docs, len(doc_freq), len(exclusions & set(doc_freq)))
    log.info("Vocabulary at min_doc_freq=%d: %d terms", min_doc_freq, len(terms))

    result = {
        "method": "stage 8a: doc-frequency threshold + curated entity/citation exclusion "
                  "(see src/build_vocabulary.py docstring)",
        "volume_scope": volume or "all",
        "n_consilia_with_keyword_data": n_docs,
        "min_doc_freq": min_doc_freq,
        "n_raw_canonical_labels": len(doc_freq),
        "n_excluded_entity_noise": len(exclusions & set(doc_freq)),
        "n_vocabulary_terms": len(terms),
        "terms": terms,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Induce a closed controlled keyword vocabulary (stage 8a)")
    ap.add_argument("--consilia", type=Path, default=OUTPUT_DIR / "consilia.json")
    ap.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    ap.add_argument("--min-doc-freq", type=int, default=DEFAULT_MIN_DOC_FREQ,
                     help="Optional secondary floor: minimum number of distinct consilia a label must "
                          "appear in. Default 1 (off) — curation via vocabulary_exclusions.json is the "
                          "primary gate, not this.")
    ap.add_argument("--volume", default=None, help="Restrict to one volume (default: whole corpus)")
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "vocabulary.json")
    ap.add_argument("--list-candidates", action="store_true",
                     help="Print capitalized-word labels not yet reviewed in vocabulary_exclusions.json, then exit")
    args = ap.parse_args()
    run(args.consilia, args.exclusions, args.min_doc_freq, args.volume, args.out, args.list_candidates)


if __name__ == "__main__":
    main()
