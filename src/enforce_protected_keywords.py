"""
Stage 9d: force-include protected keyword categories (src/protected_keyword_categories.json)
into every consilium's keywords_final, regardless of what src/gpu/reduce_keywords.py's
two-pass selection picked.

Why this exists: spot-checking the reduction (10 random consilia, plus a full sweep
of the sexuality/reproduction area) found the two-pass selection itself is precise —
9/10 random samples were accurate, comprehensive matches to their actual content.
But the sweep also found that a hard ~10-term cap means a real, minor, non-central
fact (e.g. a marriage's consummation status, mentioned only to compute a dowry
restitution amount) can lose out to other equally-valid candidates competing for the
same few "distinctive" slots. That's an acceptable trade-off for most of the
vocabulary, but not for a category whose discoverability specifically matters for
research — hence a deterministic, no-GPU-needed safety net for the curated
"intimate and reproductive body" list.

This is pure local post-processing on already-merged data (consilia.json's existing
`keywords` and `keywords_final` fields) — no GPU pass, no new extraction. It is
idempotent (safe to re-run) and additive only: it only ever grows keywords_final by
terms that were already present in that consilium's pre-reduction keywords, never
removes anything.

Usage:
    python src/enforce_protected_keywords.py
    python src/enforce_protected_keywords.py --categories src/protected_keyword_categories.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output"
DEFAULT_CATEGORIES = ROOT / "src" / "protected_keyword_categories.json"


def load_protected_terms(path: Path) -> dict[str, set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        cat: set(info["terms"])
        for cat, info in data.items()
        if not cat.startswith("_")
    }


def enforce(consilia: dict, protected_by_category: dict[str, set[str]]) -> dict[str, dict]:
    """Returns {consilium_id: added_terms_by_category} for consilia that changed."""
    all_protected = set().union(*protected_by_category.values())
    changed: dict[str, dict] = {}

    for cid, c in consilia.items():
        kw = c.get("keywords") or {}
        pre_reduction = set(kw.get("argument") or []) | set(kw.get("body") or [])
        present_protected = pre_reduction & all_protected
        if not present_protected:
            continue

        current_final = c.get("keywords_final") or []
        missing = [t for t in present_protected if t not in current_final]
        if not missing:
            continue

        c["keywords_final"] = current_final + missing
        added_by_cat = {}
        for cat, terms in protected_by_category.items():
            hit = [t for t in missing if t in terms]
            if hit:
                added_by_cat[cat] = hit
        changed[cid] = added_by_cat

    return changed


def merge_into(path: Path, changed_ids: set[str], consilia: dict) -> int:
    if not path.exists():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    file_consilia = data.get("consilia", {})
    n_updated = 0
    for cid in changed_ids:
        if cid in file_consilia:
            file_consilia[cid]["keywords_final"] = consilia[cid]["keywords_final"]
            n_updated += 1
    if n_updated:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return n_updated


def run(categories_path: Path) -> None:
    merged_path = OUTPUT_DIR / "consilia.json"
    consilia: dict = json.loads(merged_path.read_text(encoding="utf-8"))["consilia"]
    protected_by_category = load_protected_terms(categories_path)

    changed = enforce(consilia, protected_by_category)
    log.info("%d consilia gained protected term(s)", len(changed))
    for cat in protected_by_category:
        n = sum(1 for added in changed.values() if cat in added)
        n_terms = sum(len(added[cat]) for added in changed.values() if cat in added)
        log.info("  %s: %d consilia affected, %d term-instances added", cat, n, n_terms)

    if not changed:
        log.info("Nothing to do.")
        return

    merged_path.write_text(json.dumps({"consilia": consilia}, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote %s", merged_path)

    changed_ids = set(changed)
    by_author: dict[str, set[str]] = {}
    by_print: dict[tuple[str, str], set[str]] = {}
    for cid in changed_ids:
        c = consilia[cid]
        by_author.setdefault(c.get("author_viaf", ""), set()).add(cid)
        by_print.setdefault((c.get("author_viaf", ""), c.get("volume", "")), set()).add(cid)

    for author_viaf, ids in by_author.items():
        n = merge_into(OUTPUT_DIR / author_viaf / "consilia.json", ids, consilia)
        log.info("Merged into %s: %d consilia updated", OUTPUT_DIR / author_viaf / "consilia.json", n)
    for (author_viaf, volume), ids in by_print.items():
        path = OUTPUT_DIR / author_viaf / f"{volume}.json"
        n = merge_into(path, ids, consilia)
        log.info("Merged into %s: %d consilia updated", path, n)


def main() -> None:
    ap = argparse.ArgumentParser(description="Force-include protected keyword categories into keywords_final")
    ap.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    args = ap.parse_args()
    run(args.categories)


if __name__ == "__main__":
    main()
