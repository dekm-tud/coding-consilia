"""
Stage 7c of keyword refinement: map every consilium's raw keywords onto the
canonical vocabulary built in stages 7a (src/refine_keywords.py) and 7b
(src/gpu/name_clusters.py), and merge the result into consilia.json.

Usage:
    python src/apply_keyword_mapping.py
    python src/apply_keyword_mapping.py --named output/keyword_clusters_named.json

Adds to every consilium that had `keywords_raw`:
    "keywords": {"argument": [...canonical labels, deduped...],
                 "body": [...canonical labels, deduped...]}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from merge_llm_output import merge_into  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output"


def _normalize(kw: str) -> str:
    return " ".join(kw.lower().split())


def build_lookup(named_clusters: list[dict]) -> dict[str, str]:
    """Map normalised raw keyword phrase -> canonical label."""
    lookup: dict[str, str] = {}
    for rec in named_clusters:
        label = rec["label"]
        for member in rec.get("members", []):
            lookup[_normalize(member)] = label
        lookup[_normalize(label)] = label
    return lookup


def map_keywords(raw: list[str], lookup: dict[str, str]) -> list[str]:
    """Map raw keywords to canonical labels, deduping while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    unmapped = 0
    for kw in raw:
        label = lookup.get(_normalize(kw))
        if label is None:
            unmapped += 1
            continue
        if label not in seen:
            seen.add(label)
            result.append(label)
    if unmapped:
        log.debug("%d/%d raw keyword(s) had no cluster mapping (skipped)", unmapped, len(raw))
    return result


def run(named_path: Path) -> None:
    if not named_path.exists():
        print(f"error: {named_path} not found — run src/refine_keywords.py and "
              f"src/gpu/name_clusters.py first", file=sys.stderr)
        sys.exit(1)

    named_clusters = json.loads(named_path.read_text(encoding="utf-8"))
    lookup = build_lookup(named_clusters)
    log.info("Loaded %d canonical label(s) covering %d raw phrase variants",
              len({r['label'] for r in named_clusters}), len(lookup))

    merged_path = OUTPUT_DIR / "consilia.json"
    consilia = json.loads(merged_path.read_text(encoding="utf-8"))["consilia"]

    updates: dict[str, dict] = {}
    for cid, c in consilia.items():
        raw = c.get("keywords_raw")
        if not raw:
            continue
        updates[cid] = {
            "keywords": {
                "argument": map_keywords(raw.get("argument") or [], lookup),
                "body": map_keywords(raw.get("body") or [], lookup),
            }
        }

    if not updates:
        log.warning("No consilia had 'keywords_raw' — run the GPU stage + merge_llm_output.py first.")
        return

    n_global = merge_into(merged_path, updates)
    log.info("Merged into %s: %d consilia updated", merged_path, n_global)

    merged_data = json.loads(merged_path.read_text(encoding="utf-8"))["consilia"]
    by_author: dict[str, dict[str, dict]] = {}
    by_print: dict[tuple[str, str], dict[str, dict]] = {}
    for cid in updates:
        c = merged_data.get(cid)
        if not c:
            continue
        author_viaf = c.get("author_viaf", "")
        volume = c.get("volume", "")
        by_author.setdefault(author_viaf, {})[cid] = updates[cid]
        by_print.setdefault((author_viaf, volume), {})[cid] = updates[cid]

    for author_viaf, author_updates in by_author.items():
        path = OUTPUT_DIR / author_viaf / "consilia.json"
        n = merge_into(path, author_updates)
        log.info("Merged into %s: %d consilia updated", path, n)

    for (author_viaf, volume), print_updates in by_print.items():
        path = OUTPUT_DIR / author_viaf / f"{volume}.json"
        n = merge_into(path, print_updates)
        log.info("Merged into %s: %d consilia updated", path, n)


def main() -> None:
    ap = argparse.ArgumentParser(description="Map raw keywords to canonical labels (stage 7c)")
    ap.add_argument("--named", type=Path, default=OUTPUT_DIR / "keyword_clusters_named.json")
    args = ap.parse_args()
    run(args.named)


if __name__ == "__main__":
    main()
