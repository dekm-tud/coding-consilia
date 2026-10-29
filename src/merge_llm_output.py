"""
Merge GPU-produced translations + raw keywords (src/gpu/qwen_process.py's
output) back into consilia.json (global, per-author, and per-print copies).

Usage:
    python src/merge_llm_output.py
    python src/merge_llm_output.py --llm-output output/llm_output.jsonl

Adds two fields to every matched consilium:
    "translations": {"en": {"argument": "...", "body": [...]},
                      "de": {"argument": "...", "body": [...]}},
    "keywords_raw": {"argument": [...], "body": [...]}

Idempotent — safe to re-run as more GPU output trickles in (only overwrites
the fields above; everything else is left untouched).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_updates(llm_records: list[dict]) -> dict[str, dict]:
    updates: dict[str, dict] = {}
    for rec in llm_records:
        updates[rec["id"]] = {
            "translations": {
                "en": {"argument": rec.get("argument_en", ""), "body": rec.get("body_chunks_en") or []},
                "de": {"argument": rec.get("argument_de", ""), "body": rec.get("body_chunks_de") or []},
            },
            "keywords_raw": {
                "argument": rec.get("keywords_argument_raw") or [],
                "body": rec.get("keywords_body_raw") or [],
            },
        }
    return updates


def merge_into(path: Path, updates: dict[str, dict]) -> int:
    """Apply `updates` (keyed by consilium id) to the consilia dict in `path`."""
    if not path.exists():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    consilia = data.get("consilia", {})
    n_updated = 0
    for cid, extra in updates.items():
        if cid in consilia:
            consilia[cid].update(extra)
            n_updated += 1
    if n_updated:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return n_updated


def run(llm_output_path: Path) -> None:
    if not llm_output_path.exists():
        print(f"error: {llm_output_path} not found", file=sys.stderr)
        sys.exit(1)

    llm_records = _read_jsonl(llm_output_path)
    if not llm_records:
        log.warning("%s is empty — nothing to merge.", llm_output_path)
        return
    updates = build_updates(llm_records)
    log.info("Loaded %d LLM record(s) from %s", len(updates), llm_output_path)

    merged_path = OUTPUT_DIR / "consilia.json"
    n_global = merge_into(merged_path, updates)
    log.info("Merged into %s: %d consilia updated", merged_path, n_global)
    if n_global == 0:
        log.warning("No matching consilium ids found in %s — check that llm_output.jsonl "
                    "was produced from the same consilia.json.", merged_path)
        return

    # Propagate into per-author and per-print copies, using volume/author_viaf
    # looked up from the just-updated global file.
    merged_data = json.loads(merged_path.read_text(encoding="utf-8"))["consilia"]
    touched = {cid: merged_data[cid] for cid in updates if cid in merged_data}

    by_author: dict[str, dict[str, dict]] = {}
    by_print: dict[tuple[str, str], dict[str, dict]] = {}
    for cid, c in touched.items():
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
    ap = argparse.ArgumentParser(description="Merge GPU LLM output into consilia.json")
    ap.add_argument("--llm-output", type=Path, default=OUTPUT_DIR / "llm_output.jsonl")
    args = ap.parse_args()
    run(args.llm_output)


if __name__ == "__main__":
    main()
