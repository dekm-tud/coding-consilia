"""
Export consilia to a JSONL file for GPU-side LLM processing (translation +
keyword extraction — see src/gpu/qwen_process.py, which is meant to be copied
to and run on a separate GPU server).

Usage:
    python src/prepare_llm_input.py
    python src/prepare_llm_input.py --author Baldo_29618397
    python src/prepare_llm_input.py --print Baldo_Cons_Print_Venice_1575_v1
    python src/prepare_llm_input.py --out output/llm_input.jsonl

Each output line:
    {"id": "...", "n": 27, "title": "CONSILIVM XXVII.",
     "volume": "...", "author_viaf": "...",
     "argument_block": "1. Clausula, praesentes literas...\\n2. Quaestuarii qui dicantur.",
     "body_chunks": ["In Christi nomine. Casus talis est...", "† Item non obstat..."]}

`argument_block` is built from the structured `argument` field when present
(annotation-based prints); for prints without it, the flattened `summary`
string is used as a best-effort fallback (not necessarily as cleanly
numbered).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output"


def _argument_block(consilium: dict) -> str:
    items = consilium.get("argument") or []
    if items:
        return "\n".join(f"{item['n']}. {item['text']}".strip() for item in items)
    return consilium.get("summary", "") or ""


def build_records(consilia: dict) -> list[dict]:
    records = []
    for cid, c in consilia.items():
        records.append({
            "id": cid,
            "n": c.get("n"),
            "title": c.get("title", ""),
            "volume": c.get("volume", ""),
            "author_viaf": c.get("author_viaf", ""),
            "argument_block": _argument_block(c),
            "body_chunks": c.get("body") or [],
        })
    return records


def run(author_filter: str | None, print_filter: str | None, out_path: Path) -> None:
    merged_path = OUTPUT_DIR / "consilia.json"
    if not merged_path.exists():
        print(f"error: {merged_path} not found — run src/build.py first", file=sys.stderr)
        sys.exit(1)

    consilia: dict = json.loads(merged_path.read_text(encoding="utf-8"))["consilia"]

    if author_filter:
        consilia = {k: v for k, v in consilia.items() if v.get("author_viaf") == author_filter}
    if print_filter:
        consilia = {k: v for k, v in consilia.items() if v.get("volume") == print_filter}

    if not consilia:
        print("error: no consilia matched the given filters", file=sys.stderr)
        sys.exit(1)

    records = build_records(consilia)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_with_argument = sum(1 for c in consilia.values() if c.get("argument"))
    print(f"Wrote {len(records)} record(s) to {out_path}")
    print(f"  {n_with_argument} of {len(records)} have structured 'argument' data "
          f"(rest fall back to 'summary')")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export consilia to JSONL for GPU LLM processing")
    ap.add_argument("--author", metavar="NAME", help="Restrict to this author_viaf")
    ap.add_argument("--print", dest="print_filter", metavar="NAME", help="Restrict to this print/volume")
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "llm_input.jsonl", help="Output JSONL path")
    args = ap.parse_args()
    run(author_filter=args.author, print_filter=args.print_filter, out_path=args.out)


if __name__ == "__main__":
    main()
