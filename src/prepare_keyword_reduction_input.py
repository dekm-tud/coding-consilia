"""
Stage 9a: export per-consilium candidates for keyword reduction (GPU server —
see src/gpu/reduce_keywords.py).

Rationale: the curated vocabulary (output/vocabulary.json, 3,463 terms) is
too large to hand an LLM in one prompt and expect reliable selection — tested
empirically (see project history): embedding-similarity retrieval against the
full vocabulary systematically misses topics that are real but only briefly
mentioned in a long consilium (a whole-document embedding averages over
everything the text discusses, diluting a minor mention below the shortlist
cutoff — confirmed on documents already correctly tagged "rape" by the
original extraction, where embedding retrieval ranked it anywhere from #3 to
#1525 out of 3,463 depending on the document).

So this does NOT retrieve from the full vocabulary. Instead it uses each
consilium's OWN existing keywords.argument/keywords.body (already produced by
reading the actual text — not retrieval) filtered down to terms that survived
curation. That candidate set is small (typically 13-39 terms) and already
validated as relevant to this specific document, including minor/rare topics
the original extraction already found. Stage 9b then just has to pick the
best ~10 of those ~30, which is a far more tractable task than picking from
the full vocabulary.

Usage:
    python src/prepare_keyword_reduction_input.py
    python src/prepare_keyword_reduction_input.py --print Baldo_Cons_Print_Venice_1575_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output"


def _text_en(c: dict) -> str:
    tr = c.get("translations", {}).get("en", {})
    argument = tr.get("argument", "") or ""
    body = " ".join(tr.get("body") or [])
    return (argument + "\n\n" + body).strip()


def build_records(consilia: dict, vocab_freq: dict[str, int]) -> list[dict]:
    records = []
    for cid, c in consilia.items():
        kw = c.get("keywords") or {}
        labels = set(kw.get("argument") or []) | set(kw.get("body") or [])
        candidates = sorted(
            ({"label": l, "doc_freq": vocab_freq[l]} for l in labels if l in vocab_freq),
            key=lambda t: (-t["doc_freq"], t["label"]),
        )
        if not candidates:
            continue
        records.append({
            "id": cid,
            "text_en": _text_en(c),
            "candidates": candidates,
        })
    return records


def run(author_filter: str | None, print_filter: str | None, out_path: Path) -> None:
    merged_path = OUTPUT_DIR / "consilia.json"
    vocab_path = OUTPUT_DIR / "vocabulary.json"
    if not merged_path.exists() or not vocab_path.exists():
        print(f"error: {merged_path} and {vocab_path} must both exist — run the "
              f"translation/keyword/vocabulary pipeline first", file=sys.stderr)
        sys.exit(1)

    consilia: dict = json.loads(merged_path.read_text(encoding="utf-8"))["consilia"]
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    vocab_freq = {t["label"]: t["doc_freq"] for t in vocab["terms"]}

    if author_filter:
        consilia = {k: v for k, v in consilia.items() if v.get("author_viaf") == author_filter}
    if print_filter:
        consilia = {k: v for k, v in consilia.items() if v.get("volume") == print_filter}
    if not consilia:
        print("error: no consilia matched the given filters", file=sys.stderr)
        sys.exit(1)

    records = build_records(consilia, vocab_freq)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_small = sum(1 for r in records if len(r["candidates"]) <= 10)
    print(f"Wrote {len(records)} record(s) to {out_path}")
    print(f"  {n_small} already have <=10 candidates (reduce_keywords.py will pass these through "
          f"without an LLM call)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export per-consilium candidate keywords for GPU-side reduction")
    ap.add_argument("--author", metavar="NAME", help="Restrict to this author_viaf")
    ap.add_argument("--print", dest="print_filter", metavar="NAME", help="Restrict to this print/volume")
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "keyword_reduction_input.jsonl")
    args = ap.parse_args()
    run(args.author, args.print_filter, args.out)


if __name__ == "__main__":
    main()
