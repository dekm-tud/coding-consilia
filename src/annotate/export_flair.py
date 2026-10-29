#!/usr/bin/env python3
"""Turn span annotations (src/annotate/spans.py output or the gold file) into a Flair ColumnCorpus:
output/flair/{train,dev,test}.txt, one token per line ("token<TAB>BIO-tag"), a blank line between
sentences. The split is by consilium (hash of the doc id), so no document is in two sets.

  python src/annotate/export_flair.py output/annotations/spans_<model>_p1.jsonl [more.jsonl] [--gold eval/gold/spans_gold.jsonl]

Records from --gold override model records of the same unit. Tokens are maximal runs of letters/digits
or single punctuation marks; a token is B-/I- when it overlaps a span. Sentences are cut at ". ", ": "
or "; " followed by a space, but never inside a span, and at most ~120 tokens.
Training script: tools/train_flair_ner.py (GPU server).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UNITS = ROOT / "output" / "units.jsonl"
OUT = ROOT / "output" / "flair"
TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
MAX_TOKENS = 120


def tag_tokens(text: str, spans: list[dict]) -> list[tuple[str, str, int]]:
    """[(token, tag, end_offset)] with BIO tags."""
    spans = sorted(spans, key=lambda s: s["start"])
    out, si, started = [], 0, set()
    for m in TOKEN.finditer(text):
        a, b = m.span()
        while si < len(spans) and spans[si]["end"] <= a:
            si += 1
        tag = "O"
        if si < len(spans) and spans[si]["start"] < b and a < spans[si]["end"]:
            tag = ("I-" if si in started else "B-") + spans[si]["label"]; started.add(si)
        out.append((m.group(0), tag, b))
    return out


def sentences(tokens: list[tuple[str, str, int]], text: str) -> list[list[tuple[str, str]]]:
    sents, cur = [], []
    for i, (tok, tag, end) in enumerate(tokens):
        cur.append((tok, tag))
        nxt_tag = tokens[i + 1][1] if i + 1 < len(tokens) else "O"
        boundary = tok in ".:;" and tag == "O" and not nxt_tag.startswith("I-") and text[end:end + 1].isspace()
        if (boundary and len(cur) >= 8) or len(cur) >= MAX_TOKENS and tag == "O" and not nxt_tag.startswith("I-"):
            sents.append(cur); cur = []
    if cur:
        sents.append(cur)
    return sents


def split_of(doc: str) -> str:
    h = int(hashlib.sha1(doc.encode()).hexdigest(), 16) % 10
    return "train" if h < 8 else "dev" if h == 8 else "test"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--gold", type=Path)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--units", type=Path, default=UNITS, help="units file the runs were made on (default output/units.jsonl)")
    a = ap.parse_args()
    units = {json.loads(l)["unit"]: json.loads(l) for l in a.units.open(encoding="utf-8")}
    recs = {}
    for f in a.runs:
        for l in f.open(encoding="utf-8"):
            r = json.loads(l)
            if r.get("ok") and r["unit"] in units:
                recs[r["unit"]] = r
    if a.gold:
        for l in a.gold.open(encoding="utf-8"):
            r = json.loads(l); recs[r["unit"]] = r
    a.out.mkdir(parents=True, exist_ok=True)
    files = {s: (a.out / f"{s}.txt").open("w", encoding="utf-8") for s in ("train", "dev", "test")}
    n = Counter(); tags = Counter()
    for unit, r in recs.items():
        text = units[unit]["la"]; split = split_of(r.get("doc") or unit.split("#")[0])
        for sent in sentences(tag_tokens(text, r["spans"]), text):
            files[split].write("\n".join(f"{t}\t{g}" for t, g in sent) + "\n\n")
            n[split] += 1; tags.update(g for _, g in sent if g != "O")
    for fh in files.values():
        fh.close()
    (a.out / "labels.json").write_text(json.dumps(sorted({t[2:] for t in tags}), indent=1), encoding="utf-8")
    print(f"{len(recs)} units -> {dict(n)} sentences in {a.out}; tagged tokens: {dict(tags.most_common())}")


if __name__ == "__main__":
    main()
