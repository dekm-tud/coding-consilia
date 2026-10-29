#!/usr/bin/env python3
"""
Give the lines of columns that had no Gemini transcription their Gemini reading (output/ocr_gemini/<print>/<page>_c<n>.txt),
so that tools/final_text.py can vote with three readings there. Until now the "gemini" field of those lines in
output/merge/kraken_current.jsonl was a copy of the PyLaia reading. Lines are paired by banded alignment of the
loosely normalised text (as in src/qa/relayout_from_gemini.py); a pair differing by more than 25 % is not taken,
the line keeps its copy. The geometry and the PageXML are not touched.

    python tools/add_gemini_columns.py v1:0069:2 v1:0166:2 …            # report only
    python tools/add_gemini_columns.py v1:0069:2 … --write              # update kraken_current.jsonl (backup kept)
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "qa")); sys.path.insert(0, str(ROOT / "src"))
from compare_ocr import align, dist, normalise  # noqa: E402
from relayout_from_gemini import T, gemini_info, read_lines, split_inline_headings  # noqa: E402

MERGE = ROOT / "output" / "merge"
DATA = ROOT / "data" / "Baldo_29618397"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("columns", nargs="+", help="v<vol>:<page stem>:<column>, e.g. v1:0069:2 or v2:page_216:1")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--max-diff", type=float, default=0.25)
    a = ap.parse_args()
    recs = [json.loads(l) for l in (MERGE / "kraken_current.jsonl").open(encoding="utf-8")]
    by_key = {(r["print"], r["page"], r["line_id"]): r for r in recs}
    changed = 0
    for col in a.columns:
        v, stem, n = col.split(":")
        prt = f"Baldo_Cons_Print_Venice_1575_{v}"
        xml = DATA / prt / "page" / f"{stem}.xml"
        region = next(tr for tr in ET.parse(xml).getroot().find(T + "Page").findall(T + "TextRegion")
                      if re.search(rf"type:column_{n}\b", tr.get("custom", "") or ""))
        X = read_lines(region)
        raw = (ROOT / "output" / "ocr_gemini" / prt / f"{stem}_c{n}.txt").read_text(encoding="utf-8").splitlines()
        G = [gemini_info(s) for s in split_inline_headings([s for s in raw if s.strip()])]
        xl = [normalise(l["text"], True) for l in X]
        took = same = skipped = 0
        for i, j in align(xl, [g["loose"] for g in G]):
            if i is None or j is None:
                continue
            r = by_key.get((prt, f"{stem}.xml", X[i]["id"]))
            if r is None:
                continue
            d = dist(xl[i], G[j]["loose"]) / max(len(xl[i]), len(G[j]["loose"]), 1)
            if d > a.max_diff or G[j]["abbr"]:
                skipped += 1; continue
            g = re.sub(r"^\{S\}\s*", "", G[j]["conv"]).strip()
            if g == r["gemini"]:
                same += 1; continue
            r["gemini"] = g; r["comment"] = (r.get("comment") or "") + "; gemini column 2026-10-29 (gemini-4-argon)"
            took += 1
        changed += took
        print(f"{col}: {len(X)} lines, {len(G)} Gemini lines; reading changed {took}, same {same}, not paired/too different {skipped}")
    if a.write and changed:
        shutil.copy(MERGE / "kraken_current.jsonl", MERGE / "recheck_2026-10-29" / "kraken_current.before.jsonl")
        with (MERGE / "kraken_current.jsonl").open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"kraken_current.jsonl: {changed} Gemini readings written")


if __name__ == "__main__":
    main()
