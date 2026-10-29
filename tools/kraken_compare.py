#!/usr/bin/env python3
"""
Compare the three readings per line (PyLaia, Gemini, kraken) from tools/kraken_read_all.py and write a
report. Nothing is written to the PageXML.

    python tools/kraken_compare.py [--in output/qa/kraken_all_e31.jsonl] [--out output/qa/kraken_compare_e31.md]

Comparison after the loose normalisation of src/qa/compare_ocr.py (no spaces, no punctuation, long s,
ligatures). Groups per line: all three equal, kraken = Gemini only, kraken = PyLaia only, Gemini = PyLaia
but kraken differs, all three differ. Reported per print, per kraken split (train / val / other) and per
earlier decision (agent, lexicon, kraken epoch 0, manual review, none); CER of kraken against each
reading; samples per group. A JSON summary is written next to the report.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "qa"))
from compare_ocr import normalise, dist  # noqa: E402


def decided_by(cm: str) -> str:
    if "reviewed " not in cm:
        return "undecided" if True else ""
    tail = cm.split("reviewed ", 1)[1]
    for k in ("agent", "lexicon", "kraken"):
        if f": {k} " in tail or f": {k}" in tail[:30]:
            return k
    return "manual"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in", dest="inp", type=Path, default=ROOT / "output" / "qa" / "kraken_all_e31.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "qa" / "kraken_compare_e31.md")
    a = ap.parse_args()
    rows = [json.loads(l) for l in a.inp.open(encoding="utf-8")]
    groups = collections.Counter(); by_print = collections.defaultdict(collections.Counter)
    by_split = collections.defaultdict(collections.Counter); by_dec = collections.defaultdict(collections.Counter)
    err = collections.Counter(); samples = collections.defaultdict(list)
    for r in rows:
        g, p, k = normalise(r["gemini"], True), normalise(r["pylaia"], True), normalise(r["kraken"], True)
        if g == p == k:
            grp = "all three equal"
        elif g == p:
            grp = "Gemini = PyLaia, kraken differs"
        elif k == g:
            grp = "kraken = Gemini"
        elif k == p:
            grp = "kraken = PyLaia"
        else:
            grp = "all three differ"
        r["group"] = grp
        groups[grp] += 1; by_print[r["print"]][grp] += 1
        by_split[r.get("kraken_split") or "not in kraken data"][grp] += 1
        disputed = g != p
        dec = decided_by(r.get("comment", "")) if disputed else "readings agreed"
        by_dec[dec][grp] += 1
        err["chars_g"] += len(g); err["kg"] += dist(k, g); err["chars_p"] += len(p); err["kp"] += dist(k, p)
        if disputed:
            err["d_chars_g"] += len(g); err["d_kg"] += dist(k, g); err["d_chars_p"] += len(p); err["d_kp"] += dist(k, p)
        samples[grp].append(r)
    random.seed(11)
    order = ["all three equal", "Gemini = PyLaia, kraken differs", "kraken = Gemini", "kraken = PyLaia", "all three differ"]
    L = [f"# Three readings per line: PyLaia, Gemini, kraken (epoch 31)\n", f"Lines read: {len(rows):,}\n",
         "Comparison after loose normalisation (spaces, punctuation, long s and ligatures ignored).\n",
         "\n## Groups\n", "| group | lines | share |", "|---|---|---|"]
    for o in order:
        L.append(f"| {o} | {groups[o]:,} | {groups[o] / max(len(rows), 1) * 100:.1f} % |")
    L += ["\n## CER of kraken\n", "| against | all lines | lines where Gemini and PyLaia differ |", "|---|---|---|",
          f"| Gemini | {err['kg'] / max(err['chars_g'], 1) * 100:.2f} % | {err['d_kg'] / max(err['d_chars_g'], 1) * 100:.2f} % |",
          f"| PyLaia | {err['kp'] / max(err['chars_p'], 1) * 100:.2f} % | {err['d_kp'] / max(err['d_chars_p'], 1) * 100:.2f} % |"]
    def table(title, d):
        out = [f"\n## {title}\n", "| | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
        for key in sorted(d):
            out.append(f"| {key} | " + " | ".join(f"{d[key][o]:,}" for o in order) + " |")
        return out
    L += table("Per print", by_print)
    L += table("Per kraken split (train lines are circular: kraken learnt them from the Gemini/PyLaia consensus)", by_split)
    L += table("Lines where Gemini and PyLaia differ, by earlier decision", {k: v for k, v in by_dec.items() if k != "readings agreed"})
    L.append("\n## Samples\n")
    for o in order[1:]:
        L.append(f"\n### {o}\n")
        for r in random.sample(samples[o], min(8, len(samples[o]))):
            L.append(f"- {r['print'][-2:]} {r['page']} {r['line_id']} (valid: {r['valid']}, conf {r['conf']})  ")
            L.append(f"  G: `{r['gemini'][:110]}`  ")
            L.append(f"  P: `{r['pylaia'][:110]}`  ")
            L.append(f"  K: `{r['kraken'][:110]}`")
    a.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump({"lines": len(rows), "groups": groups, "per_print": by_print, "per_split": by_split, "per_decision": by_dec,
               "cer": {k: v for k, v in err.items()}}, a.out.with_suffix(".json").open("w"), indent=1, default=dict)
    print(f"report: {a.out}")


if __name__ == "__main__":
    main()
