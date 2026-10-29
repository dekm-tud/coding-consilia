#!/usr/bin/env python3
"""
Decide the marginal number of every body line from the readings and the numbering sequence.

Within a consilium the marginal numbers run 1, 2, 3 ... in reading order, and each † in the body text
marks the place of one. For every cons_body line of the working copy (output/merge) the candidates are
the numbers Gemini and kraken read at the line start (PyLaia is not used: its line crops do not reach
the gutter). The expected number is the previous numbered body line + 1 and the next one - 1 (reset at a
consilium title).
  opening line:     the first body line after title and summaries carries 1 when a reading has it
  line with a †:    the candidate that fits the sequence wins; otherwise a single candidate, otherwise
                    Gemini's; no reading: no number (inferring it from the sequence alone, or from the
                    count of summaries, was wrong too often on the checked lines: numbers shift by a line
                    where two † stand close together)
  line without a †: a number only when Gemini and kraken read the same one and it fits the sequence,
                    otherwise none
    python tools/margin_numbers.py [--check]     # --check scores the rule against the gold and review answers

Writes output/merge/margin_numbers.jsonl (print, page, line_id, cross, gemini, kraken, expected, decision,
how). Nothing is written to the PageXML.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NS = "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}"
MERGE = ROOT / "output" / "merge"
NUM = re.compile(r"\s*(\d{1,2})\s")


def num(t):
    m = NUM.match(t or "")
    return int(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    cur = {}
    for l in (MERGE / "kraken_current.jsonl").open(encoding="utf-8"):
        r = json.loads(l); cur[(r["print"], r["page"], r["line_id"])] = r
    out = []
    for prt_dir in sorted((MERGE / "Baldo_29618397").iterdir()):
        seq = []
        for f in sorted((prt_dir / "page").glob("*.xml")):
            for reg in ET.parse(f).getroot().iter(NS + "TextRegion"):
                for tl in reg.findall(NS + "TextLine"):
                    ty = (re.search(r"type:(\w+)", tl.get("custom") or "") or [None, ""])[1]
                    seq.append(((prt_dir.name, f.name, tl.get("id")), ty))
        # first pass: numbers that two sources share are anchors of the sequence
        read = {}
        for k, ty in seq:
            r = cur.get(k)
            if r and ty == "cons_body":
                read[k] = (num(r["gemini"]), num(r["kraken"]), any("†" in r[s] for s in ("gemini", "pylaia", "kraken")))

        # the opening line of a consilium (first body line after its title and summaries) carries number 1
        opener = set(); last = None
        for k, ty in seq:
            if ty == "cons_title":
                last = "title"
            elif ty == "cons_argument" and last in ("title", "arg"):
                last = "arg"
            elif ty == "cons_body":
                if last in ("title", "arg"):
                    opener.add(k)
                last = "body"

        # consilia whose numbered summaries match the † of the body one to one
        by_count = {}
        cons, c = [], None
        for k, ty in seq:
            if ty == "cons_title":
                c = dict(args=[], body=[]); cons.append(c)
            elif c is not None and k in cur:
                if ty == "cons_argument":
                    c["args"].append(k)
                elif ty == "cons_body":
                    c["body"].append(k)
        for c in cons:
            ns = []
            for k in c["args"]:
                r = cur[k]
                v = collections.Counter(x for x in (num(r["gemini"]), num(r["pylaia"]), num(r["kraken"])) if x).most_common(1)
                if v:
                    ns.append(v[0][0])
            crosses = [(k, max(cur[k][s].count("†") for s in ("gemini", "pylaia", "kraken"))) for k in c["body"]]
            if ns and max(ns) == sum(n for _, n in crosses):
                j = 0
                for k, n in crosses:
                    if n:
                        by_count[k] = j + 1; j += n

        def neighbour(i, step):
            j = i + step
            while 0 <= j < len(seq):
                k, ty = seq[j]
                if ty == "cons_title":
                    return 0 if step < 0 else None
                if k in opener and step < 0 and k in read and 1 in read[k][:2]:
                    return 1
                if k in read:
                    g, kr, _ = read[k]
                    n = g if g is not None and (kr is None or kr == g) else (kr if g is None else None)
                    if n:
                        return n
                j += step
            return None
        for i, (k, ty) in enumerate(seq):
            if k not in read:
                continue
            g, kr, cross = read[k]
            p, n = neighbour(i, -1), neighbour(i, 1)
            exp = {x for x in ((p + 1) if p is not None else None, (n - 1) if n is not None else None) if x}
            cands = {x for x in (g, kr) if x}
            fit = [x for x in cands if x in exp]
            if k in opener and 1 in cands:
                d, how = 1, "opening line, read as 1"
            elif cross:
                if len(fit) == 1:
                    d, how = fit[0], "read, fits the sequence"
                elif not cands:
                    d, how = None, "† but no number read"
                elif len(cands) == 1:
                    d, how = next(iter(cands)), "read once, does not fit the sequence"
                elif g is not None:
                    d, how = g, "readings differ, neither fits: Gemini"
                else:
                    d, how = "open", "open"
            else:
                if g is not None and g == kr and g in exp:
                    d, how = g, "no †, read twice, fits"
                else:
                    d, how = None, "no †, no number"
            out.append(dict(print=k[0], page=k[1], line_id=k[2], cross=cross, gemini=g, kraken=kr,
                            expected=sorted(exp), decision=d, how=how))
    with (MERGE / "margin_numbers.jsonl").open("w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    c = collections.Counter(o["how"] for o in out)
    print("body lines:", len(out), dict(c))
    changed = sum(1 for o in out if o["decision"] not in ("open", o["gemini"]))
    print("decision differs from Gemini's number:", changed)
    if a.check:
        D = {(o["print"], o["page"], o["line_id"]): o for o in out}
        for name, sp, ap_ in (("gold", "output/gold/sample.json", "output/gold/gold.jsonl"),
                              ("review", "output/gold/review/sample.json", "output/gold/review/gold.jsonl")):
            S = {s["id"]: s for s in json.load(open(ROOT / sp))}
            A = {}
            for l in open(ROOT / ap_):
                x = json.loads(l); A[x["id"]] = x
            cc = collections.Counter(); wrong = []
            for i, x in A.items():
                s = S[i]; o = D.get((s["print"], s["page"], s["line_id"]))
                if not o:
                    continue
                t = num(x["gold"]); cc["lines"] += 1
                if o["decision"] == "open":
                    cc["open"] += 1
                elif o["decision"] == t:
                    cc["right"] += 1
                else:
                    cc["wrong"] += 1; wrong.append((i, o["how"], o["gemini"], o["kraken"], o["expected"], t, x["gold"][:45]))
            print(name, dict(cc))
            for w in wrong:
                print("   ", w)


if __name__ == "__main__":
    main()
