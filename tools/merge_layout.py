#!/usr/bin/env python3
"""
Working copy of the Baldus PageXML that merges the local state (data/) with the Transkribus state and
extends the line polygons that miss their marginal number.

Base: data/Baldo_29618397/<print>/page/*.xml (all local improvements: relayout, Gemini reading, joined
opener lines, structure fixes). From the Transkribus download (tools/transkribus_download.py --out …):
  A  geometry (Coords, Baseline) of a line whose Transkribus polygon starts more than 20 px further left
     than the local one, where the local geometry is still the PageXML one (not a union from the relayout)
  B  structure type page_number where Transkribus has it and data/ turned it into header
Then:
  D  a polygon that stops short of its baseline (mainly v1: baselines were corrected, polygons not
     recomputed) is extended on that side to the baseline end plus 6 px, when the gap exceeds 15 px
  C  every cons_body line with a † in one of its three readings (PyLaia, Gemini, kraken) whose polygon
     ends at the text is extended into the gutter: the left edge moves to the median start of the
     already extended lines of the same column (at least three of them), else 60 px to the left, by at
     least 40 and at most 120 px; the baseline gets a first point there. The † marks the place of a
     marginal number. Line pieces that start more than 40 px inside the column are not moved (they
     would cover the text before them) but listed for review.

    python tools/merge_layout.py --transkribus <download dir> [--out output/merge/Baldo_29618397]

Writes the copy (PageXML only; images stay in data/), output/merge/changes.jsonl (one record per changed
line: print, page, line_id, step, old and new left x) and output/merge/review.jsonl (lines for review:
† with polygon already in the gutter but no number read; number read but no † in any reading).
data/ is not touched.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"; T = "{%s}" % NS
DATA = ROOT / "data" / "Baldo_29618397"
READINGS = ROOT / "output" / "qa" / "kraken_all_e31.jsonl"
EXTENDED = 25          # a line counts as extended when it starts this far left of the column's median start
NUM = re.compile(r"\s*(\d{1,2})\s")
# step A candidates rejected on the image (2026-10-23): the Transkribus polygon takes in text that is not
# part of the line ("Ad idem." beside a heading, a neighbouring line beside an opener initial)
REJECT_A = {
    ("Baldo_Cons_Print_Venice_1575_v1", "0228.xml", "tr_1765976316_tl_2"),
    ("Baldo_Cons_Print_Venice_1575_v1", "0278.xml", "tr_1765976242_tl_33_right2"),
    ("Baldo_Cons_Print_Venice_1575_v2", "page_214.xml", "line_1780406090053_1224"),
    ("Baldo_Cons_Print_Venice_1575_v2", "page_235.xml", "line_1780410093515_1668"),
}
ET.register_namespace("", NS)


def pts(s: str) -> list[list[int]]:
    return [[int(float(v)) for v in p.split(",")] for p in s.split()]


def fmt(p: list[list[int]]) -> str:
    return " ".join(f"{x},{y}" for x, y in p)


def ltype(tl) -> str:
    m = re.search(r"structure \{[^}]*type:([\w-]+)", tl.get("custom") or "")
    return m.group(1) if m else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--transkribus", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "merge" / "Baldo_29618397")
    a = ap.parse_args()
    readings = collections.defaultdict(dict)
    for l in READINGS.open(encoding="utf-8"):
        r = json.loads(l)
        readings[(r["print"], r["page"])][r["line_id"]] = r
    changes, review = [], []
    stats = collections.Counter()
    for src in sorted(DATA.glob("*/page/*.xml")):
        prt, page = src.parts[-3], src.name
        tree = ET.parse(src); root = tree.getroot()
        tk = {}
        tkf = a.transkribus / prt / "page" / page
        if tkf.exists():
            for tl in ET.parse(tkf).getroot().iter(T + "TextLine"):
                tk[tl.get("id")] = tl
        rd = readings.get((prt, page), {})
        for reg in root.iter(T + "TextRegion"):
            lines = [tl for tl in reg.findall(T + "TextLine") if tl.find(T + "Coords") is not None]
            for tl in lines:
                o = tk.get(tl.get("id"))
                if o is None or o.find(T + "Coords") is None:
                    continue
                # A: Transkribus polygon reaches further left
                if "geometry=union" not in (tl.get("comments") or "") and "geometry=interpolated" not in (tl.get("comments") or ""):
                    xl = min(x for x, _ in pts(tl.find(T + "Coords").get("points")))
                    xt = min(x for x, _ in pts(o.find(T + "Coords").get("points")))
                    if xt < xl - 20 and (prt, page, tl.get("id")) not in REJECT_A:
                        tl.find(T + "Coords").set("points", o.find(T + "Coords").get("points"))
                        if o.find(T + "Baseline") is not None and tl.find(T + "Baseline") is not None:
                            tl.find(T + "Baseline").set("points", o.find(T + "Baseline").get("points"))
                        changes.append(dict(print=prt, page=page, line_id=tl.get("id"), step="A transkribus geometry", old_x=xl, new_x=xt))
                        stats["A"] += 1
                # B: page_number tag
                if ltype(o) == "page_number" and ltype(tl) == "header":
                    tl.set("custom", (tl.get("custom") or "").replace("type:header", "type:page_number"))
                    changes.append(dict(print=prt, page=page, line_id=tl.get("id"), step="B page_number tag"))
                    stats["B"] += 1
                # D: polygon shorter than its baseline
                b = tl.find(T + "Baseline")
                if b is not None and len(pts(b.get("points"))) >= 1:
                    c = pts(tl.find(T + "Coords").get("points")); bp = pts(b.get("points"))
                    cx0, cx1 = min(x for x, _ in c), max(x for x, _ in c)
                    bx0, bx1 = min(x for x, _ in bp), max(x for x, _ in bp)
                    if cx0 - bx0 > 15:
                        c = [[bx0 - 6, y] if x <= cx0 + 15 else [x, y] for x, y in c]
                        changes.append(dict(print=prt, page=page, line_id=tl.get("id"), step="D polygon to baseline start", old_x=cx0, new_x=bx0 - 6)); stats["D left"] += 1
                    if bx1 - cx1 > 15:
                        c = [[bx1 + 6, y] if x >= cx1 - 15 else [x, y] for x, y in c]
                        changes.append(dict(print=prt, page=page, line_id=tl.get("id"), step="D polygon to baseline end", old_x=cx1, new_x=bx1 + 6)); stats["D right"] += 1
                    if cx0 - bx0 > 15 or bx1 - cx1 > 15:
                        tl.find(T + "Coords").set("points", fmt(c))
                        tl.set("comments", ((tl.get("comments") or "") + "; polygon extended to the baseline").lstrip("; "))
            if "column" not in (reg.get("custom") or "") or len(lines) < 10:
                continue
            xs = {tl.get("id"): min(x for x, _ in pts(tl.find(T + "Coords").get("points"))) for tl in lines}
            med = statistics.median(xs.values())
            ext = [x for x in xs.values() if x < med - EXTENDED]
            target = int(statistics.median(ext)) if len(ext) >= 3 else int(med - 60)
            for tl in lines:
                r = rd.get(tl.get("id"))
                if r is None or ltype(tl) != "cons_body":
                    continue
                texts = [r["gemini"], r["pylaia"], r["kraken"]]
                cross = any("†" in t for t in texts)
                number = any(NUM.match(t) for t in texts)
                extended = xs[tl.get("id")] < med - EXTENDED
                if cross and not extended and xs[tl.get("id")] > med + 40:
                    review.append(dict(print=prt, page=page, line_id=tl.get("id"), reason="† in a line piece that starts inside the column", **{k: r[k] for k in ("gemini", "pylaia", "kraken")}))
                elif cross and not extended:
                    # C: move the left edge into the gutter
                    c = pts(tl.find(T + "Coords").get("points")); x0 = min(x for x, _ in c)
                    new_x = max(min(target, x0 - 40), x0 - 120)
                    c = [[new_x, y] if x <= x0 + 15 else [x, y] for x, y in c]
                    tl.find(T + "Coords").set("points", fmt(c))
                    b = tl.find(T + "Baseline")
                    if b is not None:
                        bp = pts(b.get("points"))
                        if bp and bp[0][0] > new_x:
                            b.set("points", fmt([[new_x, bp[0][1]]] + bp))
                    tl.set("comments", ((tl.get("comments") or "") + f"; polygon extended to the gutter (†) {x0}->{new_x}").lstrip("; "))
                    changes.append(dict(print=prt, page=page, line_id=tl.get("id"), step="C extended for marginal number",
                                        old_x=x0, new_x=new_x, number_read=bool(number)))
                    stats["C"] += 1
                elif cross and extended and not number:
                    review.append(dict(print=prt, page=page, line_id=tl.get("id"), reason="† and polygon in the gutter, no number read", **{k: r[k] for k in ("gemini", "pylaia", "kraken")}))
                elif number and not cross:
                    review.append(dict(print=prt, page=page, line_id=tl.get("id"), reason="number read, no † († missing or the line is a summary)", **{k: r[k] for k in ("gemini", "pylaia", "kraken")}))
        dst = a.out / prt / "page" / page
        dst.parent.mkdir(parents=True, exist_ok=True)
        tree.write(dst, encoding="UTF-8", xml_declaration=True)
    for prt_dir in DATA.iterdir():
        for f in ("metadata.xml", "mets.xml"):
            if (prt_dir / f).exists() and (a.out / prt_dir.name).exists():
                shutil.copy(prt_dir / f, a.out / prt_dir.name / f)
    base = a.out.parent
    with (base / "changes.jsonl").open("w", encoding="utf-8") as f:
        for c in changes:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with (base / "review.jsonl").open("w", encoding="utf-8") as f:
        for c in review:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(dict(stats), "review:", dict(collections.Counter(r["reason"] for r in review)))


if __name__ == "__main__":
    main()
