#!/usr/bin/env python3
"""
Segmentation check against the column: lines that span both columns, lines of two printed lines,
and lines whose polygon stops short of text that stands in the column but is covered by no polygon.

Per column region the reference is taken from its full lines: median left edge, median right edge,
median width, median height. Per line (cons_body, cons_argument):
  wide     wider than 1.3 x the median width (runs into the other column)
  two_lines  Gemini's text has more than 1.5 x the column's median letters per pixel of width: the
           polygon holds the text of two printed lines (height alone misleads: opener lines with a large
           first word are rightly taller)
  short_r  right edge more than 40 px left of the column's right edge and the line goes on: uncovered
           ink (inside no polygon) starts within 30 px of the edge and runs on with gaps below 35 px
  short_l  the same at the left edge
  number   a line with a † or a number in Gemini's reading and a small block of uncovered ink (4 to 80 px
           wide) within 130 px left of the polygon: a marginal number outside the polygon
  text     kraken's reading shorter than 0.8 of Gemini's (letters only), a sign of a crop that misses text
--fix extends number polygons, and short_l / short_r ones only when the text flag confirms the loss
(on the image alone, show-through and neighbouring initials look like cut text), to the outermost uncovered ink in that band (+6 px) and writes
the PageXML in place (working copy only).

    python tools/audit_segmentation.py [--data output/merge/Baldo_29618397] [--fix]

Writes output/qa/segmentation.jsonl (one record per flagged line) and prints counts per print.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"; T = "{%s}" % NS
DATA = ROOT / "data" / "Baldo_29618397"
MERGE = ROOT / "output" / "merge"
INK = 0.02


def pts(s):
    return [(int(float(x)), int(float(y))) for x, y in (p.split(",") for p in s.split())]


def otsu(a):
    h = np.bincount(a.ravel(), minlength=256).astype(float); p = h / h.sum()
    w = np.cumsum(p); m = np.cumsum(p * np.arange(256))
    return int(np.argmax((m[-1] * w - m) ** 2 / (w * (1 - w) + 1e-12)))


def letters(t):
    return len(re.sub(r"[^A-Za-z]", "", t or ""))


READ = {}


def load_readings():
    f = MERGE / "kraken_current.jsonl"
    if f.exists():
        for l in f.open(encoding="utf-8"):
            r = json.loads(l); READ[(r["print"], r["page"], r["line_id"])] = (r["gemini"], r["kraken"])


def page(args):
    xml, fix = args
    if not READ:
        load_readings()
    xml = Path(xml); tree = ET.parse(xml); root = tree.getroot(); pg = root.find(T + "Page")
    img = next(p for p in (xml.parents[1] / pg.get("imageFilename"), DATA / xml.parts[-3] / pg.get("imageFilename")) if p.exists())
    g = np.asarray(Image.open(img).convert("L")); dark = g < otsu(g[::4, ::4]); H, W = dark.shape
    allp = [pts(tl.find(T + "Coords").get("points")) for tl in root.iter(T + "TextLine") if tl.find(T + "Coords") is not None]
    cover = Image.new("1", (W, H), 0); d = ImageDraw.Draw(cover)
    for p in allp:
        if len(p) >= 3:
            d.polygon(p, fill=1)
    cover = np.asarray(cover)
    out, changed = [], False
    for reg in root.iter(T + "TextRegion"):
        if "column" not in (reg.get("custom") or ""):
            continue
        L = []
        for tl in reg.findall(T + "TextLine"):
            c, b = tl.find(T + "Coords"), tl.find(T + "Baseline")
            if c is None or b is None:
                continue
            cp, bp = pts(c.get("points")), pts(b.get("points"))
            if len(cp) < 3 or not bp:
                continue
            ty = (re.search(r"type:(\w+)", tl.get("custom") or "") or [None, ""])[1]
            L.append((tl, cp, bp, ty))
        body = [(tl, cp) for tl, cp, bp, ty in L if ty == "cons_body"]
        if len(body) < 8:
            continue
        x0s = [min(x for x, _ in cp) for _, cp in body]; x1s = [max(x for x, _ in cp) for _, cp in body]
        left, right = statistics.median(x0s), statistics.median(x1s)
        width = statistics.median(b - a for a, b in zip(x0s, x1s))
        height = statistics.median(max(y for _, y in cp) - min(y for _, y in cp) for _, cp in body)
        dl = [letters(READ[(xml.parts[-3], xml.name, tl.get("id"))][0]) / max(max(x for x, _ in cp) - min(x for x, _ in cp), 1)
              for tl, cp, bp, ty in L if ty == "cons_body" and (xml.parts[-3], xml.name, tl.get("id")) in READ]
        dens = statistics.median(dl) if dl else None
        for tl, cp, bp, ty in L:
            if ty not in ("cons_body", "cons_argument"):
                continue
            x0, x1 = min(x for x, _ in cp), max(x for x, _ in cp)
            h = max(y for _, y in cp) - min(y for _, y in cp)
            by = int(statistics.median(y for _, y in bp))
            y0, y1 = max(0, by - int(0.45 * height)), min(H, by + int(0.1 * height))
            band = dark[y0:y1] & ~cover[y0:y1]
            k = (xml.parts[-3], xml.name, tl.get("id"))
            r = dict(print=k[0], page=k[1], line_id=k[2], ltype=ty, x0=x0, x1=x1, w=x1 - x0, h=h, col_w=round(width), col_h=round(height))
            flags = []
            if x1 - x0 > 1.3 * width:
                flags.append("wide")
            fix_l = fix_r = None

            def run(start, step, limit):
                """continuation of the line: uncovered ink that starts within 30 px of the edge and goes on
                with gaps below 35 px (word spaces); returns the far end or None"""
                x, last, gap, first = start, None, 0, None
                while 0 <= x < W and abs(x - start) < limit:
                    if band[:, x].mean() > 0.08:
                        if first is None:
                            first = abs(x - start)
                            if first > 30:
                                return None
                        last = x; gap = 0
                    else:
                        gap += 1
                        if first is not None and gap >= 35:
                            break
                        if first is None and abs(x - start) > 30:
                            return None
                    x += step
                return last if last is not None and abs(last - start) >= 25 else None
            if x1 < right - 40:
                e = run(x1 + 4, 1, int(right) + 20 - x1)
                if e is not None:
                    flags.append("short_r"); fix_r = e + 6
            if x0 > left + 40:
                e = run(x0 - 4, -1, x0 - int(left) + 20)
                if e is not None:
                    flags.append("short_l"); fix_l = e - 6
            gk = READ.get(k)
            # number: a small block of uncovered ink in the gutter beside a line that has a † or a number
            if gk and ty == "cons_body" and ("†" in gk[0] or "†" in gk[1] or re.match(r"\s*\d{1,2}\s", gk[0])):
                a = max(0, x0 - 130)
                cols = band[:, a:x0 - 4].mean(axis=0) if x0 - 4 > a else np.zeros(0)
                hit = np.nonzero(cols > 0.08)[0]
                if 4 <= hit.size <= 80 and hit.max() - hit.min() < 90:
                    flags.append("number"); fix_l = min(fix_l or 10 ** 9, a + int(hit.min()) - 6)
            if gk and letters(gk[0]) >= 20 and letters(gk[1]) < 0.8 * letters(gk[0]):
                flags.append("text")
            if gk and letters(gk[0]) >= 30 and dens and letters(gk[0]) / max(x1 - x0, 1) > 1.5 * dens:
                flags.append("two_lines")
            if not flags:
                continue
            r["flags"] = flags
            if fix_r:
                r["cand_r"] = fix_r
            if fix_l:
                r["cand_l"] = fix_l
            if "text" not in flags:          # a short edge is only trusted when kraken's text confirms the loss
                fix_r = None
                fix_l = fix_l if "number" in flags else None
            if fix and "wide" not in flags and "two_lines" not in flags and (fix_l or fix_r):
                c = [list(p) for p in cp]
                if fix_r:
                    c = [[fix_r, y] if x >= x1 - 15 else [x, y] for x, y in c]; r["fixed_r"] = fix_r
                if fix_l:
                    c = [[fix_l, y] if x <= x0 + 15 else [x, y] for x, y in c]; r["fixed_l"] = fix_l
                tl.find(T + "Coords").set("points", " ".join(f"{x},{y}" for x, y in c))
                tl.set("comments", ((tl.get("comments") or "") + "; polygon extended to uncovered text in the column").lstrip("; "))
                changed = True
            out.append(r)
    if changed:
        ET.register_namespace("", NS); tree.write(xml, encoding="UTF-8", xml_declaration=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=MERGE / "Baldo_29618397")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "qa" / "segmentation.jsonl")
    ap.add_argument("--fix", action="store_true")
    a = ap.parse_args()
    if a.fix and a.data.resolve() == DATA.resolve():
        raise SystemExit("--fix refuses to write data/")
    files = sorted(str(p) for p in a.data.glob("*/page/*.xml"))
    st = collections.Counter()
    with ProcessPoolExecutor(8) as ex, a.out.open("w", encoding="utf-8") as f:
        for recs in ex.map(page, [(x, a.fix) for x in files], chunksize=4):
            for r in recs:
                f.write(json.dumps(r) + "\n")
                for fl in r["flags"]:
                    st[(r["print"][-2:], fl)] += 1
                st[(r["print"][-2:], "fixed")] += ("fixed_l" in r or "fixed_r" in r)
    for v in sorted({k[0] for k in st}):
        print(v, {k[1]: n for k, n in sorted(st.items()) if k[0] == v})


if __name__ == "__main__":
    main()
