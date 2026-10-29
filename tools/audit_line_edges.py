#!/usr/bin/env python3
"""
Image check of the line polygons: is a printed character cut off at the left or right end of a line,
or is there ink in the gutter beside the line that no polygon covers (a marginal number)?

For every TextLine the letter band is taken from the baseline (from the baseline up to 0.6 of the
polygon height, plus 0.15 below it). The page is binarised with Otsu's threshold. Measured per side:
  edge    dark share in the 8 px just outside the polygon and in the 4 px just inside it; both above
          4 % means ink runs across the boundary: a cut character
  overhang  length of uncovered ink (not inside any other polygon) that continues the line beyond
          the polygon edge without a gap of 14 px; more than 24 px (the side margin of the kraken crop)
          counts as cut (cut_left / cut_right)
  gutter  dark share in the strip 12 to 90 px outside the polygon (left side only), excluding pixels
          covered by any other line polygon of the page: uncovered ink beside the line
    python tools/audit_line_edges.py [--data output/merge/Baldo_29618397] [--out output/qa/line_edges.jsonl] [--fix]

--fix moves the cut edge of a polygon out to the end of the overhanging ink (at most 150 px) and writes
the PageXML in place; it refuses to run on data/.

One record per line with the measures and flags; a summary per print on stdout.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"; T = "{%s}" % NS
DATA = ROOT / "data" / "Baldo_29618397"
EDGE_OUT, EDGE_IN, GUT0, GUT1, INK = 8, 4, 12, 90, 0.04
GAP = 14       # px of empty columns that end a run of ink (word space ~ 15-25 px on these scans)
FIX_MAX = 150  # --fix extends a polygon over at most this much overhanging ink
SIDE = 24      # kraken crops add this much on both sides; ink beyond it is lost to the recogniser


def pts(s):
    return [(int(float(x)), int(float(y))) for x, y in (p.split(",") for p in s.split())]


def otsu(a: np.ndarray) -> int:
    h = np.bincount(a.ravel(), minlength=256).astype(float); p = h / h.sum()
    w = np.cumsum(p); m = np.cumsum(p * np.arange(256)); mt = m[-1]
    s = (mt * w - m) ** 2 / (w * (1 - w) + 1e-12)
    return int(np.argmax(s))


def page(xml: str, fix: bool = False) -> list[dict]:
    xml = Path(xml); tree = ET.parse(xml); root = tree.getroot(); pg = root.find(T + "Page")
    img = next(p for p in (xml.parents[1] / pg.get("imageFilename"), DATA / xml.parts[-3] / pg.get("imageFilename")) if p.exists())
    g = np.asarray(Image.open(img).convert("L"))
    dark = g < otsu(g[::4, ::4])
    H, W = dark.shape
    lines = []
    for reg in root.iter(T + "TextRegion"):
        for tl in reg.findall(T + "TextLine"):
            c, b = tl.find(T + "Coords"), tl.find(T + "Baseline")
            if c is None or b is None:
                continue
            cp, bp = pts(c.get("points")), pts(b.get("points"))
            if len(cp) < 3 or not bp:
                continue
            lines.append((reg, tl, cp, bp))
    cover = Image.new("1", (W, H), 0); d = ImageDraw.Draw(cover)
    for _, _, cp, _ in lines:
        d.polygon(cp, fill=1)
    cover = np.asarray(cover)
    out = []
    for reg, tl, cp, bp in lines:
        x0, x1 = min(x for x, _ in cp), max(x for x, _ in cp)
        h = max(y for _, y in cp) - min(y for _, y in cp)
        by = int(statistics.median(y for _, y in bp))
        y0, y1 = max(0, by - int(0.6 * h)), min(H, by + int(0.15 * h))
        if y1 - y0 < 5:
            continue

        def share(a, b2, excl=False):
            a, b2 = max(0, a), min(W, b2)
            if b2 <= a:
                return 0.0
            blk = dark[y0:y1, a:b2]
            if excl:
                blk = blk & ~cover[y0:y1, a:b2]
            return float(blk.mean())
        r = dict(print=xml.parts[-3], page=xml.name, line_id=tl.get("id"), region=reg.get("id"),
                 ltype=(tl.get("custom") or "").split("type:")[-1].split(";")[0].split("}")[0] if "type:" in (tl.get("custom") or "") else "",
                 out_l=round(share(x0 - EDGE_OUT, x0), 3), in_l=round(share(x0, x0 + EDGE_IN), 3),
                 out_r=round(share(x1, x1 + EDGE_OUT), 3), in_r=round(share(x1 - EDGE_IN, x1), 3),
                 gutter=round(share(x0 - GUT1, x0 - GUT0, excl=True), 3))
        # overhang: uncovered ink that continues the line beyond its edge without a gap of GAP px
        def overhang(start, step):
            run = last = 0; x = start
            while 0 <= x < W and run < GAP and abs(x - start) < 400:
                col = dark[y0:y1, x] & ~cover[y0:y1, x]
                if col.mean() > 0.06:
                    last = abs(x - start) + 1; run = 0
                else:
                    run += 1
                x += step
            return last
        r["over_l"] = overhang(x0 - 1, -1); r["over_r"] = overhang(x1 + 1, 1)
        r["cut_left"] = r["over_l"] > SIDE
        r["cut_right"] = r["over_r"] > SIDE
        if fix and (r["cut_left"] or r["cut_right"]):
            # move the edge points out to the end of the overhanging ink, at most FIX_MAX px
            c = [list(p) for p in cp]
            if r["cut_left"] and r["over_l"] <= FIX_MAX:
                nx = x0 - r["over_l"] - 6
                c = [[nx, y] if x <= x0 + 15 else [x, y] for x, y in c]; r["fixed_left"] = nx
            if r["cut_right"] and r["over_r"] <= FIX_MAX:
                nx = x1 + r["over_r"] + 6
                c = [[nx, y] if x >= x1 - 15 else [x, y] for x, y in c]; r["fixed_right"] = nx
            if "fixed_left" in r or "fixed_right" in r:
                tl.find(T + "Coords").set("points", " ".join(f"{x},{y}" for x, y in c))
                tl.set("comments", ((tl.get("comments") or "") + "; polygon extended over cut ink").lstrip("; "))
        out.append(r)
    if fix and any("fixed_left" in r or "fixed_right" in r for r in out):
        ET.register_namespace("", NS)
        tree.write(xml, encoding="UTF-8", xml_declaration=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=ROOT / "output" / "merge" / "Baldo_29618397")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "qa" / "line_edges.jsonl")
    ap.add_argument("--fix", action="store_true", help="extend cut polygons in place (only on a working copy, never on data/)")
    a = ap.parse_args()
    if a.fix and a.data.resolve() == DATA.resolve():
        raise SystemExit("--fix refuses to write data/")
    files = sorted(str(p) for p in a.data.glob("*/page/*.xml"))
    stats = collections.Counter()
    with ProcessPoolExecutor(8) as ex, a.out.open("w", encoding="utf-8") as f:
        for recs in ex.map(page, files, [a.fix] * len(files), chunksize=4):
            for r in recs:
                f.write(json.dumps(r) + "\n")
                v = r["print"][-2:]
                stats[(v, "lines")] += 1; stats[(v, "cut left")] += r["cut_left"]; stats[(v, "cut right")] += r["cut_right"]
                stats[(v, "gutter ink")] += r["gutter"] > 0.03
                stats[(v, "fixed")] += ("fixed_left" in r or "fixed_right" in r)
    for v in sorted({k[0] for k in stats}):
        print(v, {k[1]: stats[k] for k in stats if k[0] == v})


if __name__ == "__main__":
    main()
