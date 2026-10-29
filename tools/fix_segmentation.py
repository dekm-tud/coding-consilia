#!/usr/bin/env python3
"""
Repair the segmentation errors listed by tools/audit_segmentation.py in the working copy (output/merge).

  wide       a polygon that runs across the gutter into the other column, where that column has its own
             line at the same height: the polygon and baseline are clipped to the line's own column (at
             the middle of the gutter). If the line's Gemini text begins or ends with the other column's
             line, that part is removed from the text.
  tall       a polygon that reaches down (or up) over a neighbouring line of its own column: clipped at
             the neighbour's edge
  duplicate  a line whose Gemini text is the text of another line on the page and whose polygon lies in
             a different column than that line: removed
  short_r/l  a polygon that stops before the end of its line, confirmed by kraken's shorter text:
             extended to the end of the continuing ink (as found by audit_segmentation)

    python tools/fix_segmentation.py [--list output/merge/split_needed.jsonl] [--audit output/qa/segmentation_check.jsonl]

Writes the PageXML of output/merge in place and output/merge/segmentation_fixes.jsonl (one record per
change, with the old geometry). data/ is not touched. Re-read the changed lines with kraken afterwards.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from gold_eval import fold  # noqa: E402

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"; T = "{%s}" % NS
MERGE = ROOT / "output" / "merge"
ET.register_namespace("", NS)


def pts(s):
    return [(float(x), float(y)) for x, y in (p.split(",") for p in s.split())]


def fmt(p):
    return " ".join(f"{int(round(x))},{int(round(y))}" for x, y in p)


def clip(poly, keep):
    """Sutherland-Hodgman against one half-plane; keep(p) -> signed distance (>= 0 inside)"""
    out = []
    for i, cur in enumerate(poly):
        prev = poly[i - 1]
        dc, dp = keep(cur), keep(prev)
        if dc >= 0:
            if dp < 0:
                t = dp / (dp - dc); out.append((prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])))
            out.append(cur)
        elif dp >= 0:
            t = dp / (dp - dc); out.append((prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])))
    return out


def clip_line(points, x_lo, x_hi):
    """baseline clipped to x_lo..x_hi, with interpolated end points"""
    out = []
    for i, (x, y) in enumerate(points):
        if x_lo <= x <= x_hi:
            out.append((x, y))
        if i + 1 < len(points):
            x2, y2 = points[i + 1]
            for b in (x_lo, x_hi):
                if (x - b) * (x2 - b) < 0:
                    out.append((b, y + (y2 - y) * (b - x) / (x2 - x)))
    return sorted(out)


def gemini_of(tl):
    tes = tl.findall(T + "TextEquiv")
    for te in tes:
        if te.get("comments") == "gemini":
            return te
    return tes[0] if "text=gemini" in (tl.get("comments") or "") and tes else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", type=Path, default=MERGE / "split_needed.jsonl")
    ap.add_argument("--audit", type=Path, default=ROOT / "output" / "qa" / "segmentation_check.jsonl")
    a = ap.parse_args()
    todo = {}
    for l in a.list.open(encoding="utf-8"):
        r = json.loads(l); todo.setdefault((r["print"], r["page"]), []).append(r)
    for l in a.audit.open(encoding="utf-8"):
        r = json.loads(l)
        if ("short_r" in r["flags"] or "short_l" in r["flags"]) and "text" in r["flags"] and "wide" not in r["flags"]:
            todo.setdefault((r["print"], r["page"]), []).append(dict(r, flags=[f for f in r["flags"] if f != "two_lines"] + ["edge"]))
    log = []
    for (prt, page), items in sorted(todo.items()):
        f = MERGE / "Baldo_29618397" / prt / "page" / page
        tree = ET.parse(f); root = tree.getroot()
        regs = [(reg, (re.search(r"type:(\w+)", reg.get("custom") or "") or [0, ""])[1]) for reg in root.iter(T + "TextRegion")]
        cols = {ty: reg for reg, ty in regs if ty in ("column_1", "column_2")}
        if len(cols) == 2:
            r1 = max(x for x, _ in pts(cols["column_1"].find(T + "Coords").get("points")))
            l2 = min(x for x, _ in pts(cols["column_2"].find(T + "Coords").get("points")))
            r1_lines = [max(x for x, _ in pts(t.find(T + "Coords").get("points"))) for t in cols["column_1"].findall(T + "TextLine")]
            l2_lines = [min(x for x, _ in pts(t.find(T + "Coords").get("points"))) for t in cols["column_2"].findall(T + "TextLine")]
            import statistics
            gx = (statistics.median(r1_lines) + statistics.median(l2_lines)) / 2 if r1_lines and l2_lines else (r1 + l2) / 2
        else:
            gx = None
        changed = False
        for it in items:
            home = next(((reg, ty) for reg, ty in regs if any(t.get("id") == it["line_id"] for t in reg.findall(T + "TextLine"))), None)
            if home is None:
                continue
            reg, ty = home
            tl = next(t for t in reg.findall(T + "TextLine") if t.get("id") == it["line_id"])
            c_el, b_el = tl.find(T + "Coords"), tl.find(T + "Baseline")
            old = dict(coords=c_el.get("points"), baseline=b_el.get("points") if b_el is not None else None)
            cp = pts(c_el.get("points"))
            ya, yb = min(y for _, y in cp), max(y for _, y in cp)
            # duplicate of a line in the other column
            g_el = gemini_of(tl); g = fold((g_el.find(T + "Unicode").text or "") if g_el is not None else "", "letters")
            others = [(t, r2, ty2) for r2, ty2 in regs for t in r2.findall(T + "TextLine") if t is not tl]
            dup = None
            for t, r2, ty2 in others:
                ge = gemini_of(t)
                if ge is not None and g and len(g) > 15 and fold(ge.find(T + "Unicode").text or "", "letters") == g and r2 is not reg:
                    dup = t
            if dup is not None:
                reg.remove(tl); changed = True
                log.append(dict(print=prt, page=page, line_id=it["line_id"], fix="duplicate removed", same_as=dup.get("id"), **old)); continue
            if "wide" in it["flags"] and gx is not None:
                xs = [x for x, _ in cp]
                own_left = (sum(1 for x in xs if x < gx) >= sum(1 for x in xs if x >= gx)) if ty not in cols else ty == "column_1"
                keep = (lambda p: gx - p[0]) if own_left else (lambda p: p[0] - gx)
                new = clip(cp, keep)
                if len(new) >= 3:
                    c_el.set("points", fmt(new))
                    if b_el is not None:
                        bl = clip_line(pts(b_el.get("points")), -1e9 if own_left else gx, gx if own_left else 1e9)
                        if len(bl) >= 2:
                            b_el.set("points", fmt(bl))
                    # text: drop the other column's line if Gemini's reading carries it
                    other = cols.get("column_2" if own_left else "column_1")
                    if other is not None and g_el is not None:
                        u = g_el.find(T + "Unicode"); text = u.text or ""
                        for t in other.findall(T + "TextLine"):
                            tp = pts(t.find(T + "Coords").get("points"))
                            if min(yb, max(y for _, y in tp)) - max(ya, min(y for _, y in tp)) <= 0:
                                continue
                            ge = gemini_of(t)
                            ot = (ge.find(T + "Unicode").text or "").strip() if ge is not None else ""
                            if ot and text.strip().startswith(ot):
                                u.text = text.strip()[len(ot):].strip()
                            elif ot and text.strip().endswith(ot):
                                u.text = text.strip()[: -len(ot)].strip()
                    tl.set("comments", ((tl.get("comments") or "") + "; clipped to its column (polygon ran across the gutter)").lstrip("; "))
                    changed = True
                    log.append(dict(print=prt, page=page, line_id=it["line_id"], fix="clipped to its column", gutter_x=round(gx), **old))
            if "two_lines" in it["flags"]:
                # clip at the neighbour of the same column that the polygon overlaps most
                best = None
                for t in reg.findall(T + "TextLine"):
                    if t is tl:
                        continue
                    tp = pts(t.find(T + "Coords").get("points")); ta, tb = min(y for _, y in tp), max(y for _, y in tp)
                    ov = min(yb, tb) - max(ya, ta)
                    if ov > 0.4 * (tb - ta) and (best is None or ov > best[0]):
                        best = (ov, ta, tb)
                if best:
                    _, ta, tb = best
                    if ta > ya:
                        new = clip(cp, lambda p, cut=ta + 2: cut - p[1])
                    else:
                        new = clip(cp, lambda p, cut=tb - 2: p[1] - cut)
                    if len(new) >= 3:
                        c_el.set("points", fmt(new)); changed = True
                        tl.set("comments", ((tl.get("comments") or "") + "; clipped at the neighbouring line (polygon covered two lines)").lstrip("; "))
                        log.append(dict(print=prt, page=page, line_id=it["line_id"], fix="clipped at the neighbouring line", **old))
            if "edge" in it["flags"]:
                # same continuation rule as audit_segmentation: re-derived there; here only the right/left edge is moved
                fr = it.get("cand_r"); fl = it.get("cand_l")
                if fr or fl:
                    x0, x1 = min(x for x, _ in cp), max(x for x, _ in cp)
                    new = [((fr if fr and x >= x1 - 15 else (fl if fl and x <= x0 + 15 else x)), y) for x, y in cp]
                    c_el.set("points", fmt(new)); changed = True
                    tl.set("comments", ((tl.get("comments") or "") + "; polygon extended to the end of its line").lstrip("; "))
                    log.append(dict(print=prt, page=page, line_id=it["line_id"], fix="extended to the end of its line", **old))
        if changed:
            tree.write(f, encoding="UTF-8", xml_declaration=True)
    with (MERGE / "segmentation_fixes.jsonl").open("w", encoding="utf-8") as fo:
        for r in log:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    import collections
    print(dict(collections.Counter(r["fix"] for r in log)))


if __name__ == "__main__":
    main()
