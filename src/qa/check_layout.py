"""
Layout audit of the Transkribus PageXML of every print.

Per print it writes output/qa/layout_<print>.json (every finding with page and ids) and prints a
summary table. Findings are heuristics for an editor to look at, not errors by themselves:

  no_text_lines        page has regions but no TextLine (prebuilt prints are expected to look so)
  no_columns           page has no region of type column_*
  odd_region_types     region types other than column_1/2, header, footer, marginalia, …
  empty_region         region without lines
  empty_line           TextLine without text
  untagged_line        annotated print: line without cons_title/cons_argument/cons_body tag
                       (such lines are dropped by AnnotatedConsiliumSegmenter)
  line_outside_region  line bbox centre lies outside its region bbox
  column_overlap       two column regions overlap by more than 20 % of the smaller one
  reading_order        column_2 ordered before column_1
  tall_line            line taller than 2.2 x the page median (merged initial, see fix_layout.py)
  heading_sequence     consilium numbers in cons_title lines that jump or repeat

Usage:
    python src/qa/check_layout.py                 # all prints
    python src/qa/check_layout.py --print Baldo_Cons_Print_Venice_1575_v4
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import find_prints, load_pages  # noqa: E402
from segment_consilia import _RE_HEADING, roman_to_int  # noqa: E402

OUT = ROOT / "output" / "qa"
KNOWN_REGION_TYPES = {"column_1", "column_2", "column_3", "header", "footer", "marginalia",
                      "page-number", "page_number", "catch-word", "signature-mark", "heading", "paragraph", ""}
ANNOTATED = {"cons_title", "cons_argument", "cons_body"}


def bbox(coords):
    pts = coords.points if coords else []
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def overlap_ratio(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])) or 1
    return inter / small


def audit_print(print_dir: Path) -> dict:
    pages = load_pages(print_dir)
    findings: list[dict] = []
    counts: Counter = Counter()
    n_lines_total = 0
    annotated = any(l.line_type in ANNOTATED for p in pages for r in p.regions for l in r.lines)
    headings: list[tuple[str, int]] = []

    def add(kind, page, **kw):
        counts[kind] += 1
        findings.append({"kind": kind, "page": page, **kw})

    for page in pages:
        pg = page.filename
        lines = [l for r in page.regions for l in r.lines]
        n_lines_total += len(lines)
        if page.regions and not lines:
            add("no_text_lines", pg)
        cols = [r for r in page.regions if r.region_type.startswith("column_")]
        if page.regions and not cols:
            add("no_columns", pg, types=[r.region_type for r in page.regions])
        for r in page.regions:
            if r.region_type not in KNOWN_REGION_TYPES and not r.region_type.startswith("column_"):
                add("odd_region_types", pg, region=r.region_id, type=r.region_type)
            if lines and not r.lines and r.region_type.startswith("column_"):
                add("empty_region", pg, region=r.region_id, type=r.region_type)
        # reading order of columns
        order = [(r.reading_order, r.region_type) for r in cols]
        if len(cols) >= 2:
            ro = {t: o for o, t in order}
            if "column_1" in ro and "column_2" in ro and ro["column_2"] < ro["column_1"]:
                add("reading_order", pg, order=order)
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    a, b = bbox(cols[i].coords), bbox(cols[j].coords)
                    if a and b and overlap_ratio(a, b) > 0.2:
                        add("column_overlap", pg, regions=[cols[i].region_id, cols[j].region_id],
                            ratio=round(overlap_ratio(a, b), 2))
        heights = []
        for r in page.regions:
            rb = bbox(r.coords)
            for l in r.lines:
                if not l.text.strip():
                    add("empty_line", pg, region=r.region_id, line=l.line_id)
                if annotated and r.region_type.startswith("column_") and l.line_type not in ANNOTATED \
                        and l.text.strip():
                    add("untagged_line", pg, region=r.region_id, line=l.line_id, text=l.text[:80])
                lb = bbox(l.coords)
                if lb:
                    heights.append(lb[3] - lb[1])
                    if rb:
                        cx, cy = (lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2
                        if not (rb[0] <= cx <= rb[2] and rb[1] <= cy <= rb[3]):
                            add("line_outside_region", pg, region=r.region_id, line=l.line_id)
                if l.line_type == "cons_title":
                    m = _RE_HEADING.match(l.text)
                    if m:
                        n = roman_to_int(m.group(1)) if not m.group(1).isdigit() else int(m.group(1))
                        if n:
                            headings.append((pg, n))
        if heights:
            med = statistics.median(heights)
            for r in page.regions:
                if not r.region_type.startswith("column_"):
                    continue
                for l in r.lines:
                    if l.line_type == "cons_title":
                        continue
                    lb = bbox(l.coords)
                    if lb and (lb[3] - lb[1]) > 2.2 * med and len(l.text) > 3:
                        add("tall_line", pg, line=l.line_id, height=lb[3] - lb[1], median=med,
                            text=l.text[:60])
    prev = 0
    for pg, n in headings:
        if n == prev + 1 or (prev == 0):
            pass
        elif n == prev:
            add("heading_sequence", pg, n=n, prev=prev, note="repeated")
        else:
            add("heading_sequence", pg, n=n, prev=prev, note="jump")
        prev = n
    return {
        "print": print_dir.name, "pages": len(pages), "lines": n_lines_total,
        "annotated": annotated, "headings": len(headings),
        "counts": dict(counts), "findings": findings,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--print", dest="print_filter")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for author, print_id, print_dir in find_prints(ROOT / "data"):
        if args.print_filter and print_id != args.print_filter:
            continue
        rep = audit_print(print_dir)
        (OUT / f"layout_{print_id}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                                     encoding="utf-8")
        rows.append(rep)
    kinds = sorted({k for r in rows for k in r["counts"]})
    print("print".ljust(38), "pages lines  " + " ".join(k[:14].rjust(14) for k in kinds))
    for r in rows:
        print(r["print"].ljust(38), str(r["pages"]).rjust(5), str(r["lines"]).rjust(6),
              " ".join(str(r["counts"].get(k, 0)).rjust(14) for k in kinds))
    md = ["# Layout audit", "", "| print | pages | lines | " + " | ".join(kinds) + " |",
          "|---|---|---|" + "---|" * len(kinds)]
    for r in rows:
        md.append(f"| {r['print']} | {r['pages']} | {r['lines']} | " + " | ".join(str(r["counts"].get(k, 0)) for k in kinds) + " |")
    md += ["", "Meaning of the columns: see the docstring of src/qa/check_layout.py. Details per finding in layout_<print>.json."]
    (OUT / "layout_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nDetails: {OUT}/layout_<print>.json, summary in layout_summary.md")


if __name__ == "__main__":
    main()
