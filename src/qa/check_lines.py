"""
Line-level layout check of the PageXML: are the lines of every column evenly spaced and complete?

check_layout.py looks at regions and tags; this script looks at the vertical rhythm of the text
lines inside each column region. A column of a sixteenth-century folio print is set on a fixed
line pitch, so missing, merged or misplaced lines show up as irregularities of that rhythm.

Per column (region of type column_*), from the baselines (fallback: the median y of the line
polygon):
  pitch           median distance between consecutive lines in this column
  expected        column height / pitch, compared with the number of lines actually present
Findings (heuristics; kinds and the threshold that fires them):
  missing_line    gap of >= 1.85 pitch between two body lines of normal length that are not next
                  to a heading or argumentum line (a line was not detected, or two were merged)
  merged_line     more than 1.6 x the characters per pixel typical of the column, or a polygon
                  taller than 1.8 x the column's median line height with 1.3 x the typical
                  characters (two printed lines read as one); tall polygons alone are normal
                  around initials and summaria
  overlap_line    two consecutive text lines (both > 20 characters, horizontally overlapping)
                  closer than 0.45 pitch (duplicate detection)
  wide_line       line wider than 1.12 x the column width (runs into the gutter or the other column)
  sparse_text     full-width line of normal height with fewer than 55 % of the characters per pixel
                  typical of the column (text recognition dropped part of the line); large-type
                  openers such as "In Christi nomine" are excluded by their height
  top_gap / bottom_gap
                  first/last baseline more than 1.5 pitch higher/lower than on the neighbouring
                  pages of the same parity (text block edge drifts slowly through a volume), so
                  lines at the top or bottom margin were probably not detected
  short_column    column region shorter than 80 % of the volume's median column height while the
                  page has a normal number of lines in the other column (region cut off)
  few_lines       fewer than 82 % of the expected lines for the region height, or more than 8
                  lines fewer than the same column on the neighbouring pages

Every finding carries page, region, line ids and a short text sample. Pages are ranked by a score
(sum of findings weighted by severity) so that an editor can look at the worst pages first.

Usage:
    python src/qa/check_lines.py                              # all prints with text lines
    python src/qa/check_lines.py --print Baldo_Cons_Print_Venice_1575_v1 --images 20
        --images N   writes output/qa/lines/<print>/<page>.jpg for the N worst pages, baselines in
                     green, flagged lines in red, columns in blue (needs the page JPGs in data/)
Output: output/qa/lines_<print>.json, output/qa/lines_summary.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pipeline import find_prints  # noqa: E402

OUT = ROOT / "output" / "qa"
NS = "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}"
WEIGHT = {"missing_line": 3, "merged_line": 3, "overlap_line": 1, "wide_line": 2, "sparse_text": 2,
          "top_gap": 2, "bottom_gap": 2, "short_column": 4, "few_lines": 4}
HEAD_TYPES = {"cons_title", "cons_argument"}


def pts(s: str) -> list[tuple[int, int]]:
    return [tuple(int(v) for v in p.split(",")) for p in (s or "").split() if "," in p]


def custom_type(custom: str) -> str:
    import re
    m = re.search(r"structure\s*\{[^}]*type:([^;}]+)", custom or "")
    return m.group(1).strip() if m else ""


def read_page(xml: Path) -> dict:
    """Columns with their lines: baseline y, polygon bbox, text, type."""
    root = ET.parse(xml).getroot()
    page = root.find(f"{NS}Page")
    out = {"file": xml.name, "image": page.get("imageFilename"), "W": int(page.get("imageWidth", 0)),
           "H": int(page.get("imageHeight", 0)), "columns": []}
    for tr in page.findall(f"{NS}TextRegion"):
        rtype = custom_type(tr.get("custom", ""))
        if not rtype.startswith("column_"):
            continue
        c = tr.find(f"{NS}Coords")
        rp = pts(c.get("points")) if c is not None else []
        if not rp:
            continue
        col = {"id": tr.get("id"), "type": rtype,
               "bbox": (min(x for x, _ in rp), min(y for _, y in rp), max(x for x, _ in rp), max(y for _, y in rp)),
               "lines": []}
        for tl in tr.findall(f"{NS}TextLine"):
            lc = tl.find(f"{NS}Coords")
            lp = pts(lc.get("points")) if lc is not None else []
            if not lp:
                continue
            bl = tl.find(f"{NS}Baseline")
            bp = pts(bl.get("points")) if bl is not None else []
            y = statistics.median([py for _, py in bp]) if bp else statistics.median([py for _, py in lp])
            te = tl.find(f"{NS}TextEquiv/{NS}Unicode")
            col["lines"].append({
                "id": tl.get("id"), "y": y, "type": custom_type(tl.get("custom", "")),
                "bbox": (min(x for x, _ in lp), min(py for _, py in lp), max(x for x, _ in lp), max(py for _, py in lp)),
                "text": (te.text or "").strip() if te is not None else "", "baseline": bool(bp)})
        col["lines"].sort(key=lambda l: l["y"])
        out["columns"].append(col)
    return out


def check_print(print_dir: Path) -> dict:
    pages = [read_page(p) for p in sorted((print_dir / "page").glob("*.xml"))]
    findings: list[dict] = []
    counts: Counter = Counter()
    page_score: Counter = Counter()

    def add(kind, page, **kw):
        counts[kind] += 1
        page_score[page] += WEIGHT[kind]
        findings.append({"kind": kind, "page": page, **kw})

    col_heights = [c["bbox"][3] - c["bbox"][1] for p in pages for c in p["columns"] if len(c["lines"]) >= 20]
    med_height = statistics.median(col_heights) if col_heights else 0
    n_lines = 0
    # per column type: first/last baseline y and line count per page, for the neighbour comparison
    stats: dict[str, dict[int, tuple[float, float, int]]] = defaultdict(dict)
    for i, p in enumerate(pages):
        for c in p["columns"]:
            if len(c["lines"]) >= 10:
                stats[c["type"]][i] = (c["lines"][0]["y"], c["lines"][-1]["y"], len(c["lines"]))

    def neighbour_median(ctype: str, i: int, k: int, window: int = 8):
        vals = [v[k] for j, v in stats[ctype].items() if j != i and j % 2 == i % 2 and abs(j - i) <= 2 * window]
        return statistics.median(vals) if len(vals) >= 3 else None

    for pi, p in enumerate(pages):
        pg = p["file"]
        per_col_lines = {c["type"]: len(c["lines"]) for c in p["columns"]}
        for c in p["columns"]:
            L = c["lines"]
            n_lines += len(L)
            x0, y0, x1, y1 = c["bbox"]
            width, height = x1 - x0, y1 - y0
            if len(L) < 3:
                continue
            gaps = [b["y"] - a["y"] for a, b in zip(L, L[1:])]
            pitch = statistics.median(gaps)
            if pitch <= 0:
                continue
            expected = height / pitch
            # characters per pixel of full-width lines, the column's own norm
            full = [l for l in L if (l["bbox"][2] - l["bbox"][0]) >= 0.85 * width and l["text"]]
            cpp = statistics.median([len(l["text"]) / (l["bbox"][2] - l["bbox"][0]) for l in full]) if len(full) >= 5 else None
            sample = lambda l: l["text"][:50]
            med_lh = statistics.median([l["bbox"][3] - l["bbox"][1] for l in L])
            nb_count = neighbour_median(c["type"], pi, 2)
            if len(L) < 0.82 * expected or (nb_count and len(L) < nb_count - 8):
                add("few_lines", pg, region=c["id"], type=c["type"], lines=len(L), expected=round(expected, 1),
                    neighbours=nb_count)
            other = [n for t, n in per_col_lines.items() if t != c["type"]]
            if med_height and height < 0.8 * med_height and other and max(other) >= 0.9 * (med_height / pitch):
                add("short_column", pg, region=c["id"], type=c["type"], height=height, median=round(med_height))
            nb_top, nb_bot = neighbour_median(c["type"], pi, 0), neighbour_median(c["type"], pi, 1)
            if nb_top is not None and L[0]["y"] - nb_top > 1.5 * pitch:
                add("top_gap", pg, region=c["id"], type=c["type"], first_y=round(L[0]["y"]), neighbours=round(nb_top), pitch=round(pitch), first=sample(L[0]))
            if nb_bot is not None and nb_bot - L[-1]["y"] > 1.5 * pitch and pi < len(pages) - 1:
                add("bottom_gap", pg, region=c["id"], type=c["type"], last_y=round(L[-1]["y"]), neighbours=round(nb_bot), pitch=round(pitch), last=sample(L[-1]))
            for i, l in enumerate(L):
                lw = l["bbox"][2] - l["bbox"][0]
                lh = l["bbox"][3] - l["bbox"][1]
                if lw > 1.12 * width:
                    add("wide_line", pg, region=c["id"], line=l["id"], width=lw, column=width, text=sample(l))
                dense = cpp and lw > 0 and l["text"] and len(l["text"]) / lw > 1.6 * cpp
                tall = cpp and lw > 0 and l["text"] and lh > 1.8 * med_lh and len(l["text"]) / lw > 1.3 * cpp
                if l["text"] and l["type"] != "cons_title" and (tall or dense):
                    add("merged_line", pg, region=c["id"], line=l["id"], height=lh, pitch=round(pitch), chars=len(l["text"]),
                        typical=round(cpp * lw) if cpp else None, text=sample(l))
                if cpp and lw >= 0.85 * width and l["text"] and len(l["text"]) / lw < 0.55 * cpp \
                        and l["type"] not in HEAD_TYPES and lh <= 1.25 * med_lh:
                    add("sparse_text", pg, region=c["id"], line=l["id"], chars=len(l["text"]), typical=round(cpp * lw), text=sample(l))
                if i == 0:
                    continue
                a = L[i - 1]
                g = l["y"] - a["y"]
                h_overlap = min(a["bbox"][2], l["bbox"][2]) - max(a["bbox"][0], l["bbox"][0])
                if g < 0.45 * pitch and len(a["text"]) > 20 and len(l["text"]) > 20 \
                        and h_overlap > 0.5 * min(a["bbox"][2] - a["bbox"][0], lw):
                    add("overlap_line", pg, region=c["id"], lines=[a["id"], l["id"]], gap=round(g), pitch=round(pitch), text=sample(l))
                heads_near = {a["type"], l["type"]} & HEAD_TYPES or \
                    (i >= 2 and L[i - 2]["type"] in HEAD_TYPES) or (i + 1 < len(L) and L[i + 1]["type"] in HEAD_TYPES)
                if g >= 1.85 * pitch and not heads_near and len(a["text"]) > 30 and len(l["text"]) > 30:
                    add("missing_line", pg, region=c["id"], lines=[a["id"], l["id"]], gap=round(g), pitch=round(pitch),
                        before=sample(a), after=sample(l))
    ranked = sorted(page_score.items(), key=lambda kv: -kv[1])
    return {"print": print_dir.name, "pages": len(pages), "lines": n_lines, "median_column_height": med_height,
            "counts": dict(counts), "pages_flagged": len(page_score),
            "worst_pages": [{"page": p, "score": s} for p, s in ranked[:40]], "findings": findings}


def draw_pages(print_dir: Path, rep: dict, n: int) -> None:
    from PIL import Image, ImageDraw
    out = OUT / "lines" / print_dir.name
    out.mkdir(parents=True, exist_ok=True)
    by_page = defaultdict(list)
    for f in rep["findings"]:
        by_page[f["page"]].append(f)
    for w in rep["worst_pages"][:n]:
        xml = print_dir / "page" / w["page"]
        p = read_page(xml)
        img = next((c for c in [print_dir / p["image"], print_dir / "page" / p["image"]] if c.exists()), None)
        if not img:
            continue
        im = Image.open(img).convert("RGB")
        d = ImageDraw.Draw(im)
        flagged = set()
        for f in by_page[w["page"]]:
            for k in ("line", "lines"):
                v = f.get(k)
                flagged.update(v if isinstance(v, list) else ([v] if v else []))
        for c in p["columns"]:
            d.rectangle(c["bbox"], outline=(0, 90, 220), width=6)
            for l in c["lines"]:
                col = (220, 0, 0) if l["id"] in flagged else (0, 160, 0)
                d.line([(l["bbox"][0], l["y"]), (l["bbox"][2], l["y"])], fill=col, width=5 if l["id"] in flagged else 2)
        d.text((40, 40), f"{w['page']}  score {w['score']}  " + ", ".join(sorted({f['kind'] for f in by_page[w['page']]})), fill=(220, 0, 0))
        im.thumbnail((im.width // 2, im.height // 2))
        im.save(out / (Path(w["page"]).stem + ".jpg"), quality=80)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--print", dest="print_filter")
    ap.add_argument("--images", type=int, default=0, help="debug images for the N worst pages")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for _author, print_id, print_dir in find_prints(ROOT / "data"):
        if args.print_filter and print_id != args.print_filter:
            continue
        rep = check_print(print_dir)
        if not rep["lines"]:
            continue
        (OUT / f"lines_{print_id}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        if args.images:
            draw_pages(print_dir, rep, args.images)
        rows.append(rep)
    kinds = sorted(WEIGHT)
    md = ["# Line rhythm check", "", "| print | pages | lines | flagged pages | " + " | ".join(kinds) + " |",
          "|---|---|---|---|" + "---|" * len(kinds)]
    print("print".ljust(36), "pages lines flagged " + " ".join(k[:12].rjust(12) for k in kinds))
    for r in rows:
        print(r["print"].ljust(36), str(r["pages"]).rjust(5), str(r["lines"]).rjust(6), str(r["pages_flagged"]).rjust(7),
              " ".join(str(r["counts"].get(k, 0)).rjust(12) for k in kinds))
        md.append(f"| {r['print']} | {r['pages']} | {r['lines']} | {r['pages_flagged']} | " + " | ".join(str(r["counts"].get(k, 0)) for k in kinds) + " |")
        md.append("")
        md.append("Worst pages: " + ", ".join(f"{w['page']} ({w['score']})" for w in r["worst_pages"][:12]))
        md.append("")
    md += ["Meaning of the columns and thresholds: docstring of src/qa/check_lines.py. Details in lines_<print>.json; debug images with --images N in output/qa/lines/<print>/."]
    (OUT / "lines_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nDetails: {OUT}/lines_<print>.json, summary lines_summary.md")


if __name__ == "__main__":
    main()
