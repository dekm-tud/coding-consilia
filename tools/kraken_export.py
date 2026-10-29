#!/usr/bin/env python3
"""
Training lines for a kraken recognition model from the consensus of the two readings.

A line counts as consensus when its PageXML holds one reading from the relayout with Gemini
(the two readings were identical, so only one was stored) or two readings that are equal after
the loose normalisation of src/qa/compare_ocr.py. Lines with a review comment, one source only
("pylaia only", "geometry=interpolated") or an empty text are left out.

    python tools/kraken_export.py --out output/kraken --val-share 0.08 [--print <print id>] [--limit N]

Writes <out>/{train,val}/<print>_<page>_<line>.png (line crop, the polygon masked white outside,
a small margin) and the matching .gt.txt with the edition text (index-0 TextEquiv), plus
<out>/manifest.jsonl (one record per line: file, print, page, line id, text, kind) and
<out>/stats.json. The split is by page, so no page is in both sets.

Train afterwards (separate environment .venv-kraken):
    ketos train -f path -o output/kraken/model/consilia -d cuda:0 --augment \\
        -t output/kraken/train.txt -e output/kraken/val.txt [-i <base model>.mlmodel --resize new]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "qa"))
from compare_ocr import normalise  # noqa: E402

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"; T = "{%s}" % NS
DATA = ROOT / "data" / "Baldo_29618397"
MARGIN = 8


def points(el):
    c = el.find(T + "Coords")
    return [tuple(int(float(v)) for v in p.split(",")) for p in c.get("points").split()] if c is not None else []


def consensus_text(tl) -> tuple[str, str] | None:
    cm = tl.get("comments") or ""
    if "reviewed " in cm or "pylaia only" in cm or "geometry=interpolated" in cm or "relayout" not in cm:
        return None
    tes = tl.findall(T + "TextEquiv")
    if not tes:
        return None
    t0 = (tes[0].find(T + "Unicode").text or "").strip()
    if not t0:
        return None
    if len(tes) == 1:
        return t0, "identical"
    t1 = (tes[1].find(T + "Unicode").text or "").strip()
    if t1 and normalise(t0, True) == normalise(t1, True):
        return t0, "agree"
    return None


def crop_line(im: Image.Image, poly: list[tuple[int, int]], margin: int = MARGIN, side: int | None = None) -> Image.Image:
    """line tile: polygon dilated by `margin`, plus `side` extra pixels left and right (the PyLaia
    polygons often start a few pixels into the first letter)"""
    side = margin if side is None else side
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    x0, y0 = max(0, min(xs) - side), max(0, min(ys) - margin)
    x1, y1 = min(im.width, max(xs) + side), min(im.height, max(ys) + margin)
    tile = im.crop((x0, y0, x1, y1)).convert("L")
    mask = Image.new("L", tile.size, 0); d = ImageDraw.Draw(mask); shifted = [(x - x0, y - y0) for x, y in poly]
    d.polygon(shifted, fill=255)
    d.line(shifted + shifted[:1], fill=255, width=2 * margin, joint="curve")   # dilate: the PyLaia polygons clip ascenders and initials
    if side > margin:
        d.rectangle((0, 0, side, tile.height), fill=255); d.rectangle((tile.width - side, 0, tile.width, tile.height), fill=255)
    white = Image.new("L", tile.size, 255)
    return Image.composite(tile, white, mask)


def export_page(args):
    """one page: (records, stats) — runs in a worker process"""
    xml, out, val_share = args
    print_id = xml.parts[-3]
    st = {"identical": 0, "agree": 0, "skipped": 0, "pages": 0, "train": 0, "val": 0}; recs = []
    root = ET.parse(xml).getroot(); page = root.find(T + "Page")
    img = next((p for p in (xml.parents[1] / page.get("imageFilename"), xml.parent / page.get("imageFilename")) if p.exists()), None)
    if img is None:
        return recs, st
    split = "val" if int(hashlib.sha1(f"{print_id}/{xml.stem}".encode()).hexdigest(), 16) % 1000 < val_share * 1000 else "train"
    im = None
    for tr in page.findall(T + "TextRegion"):
        for tl in tr.findall(T + "TextLine"):
            ct = consensus_text(tl)
            if not ct:
                st["skipped"] += 1; continue
            text, kind = ct
            poly = points(tl)
            if len(poly) < 3:
                st["skipped"] += 1; continue
            if im is None:
                im = Image.open(img); im.load(); st["pages"] += 1
            tile = crop_line(im, poly)
            if tile.width < 20 or tile.height < 10:
                st["skipped"] += 1; continue
            short = print_id.rsplit("_", 1)[-1]           # v1 .. v5
            name = f"{short}_{xml.stem}_{tl.get('id')}"
            png = out / split / f"{name}.png"
            tile.save(png); (out / split / f"{name}.gt.txt").write_text(text, encoding="utf-8")
            recs.append({"file": str(png), "print": print_id, "page": xml.name, "line_id": tl.get("id"), "text": text, "kind": kind, "split": split})
            st[kind] += 1; st[split] += 1
    return recs, st


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "kraken")
    ap.add_argument("--val-share", type=float, default=0.08)
    ap.add_argument("--print", dest="print_filter")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N/100 pages")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    for d in ("train", "val"):
        (a.out / d).mkdir(parents=True, exist_ok=True)
    stats = {"identical": 0, "agree": 0, "skipped": 0, "pages": 0, "train": 0, "val": 0}
    lists = {"train": [], "val": []}
    xmls = [x for x in sorted(DATA.glob("*/page/*.xml")) if not a.print_filter or x.parts[-3] == a.print_filter]
    if a.limit:
        xmls = xmls[:max(1, a.limit // 100)]
    from concurrent.futures import ProcessPoolExecutor
    with (a.out / "manifest.jsonl").open("w", encoding="utf-8") as man, ProcessPoolExecutor(max_workers=a.workers) as ex:
        for recs, st in ex.map(export_page, [(x, a.out, a.val_share) for x in xmls], chunksize=4):
            for k, v in st.items():
                stats[k] += v
            for r in recs:
                lists[r["split"]].append(r["file"]); man.write(json.dumps(r, ensure_ascii=False) + "\n")
    for split, files in lists.items():
        (a.out / f"{split}.txt").write_text("\n".join(files) + "\n", encoding="utf-8")
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(stats)


if __name__ == "__main__":
    main()
