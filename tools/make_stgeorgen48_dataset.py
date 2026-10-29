#!/usr/bin/env python3
"""
Build the public layout dataset of Karlsruhe, BLB, Cod. St. Georgen 48 (PageXML without images) and the
kraken segmentation training set from the Transkribus export in dist/hf-censilia-med-1466/.

Dataset (default dist/hf-blb-st-georgen-48-pagexml/):
  page/<nnnn>.xml         clean PAGE 2013 files: regions with a structure type (and PAGE @type), text
                          lines with polygon, baseline and a line type inherited from their region;
                          no text, no Transkribus metadata; the image is referenced by file name only
  lines.parquet           one row per text line (ids, types, geometry, split, in_training)
  pages.csv               one row per page (size, split, region/line counts, annotation state, in_training)
  splits/{train,val,test}.txt   page lists of the published split (paths relative to the dataset root)
  stats.json              the counts quoted in the dataset card

kraken (default output/kraken_stgeorgen48/):
  pages/<nnnn>.xml + .jpg the training version of each page used for training (image hard-linked):
                          region-only pages and pages without writing are left out, text regions
                          without lines are removed, rare region types are mapped (see REGION_MAP)
  seg_{train,val,test}.txt   the lists for `ketos segtrain` / `ketos segtest`

    python tools/make_stgeorgen48_dataset.py [--src DIR] [--out DIR] [--kraken-out DIR]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
XSI = "http://www.w3.org/2001/XMLSchema-instance"
T = "{%s}" % NS
NSMAP = {None: NS, "xsi": XSI}
SRC = ROOT / "dist" / "hf-censilia-med-1466" / "BLB_Karlsruhe_St__Georgen_48" / "BLB_Karlsruhe_St__Georgen_48"
STRUCT = re.compile(r"structure \{[^}]*type:([^;}]+)")
TEST_PERMILLE, VAL_PERMILLE = 50, 130
CREATOR = "Michael Schonhardt (TU Darmstadt): layout annotation in Transkribus, 2025/26; see the dataset card"

# spelling fixes applied to the published dataset
RENAME = {"quire_number": "quire-number", "colophone": "colophon"}
# PAGE @type for the structure types
# kraken reads the PAGE @type before the custom structure type, so the columns get no @type: otherwise a model
# trained on the files learns "paragraph" instead of column_1 / column_2 / column_single
PAGE_TYPE = {"paragraph": "paragraph", "marginalia": "marginalia", "page-number": "page-number", "quire-number": "signature-mark",
             "running-title": "header", "TOC-entry": "TOC-entry", "index": "paragraph", "colophon": "paragraph"}
# text blocks: a page is annotated only if all of them carry lines
TEXT_BLOCKS = {"column_1", "column_2", "column_single", "index", "paragraph", "TOC-entry", "colophon", "running-title"}
# page-level descriptions, not layout: kept in the dataset, dropped from training
NON_LAYOUT = {"flyleaf", "empty-page", "binding", "pastedown"}
# region classes of the model; everything else is mapped here for training
REGION_MAP = {"column_1": "column_1", "column_2": "column_2", "column_single": "column_single", "marginalia": "marginalia",
              "page-number": "page-number", "quire-number": "quire-number"}
REGION_OTHER = "paragraph"
# line classes of the model, inherited from the region
LINE_OF_REGION = {"marginalia": "marginalia", "page-number": "page-number", "quire-number": "quire-number"}
LINE_DEFAULT = "body"


def split_of(stem: str) -> str:
    h = int(hashlib.sha1(f"stgeorgen48/{stem}".encode()).hexdigest(), 16) % 1000
    return "test" if h < TEST_PERMILLE else "val" if h < VAL_PERMILLE else "train"


def struct_type(el) -> str | None:
    m = STRUCT.search(el.get("custom") or "")
    t = m.group(1).strip() if m else el.get("type")
    return RENAME.get(t, t) if t else None


def pts(el) -> str:
    c = el.find(T + "Coords") if el.tag != T + "Baseline" else el
    return (c.get("points") or "").strip() if c is not None else ""


def bbox(points: str) -> tuple[int, int, int, int]:
    xy = [tuple(int(float(v)) for v in p.split(",")) for p in points.split()]
    xs, ys = [p[0] for p in xy], [p[1] for p in xy]
    return min(xs), min(ys), max(xs), max(ys)


def new_doc(image_name: str, w: str, h: str, created: str | None, changed: str | None):
    root = etree.Element(T + "PcGts", nsmap=NSMAP)
    root.set("{%s}schemaLocation" % XSI, f"{NS} {NS}/pagecontent.xsd")
    md = etree.SubElement(root, T + "Metadata")
    etree.SubElement(md, T + "Creator").text = CREATOR
    etree.SubElement(md, T + "Created").text = created or ""
    etree.SubElement(md, T + "LastChange").text = changed or ""
    pg = etree.SubElement(root, T + "Page", imageFilename=image_name, imageWidth=w, imageHeight=h)
    return root, pg


def add_region(pg, og, r, rtype: str | None, lines: list, line_type_of, index: int, region_type_out: str | None):
    rid = r.get("id")
    custom = f"readingOrder {{index:{index};}}"
    if region_type_out:
        custom += f" structure {{type:{region_type_out};}}"
    el = etree.SubElement(pg, T + "TextRegion", id=rid, custom=custom)
    if region_type_out in PAGE_TYPE:
        el.set("type", PAGE_TYPE[region_type_out])
    etree.SubElement(el, T + "Coords", points=pts(r))
    for i, tl in enumerate(lines):
        lt = line_type_of(rtype)
        le = etree.SubElement(el, T + "TextLine", id=tl.get("id"), custom=f"readingOrder {{index:{i};}} structure {{type:{lt};}}")
        etree.SubElement(le, T + "Coords", points=pts(tl))
        bl = tl.find(T + "Baseline")
        if bl is not None and pts(bl):
            etree.SubElement(le, T + "Baseline", points=pts(bl))
    etree.SubElement(og, T + "RegionRefIndexed", index=str(index), regionRef=rid)


def write(root, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    etree.ElementTree(root).write(str(path), xml_declaration=True, encoding="UTF-8", pretty_print=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=ROOT / "dist" / "hf-blb-st-georgen-48-pagexml")
    ap.add_argument("--kraken-out", type=Path, default=ROOT / "output" / "kraken_stgeorgen48")
    a = ap.parse_args()
    rows, pages = [], []
    total: Counter = Counter()
    lists: dict[str, list[str]] = {k: [] for k in ("train", "val", "test")}
    rtypes_ds, ltypes_ds, rtypes_tr, excluded = Counter(), Counter(), Counter(), Counter()
    for xml in sorted((a.src / "page").glob("*.xml"), key=lambda p: int(p.stem)):
        stem = f"{int(xml.stem):04d}"  # 086 -> 0086
        img = a.src / f"{xml.stem}.jpg"
        if not img.exists():
            print(f"no image for {xml}", file=sys.stderr)
            continue
        split = split_of(stem)
        src = etree.parse(str(xml)).getroot()
        page = src.find(T + "Page")
        w, h = page.get("imageWidth"), page.get("imageHeight")
        md = src.find(T + "Metadata")
        created = md.findtext(T + "Created") if md is not None else None
        changed = md.findtext(T + "LastChange") if md is not None else None
        regions = page.findall(T + "TextRegion")
        # order regions by the reading order index
        def ro(r):
            m = re.search(r"readingOrder \{index:(\d+);", r.get("custom") or "")
            return int(m.group(1)) if m else 10**6
        regions.sort(key=ro)
        info = [(r, struct_type(r), r.findall(T + "TextLine")) for r in regions]
        n_lines = sum(len(ls) for _, _, ls in info)
        text_regions = [t for _, t, _ in info if t not in NON_LAYOUT]
        # a page counts as annotated only if every text block (column etc.) has lines; a page-number line alone is not enough
        blocks_empty = [t for _, t, ls in info if t in TEXT_BLOCKS and not ls]
        state = ("regions_only" if blocks_empty or (text_regions and not n_lines) else "lines" if n_lines else "no_text")
        # training eligibility
        train_ok = state == "lines"
        if not train_ok:
            excluded[state] += 1
        # --- dataset page (everything, cleaned)
        root, pg = new_doc(f"{stem}.jpg", w, h, created, changed)
        og = etree.SubElement(etree.SubElement(pg, T + "ReadingOrder"), T + "OrderedGroup", id="ro_1", caption="Regions reading order")
        line_type = lambda rt: LINE_OF_REGION.get(rt, LINE_DEFAULT)
        for i, (r, rt, ls) in enumerate(info):
            add_region(pg, og, r, rt, ls, line_type, i, rt)
            rtypes_ds[rt] += 1
            for j, tl in enumerate(ls):
                lt = line_type(rt)
                ltypes_ds[lt] += 1
                poly, bl = pts(tl), pts(tl.find(T + "Baseline")) if tl.find(T + "Baseline") is not None else ""
                x0, y0, x1, y1 = bbox(poly) if poly else (None, None, None, None)
                rows.append({"page": stem, "image": f"{stem}.jpg", "split": split, "region_id": r.get("id"), "region_type": rt,
                             "region_index": i, "line_id": tl.get("id"), "line_index": j, "line_type": lt, "polygon": poly,
                             "baseline": bl, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                             "height": (y1 - y0) if poly else None, "in_training": train_ok})
        if not len(og):
            pg.remove(pg.find(T + "ReadingOrder"))
        write(root, a.out / "page" / f"{stem}.xml")
        # --- training page
        if train_ok:
            troot, tpg = new_doc(f"{stem}.jpg", w, h, created, changed)
            tog = etree.SubElement(etree.SubElement(tpg, T + "ReadingOrder"), T + "OrderedGroup", id="ro_1", caption="Regions reading order")
            k = 0
            for r, rt, ls in info:
                if rt in NON_LAYOUT or (not ls):  # drop page descriptions and text regions without lines
                    continue
                out_t = REGION_MAP.get(rt, REGION_OTHER)
                rtypes_tr[out_t] += 1
                add_region(tpg, tog, r, rt, ls, line_type, k, out_t)
                k += 1
            dst = a.kraken_out / "pages" / f"{stem}.xml"
            write(troot, dst)
            jpg = a.kraken_out / "pages" / f"{stem}.jpg"
            if not jpg.exists():
                try:
                    os.link(img, jpg)
                except OSError:
                    shutil.copyfile(img, jpg)
            lists[split].append(str(dst))
            total[f"train:{split}:pages"] += 1
            total[f"train:{split}:lines"] += n_lines
        pages.append({"page": stem, "image": f"{stem}.jpg", "width": int(w), "height": int(h), "split": split, "state": state,
                      "n_regions": len(info), "n_lines": n_lines, "in_training": train_ok})
        total["pages"] += 1
        total["lines"] += n_lines
        total["regions"] += len(info)
        total[f"split:{split}:pages"] += 1
        total[f"split:{split}:lines"] += n_lines
    (a.out / "splits").mkdir(parents=True, exist_ok=True)
    a.kraken_out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (a.out / "splits" / f"{split}.txt").write_text("".join(f"page/{p['page']}.xml\n" for p in pages if p["split"] == split), encoding="utf-8")
        (a.kraken_out / f"seg_{split}.txt").write_text("".join(x + "\n" for x in lists[split]), encoding="utf-8")
    with (a.out / "pages.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(pages[0].keys()))
        wr.writeheader(); wr.writerows(pages)
    import pyarrow as pa
    import pyarrow.parquet as pq
    cols = list(rows[0].keys())
    pq.write_table(pa.table({c: [r[c] for r in rows] for c in cols}), str(a.out / "lines.parquet"), compression="zstd")
    total["region_types"] = dict(rtypes_ds.most_common())
    total["line_types"] = dict(ltypes_ds.most_common())
    total["training_region_classes"] = dict(rtypes_tr.most_common())
    total["pages_excluded_from_training"] = dict(excluded)
    total["lines_with_baseline"] = sum(1 for r in rows if r["baseline"])
    total["states"] = dict(Counter(p["state"] for p in pages))
    (a.out / "stats.json").write_text(json.dumps(dict(sorted(total.items())), ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(dict(sorted(total.items())), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
