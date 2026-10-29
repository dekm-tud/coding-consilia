#!/usr/bin/env python3
"""
Build the public PageXML dataset of the Baldus print (layout and transcription ground truth) and the
training lists for the kraken models from the edition's PageXML in data/.

Dataset (default dist/hf-consilia-baldus-venice1575-pagexml/, the Hugging Face dataset folder):
  <print>/<page>.xml      clean PAGE 2013 files: regions with a structure type, text lines with
                          polygon, baseline, structure type and the edition text (one TextEquiv);
                          the Transkribus metadata, the alternative readings and the working
                          comments of data/ are left out
  (no images)             the page images stay in the Internet Archive items of the volumes, the same files
                          the PageXML was made on: pages.csv gives their URL and SHA-1, fetch_images.py
                          (copied from tools/pagexml_dataset/) downloads them next to the XML files;
                          --with-images puts them into the folder instead (hard links, or copies)
  lines.parquet           one row per text line: ids, types, geometry, text, agreement of the
                          three readings with the edition text, review flag, split
  pages.csv               one row per page with counts, the split and the image (URL, SHA-1, bytes, IIIF manifest,
                          Internet Archive viewer page); the URLs come from output/ia_images.json (tools/ia_image_map.py)
  splits/{train,val,test}.txt   page lists (paths relative to the dataset root; split by page hash,
                          5 % test, 8 % validation; the same split as the kraken training)
  stats.json              the counts quoted in the dataset card

kraken (default output/kraken_final/):
  seg_{train,val,test}.txt   the dataset's XML files per split (segmentation: every line counts)
  rec/<print>/<page>.xml      recognition ground truth: only lines with a non-empty text, a baseline,
                              and either a human decision or at least two of the three readings
                              agreeing with the edition text (no interpolated geometry, no single
                              reading); images hard-linked next to the files
  rec_{train,val,test}.txt    the lists for `ketos compile`

    python tools/make_pagexml_dataset.py [--out DIR] [--kraken-out DIR] [--with-images [--copy-images]]

Region types: Transkribus typed the regions of 78 pages (header, column_1, column_2, page_number,
footer). On the other pages the type is set here from the lines a region holds: header, page_number,
and the text columns numbered from left to right (column_1, column_2). The PAGE @type attribute is
set accordingly (header, page-number, footer, paragraph). Line types are those of the edition
(cons_title, cons_argument, cons_body, header, page_number, catch-word).
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
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "qa"))
from compare_ocr import normalise  # noqa: E402

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
XSI = "http://www.w3.org/2001/XMLSchema-instance"
T = "{%s}" % NS
NSMAP = {None: NS, "xsi": XSI}
DATA = ROOT / "data" / "Baldo_29618397"
PRINTS = sorted(p.name for p in DATA.iterdir() if p.is_dir() and (p / "page").is_dir())
STRUCT = re.compile(r"structure \{[^}]*type:([^;}]+)")
# kraken reads the PAGE @type before the custom structure type; the columns therefore get no @type, otherwise a
# segmentation model trained on the files learns one class "paragraph" and merges the two adjacent columns
PAGE_TYPE = {"header": "header", "page_number": "page-number", "footer": "footer"}
TEST_PERMILLE, VAL_PERMILLE = 50, 130
CREATOR = "Consilia edition (Michael Schonhardt, TU Darmstadt): Transkribus layout, corrected; text from three readings, see the dataset card"


def split_of(print_id: str, stem: str) -> str:
    h = int(hashlib.sha1(f"{print_id}/{stem}".encode()).hexdigest(), 16) % 1000
    return "test" if h < TEST_PERMILLE else "val" if h < VAL_PERMILLE else "train"


def struct_type(el) -> str | None:
    m = STRUCT.search(el.get("custom") or "")
    return m.group(1).strip() if m else el.get("type")


def clean_text(s: str | None) -> str:
    s = (s or "").strip()
    return "".join(ch for ch in s if unicodedata.category(ch) != "Cc")


def pts(el) -> str:
    c = el.find(T + "Coords") if el.tag != T + "Baseline" else el
    return (c.get("points") or "").strip() if c is not None else ""


def bbox(points: str) -> tuple[int, int, int, int]:
    xy = [tuple(int(float(v)) for v in p.split(",")) for p in points.split()]
    xs, ys = [p[0] for p in xy], [p[1] for p in xy]
    return min(xs), min(ys), max(xs), max(ys)


def line_record(tl) -> dict:
    tes = tl.findall(T + "TextEquiv")
    text = clean_text(tes[0].find(T + "Unicode").text) if tes else ""
    f0 = normalise(text, True) if text else None
    agree = sum(1 for te in tes[1:] if f0 and normalise(te.find(T + "Unicode").text or "", True) == f0)
    cm = tl.get("comments") or ""
    flags = []
    if "reviewed " in cm:
        flags.append("human_decision")
    if "pylaia only" in cm:
        flags.append("single_reading")
    if "geometry=interpolated" in cm:
        flags.append("interpolated_geometry")
    bl = tl.find(T + "Baseline")
    return {"text": text, "readings_agree": agree if text else None, "flags": flags,
            "polygon": pts(tl), "baseline": pts(bl) if bl is not None else "", "n_readings": max(0, len(tes) - 1)}


def build_page(xml: Path, print_id: str, out_xml: Path, rec_xml: Path | None, image_name: str, rows: list, split: str) -> dict:
    src = etree.parse(str(xml)).getroot()
    page = src.find(T + "Page")
    w, h = page.get("imageWidth"), page.get("imageHeight")
    regions = page.findall(T + "TextRegion")
    # region types: Transkribus' where present, else from the lines
    typed, columns = {}, []
    for r in regions:
        t = struct_type(r)
        lts = Counter(struct_type(tl) for tl in r.findall(T + "TextLine"))
        if t in ("column_1", "column_2"):
            typed[r] = t
        elif t:
            typed[r] = t
        elif lts and set(lts) <= {"header"}:
            typed[r] = "header"
        elif lts and set(lts) <= {"page_number"}:
            typed[r] = "page_number"
        elif lts:
            columns.append(r)
        else:
            typed[r] = None
    for i, r in enumerate(sorted(columns, key=lambda r: sum(bbox(pts(r))[0::2]) / 2)):
        typed[r] = f"column_{i + 1}"

    def new_doc():
        root = etree.Element(T + "PcGts", nsmap=NSMAP)
        root.set("{%s}schemaLocation" % XSI, f"{NS} {NS}/pagecontent.xsd")
        md = etree.SubElement(root, T + "Metadata")
        etree.SubElement(md, T + "Creator").text = CREATOR
        created = src.find(T + "Metadata/" + T + "Created")
        etree.SubElement(md, T + "Created").text = created.text if created is not None else "2025-10-30T00:00:00"
        changed = src.find(T + "Metadata/" + T + "LastChange")
        etree.SubElement(md, T + "LastChange").text = changed.text if changed is not None else "2026-10-26T00:00:00"
        pg = etree.SubElement(root, T + "Page", imageFilename=image_name, imageWidth=w, imageHeight=h)
        ro = etree.SubElement(pg, T + "ReadingOrder")
        og = etree.SubElement(ro, T + "OrderedGroup", id="ro_regions", caption="Regions reading order")
        return root, pg, og

    doc, pg, og = new_doc()
    rdoc, rpg, rog = new_doc() if rec_xml else (None, None, None)
    st = Counter()
    rtypes = Counter()
    for ri, r in enumerate(regions):
        rt = typed.get(r)
        rtypes[rt or "untyped"] += 1
        lines = r.findall(T + "TextLine")
        recs = [line_record(tl) for tl in lines]
        for tgt, group, keep in ((pg, og, None), (rpg, rog, "rec")):
            if tgt is None:
                continue
            sel = [(tl, rec) for tl, rec in zip(lines, recs)
                   if keep is None or (rec["text"] and rec["baseline"] and "interpolated_geometry" not in rec["flags"]
                                       and "single_reading" not in rec["flags"] and (rec["readings_agree"] >= 2 or "human_decision" in rec["flags"]))]
            if keep and not sel:
                continue
            etree.SubElement(group, T + "RegionRefIndexed", index=str(len(group)), regionRef=r.get("id"))
            attrs = {"id": r.get("id")}
            if rt and PAGE_TYPE.get(rt):
                attrs["type"] = PAGE_TYPE[rt]
            attrs["custom"] = f"readingOrder {{index:{len(group) - 1};}}" + (f" structure {{type:{rt};}}" if rt else "")
            tr = etree.SubElement(tgt, T + "TextRegion", **attrs)
            etree.SubElement(tr, T + "Coords", points=pts(r))
            for li, (tl, rec) in enumerate(sel):
                lt = struct_type(tl)
                el = etree.SubElement(tr, T + "TextLine", id=tl.get("id"), custom=f"readingOrder {{index:{li};}}" + (f" structure {{type:{lt};}}" if lt else ""))
                etree.SubElement(el, T + "Coords", points=rec["polygon"])
                if rec["baseline"]:
                    etree.SubElement(el, T + "Baseline", points=rec["baseline"])
                te = etree.SubElement(el, T + "TextEquiv")
                etree.SubElement(te, T + "Unicode").text = rec["text"]
            te = etree.SubElement(tr, T + "TextEquiv")
            etree.SubElement(te, T + "Unicode").text = "\n".join(rec["text"] for _, rec in sel)
            if keep:
                st["rec_lines"] += len(sel)
        for li, (tl, rec) in enumerate(zip(lines, recs)):
            lt = struct_type(tl)
            x0, y0, x1, y1 = bbox(rec["polygon"]) if rec["polygon"] else (None, None, None, None)
            rows.append({"print": print_id, "page": xml.stem, "image": f"{print_id}/{image_name}", "split": split,
                         "region_id": r.get("id"), "region_type": rt, "region_index": ri, "line_id": tl.get("id"), "line_index": li,
                         "line_type": lt, "polygon": rec["polygon"], "baseline": rec["baseline"],
                         "x0": x0, "y0": y0, "x1": x1, "y1": y1, "height": (y1 - y0) if y0 is not None else None,
                         "text": rec["text"], "n_chars": len(rec["text"]), "readings_agree": rec["readings_agree"],
                         "human_decision": "human_decision" in rec["flags"], "flags": ",".join(f for f in rec["flags"] if f != "human_decision")})
            st["lines"] += 1
            st["lines_with_text"] += bool(rec["text"])
            st["lines_with_baseline"] += bool(rec["baseline"])
            st[f"line_type:{lt}"] += 1
            if rec["text"]:
                st[f"agree:{rec['readings_agree']}"] += 1
            st["human_decision"] += "human_decision" in rec["flags"]
    for k, v in rtypes.items():
        st[f"region_type:{k}"] += v
    st["regions"] = len(regions)
    if not len(og):                                   # a page without regions: no ReadingOrder (the schema wants a non-empty group)
        pg.remove(pg.find(T + "ReadingOrder"))
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    etree.ElementTree(doc).write(str(out_xml), xml_declaration=True, encoding="utf-8", pretty_print=True)
    if rec_xml and len(rog):
        rec_xml.parent.mkdir(parents=True, exist_ok=True)
        etree.ElementTree(rdoc).write(str(rec_xml), xml_declaration=True, encoding="utf-8", pretty_print=True)
        st["rec_pages"] += 1
    return st


def link(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copyfile(src, dst)
    else:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "dist" / "hf-consilia-baldus-venice1575-pagexml")
    ap.add_argument("--kraken-out", type=Path, default=ROOT / "output" / "kraken_final")
    ap.add_argument("--with-images", action="store_true", help="put the page images into the dataset folder (default: linked only)")
    ap.add_argument("--copy-images", action="store_true", help="with --with-images: copies instead of hard links")
    ap.add_argument("--print", dest="print_filter")
    a = ap.parse_args()
    ia = json.loads((ROOT / "output" / "ia_images.json").read_text(encoding="utf-8"))
    reg = json.loads((ROOT / "site-data" / "iiif_manifests.json").read_text(encoding="utf-8"))
    rows: list[dict] = []
    pages: list[dict] = []
    total = Counter()
    lists = defaultdict(list)
    for print_id in PRINTS:
        if a.print_filter and print_id != a.print_filter:
            continue
        for xml in sorted((DATA / print_id / "page").glob("*.xml")):
            img = DATA / print_id / f"{xml.stem}.jpg"
            if not img.exists():
                print(f"no image for {xml}", file=sys.stderr)
                continue
            split = split_of(print_id, xml.stem)
            out_xml = a.out / print_id / f"{xml.stem}.xml"
            rec_xml = a.kraken_out / "rec" / print_id / f"{xml.stem}.xml"
            st = build_page(xml, print_id, out_xml, rec_xml, f"{xml.stem}.jpg", rows, split)
            if a.with_images:
                link(img, a.out / print_id / f"{xml.stem}.jpg", a.copy_images)
            else:
                (a.out / print_id / f"{xml.stem}.jpg").unlink(missing_ok=True)      # left from an earlier build with images
            link(img, a.kraken_out / "rec" / print_id / f"{xml.stem}.jpg", False)
            total.update(st)
            total[f"split:{split}:pages"] += 1
            total[f"split:{split}:lines"] += st["lines"]
            total[f"split:{split}:rec_lines"] += st["rec_lines"]
            lists[f"seg_{split}"].append(str(out_xml))
            if rec_xml.exists() and st["rec_lines"]:
                lists[f"rec_{split}"].append(str(rec_xml))
            lists[f"split_{split}"].append(f"{print_id}/{xml.stem}.xml")
            src = ia[print_id][xml.stem]
            item = reg[print_id]["manifest"].split("/")[-2]
            idx = int(re.sub(r"\D", "", xml.stem)) - 1 + reg[print_id].get("canvas_offset", 0)
            pages.append({"print": print_id, "page": xml.stem, "image": f"{print_id}/{xml.stem}.jpg", "width": None, "height": None,
                          "split": split, "n_regions": st["regions"], "n_lines": st["lines"], "n_lines_with_text": st["lines_with_text"],
                          "image_url": src["url"], "image_sha1": src["sha1"], "image_bytes": src["bytes"],
                          "iiif_manifest": reg[print_id]["manifest"], "ia_view": f"https://archive.org/details/{item}/page/n{idx}/mode/1up"})
        print(print_id, "done", file=sys.stderr)
    # page sizes from the lines table source (cheap: read the xml again would be slow; take from PageXML attributes already parsed)
    sizes = {}
    for p in pages:
        root = etree.parse(str(a.out / p["print"] / f"{p['page']}.xml")).getroot().find(T + "Page")
        p["width"], p["height"] = int(root.get("imageWidth")), int(root.get("imageHeight"))
        sizes[(p["print"], p["page"])] = (p["width"], p["height"])
    (a.out / "splits").mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (a.out / "splits" / f"{split}.txt").write_text("\n".join(lists[f"split_{split}"]) + "\n", encoding="utf-8")
        a.kraken_out.mkdir(parents=True, exist_ok=True)
        (a.kraken_out / f"seg_{split}.txt").write_text("\n".join(lists[f"seg_{split}"]) + "\n", encoding="utf-8")
        (a.kraken_out / f"rec_{split}.txt").write_text("\n".join(lists[f"rec_{split}"]) + "\n", encoding="utf-8")
    shutil.copyfile(ROOT / "tools" / "pagexml_dataset" / "fetch_images.py", a.out / "fetch_images.py")
    with (a.out / "pages.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(pages[0].keys()))
        wr.writeheader(); wr.writerows(pages)
    import pyarrow as pa
    import pyarrow.parquet as pq
    cols = list(rows[0].keys())
    table = pa.table({c: [r[c] for r in rows] for c in cols})
    pq.write_table(table, str(a.out / "lines.parquet"), compression="zstd")
    chars = Counter()
    for r in rows:
        chars.update(r["text"])
    total["pages"] = len(pages)
    total["characters"] = sum(r["n_chars"] for r in rows)
    total["words"] = sum(len(r["text"].split()) for r in rows)
    total["charset_size"] = len(chars)
    total["charset"] = "".join(sorted(chars))
    total["rec_charset_size"] = len({ch for r in rows if r["text"] and r["baseline"] and not r["flags"] and (r["readings_agree"] >= 2 or r["human_decision"]) for ch in r["text"]})
    (a.out / "stats.json").write_text(json.dumps(dict(sorted(total.items())), ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in sorted(total.items()) if k != "charset"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
