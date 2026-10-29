#!/usr/bin/env python3
"""
Read every text line of the Baldus prints with a kraken model and store the reading next to the two
existing ones (PyLaia/Transkribus and Gemini), without changing the PageXML.

    .venv-kraken/bin/python tools/kraken_read_all.py --model output/kraken/model/consilia_e31.mlmodel \
        [--out output/qa/kraken_all_e31.jsonl] [--print Baldo_Cons_Print_Venice_1575_v1]

--data reads another copy of the PageXML (same layout <print>/page/*.xml; images are taken from data/
when the copy has none); --only reads just the lines listed in a JSONL file (fields print, page, line_id).

One record per line: file, region, line_id, valid source, gemini, pylaia (empty when the line has only
one reading), kraken, mean confidence, the line's comment (to see earlier decisions), and whether the
line is in the kraken training or validation set. Resumable: lines already in the output are skipped.
Run tools/kraken_compare.py afterwards.
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kraken_export import crop_line, points, T  # noqa: E402

DATA = ROOT / "data" / "Baldo_29618397"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "qa" / "kraken_all_e31.jsonl")
    ap.add_argument("--print", dest="print_filter")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--only", type=Path, help="JSONL with print, page, line_id: read only these lines")
    a = ap.parse_args()
    only = None
    if a.only:
        only = {(r["print"], r["page"], r["line_id"]) for r in map(json.loads, a.only.open(encoding="utf-8"))}
    from kraken.lib import models
    from kraken import rpred
    from kraken.containers import Segmentation, BBoxLine
    model = models.load_any(str(a.model), device="cuda:0")
    split = {}
    for name in ("train", "val"):
        f = ROOT / "output" / "kraken" / f"{name}.txt"
        if f.exists():
            for p in f.read_text().split():
                split[Path(p).stem] = name
    done = set()
    if a.out.exists():
        for l in a.out.open(encoding="utf-8"):
            r = json.loads(l); done.add((r["file"], r["line_id"]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with a.out.open("a", encoding="utf-8") as out:
        for xml in sorted(a.data.glob("*/page/*.xml")):
            print_id = xml.parts[-3]
            if a.print_filter and print_id != a.print_filter:
                continue
            if only is not None and not any(k[0] == print_id and k[1] == xml.name for k in only):
                continue
            root = ET.parse(xml).getroot(); page = root.find(T + "Page")
            img = next((p for p in (xml.parents[1] / page.get("imageFilename"), xml.parent / page.get("imageFilename"),
                                     DATA / print_id / page.get("imageFilename")) if p.exists()), None)
            if img is None:
                continue
            im = None
            for tr in page.findall(T + "TextRegion"):
                for tl in tr.findall(T + "TextLine"):
                    if (str(xml), tl.get("id")) in done:
                        continue
                    if only is not None and (print_id, xml.name, tl.get("id")) not in only:
                        continue
                    poly = points(tl)
                    tes = tl.findall(T + "TextEquiv")
                    if len(poly) < 3 or not tes:
                        continue
                    cm = tl.get("comments") or ""
                    src0 = "gemini" if "text=gemini" in cm else "pylaia"
                    t0 = (tes[0].find(T + "Unicode").text or "").strip()
                    t1 = (tes[1].find(T + "Unicode").text or "").strip() if len(tes) > 1 else ""
                    if not t0:
                        continue
                    gem, pyl = (t0, t1) if src0 == "gemini" else (t1, t0)
                    if len(tes) == 1:
                        gem = pyl = t0          # identical readings were stored once
                    if im is None:
                        im = Image.open(img); im.load()
                    tile = crop_line(im, poly, margin=10, side=24)
                    seg = Segmentation(type="bbox", imagename=str(xml), text_direction="horizontal-lr", script_detection=False,
                                       lines=[BBoxLine(id="l", bbox=(0, 0, tile.width, tile.height))])
                    try:
                        rec = list(rpred.rpred(model, tile, seg))[0]
                        k = unicodedata.normalize("NFC", rec.prediction.strip()); conf = sum(rec.confidences) / max(len(rec.confidences), 1)   # model trained in NFD
                    except Exception:  # noqa: BLE001
                        k, conf = "", 0.0
                    stem = f"{print_id.rsplit('_', 1)[-1]}_{xml.stem}_{tl.get('id')}"
                    out.write(json.dumps({"file": str(xml), "print": print_id, "page": xml.name, "region": tr.get("id"), "line_id": tl.get("id"),
                                          "valid": src0, "single": len(tes) == 1, "gemini": gem, "pylaia": pyl, "kraken": k, "conf": round(conf, 3),
                                          "comment": cm, "kraken_split": split.get(stem, "")}, ensure_ascii=False) + "\n")
                    n += 1
                    if n % 5000 == 0:
                        out.flush(); print(f"{n} lines", flush=True)
    print(f"done: {n} new lines")


if __name__ == "__main__":
    main()
