#!/usr/bin/env python3
"""
Third vote for the disputed lines: the kraken model trained on the consensus lines
(tools/kraken_export.py, ketos train) reads every line whose two readings still differ and that
nobody has reviewed; if its reading agrees with one of the two (loose normalisation of
src/qa/compare_ocr.py), that reading becomes the valid one, with the comment
"reviewed <date>: kraken gemini|pylaia". Lines where it agrees with neither stay for the tool.

    python tools/kraken_arbitrate.py --model output/kraken/model/consilia_e0.mlmodel [--max-cer 1.0] [--limit N] [--apply]

Without --apply only the report output/qa/kraken_arbitration.jsonl is written (one record per line:
file, line id, readings, kraken reading, decision). Run in the kraken environment (.venv-kraken).
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "src" / "qa"))
from kraken_export import crop_line, points, T  # noqa: E402
from compare_ocr import normalise, dist  # noqa: E402

DATA = ROOT / "data" / "Baldo_29618397"


def disputed_lines():
    for xml in sorted(DATA.glob("*/page/*.xml")):
        root = ET.parse(xml).getroot(); page = root.find(T + "Page")
        img = next((p for p in (xml.parents[1] / page.get("imageFilename"), xml.parent / page.get("imageFilename")) if p.exists()), None)
        if img is None:
            continue
        im = None
        for tr in page.findall(T + "TextRegion"):
            for tl in tr.findall(T + "TextLine"):
                cm = tl.get("comments") or ""
                tes = tl.findall(T + "TextEquiv")
                if len(tes) < 2 or "reviewed " in cm:
                    continue
                t0 = (tes[0].find(T + "Unicode").text or "").strip(); t1 = (tes[1].find(T + "Unicode").text or "").strip()
                if not t0 or not t1 or normalise(t0, True) == normalise(t1, True):
                    continue
                poly = points(tl)
                if len(poly) < 3:
                    continue
                if im is None:
                    im = Image.open(img); im.load()
                yield xml, tr.get("id"), tl, t0, t1, cm, crop_line(im, poly, margin=10, side=24)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--max-cer", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--tolerance", type=float, default=0.06, help="max distance kraken -> chosen reading")
    ap.add_argument("--gap", type=float, default=0.03, help="min lead over the other reading")
    ap.add_argument("--report", type=Path, default=ROOT / "output" / "qa" / "kraken_arbitration.jsonl")
    a = ap.parse_args()
    from kraken.lib import models
    from kraken import rpred
    from kraken.containers import Segmentation, BBoxLine
    model = models.load_any(str(a.model), device="cuda:0")
    if a.apply:
        from review_server import apply_decision
    stats = {"lines": 0, "gemini": 0, "pylaia": 0, "neither": 0, "both": 0}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    with a.report.open("w", encoding="utf-8") as rep:
        for xml, region, tl, t0, t1, cm, tile in disputed_lines():
            src0 = "gemini" if "text=gemini" in cm else "pylaia"
            gem, pyl = (t0, t1) if src0 == "gemini" else (t1, t0)
            g, p = normalise(gem, True), normalise(pyl, True)
            cer = dist(g, p) / max(len(g), len(p), 1)
            if cer > a.max_cer:
                continue
            seg = Segmentation(type="bbox", imagename=str(xml), text_direction="horizontal-lr", script_detection=False,
                               lines=[BBoxLine(id="l1", bbox=(0, 0, tile.width, tile.height))])
            try:
                rec = list(rpred.rpred(model, tile, seg))[0]
                k = unicodedata.normalize("NFC", rec.prediction.strip()); conf = sum(rec.confidences) / max(len(rec.confidences), 1)   # model trained in NFD
            except Exception as e:  # noqa: BLE001
                k, conf = "", 0.0
            kn = normalise(k, True)
            # the first letters of a tile are often clipped by the polygon: compare from the fourth character on
            kk, gg, pp = kn[3:], g[3:], p[3:]
            dg, dp = dist(kk, gg) / max(len(kk), len(gg), 1), dist(kk, pp) / max(len(kk), len(pp), 1)
            # the closer reading wins when it is close and clearly closer than the other one
            if kn and dg == 0 and dp == 0:
                choice = "both"
            elif kn and dg <= a.tolerance and dg + a.gap <= dp:
                choice = "gemini"
            elif kn and dp <= a.tolerance and dp + a.gap <= dg:
                choice = "pylaia"
            else:
                choice = "neither"
            stats["lines"] += 1; stats[choice] += 1
            rec_out = {"file": str(xml), "region": region, "line_id": tl.get("id"), "valid": src0, "gemini": gem, "pylaia": pyl,
                       "kraken": k, "conf": round(conf, 3), "cer_g": round(dg, 3), "cer_p": round(dp, 3), "cer_readings": round(cer, 3), "choice": choice}
            rep.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            if a.apply and choice in ("gemini", "pylaia"):
                apply_decision({"file": str(xml), "region": region, "line_id": tl.get("id"), "kind": "readings differ", "valid": src0, "gemini": gem, "pylaia": pyl},
                               choice, "", who="kraken")
            if a.limit and stats["lines"] >= a.limit:
                break
    print(stats)


if __name__ == "__main__":
    main()
