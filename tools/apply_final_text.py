#!/usr/bin/env python3
"""
Write the edition text (output/merge/final_text.jsonl from tools/final_text.py) into a PageXML copy.

Every TextLine with a decision gets its TextEquiv elements rebuilt:
  index 0  the edition text (conf 1.0, comments "final")          <- read by src/parse_pagexml.py
  index 1  Gemini's reading (comments "gemini")
  index 2  PyLaia's reading (comments "pylaia")
  index 3  kraken's reading (comments "kraken", conf = kraken's mean confidence)
and "; final <date>: <how the text was decided>" is appended to its comments. Lines without a decision
are left alone.

    python tools/apply_final_text.py [--target output/merge/Baldo_29618397]

Writing to data/ needs --target data/Baldo_29618397 --i-mean-data (done only on the editor's word).
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"; T = "{%s}" % NS
MERGE = ROOT / "output" / "merge"
ET.register_namespace("", NS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", type=Path, default=MERGE / "Baldo_29618397")
    ap.add_argument("--i-mean-data", action="store_true")
    a = ap.parse_args()
    if "data" in a.target.parts and not a.i_mean_data:
        raise SystemExit("refusing to write data/ without --i-mean-data")
    final, reads = {}, {}
    for l in (MERGE / "final_text.jsonl").open(encoding="utf-8"):
        r = json.loads(l); final[(r["print"], r["page"], r["line_id"])] = r
    for l in (MERGE / "kraken_current.jsonl").open(encoding="utf-8"):
        r = json.loads(l); reads[(r["print"], r["page"], r["line_id"])] = r
    today = datetime.date.today().isoformat()
    n = 0
    for f in sorted(a.target.glob("*/page/*.xml")):
        prt, page = f.parts[-3], f.name
        tree = ET.parse(f); changed = False
        for tl in tree.getroot().iter(T + "TextLine"):
            k = (prt, page, tl.get("id"))
            if k not in final or k not in reads:
                continue
            fr, rd = final[k], reads[k]
            for te in tl.findall(T + "TextEquiv"):
                tl.remove(te)
            for i, (txt, cm, conf) in enumerate(((fr["text"], "final", "1.0"), (rd["gemini"], "gemini", "0.5"),
                                                  (rd["pylaia"], "pylaia", "0.5"), (rd["kraken"], "kraken", str(rd.get("conf", 0))))):
                te = ET.SubElement(tl, T + "TextEquiv", {"index": str(i), "conf": conf, "comments": cm})
                ET.SubElement(te, T + "Unicode").text = txt
            base = re.sub(r"; final [0-9-]+: [^;]*(; [^;]*kept as it is)?", "", tl.get("comments") or "")
            base = re.sub(r"(; final [0-9-]+:.*)$", "", base)          # a final note is always the last one
            tl.set("comments", (base + f"; final {today}: {fr['how']}").lstrip("; "))
            changed = True; n += 1
        if changed:
            tree.write(f, encoding="UTF-8", xml_declaration=True)
    print(f"{n} lines written to {a.target}")


if __name__ == "__main__":
    main()
