#!/usr/bin/env python3
"""
Download the latest PageXML of every page of the Baldus documents from Transkribus.

Credentials: ~/.config/transkribus.env (TRANSKRIBUS_USER, TRANSKRIBUS_PASSWORD). Read-only: only
login, fulldoc and the transcript files are requested.

    python tools/transkribus_download.py [--out data_transkribus/2026-10-23]

Writes <out>/<print>/page/<image stem>.xml and <out>/<print>/transcripts.json (page number, image,
transcript id, timestamp, tool, status). data/ is not touched.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
B = "https://transkribus.eu/TrpServer/rest"
COLLECTION = 2217442
DOCS = {
    "Baldo_Cons_Print_Venice_1575_v1": 13119248,
    "Baldo_Cons_Print_Venice_1575_v2": 16294658,
    "Baldo_Cons_Print_Venice_1575_v3": 16294539,
    "Baldo_Cons_Print_Venice_1575_v4": 16294581,
    "Baldo_Cons_Print_Venice_1575_v5": 16294564,
}


def session() -> requests.Session:
    env = dict(l.split("=", 1) for l in (pathlib.Path.home() / ".config" / "transkribus.env").read_text().splitlines() if "=" in l)
    env = {k.upper(): v.strip().strip('"') for k, v in env.items()}
    s = requests.Session()
    r = s.post(B + "/auth/login", data={"user": env["TRANSKRIBUS_USER"], "pw": env["TRANSKRIBUS_PASSWORD"]})
    r.raise_for_status()
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "data_transkribus" / "2026-10-23")
    a = ap.parse_args()
    s = session()
    for prt, doc in DOCS.items():
        fd = s.get(f"{B}/collections/{COLLECTION}/{doc}/fulldoc", headers={"Accept": "application/json"}).json()
        out = a.out / prt / "page"; out.mkdir(parents=True, exist_ok=True)
        meta = []
        for p in fd["pageList"]["pages"]:
            ts = p["tsList"]["transcripts"][0]
            meta.append({"page": p["pageNr"], "image": p["imgFileName"], "tsId": ts["tsId"], "timestamp": ts["timestamp"],
                         "tool": ts.get("toolName"), "status": ts.get("status"), "url": ts["url"],
                         "file": pathlib.Path(p["imgFileName"]).stem + ".xml"})

        def get(m):
            f = out / m["file"]
            if not f.exists():
                r = s.get(m["url"]); r.raise_for_status(); f.write_bytes(r.content)
            return m["file"]

        with ThreadPoolExecutor(8) as ex:
            n = sum(1 for _ in ex.map(get, meta))
        (a.out / prt / "transcripts.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"{prt}: {n} pages", file=sys.stderr)


if __name__ == "__main__":
    main()
