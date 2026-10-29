#!/usr/bin/env python3
"""
Download the page images of this dataset from the Internet Archive and put each next to its PageXML file
(<print>/<page>.jpg, the file name the XML refers to). The URLs in pages.csv point to the original files the
PageXML was made on (same bytes, same pixel size); every download is checked against the SHA-1 in pages.csv.
Files already present and correct are skipped, so the script can be run again after an interruption.

    python fetch_images.py                       # all 1,392 pages (about 5.2 GB)
    python fetch_images.py --split test          # only the pages of one split (splits/<split>.txt)
    python fetch_images.py --print Baldo_Cons_Print_Venice_1575_v3 --workers 8
Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(row: dict, tries: int = 4) -> str:
    dst = ROOT / row["print"] / f"{row['page']}.jpg"
    if dst.exists() and sha1(dst) == row["image_sha1"]:
        return "present"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    for k in range(tries):
        try:
            req = urllib.request.Request(row["image_url"], headers={"User-Agent": "consilia-pagexml-dataset/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
                while block := r.read(1 << 20):
                    f.write(block)
            if sha1(tmp) != row["image_sha1"]:
                raise ValueError("checksum differs")
            tmp.replace(dst)
            return "downloaded"
        except Exception as e:  # noqa: BLE001 - retried, reported at the end
            err = e
            time.sleep(5 * (k + 1))
    tmp.unlink(missing_ok=True)
    return f"FAILED: {err}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--print", dest="print_filter")
    ap.add_argument("--split", choices=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    rows = list(csv.DictReader((ROOT / "pages.csv").open(encoding="utf-8")))
    if a.print_filter:
        rows = [r for r in rows if r["print"] == a.print_filter]
    if a.split:
        rows = [r for r in rows if r["split"] == a.split]
    failed = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch, r): r for r in rows}
        for i, f in enumerate(as_completed(futs), 1):
            res = f.result(); r = futs[f]
            if res.startswith("FAILED"):
                failed += 1; print(f"{r['print']}/{r['page']}: {res}", file=sys.stderr)
            if i % 50 == 0 or i == len(rows):
                print(f"{i}/{len(rows)} pages", flush=True)
    sys.exit(f"{failed} pages failed; run again" if failed else 0)


if __name__ == "__main__":
    main()
