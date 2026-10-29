#!/usr/bin/env python3
"""
Map every Baldus page image (data/Baldo_29618397/<print>/<page>.jpg) to its original file in the Internet Archive
item of the volume (<item>_images.zip, uploaded by the project; the files there are the same bytes). The zip
members are listed by archive.org; member k of the sorted list is page k of the sorted local pages, and every pair
is checked by file size (and, with --sha1, by hash of the local file only; the remote hash is checked by the
dataset's fetch script). Writes output/ia_images.json: {print: {page: {"url", "bytes", "sha1"}}}.

    python tools/ia_image_map.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Baldo_29618397"
OUT = ROOT / "output" / "ia_images.json"
ROW = re.compile(r'<a href="//archive\.org/download/([^"]+\.jpg)">[^<]*</a><td><td>[^<]*<td id="size">(\d+)</tr>')


def main() -> None:
    reg = json.loads((ROOT / "site-data" / "iiif_manifests.json").read_text(encoding="utf-8"))
    out, bad = {}, 0
    for prt in sorted(p.name for p in DATA.iterdir() if p.is_dir() and p.name.startswith("Baldo_Cons_Print_")):
        item = reg[prt]["manifest"].split("/")[-2]
        html = urllib.request.urlopen(f"https://archive.org/download/{item}/{item}_images.zip/", timeout=120).read().decode("utf-8")
        members = sorted(ROW.findall(html))
        pages = sorted(DATA.glob(f"{prt}/*.jpg"), key=lambda p: int(re.sub(r"\D", "", p.stem)))
        if len(members) != len(pages):
            sys.exit(f"{prt}: {len(members)} files in the archive, {len(pages)} local pages")
        m = {}
        for (path, size), img in zip(members, pages):
            n = img.stat().st_size
            if int(size) != n:
                bad += 1; print(f"  size differs: {prt} {img.stem} local {n} archive {size}")
            m[img.stem] = {"url": "https://archive.org/download/" + urllib.parse.quote(urllib.parse.unquote(path), safe="/"),
                           "bytes": n, "sha1": hashlib.sha1(img.read_bytes()).hexdigest()}
        out[prt] = m
        print(f"{prt}: {len(m)} pages, item {item}")
    if bad:
        sys.exit(f"{bad} pages differ in size; nothing written")
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("->", OUT)


if __name__ == "__main__":
    main()
