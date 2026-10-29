"""
Write web derivatives of the page scans and check that every image
matches the dimensions its PageXML declares.

For each print under data/{author}/{print}/ this writes  img/{stem}.avif  (default 1400 px
wide, AVIF quality 45, metadata stripped; ~230 KB instead of ~3.6 MB) and records the
dimensions per page in site-data/image_dims.json:

    {"<print id>": {"<stem>": {"w": orig_w, "h": orig_h, "dw": deriv_w, "dh": deriv_h,
                               "xml_w": ..., "xml_h": ...}}}

Overlays (regions from PageXML) are scaled by  dw / xml_w  (== dw / w when the check passes).
Idempotent: existing derivatives are skipped unless --force.

Usage:
    python src/make_image_derivatives.py                 # all prints
    python src/make_image_derivatives.py --check         # dimension check only, no writing
    python src/make_image_derivatives.py --print Baldo_Cons_Print_Venice_1575_v4 --limit 5
    python src/make_image_derivatives.py --width 1400 --quality 45 --workers 4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
DIMS_PATH = ROOT / "site-data" / "image_dims.json"
_RE_IMG = re.compile(r'imageFilename="([^"]+)"\s+imageWidth="(\d+)"\s+imageHeight="(\d+)"')


def find_prints(author_filter: str | None, print_filter: str | None) -> list[tuple[str, str, Path]]:
    out = []
    for author in sorted(p for p in DATA.iterdir() if p.is_dir()):
        if author_filter and author.name != author_filter:
            continue
        for pr in sorted(p for p in author.iterdir() if p.is_dir() and (p / "page").is_dir()):
            if pr.name.endswith("_backup") or (print_filter and pr.name != print_filter):
                continue
            out.append((author.name, pr.name, pr))
    return out


def _job(args: tuple) -> tuple[str, str, dict]:
    print_id, stem, jpg, xml, out, width, quality, force = args
    m = _RE_IMG.search(Path(xml).read_text(encoding="utf-8")) if xml else None
    xml_w, xml_h = (int(m.group(2)), int(m.group(3))) if m else (None, None)
    with Image.open(jpg) as im:
        w, h = im.size
        dw, dh = min(width, w), round(h * min(width, w) / w)
        if force or not Path(out).exists():
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            im.convert("RGB").resize((dw, dh), Image.LANCZOS).save(out, "AVIF", quality=quality, speed=6)
    return print_id, stem, {"w": w, "h": h, "dw": dw, "dh": dh, "xml_w": xml_w, "xml_h": xml_h}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--author")
    ap.add_argument("--print", dest="print_filter")
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--quality", type=int, default=45)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="only the first N pages per print (test)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true", help="only compare image and PageXML dimensions")
    args = ap.parse_args()

    jobs = []
    missing = []
    for author, pid, pdir in find_prints(args.author, args.print_filter):
        xmls = sorted((pdir / "page").glob("*.xml"))[: args.limit]
        for x in xmls:
            jpg = pdir / f"{x.stem}.jpg"
            if not jpg.exists():
                missing.append(f"{pid}/{x.stem}")
                continue
            jobs.append((pid, x.stem, str(jpg), str(x), str(pdir / "img" / f"{x.stem}.avif"),
                         args.width, args.quality, args.force))
    print(f"{len(jobs)} pages, {len(missing)} without a JPG")

    dims: dict[str, dict] = {}
    if args.check:
        for j in jobs:
            with Image.open(j[2]) as im:
                w, h = im.size
            m = _RE_IMG.search(Path(j[3]).read_text(encoding="utf-8"))
            dims.setdefault(j[0], {})[j[1]] = {"w": w, "h": h, "xml_w": int(m.group(2)), "xml_h": int(m.group(3))}
    else:
        with ProcessPoolExecutor(args.workers) as ex:
            for i, (pid, stem, d) in enumerate(ex.map(_job, jobs, chunksize=4), 1):
                dims.setdefault(pid, {})[stem] = d
                if i % 100 == 0:
                    print(f"  {i}/{len(jobs)}", flush=True)

    bad = [(pid, s, d) for pid, pages in dims.items() for s, d in pages.items()
           if (d["w"], d["h"]) != (d["xml_w"], d["xml_h"])]
    for pid, s, d in bad[:20]:
        print(f"MISMATCH {pid}/{s}: image {d['w']}x{d['h']} vs PageXML {d['xml_w']}x{d['xml_h']}")
    print(f"dimension check: {len(bad)} mismatches of {sum(len(p) for p in dims.values())} pages")

    if not args.check:
        DIMS_PATH.parent.mkdir(exist_ok=True)
        old = json.loads(DIMS_PATH.read_text(encoding="utf-8")) if DIMS_PATH.exists() else {}
        for pid, pages in dims.items():
            old.setdefault(pid, {}).update(pages)
        DIMS_PATH.write_text(json.dumps(old, separators=(",", ":")), encoding="utf-8")
        print("wrote", DIMS_PATH)
    if missing or bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
