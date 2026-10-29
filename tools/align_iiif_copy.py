#!/usr/bin/env python3
"""Map the pages of a print to the canvases of ANOTHER digitised copy on the Internet Archive.

The other copy has extra or missing scans (fly-leaves, plates, duplicates), so a constant offset is
not enough. The Archive publishes a text layer for every scan (<item>_djvu.xml); this script compares
it, page by page, with our transcription (sets of normalised words) and computes a monotonic
alignment: every local page gets one canvas, in increasing order. Nothing is OCRed here; the
Archive's existing text is read once and only used to find the page correspondence.

    python tools/align_iiif_copy.py --print Baldo_Cons_Print_Venice_1575_v1 --ia bub_gb_PHN5KxDdwVEC
    python tools/align_iiif_copy.py ... --dry-run          # report only

The result goes to output/iiif_manifests.json as `canvas_map` {stem: canvas index}; the reader
(assets/viewer.js) and the TEI export use it instead of a constant offset. Weak matches (blank or
damaged pages) are listed; check them in the reader. Canvas i of the IIIF v3 manifest
https://iiif.archive.org/iiif/3/<item>/manifest.json corresponds to the i-th page of the text layer.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from parse_pagexml import parse_pagexml  # noqa: E402

CACHE = ROOT / "output" / "debug" / "ia"
REG = ROOT / "output" / "iiif_manifests.json"
MIN_LEN = 5


def norm_set(text: str) -> set[str]:
    t = text.lower().replace("ſ", "s").replace("æ", "e").replace("ę", "e").replace("œ", "e").replace("ae", "e")
    t = t.replace("j", "i").replace("v", "u")
    return {w for w in re.findall(r"[a-z]+", t) if len(w) >= MIN_LEN}


def ia_pages(item: str) -> list[set[str]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{item}_djvu.xml"
    if not f.exists() or f.stat().st_size < 10000:
        url = f"https://archive.org/download/{item}/{item}_djvu.xml"
        print("downloading", url)
        req = urllib.request.Request(url, headers={"User-Agent": "consilia-align/1.0"})
        with urllib.request.urlopen(req, timeout=600) as r:
            f.write_bytes(r.read())
    root = ET.parse(f).getroot()
    return [norm_set(" ".join(w.text or "" for w in obj.iter("WORD"))) for obj in root.iter("OBJECT")]


def local_pages(print_id: str) -> list[tuple[str, set[str]]]:
    for author_dir in (ROOT / "data").iterdir():
        pdir = author_dir / print_id
        if pdir.is_dir():
            break
    else:
        sys.exit(f"print {print_id} not found")
    out = []
    for xml in sorted((pdir / "page").glob("*.xml"), key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()))):
        page = parse_pagexml(xml)
        out.append((xml.stem, norm_set(" ".join(l.text for r in page.regions for l in r.lines))))
    return out


def align(local: list[set[str]], remote: list[set[str]]):
    n, m = len(local), len(remote)
    sim = np.zeros((n, m), np.float32)
    for i, a in enumerate(local):
        if not a:
            continue
        for j, b in enumerate(remote):
            if b:
                inter = len(a & b)
                if inter:
                    sim[i, j] = inter / len(a | b)
    NEG = -1e9
    best = np.full((n, m), NEG, np.float64)
    back = np.full((n, m), -1, np.int32)
    # non-decreasing alignment: a local page missing from the other copy may share its neighbour's
    # canvas (it then shows up as a weak match) instead of forcing every later page one canvas off
    best[0] = sim[0]
    for i in range(1, n):
        prev = best[i - 1]
        run = np.maximum.accumulate(prev)
        arg = np.zeros(m, np.int32)
        cur = 0
        for j in range(m):
            if prev[j] > prev[cur]:
                cur = j
            arg[j] = cur
        best[i] = run + sim[i]
        back[i] = arg
    j = int(np.argmax(best[n - 1]))
    path = [0] * n
    for i in range(n - 1, -1, -1):
        path[i] = j
        if i > 0:
            j = int(back[i, j])
    return path, sim


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--print", dest="print_id", required=True)
    ap.add_argument("--ia", required=True, help="Internet Archive identifier")
    ap.add_argument("--source", default="Biblioteca Nazionale Centrale di Roma, via Internet Archive (Google digitisation)")
    ap.add_argument("--min-sim", type=float, default=0.12, help="matches below this Jaccard are listed as weak")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    local = local_pages(a.print_id)
    remote = ia_pages(a.ia)
    print(f"{a.print_id}: {len(local)} local pages; {a.ia}: {len(remote)} canvases with text")
    path, sim = align([s for _, s in local], remote)
    scores = [float(sim[i, path[i]]) for i in range(len(local))]
    weak = [(local[i][0], path[i], round(scores[i], 3)) for i in range(len(local)) if scores[i] < a.min_sim]
    changes = [(local[i][0], path[i - 1] - (i - 1), path[i] - i) for i in range(1, len(local)) if path[i] - i != path[i - 1] - (i - 1)]
    print(f"median Jaccard {float(np.median(scores)):.3f}; {len(weak)} weak pages (< {a.min_sim}); "
          f"offset (canvas index minus 0-based page index) starts at {path[0]} and changes at: {changes}")
    for w in weak[:25]:
        print("  weak:", w)
    if a.dry_run:
        return
    reg = json.loads(REG.read_text(encoding="utf-8"))
    reg[a.print_id] = {
        "manifest": f"https://iiif.archive.org/iiif/3/{a.ia}/manifest.json",
        "canvas_map": {local[i][0]: path[i] for i in range(len(local))},
        "image_api": "https://iiif.io/api/image/3/level2.json",
        "source": a.source,
        "rights": "Public Domain Mark 1.0 (as stated by the Internet Archive item)",
        "aligned_by": "tools/align_iiif_copy.py (our transcription matched against the Archive's text layer; no OCR run here)",
        "weak_pages": [w[0] for w in weak],
    }
    REG.write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {REG.relative_to(ROOT)} entry for {a.print_id}; now run python src/build_site_data.py")


if __name__ == "__main__":
    main()
