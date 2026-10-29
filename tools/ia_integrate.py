#!/usr/bin/env python3
"""
Switch the Baldus volumes' page images to the Internet Archive items uploaded with tools/ia_upload.py.

For every volume it fetches the item's IIIF Presentation manifest, checks that the number of canvases
equals the number of local pages (the zip was built from exactly those pages in order, so canvas index
= local page number - 1), and then writes:
  output/iiif_manifests.json     manifest URL, canvas_offset 0, Image API v3, source and rights
  site-data/image_sources.json   kind "iiif" (the AVIF derivatives are no longer needed for the reader)
  site-data/editions.json        other_copies: the Archive item and its manifest
Nothing is written unless every manifest is reachable and consistent, or --apply is given per volume.

    python tools/ia_integrate.py            # check only
    python tools/ia_integrate.py --apply    # write the registries, then run src/build_site_data.py
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from ia_upload import VOLUMES, identifier  # noqa: E402


def fetch(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "consilia-portal/1.0"}), timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    iiif_path = ROOT / "output" / "iiif_manifests.json"
    src_path = ROOT / "site-data" / "image_sources.json"
    ed_path = ROOT / "site-data" / "editions.json"
    iiif = json.loads(iiif_path.read_text(encoding="utf-8"))
    sources = json.loads(src_path.read_text(encoding="utf-8"))
    editions = json.loads(ed_path.read_text(encoding="utf-8"))
    ok = True
    plan = {}
    for print_id, v in VOLUMES.items():
        ident = identifier(print_id)
        url = f"https://iiif.archive.org/iiif/3/{ident}/manifest.json"
        n_local = len(list((ROOT / "data" / "Baldo_29618397" / print_id / "page").glob("*.xml")))
        try:
            man = fetch(url)
            n_canvas = len(man.get("items") or [])
        except Exception as e:  # noqa: BLE001
            print(f"{print_id}: manifest not available yet ({type(e).__name__})"); ok = False; continue
        status = "ok" if n_canvas == n_local else "MISMATCH"
        print(f"{print_id}: {n_canvas} canvases, {n_local} local pages -> {status}")
        if status != "ok":
            ok = False; continue
        plan[print_id] = {
            "manifest": url, "canvas_offset": 0, "image_api": "https://iiif.io/api/image/3/level2.json",
            "source": f"Universidad Complutense de Madrid (Google digitisation, via HathiTrust {v['htid']}); Internet Archive item {ident}, page images uploaded by the Consilia project in the order of this edition",
            "rights": "Public Domain Mark 1.0",
        }
    if not ok:
        print("not all manifests are ready or consistent; nothing written"); sys.exit(1)
    if not args.apply:
        print("check passed; run with --apply to write the registries"); return
    for print_id, entry in plan.items():
        old = iiif.get(print_id)
        if old and "bub_gb_" in old.get("manifest", ""):
            editions[print_id].setdefault("other_copies", []).append({"label": "Biblioteca Nazionale Centrale di Roma, via Internet Archive (IIIF, previous image source)", "url": old["manifest"].replace("https://iiif.archive.org/iiif/3/", "https://archive.org/details/").replace("/manifest.json", "")})
        iiif[print_id] = entry
        s = sources.get(print_id, {})
        sources[print_id] = {"kind": "iiif", "full_resolution": "iiif", "rights": "Public Domain Mark 1.0 (Internet Archive item)",
                             "attribution": f"Digitized by Google from the copy at Universidad Complutense de Madrid (HathiTrust {VOLUMES[print_id]['htid']}); page images served from the Internet Archive item {identifier(print_id)}.",
                             "hathitrust": s.get("hathitrust")}
        ident = identifier(print_id)
        e = editions[print_id]
        e["copy"] = f"Universidad Complutense de Madrid (Google digitisation via HathiTrust), page images from the Internet Archive item {ident}"
        e["internet_archive"] = f"https://archive.org/details/{ident}"
        e["iiif_manifest"] = entry["manifest"]
    iiif_path.write_text(json.dumps(iiif, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    src_path.write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ed_path.write_text(json.dumps(editions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("registries written; now: python src/build_site_data.py")


if __name__ == "__main__":
    main()
