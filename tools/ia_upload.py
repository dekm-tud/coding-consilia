#!/usr/bin/env python3
"""
Upload the page images of the Baldus volumes (Complutense copy, HathiTrust page images) to the
Internet Archive as public-domain book items, so that each volume gets a stable IIIF manifest.

    python tools/ia_upload.py build            # dist/ia/<identifier>/<identifier>_images.zip + metadata.json
    python tools/ia_upload.py upload           # needs `ia configure` once (your archive.org login); resumable
    python tools/ia_upload.py upload --print Baldo_Cons_Print_Venice_1575_v1
    python tools/ia_upload.py status           # item URLs, derive state, IIIF manifest URLs

What goes up: the page JPGs as they lie in data/<author>/<print>/ (the HathiTrust page images of the
copy, title page first, no Google cover sheet), zipped without compression under the name the Archive
expects for book scans (<identifier>_images.zip); the Archive derives PDF, OCR and the viewer from it.
Metadata: bibliographic data from the HathiTrust record (title, volume, imprint, extent), creator with
dates, language lat, collection opensource (community texts), Public Domain Mark, and a provenance
description naming the holding library, the HathiTrust item (htid) and the Google/Complutense
digitisation. Nothing is invented: the shelfmark is given as it is written on the title page.
After the upload, add the IIIF manifest https://iiif.archive.org/iiif/3/<identifier>/manifest.json to
site-data/editions.json (other_copies) and, if the images should be served from there, to
output/iiif_manifests.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "ia"
AUTHOR_DIR = ROOT / "data" / "Baldo_29618397"

# HathiTrust catalogue record 009264041 (API: catalog.hathitrust.org/api/volumes/full/htid/<htid>.json)
VOLUMES = {
    "Baldo_Cons_Print_Venice_1575_v1": {"n": 1, "label": "Volumen primum", "htid": "ucm.531017106x", "extent": "161 h."},
    "Baldo_Cons_Print_Venice_1575_v2": {"n": 2, "label": "Volumen secundum", "htid": "ucm.5310171050", "extent": "134 h."},
    "Baldo_Cons_Print_Venice_1575_v3": {"n": 3, "label": "Volumen tertium", "htid": "ucm.5310171023", "extent": "142 h."},
    "Baldo_Cons_Print_Venice_1575_v4": {"n": 4, "label": "Volumen quartum", "htid": "ucm.5310171014", "extent": "117 h."},
    "Baldo_Cons_Print_Venice_1575_v5": {"n": 5, "label": "Volumen quintum", "htid": "ucm.5310171005", "extent": "139 h."},
}
COLLECTION = "opensource"
PORTAL = "https://dekm-tud.github.io/coding-consilia/"


def identifier(print_id: str) -> str:
    return f"baldus-consilia-venetiis-1575-vol{VOLUMES[print_id]['n']}-ucm"


def page_images(print_id: str) -> list[Path]:
    d = AUTHOR_DIR / print_id
    imgs = sorted(d.glob("*.jpg"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    if not imgs:
        imgs = sorted((d / "page").glob("*.jpg"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    return imgs


def metadata(print_id: str, n_pages: int) -> dict:
    v = VOLUMES[print_id]
    ident = identifier(print_id)
    desc = (
        f"<p>Baldus de Ubaldis (1327–1400), <i>Consiliorum sive responsorum volumen {v['label'].split()[-1]}</i>. "
        f"Venice: Girolamo Polo (apud Hieronymum Polum), 1575. Folio, {v['extent']}, 38 cm. Volume {v['n']} of 5.</p>"
        f"<p><b>Copy:</b> Universidad Complutense de Madrid (stamp \"Universidad Central, Biblioteca Derecho\" and handwritten shelfmarks on the title page). "
        f"<b>Digitisation:</b> Google Books / Universidad Complutense de Madrid digitisation programme; page images as served by HathiTrust, "
        f"item <a href=\"https://babel.hathitrust.org/cgi/pt?id={v['htid']}\">{v['htid']}</a> (HathiTrust record 009264041). "
        f"{n_pages} page images in scan order, title page first; no cover sheet added.</p>"
        f"<p><b>Rights:</b> the work and the faithful reproduction are in the public domain (Public Domain Mark 1.0). "
        f"Uploaded for the digital edition <a href=\"{PORTAL}\">Consilia. A portal for medieval legal opinions</a> "
        f"(Michael Schonhardt, Technische Universität Darmstadt), which segments and indexes the text of this edition and links every "
        f"consilium to its page; the IIIF manifest of this item (https://iiif.archive.org/iiif/3/{ident}/manifest.json) is the image source "
        f"of the portal. Transcriptions and data: <a href=\"https://github.com/dekm-tud/coding-consilia\">github.com/dekm-tud/coding-consilia</a>.</p>"
    )
    return {
        "mediatype": "texts",
        "collection": COLLECTION,
        "title": f"Baldi Ubaldi Perusini iurisconsulti Consiliorum sive responsorum {v['label'].lower()} (Venetiis, apud Hieronymum Polum, 1575)",
        "creator": "Ubaldi, Baldo degli, 1327-1400",
        "contributor": "Universidad Complutense de Madrid; HathiTrust; Google Books",
        "publisher": "Venetiis : apud Hieronymum Polum",
        "date": "1575",
        "year": "1575",
        "language": "lat",
        "volume": str(v["n"]),
        "subject": ["consilia", "ius commune", "Roman law", "canon law", "legal opinions", "Baldus de Ubaldis", "early printed books", "Venice 1575"],
        "description": desc,
        "licenseurl": "https://creativecommons.org/publicdomain/mark/1.0/",
        "rights": "Public domain",
        "source": f"https://babel.hathitrust.org/cgi/pt?id={v['htid']}",
        "external-identifier": [f"urn:hathitrust:{v['htid']}"],
        "page-progression": "lr",
        "scanner": "Google Books / Universidad Complutense de Madrid (page images via HathiTrust); item assembled by the Consilia project",
        "related-external-id": [f"hathitrust:{v['htid']}"],
    }


def build(print_id: str, force: bool = False) -> Path:
    ident = identifier(print_id)
    d = OUT / ident
    d.mkdir(parents=True, exist_ok=True)
    imgs = page_images(print_id)
    if not imgs:
        sys.exit(f"{print_id}: no page JPGs in {AUTHOR_DIR / print_id}")
    zp = d / f"{ident}_images.zip"
    if not zp.exists() or force:
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
            for i, p in enumerate(imgs, 1):
                z.write(p, f"{ident}_images/{ident}_{i:04d}.jpg")
    (d / "metadata.json").write_text(json.dumps(metadata(print_id, len(imgs)), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{ident}: {len(imgs)} pages, {zp.stat().st_size / 1e9:.2f} GB -> {zp.relative_to(ROOT)}")
    return d


def upload(print_id: str) -> None:
    from internetarchive import get_item, get_session
    s = get_session()
    if not s.config.get("s3", {}).get("access"):
        sys.exit("No archive.org credentials: run  .venv/bin/ia configure  once (your archive.org e-mail and password).")
    ident = identifier(print_id)
    d = OUT / ident
    if not (d / "metadata.json").exists():
        build(print_id)
    md = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    item = get_item(ident)
    print(f"{ident}: uploading {d / (ident + '_images.zip')} ({'existing item' if item.exists else 'new item'}) …", flush=True)
    r = item.upload(str(d / f"{ident}_images.zip"), metadata=md, retries=10, retries_sleep=30, verbose=True, checksum=True, queue_derive=True)
    print(f"  done: {[x.status_code for x in r]}  https://archive.org/details/{ident}")


def status(print_id: str) -> None:
    from internetarchive import get_item
    ident = identifier(print_id)
    item = get_item(ident)
    if not item.exists:
        print(f"{ident}: not on archive.org yet")
        return
    files = [f.name for f in item.get_files()]
    print(f"{ident}: https://archive.org/details/{ident}  files {len(files)}  pdf {'yes' if any(f.endswith('.pdf') for f in files) else 'pending'}  "
          f"IIIF https://iiif.archive.org/iiif/3/{ident}/manifest.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["build", "upload", "status"])
    ap.add_argument("--print", dest="print_filter")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for print_id in VOLUMES:
        if args.print_filter and print_id != args.print_filter:
            continue
        {"build": lambda: build(print_id, args.force), "upload": lambda: upload(print_id), "status": lambda: status(print_id)}[args.cmd]()


if __name__ == "__main__":
    main()
