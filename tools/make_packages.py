#!/usr/bin/env python3
"""Archive packages from the TEI export:
  dist/packages/<print>_<revision>.zip   per print: TEI, per-print JSON, README with citation, checksums (Zotero attachment)
  dist/zenodo/                           one deposit: all TEI files, README.md, CITATION.cff, sha256sums.txt

    python src/export_tei.py [--doi 10.5281/zenodo.NNN] && python tools/make_packages.py
Zenodo: create the deposit from dist/zenodo/ (or let the GitHub integration pick up a release; CITATION.cff
supplies the metadata), then re-export the TEI with --doi so the header carries the DOI, and republish.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "packages"


def main() -> None:
    manifest = json.loads((ROOT / "site-data" / "manifest.json").read_text(encoding="utf-8"))
    authors = json.loads((ROOT / "output" / "authors.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    zen = ROOT / "dist" / "zenodo"
    zen.mkdir(parents=True, exist_ok=True)
    sums = []
    for tei in sorted((ROOT / "dist" / "tei").glob("*.xml")):
        (zen / tei.name).write_bytes(tei.read_bytes())
        sums.append(f"{hashlib.sha256(tei.read_bytes()).hexdigest()}  {tei.name}")
        print_id = tei.stem
        viaf = next((k for k, a in authors.items() if print_id in a.get("prints", [])), None)
        per_print = ROOT / "output" / (viaf or "") / f"{print_id}.json"
        files = [tei] + ([per_print] if per_print.exists() else [])
        readme = (f"# {print_id}\n\nAuthor: {authors.get(viaf, {}).get('name', '')} (VIAF {authors.get(viaf, {}).get('viaf', '')})\n"
                  f"Edition: Consilia. A portal for medieval legal opinions (beta), ed. Michael Schonhardt (ORCID 0000-0002-2750-1900), "
                  f"Technische Universität Darmstadt. Data revision {manifest['revision']}. https://dekm-tud.github.io/coding-consilia/\n"
                  f"Status: machine transcription and segmentation, unreviewed; machine translations and keywords.\n"
                  f"Licence: CC BY 4.0 (text data). Page images are not included; see the sourceDesc of the TEI header.\n\nFiles:\n" +
                  "".join(f"- {f.name}  sha256 {hashlib.sha256(f.read_bytes()).hexdigest()}\n" for f in files) +
                  f"\nCite as: {authors.get(viaf, {}).get('name', '')}, Consilia, {print_id}, in: Consilia. Digital edition (beta), data revision {manifest['revision']}.\n")
        zp = OUT / f"{print_id}_{manifest['revision']}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for f in files:
                z.write(f, f.name)
            z.writestr("README.md", readme)
        print(f"{zp.relative_to(ROOT)}  {zp.stat().st_size / 1e6:.1f} MB")
    (zen / "sha256sums.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    cff = ROOT / "CITATION.cff"
    if cff.exists():
        (zen / "CITATION.cff").write_bytes(cff.read_bytes())
    (zen / "README.md").write_text(
        "# Consilia: TEI data\n\n"
        "One TEI P5 file per printed volume (see the teiHeader of each file for source, responsibilities, licence and editorial declarations). "
        f"Data revision {manifest['revision']}, {manifest.get('n_consilia', '')} consilia. Edition: Michael Schonhardt (Technische Universität Darmstadt), "
        "https://dekm-tud.github.io/coding-consilia/ , code and data: https://github.com/dekm-tud/coding-consilia .\n\n"
        "Status: beta. Transcriptions are machine-made and not proof-read; translations, keywords and subject layers are machine-made working layers. "
        "Licence: CC BY 4.0 for the text data; the page images belong to the holding libraries and are linked, not included.\n\n"
        "Files: " + ", ".join(s.split("  ")[1] for s in sums) + "; checksums in sha256sums.txt; citation metadata in CITATION.cff.\n", encoding="utf-8")
    print(f"dist/zenodo: {len(sums)} TEI files, README.md, CITATION.cff, sha256sums.txt")


if __name__ == "__main__":
    main()
