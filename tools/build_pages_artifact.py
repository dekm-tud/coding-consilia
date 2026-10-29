#!/usr/bin/env python3
"""Assemble the GitHub Pages artifact and check its size.

Published: index.html, search.html, about.html, chat.html, assets/ (fonts, vendored viewer), site-data/,
data/**/img/ (AVIF derivatives) and hf-export/*/chunks.f16.bin + chunks.json (the local copy of
the embeddings until the public HF dataset is live; remove it from PUBLISH once models.json
points to Hugging Face). NOT published: PageXML, JPG originals, output/, src/, tests/.

    python tools/build_pages_artifact.py            # -> _site/, exit 1 on a limit violation
    python tools/build_pages_artifact.py --check-only _site
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
MAX_TOTAL = 900 * 1024 * 1024      # GitHub Pages: 1 GB published site
MAX_FILE = 95 * 1024 * 1024        # GitHub: 100 MB per file
FIRST_LOAD_BUDGET = 2 * 1024 * 1024  # reader first load (index + largest text shard + one page image), gzip

PUBLISH_FILES = ["index.html", "search.html", "about.html", "chat.html", "introduction.html", "baldus.html", "panormitanus.html", "bibliography.html", ".nojekyll", "robots.txt"]
PUBLISH_DIRS = ["assets", "site-data"]


def collect(all_local_images: bool = False) -> list[Path]:
    files: list[Path] = []
    for f in PUBLISH_FILES:
        if (ROOT / f).exists():
            files.append(ROOT / f)
    for d in PUBLISH_DIRS:
        files += [p for p in (ROOT / d).rglob("*") if p.is_file()]
    # local derivatives only for prints whose image_sources entry is not an absolute URL (external host)
    import json
    srcs = json.loads((ROOT / "site-data" / "image_sources.json").read_text(encoding="utf-8")) if (ROOT / "site-data" / "image_sources.json").exists() else {}
    external = {k for k, v in srcs.items() if isinstance(v, dict) and str(v.get("path_template", "")).startswith("http")}
    iiif = json.loads((ROOT / "site-data" / "iiif_manifests.json").read_text(encoding="utf-8")) if (ROOT / "site-data" / "iiif_manifests.json").exists() else {}
    if not all_local_images:
        external |= {k for k in iiif if not k.startswith("_")}      # prints served from a IIIF manifest need no local derivatives
    files += [p for p in (ROOT / "data").rglob("*.avif") if p.parent.parent.name not in external]
    for p in (ROOT / "hf-export").glob("*/*") if (ROOT / "hf-export").exists() else []:
        if p.name in ("chunks.f16.bin", "chunks.json"):
            files.append(p)
    return files


def gz(p: Path) -> int:
    return len(gzip.compress(p.read_bytes(), 6))


def check(root: Path) -> list[str]:
    problems = []
    files = [p for p in root.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    print(f"{len(files)} files, {total / 1e6:.1f} MB")
    if total > MAX_TOTAL:
        problems.append(f"artifact is {total / 1e6:.0f} MB > {MAX_TOTAL / 1e6:.0f} MB")
    for p in files:
        if p.stat().st_size > MAX_FILE:
            problems.append(f"{p.relative_to(root)} is {p.stat().st_size / 1e6:.0f} MB > 95 MB")
    # reader first load: index + the largest text shard + one page image (gzip for JSON, as Pages serves it)
    idx = root / "site-data" / "index.json"
    shards = list((root / "site-data" / "text").glob("*.json"))
    imgs = list(root.glob("data/**/img/*.avif"))
    if idx.exists() and shards and imgs:
        first = gz(idx) + max(gz(s) for s in shards) + max(i.stat().st_size for i in imgs)
        print(f"reader first load (worst case, gzip): {first / 1e6:.2f} MB (budget {FIRST_LOAD_BUDGET / 1e6:.0f} MB)")
        if first > FIRST_LOAD_BUDGET:
            problems.append(f"reader first load {first / 1e6:.2f} MB exceeds {FIRST_LOAD_BUDGET / 1e6:.0f} MB")
    for must in ("index.html", "search.html", "site-data/index.json", "site-data/manifest.json", "site-data/models.json"):
        if not (root / must).exists():
            problems.append(f"missing {must}")
    return problems


def write_offline_helpers(root: Path) -> None:
    """Start scripts and a short README for running the site from a folder (presentations, offline use)."""
    (root / "serve.bat").write_text("@echo off\r\ncd /d %~dp0\r\nstart http://localhost:8765/\r\npython -m http.server 8765\r\n", encoding="ascii")
    (root / "serve.sh").write_text("#!/usr/bin/env sh\ncd \"$(dirname \"$0\")\"\n(sleep 1; xdg-open http://localhost:8765/ 2>/dev/null || open http://localhost:8765/) &\npython3 -m http.server 8765\n", encoding="utf-8")
    (root / "serve.sh").chmod(0o755)
    (root / "README-offline.txt").write_text(
        "Consilia portal, local copy\n\n"
        "Start: double-click serve.bat (Windows, needs Python 3 on the PATH) or run ./serve.sh (macOS, Linux).\n"
        "Then open http://localhost:8765/ . Do not open index.html directly from the file system: the pages load\n"
        "their data with fetch(), which browsers block for file:// URLs.\n\n"
        "Works without internet: reader with page images of the Baldus volumes, keyword search, Index with the\n"
        "annotation layers, Similar mode, map and statistics.\n"
        "Needs internet: the Panormitanus facsimile (Harvard IIIF), the Ask page (the language-model API of your key),\n"
        "and the first use of a semantic query model (downloaded from Hugging Face once, then kept by the browser;\n"
        "load it in search.html while online before presenting offline). The cite/permalink links point to the\n"
        "public site.\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "_site")
    ap.add_argument("--check-only", type=Path)
    ap.add_argument("--offline-bundle", type=Path, help="also write serve scripts and a README into the artifact and zip it to this path")
    args = ap.parse_args()
    target = args.check_only
    if not target:
        if args.out.exists():
            shutil.rmtree(args.out)
        for f in collect(all_local_images=bool(args.offline_bundle)):
            dst = args.out / f.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dst)
        target = args.out
        if args.offline_bundle:
            # offline: prints with local derivatives are shown from them, not from a remote IIIF manifest
            import json
            reg = target / "site-data" / "iiif_manifests.json"
            srcs = json.loads((target / "site-data" / "image_sources.json").read_text(encoding="utf-8"))
            local = {k for k, v in srcs.items() if isinstance(v, dict) and v.get("kind") == "local"}
            iiif = json.loads(reg.read_text(encoding="utf-8"))
            reg.write_text(json.dumps({k: v for k, v in iiif.items() if k not in local}, ensure_ascii=False, indent=1), encoding="utf-8")
            write_offline_helpers(target)
            shutil.make_archive(str(args.offline_bundle.with_suffix("")), "zip", root_dir=target.parent, base_dir=target.name)
            print(f"offline bundle: {args.offline_bundle} ({args.offline_bundle.stat().st_size / 1e6:.0f} MB)")
    problems = check(target)
    for p in problems:
        print("FAIL:", p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
