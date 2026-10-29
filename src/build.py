"""
Incremental build system for the Consilia pipeline.

Discovers all prints under data/{author_viaf}/{print_id}/ and reprocesses
only those whose PageXML files are newer than their last recorded output.

Usage:
    python src/build.py                        # process all stale prints
    python src/build.py --force                # reprocess everything
    python src/build.py --no-embed             # skip embedding stage
    python src/build.py --author Baldo_29618397
    python src/build.py --embed-fp16
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import (
    find_prints, process_print, _embeddings_stale, carry_derived_layers, write_casus_debug,
)
from parse_pagexml import parse_pagexml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
MANIFEST_PATH = ROOT / "build_manifest.json"


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"prints": {}}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _xml_newest_mtime(print_dir: Path) -> float:
    page_dir = print_dir / "page"
    mtimes = [p.stat().st_mtime for p in page_dir.glob("*.xml")]
    pre = print_dir / "prebuilt.json"          # externally segmented print (import_prebuilt.py)
    if pre.exists():
        mtimes.append(pre.stat().st_mtime)
    return max(mtimes) if mtimes else 0.0


def is_print_stale(author_viaf: str, print_id: str, print_dir: Path, manifest: dict) -> bool:
    xml_mtime = _xml_newest_mtime(print_dir)
    key = f"{author_viaf}/{print_id}"
    entry = manifest.get("prints", {}).get(key, {})
    return xml_mtime > entry.get("xml_newest_mtime", 0)


# ---------------------------------------------------------------------------
# authors.json generation
# ---------------------------------------------------------------------------


def _build_authors_json(prints: list[tuple[str, str, Path]]) -> dict:
    """Build the authors.json dict from data folder structure + author.json files."""
    authors: dict[str, dict] = {}
    for author_viaf, print_id, _ in prints:
        if author_viaf not in authors:
            author_json_path = DATA_DIR / author_viaf / "author.json"
            if author_json_path.exists():
                meta = json.loads(author_json_path.read_text(encoding="utf-8"))
            else:
                meta = {"viaf": author_viaf, "name": author_viaf}
            authors[author_viaf] = {
                "viaf": meta.get("viaf", author_viaf),
                "name": meta.get("name", author_viaf),
                "dates": meta.get("dates", ""),
                "prints": [],
            }
        authors[author_viaf]["prints"].append(print_id)
    return authors


# ---------------------------------------------------------------------------
# Main build logic
# ---------------------------------------------------------------------------


def build(
    author_filter: str | None = None,
    print_filter: str | None = None,
    force: bool = False,
    use_lb_model: bool = True,
    embed: bool = True,
    embed_fp16: bool = False,
    embed_device: str | None = None,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    manifest = _load_manifest()

    all_prints = find_prints(DATA_DIR)
    if not all_prints:
        log.error("No prints found in %s", DATA_DIR)
        sys.exit(1)

    # Apply filters
    prints = all_prints
    if author_filter:
        prints = [p for p in prints if p[0] == author_filter]
        if not prints:
            log.error("Author '%s' not found in %s", author_filter, DATA_DIR)
            sys.exit(1)
    if print_filter:
        prints = [p for p in prints if p[1] == print_filter]
        if not prints:
            log.error("Print '%s' not found in %s", print_filter, DATA_DIR)
            sys.exit(1)

    # Determine which prints need processing
    stale = [
        (viaf, pid, pdir)
        for viaf, pid, pdir in prints
        if force or is_print_stale(viaf, pid, pdir, manifest)
    ]

    if not stale:
        log.info("All prints are up to date.")
    else:
        log.info("%d print(s) to process.", len(stale))

    processed_authors: set[str] = set()
    casus_rows: list = []
    casus_moves: list = []
    stale_kw: list = []

    for author_viaf, print_id, print_dir in stale:
        author_out_dir = OUTPUT_DIR / author_viaf
        author_out_dir.mkdir(exist_ok=True)
        per_print_path = author_out_dir / f"{print_id}.json"

        p_rows: list = []
        p_moves: list = []
        print_data = process_print(print_dir, author_viaf, use_lb_model=use_lb_model,
                                   casus_report=p_rows, casus_moves=p_moves)
        # a rebuild regenerates only the transcription: re-attach translations/keywords
        rep = carry_derived_layers(print_data, per_print_path, p_moves)
        stale_kw += rep["stale_keywords"]
        casus_rows += p_rows
        casus_moves += p_moves

        with per_print_path.open("w", encoding="utf-8") as fh:
            json.dump({"consilia": print_data}, fh, ensure_ascii=False, indent=2)
        log.info("  Wrote %s", per_print_path)

        # Update manifest entry
        key = f"{author_viaf}/{print_id}"
        manifest.setdefault("prints", {})[key] = {
            "xml_newest_mtime": _xml_newest_mtime(print_dir),
            "output_mtime": per_print_path.stat().st_mtime,
            "n_consilia": len(print_data),
        }
        processed_authors.add(author_viaf)
        _save_manifest(manifest)

    # Rebuild per-author and global merged JSONs (always, to keep in sync)
    all_consilia: dict = {}
    by_author: dict[str, dict] = {}

    for author_viaf, print_id, _ in all_prints:
        per_print_path = OUTPUT_DIR / author_viaf / f"{print_id}.json"
        if not per_print_path.exists():
            log.warning("Missing output for %s/%s — skipping from merge.", author_viaf, print_id)
            continue
        print_data = json.loads(per_print_path.read_text(encoding="utf-8"))["consilia"]
        all_consilia.update(print_data)
        by_author.setdefault(author_viaf, {}).update(print_data)

    for author_viaf, author_data in by_author.items():
        author_out_dir = OUTPUT_DIR / author_viaf
        author_out_dir.mkdir(exist_ok=True)
        per_author_path = author_out_dir / "consilia.json"
        with per_author_path.open("w", encoding="utf-8") as fh:
            json.dump({"consilia": author_data}, fh, ensure_ascii=False, indent=2)
        log.info("Wrote per-author output: %s (%d consilia)", per_author_path, len(author_data))

    merged_path = OUTPUT_DIR / "consilia.json"
    with merged_path.open("w", encoding="utf-8") as fh:
        json.dump({"consilia": all_consilia}, fh, ensure_ascii=False, indent=2)
    log.info("Wrote global output: %s (%d total consilia)", merged_path, len(all_consilia))
    if stale:
        write_casus_debug(casus_rows, casus_moves, {p[1] for p in stale}, stale_kw, all_consilia)

    # Generate authors.json from all known prints (not just the filtered subset)
    authors = _build_authors_json(all_prints)
    authors_path = OUTPUT_DIR / "authors.json"
    authors_path.write_text(json.dumps(authors, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %s", authors_path)

    # Embedding stage
    embeddings_path = OUTPUT_DIR / "embeddings.safetensors"
    meta_path = OUTPUT_DIR / "embeddings_meta.json"
    if not embed:
        log.info("Embedding skipped (--no-embed).")
    elif not _embeddings_stale(merged_path, embeddings_path):
        log.info("Embeddings are up to date — skipping.")
    else:
        log.info("Embeddings are stale or missing — running embedding stage …")
        from embed_consilia import embed as run_embed
        run_embed(
            input_path=merged_path,
            output_path=embeddings_path,
            meta_path=meta_path,
            fp16=embed_fp16,
            device=embed_device,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Incremental build for the Consilia pipeline")
    ap.add_argument("--force", action="store_true", help="Reprocess all prints regardless of staleness")
    ap.add_argument("--no-lb-model", action="store_true", help="Use heuristic dehyphenation instead of Flair")
    ap.add_argument("--author", metavar="NAME", help="Restrict to this author directory (e.g. Baldo_29618397)")
    ap.add_argument("--print", metavar="NAME", dest="print_filter", help="Restrict to this print directory")
    ap.add_argument("--no-embed", action="store_true", help="Skip the embedding stage")
    ap.add_argument("--embed-fp16", action="store_true", help="Load BGE-M3 in float16")
    ap.add_argument("--embed-device", metavar="DEVICE", default=None, help="Force embedding device")
    args = ap.parse_args()
    build(
        author_filter=args.author,
        print_filter=args.print_filter,
        force=args.force,
        use_lb_model=not args.no_lb_model,
        embed=not args.no_embed,
        embed_fp16=args.embed_fp16,
        embed_device=args.embed_device,
    )


if __name__ == "__main__":
    main()
