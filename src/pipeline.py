"""
Main pipeline: walk data/ → parse PageXML → segment consilia → write JSON → embed.

Usage:
    python pipeline.py [--no-lb-model] [--author AUTHOR_DIR] [--print PRINT_DIR] [--no-embed]

Options:
    --no-lb-model   Use heuristic dehyphenation (¬ → merge) instead of Flair.
    --author NAME   Process only prints belonging to this author directory (e.g. Baldo_29618397).
    --print NAME    Process only the named print directory (e.g. Baldo_Cons_Print_Venice_1575_v1).
    --no-embed      Skip the embedding stage even if embeddings are stale.
    --embed-fp16    Load BGE-M3 in float16 (~1 GB VRAM instead of ~2 GB).
    --embed-device  Force device for embedding: 'cuda', 'cpu', 'cuda:0', etc.

Output:
    output/{author_viaf}/{print_id}.json        per-print consilium dictionary
    output/{author_viaf}/consilia.json          per-author merged dictionary
    output/consilia.json                        merged dictionary across all authors/prints
    output/embeddings.safetensors               BGE-M3 Float32[N, 1024] (when stale)
    output/embeddings_meta.json                 embedding metadata
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running from src/ or from project root
sys.path.insert(0, str(Path(__file__).parent))

from parse_pagexml import parse_pagexml, PageData
from segment_consilia import AnnotatedConsiliumSegmenter, ConsiliumSegmenter, iter_lines

_ANNOTATION_LINE_TYPES = {"cons_title", "cons_argument", "cons_body"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"


# ---------------------------------------------------------------------------
# Print discovery
# ---------------------------------------------------------------------------


def find_prints(data_dir: Path) -> list[tuple[str, str, Path]]:
    """Return (author_viaf, print_id, print_dir) for every known print."""
    result = []
    for author_dir in sorted(data_dir.iterdir()):
        if not author_dir.is_dir():
            continue
        viaf = author_dir.name
        for print_dir in sorted(author_dir.iterdir()):
            if print_dir.is_dir() and (print_dir / "page").is_dir() \
                    and not print_dir.name.endswith("_backup"):
                result.append((viaf, print_dir.name, print_dir))
    return result


# ---------------------------------------------------------------------------
# Per-print processing
# ---------------------------------------------------------------------------


def load_pages(print_dir: Path) -> list[PageData]:
    page_dir = print_dir / "page"
    def _page_nr(p: Path) -> int:
        s = p.stem
        s = s[5:] if s.startswith("page_") else s   # strip "page_" prefix
        return int(s.lstrip("0") or "0")
    xml_files = sorted(page_dir.glob("*.xml"), key=_page_nr)
    pages: list[PageData] = []
    for xml_file in xml_files:
        try:
            pages.append(parse_pagexml(xml_file))
        except Exception as exc:
            log.warning("Could not parse %s: %s", xml_file.name, exc)
    log.info("  Loaded %d pages from %s", len(pages), print_dir.name)
    return pages


def _pages_are_annotated(pages: list[PageData]) -> bool:
    """True if any TextLine carries a cons_title/cons_argument/cons_body structure tag.

    Such pages have reliable line-level segmentation annotations, so the pipeline
    uses AnnotatedConsiliumSegmenter (direct line-type lookup) instead of the
    regex/geometry-based ConsiliumSegmenter.
    """
    for page in pages:
        for region in page.regions:
            for line in region.lines:
                if line.line_type in _ANNOTATION_LINE_TYPES:
                    return True
    return False


def process_print(
    print_dir: Path,
    author_viaf: str,
    use_lb_model: bool = True,
    casus_report: list | None = None,
    casus_moves: list | None = None,
) -> dict:
    """Process one print and return a dict keyed by consilium id.

    casus_report / casus_moves (optional out-parameters) receive the casus-correction
    candidates and accepted moves of the annotation-driven segmenter (see casus.py)."""
    log.info("Processing print: %s / %s", author_viaf, print_dir.name)
    prebuilt = print_dir / "prebuilt.json"
    if prebuilt.exists():
        # segmented outside the PageXML pipeline (e.g. src/gemini_ocr): see import_prebuilt.py
        from import_prebuilt import load_prebuilt
        data = load_prebuilt(prebuilt)
        log.info("  prebuilt.json: %d consilia (PageXML lines are not segmented)", len(data))
        return data
    pages = load_pages(print_dir)

    if _pages_are_annotated(pages):
        log.info("  Line-level annotations detected — using AnnotatedConsiliumSegmenter")
        segmenter_cls = AnnotatedConsiliumSegmenter
    else:
        segmenter_cls = ConsiliumSegmenter

    segmenter = segmenter_cls(
        volume=print_dir.name,
        author_viaf=author_viaf,
        use_lb_model=use_lb_model,
    )
    consilia = segmenter.process(iter_lines(pages))
    log.info("  Extracted %d consilia", len(consilia))
    if casus_report is not None:
        casus_report.extend(getattr(segmenter, "casus_report", []))
    if casus_moves is not None:
        casus_moves.extend(getattr(segmenter, "casus_moves", []))

    return {c.id: c.to_dict() for c in consilia}


# ---------------------------------------------------------------------------
# Derived layers and casus debug output
# ---------------------------------------------------------------------------


def carry_derived_layers(print_data: dict, per_print_path: Path, moves: list) -> dict:
    """Re-attach translations/keywords from the previous per-print output (a rebuild from
    PageXML regenerates only the transcription) and split translations for casus moves."""
    from carry_derived import carry_over

    old: dict = {}
    if per_print_path.exists():
        old = json.loads(per_print_path.read_text(encoding="utf-8")).get("consilia", {})
    rep = carry_over(print_data, old, moves)
    log.info("  carried derived layers for %d consilia (%d new without derived data)",
             rep["carried"], rep["missing"])
    return rep


def write_casus_debug(rows: list, moves: list, volumes: set[str], stale: list, all_consilia: dict) -> None:
    """output/debug/casus_review.csv       undecided candidates only (must end up empty)
    output/debug/casus_moves.json          every candidate/move per volume + ids with stale keywords
    output/debug/casus_retranslate.json    Latin pieces whose translation is missing (GPU job)"""
    import casus
    from carry_derived import find_pending_translations

    debug = OUTPUT_DIR / "debug"
    debug.mkdir(parents=True, exist_ok=True)
    casus.write_review_csv([r for r in rows if r["status"] == "review"],
                           debug / "casus_review.csv", volumes)

    moves_path = debug / "casus_moves.json"
    log_data = json.loads(moves_path.read_text(encoding="utf-8")) if moves_path.exists() else {}
    for v in volumes:
        log_data[v] = {
            "candidates": [r for r in rows if r["volume"] == v],
            "moves": [m for m in moves if m["volume"] == v],
        }
    log_data["_stale_keywords"] = sorted(set(log_data.get("_stale_keywords", [])) | set(stale))
    moves_path.write_text(json.dumps(log_data, ensure_ascii=False, indent=1), encoding="utf-8")

    pending = find_pending_translations(all_consilia)
    (debug / "casus_retranslate.json").write_text(
        json.dumps(pending, ensure_ascii=False, indent=1), encoding="utf-8")
    n_review = sum(1 for r in rows if r["status"] == "review")
    log.info("Casus: %d candidates, %d moved, %d undecided (output/debug/casus_review.csv), "
             "%d translation pieces pending", len(rows), len(moves), n_review, len(pending))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _embeddings_stale(consilia_path: Path, embeddings_path: Path) -> bool:
    """Return True if embeddings are missing or older than consilia.json."""
    if not embeddings_path.exists():
        return True
    return consilia_path.stat().st_mtime > embeddings_path.stat().st_mtime


def run(
    author_filter: str | None = None,
    print_filter: str | None = None,
    use_lb_model: bool = True,
    embed: bool = True,
    embed_fp16: bool = False,
    embed_device: str | None = None,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    prints = find_prints(DATA_DIR)
    if not prints:
        log.error("No prints found in %s", DATA_DIR)
        sys.exit(1)

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

    all_consilia: dict = {}
    by_author: dict[str, dict] = {}
    casus_rows: list = []
    casus_moves: list = []
    stale_kw: list = []

    for author_viaf, print_id, print_dir in prints:
        p_rows: list = []
        p_moves: list = []
        print_data = process_print(print_dir, author_viaf, use_lb_model=use_lb_model,
                                   casus_report=p_rows, casus_moves=p_moves)
        rep = carry_derived_layers(print_data, OUTPUT_DIR / author_viaf / f"{print_id}.json", p_moves)
        stale_kw += rep["stale_keywords"]
        casus_rows += p_rows
        casus_moves += p_moves

        # Accumulate for merged outputs
        all_consilia.update(print_data)
        by_author.setdefault(author_viaf, {}).update(print_data)

        # Write per-print JSON
        author_out_dir = OUTPUT_DIR / author_viaf
        author_out_dir.mkdir(exist_ok=True)
        per_print_path = author_out_dir / f"{print_id}.json"
        with per_print_path.open("w", encoding="utf-8") as fh:
            json.dump({"consilia": print_data}, fh, ensure_ascii=False, indent=2)
        log.info("  Wrote %s", per_print_path)

    # Write per-author merged JSONs
    for author_viaf, author_data in by_author.items():
        author_out_dir = OUTPUT_DIR / author_viaf
        author_out_dir.mkdir(exist_ok=True)
        per_author_path = author_out_dir / "consilia.json"
        with per_author_path.open("w", encoding="utf-8") as fh:
            json.dump({"consilia": author_data}, fh, ensure_ascii=False, indent=2)
        log.info("Wrote per-author output: %s (%d consilia)", per_author_path, len(author_data))

    # Write global merged JSON
    merged_path = OUTPUT_DIR / "consilia.json"
    with merged_path.open("w", encoding="utf-8") as fh:
        json.dump({"consilia": all_consilia}, fh, ensure_ascii=False, indent=2)
    log.info("Wrote merged output: %s (%d total consilia)", merged_path, len(all_consilia))
    write_casus_debug(casus_rows, casus_moves, {p[1] for p in prints}, stale_kw, all_consilia)

    # Embedding stage
    embeddings_path = OUTPUT_DIR / "embeddings.safetensors"
    meta_path = OUTPUT_DIR / "embeddings_meta.json"
    if not embed:
        log.info("Embedding skipped (--no-embed).")
    elif not _embeddings_stale(merged_path, embeddings_path):
        log.info("Embeddings are up to date — skipping (use --no-embed to suppress this check).")
    else:
        log.info("Embeddings are stale or missing — running embedding stage …")
        from embed_consilia import embed as run_embed  # imported late: heavy dependency
        run_embed(
            input_path=merged_path,
            output_path=embeddings_path,
            meta_path=meta_path,
            fp16=embed_fp16,
            device=embed_device,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract consilia from PageXML")
    parser.add_argument(
        "--no-lb-model",
        action="store_true",
        help="Use heuristic (¬ = merge) instead of the Flair LB detector",
    )
    parser.add_argument(
        "--author",
        metavar="NAME",
        help="Process only prints in this author directory (e.g. Baldo_29618397)",
    )
    parser.add_argument(
        "--print",
        metavar="NAME",
        dest="print_filter",
        help="Process only this print directory (e.g. Baldo_Cons_Print_Venice_1575_v1)",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip the embedding stage (useful during development)",
    )
    parser.add_argument(
        "--embed-fp16",
        action="store_true",
        help="Load BGE-M3 in float16 to halve GPU memory usage (~1 GB vs ~2 GB)",
    )
    parser.add_argument(
        "--embed-device",
        metavar="DEVICE",
        default=None,
        help="Force embedding device: 'cuda', 'cpu', 'cuda:0', etc.",
    )
    args = parser.parse_args()
    run(
        author_filter=args.author,
        print_filter=args.print_filter,
        use_lb_model=not args.no_lb_model,
        embed=not args.no_embed,
        embed_fp16=args.embed_fp16,
        embed_device=args.embed_device,
    )


if __name__ == "__main__":
    main()
