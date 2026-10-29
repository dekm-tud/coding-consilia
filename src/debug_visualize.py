"""
Debug visualizer for consilium boundary detection.

For every page that contains at least one detected heading, writes an annotated
JPEG to output/debug/<volume>/  with colour-coded boxes:

  GREEN  — heading matched by regex (high confidence)
  ORANGE — heading detected by geometric fallback
  CYAN   — heading produced by merging split lines
  RED    — no-body warning (heading with no detectable body opener)

Also writes output/debug/<volume>_coverage.txt with:
  - all detected consilium numbers in order
  - missing numbers (gap analysis)
  - per-page heading list

Usage:
  .venv/bin/python src/debug_visualize.py [--volume NAME] [--pages-only]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).parent))

from parse_pagexml import PageData, parse_pagexml
from segment_consilia import (
    LineRef,
    _RE_HEADING,
    _is_heading_fragment,
    _try_merge,
    _SHORT_LINE_RATIO,
    iter_lines,
    roman_to_int,
    merge_split_headings,
    _merge_colinear_splits,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

# Annotation colours (RGB)
COL_REGEX = (30, 180, 30)      # green  — regex match
COL_GEO = (220, 130, 0)        # orange — geometric fallback
COL_MERGE = (0, 180, 220)      # cyan   — merged split heading
COL_NOBODY = (200, 0, 0)       # red    — heading with no body opener
COL_NEARMISS = (160, 60, 200)  # purple — near-miss: looks like heading but not detected
COL_LABEL_BG = (255, 255, 200) # pale yellow label background

# Loose pattern for near-miss detection: any narrow line containing CONSIL/NSIL + VM variants
import re as _re
_RE_NEAR_MISS = _re.compile(r"[A-Z]{0,3}NSIL[ILR]{1,2}[IV][MV]", _re.IGNORECASE)


# ---------------------------------------------------------------------------
# Instrumented segmentation: collect heading events
# ---------------------------------------------------------------------------

@dataclass
class HeadingEvent:
    page_filename: str
    line_coords: list[list[int]]          # bounding box of the heading line(s)
    region_coords: list[list[int]]
    text: str
    n: int
    match_type: str                       # "regex" | "geo" | "merge"
    has_body: bool = True


def _collect_headings(pages: list[PageData]) -> list[HeadingEvent]:
    """Run the same detection logic as ConsiliumSegmenter but only record headings."""
    from segment_consilia import _RE_SUMMARY_ITEM, _RE_BODY_OPENER, _RE_DAGGER_LEAD
    from segment_consilia import _RE_PURE_NUMERAL, _RE_WORD_FRAGMENT

    all_lines = _merge_colinear_splits(list(iter_lines(pages)))
    all_lines = merge_split_headings(all_lines)

    events: list[HeadingEvent] = []
    state = "BEFORE"
    current_event: HeadingEvent | None = None
    body_lines_seen = 0

    for lr in all_lines:
        text = lr.text

        # --- heading detection (mirrors ConsiliumSegmenter) ---
        m = _RE_HEADING.match(text)
        match_type = None

        if m:
            match_type = "regex"
        elif state in ("BODY", "BEFORE"):
            # geometric fallback
            stripped = text.replace(".", "").replace(" ", "").replace("¬", "")
            alpha = sum(1 for c in stripped if c.isalpha())
            if (
                lr.width_ratio < _SHORT_LINE_RATIO
                and stripped.isupper()
                and alpha >= 4
                and len(text) >= 6
                and not _RE_SUMMARY_ITEM.match(text)
                and not text.rstrip().endswith("¬")
                and not _RE_PURE_NUMERAL.match(text.strip())
                and not _RE_WORD_FRAGMENT.match(text.strip())
            ):
                match_type = "geo"

        # Check if this line was produced by merge (heuristic: not in original
        # page XML — the merge pass uses the first fragment's line_id but the
        # coords are a bounding union of ≥2 fragments; we detect by width ratio
        # being suspiciously wide for an all-caps short line)
        if match_type == "regex" and m:
            roman_str = m.group(1).upper()
            # If this came from the merge pass we won't have the original split
            # info — we tag it as "merge" when the coords span differs from
            # what a natural single line would look like.  Simple proxy: if
            # the line bounding box is taller than 1.5× a typical line height
            # (> ~80 px) it was likely merged.
            xs = [p[0] for p in lr.line_coords]
            ys = [p[1] for p in lr.line_coords]
            if (max(ys) - min(ys)) > 80:
                match_type = "merge"

        if match_type is not None:
            # record previous
            if current_event is not None:
                current_event.has_body = (body_lines_seen > 0)
                events.append(current_event)

            m2 = _RE_HEADING.match(text)
            roman_str = m2.group(1).upper() if m2 else "?"
            n = roman_to_int(roman_str) or 0
            current_event = HeadingEvent(
                page_filename=lr.page_filename,
                line_coords=lr.line_coords,
                region_coords=lr.region_coords,
                text=text,
                n=n,
                match_type=match_type,
            )
            body_lines_seen = 0
            state = "SUMMARY"
            continue

        if state == "SUMMARY":
            if _RE_BODY_OPENER.search(text) or _RE_DAGGER_LEAD.match(text):
                state = "BODY"
                body_lines_seen += 1
        elif state == "BODY":
            body_lines_seen += 1

    if current_event is not None:
        current_event.has_body = (body_lines_seen > 0)
        events.append(current_event)

    return events


def _collect_near_misses(
    pages: list[PageData],
    detected_page_lines: set[str],
) -> dict[str, list[HeadingEvent]]:
    """Scan all pages for narrow all-caps lines that look like headings but
    were not detected by the main pipeline.  Keyed by page filename."""
    near_misses: dict[str, list[HeadingEvent]] = {}

    all_lines = list(iter_lines(pages))
    for lr in all_lines:
        text = lr.text.strip()
        # Only consider narrow, substantially uppercase lines
        stripped = text.replace(".", "").replace(" ", "").replace("¬", "")
        if not stripped:
            continue
        upper_frac = sum(1 for c in stripped if c.isupper()) / len(stripped)
        if (
            upper_frac >= 0.7
            and len(stripped) >= 6
            and _RE_NEAR_MISS.search(text)
            and text not in detected_page_lines
        ):
            ev = HeadingEvent(
                page_filename=lr.page_filename,
                line_coords=lr.line_coords,
                region_coords=lr.region_coords,
                text=text,
                n=0,
                match_type="near_miss",
                has_body=True,
            )
            near_misses.setdefault(lr.page_filename, []).append(ev)

    return near_misses


# ---------------------------------------------------------------------------
# Image annotation
# ---------------------------------------------------------------------------

def _annotate_page(
    image_path: Path,
    events: list[HeadingEvent],
    out_path: Path,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    # Try to get a reasonable font; fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
        small_font = font

    for ev in events:
        colour = {
            "regex": COL_REGEX,
            "geo": COL_GEO,
            "merge": COL_MERGE,
            "near_miss": COL_NEARMISS,
        }.get(ev.match_type, COL_GEO)
        if not ev.has_body and ev.match_type != "near_miss":
            colour = COL_NOBODY

        xs = [p[0] for p in ev.line_coords]
        ys = [p[1] for p in ev.line_coords]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)

        # Semi-transparent fill
        draw.rectangle([x0, y0, x1, y1], fill=(*colour, 60), outline=(*colour, 220), width=3)

        # Label above the box
        label = f"n={ev.n}  {ev.text[:40]}"
        lx, ly = x0, max(0, y0 - 34)
        draw.rectangle([lx, ly, lx + len(label) * 14, ly + 30],
                       fill=(*COL_LABEL_BG, 200))
        draw.text((lx + 2, ly + 2), label, fill=colour, font=font)

    img.save(out_path, quality=80)


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def _write_coverage(
    events: list[HeadingEvent],
    out_path: Path,
    expected_max: int = 500,
) -> None:
    found_ns = sorted(set(ev.n for ev in events if ev.n > 0))
    missing = sorted(set(range(1, expected_max + 1)) - set(found_ns))

    lines = []
    lines.append(f"=== Consilium coverage: {len(found_ns)} found, {len(missing)} missing ===\n")
    lines.append(f"Missing numbers: {missing}\n\n")

    lines.append("=== Detected headings (in page order) ===\n")
    for ev in events:
        if ev.match_type == "near_miss":
            continue  # near-misses listed separately
        flag = "" if ev.has_body else "  ← NO BODY"
        lines.append(
            f"  page={ev.page_filename}  n={ev.n:4d}  [{ev.match_type:5s}]  {ev.text}{flag}\n"
        )

    out_path.write_text("".join(lines), encoding="utf-8")
    log.info("Coverage report → %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _find_print_dirs(volume_filter: str | None) -> list[Path]:
    """Return all print directories under data/{author_viaf}/{print_id}/."""
    result = []
    for author_dir in sorted(DATA_DIR.iterdir()):
        if not author_dir.is_dir():
            continue
        for print_dir in sorted(author_dir.iterdir()):
            if print_dir.is_dir() and (print_dir / "page").is_dir():
                if volume_filter is None or print_dir.name == volume_filter:
                    result.append(print_dir)
    return result


def run(volume_filter: str | None, pages_only: bool) -> None:
    volumes = _find_print_dirs(volume_filter)

    for vol_dir in volumes:
        log.info("Analysing %s …", vol_dir.name)
        page_dir = vol_dir / "page"
        xml_files = sorted(page_dir.glob("*.xml"), key=lambda p: int(p.stem))
        pages = [parse_pagexml(f) for f in xml_files]

        # Collect heading events
        events = _collect_headings(pages)
        log.info("  Found %d heading events", len(events))

        debug_dir = OUTPUT_DIR / "debug" / vol_dir.name
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Near-miss scan: narrow all-caps lines with CONSIL/NSIL pattern not detected
        detected_texts = {ev.text for ev in events}
        near_misses = _collect_near_misses(pages, detected_texts)
        log.info("  Found near-miss candidates on %d additional pages", len(near_misses))

        # Add near-miss events to the full event list for coverage report
        all_events = events + [ev for evs in near_misses.values() for ev in evs]

        # Coverage report (always)
        _write_coverage(all_events, debug_dir / "coverage.txt")

        if pages_only:
            continue

        # Group detected events by page
        by_page: dict[str, list[HeadingEvent]] = {}
        for ev in events:
            by_page.setdefault(ev.page_filename, []).append(ev)
        # Add near-miss events to their pages
        for page_fn, nm_events in near_misses.items():
            by_page.setdefault(page_fn, []).extend(nm_events)

        # Annotate all pages that have detected headings OR near-miss candidates
        pages_written = 0
        for page_fn, page_events in by_page.items():
            img_stem = Path(page_fn).stem          # e.g. "0030"
            img_path = vol_dir / f"{img_stem}.jpg"
            if not img_path.exists():
                continue
            out_path = debug_dir / f"{img_stem}_debug.jpg"
            _annotate_page(img_path, page_events, out_path)
            pages_written += 1

        log.info("  Annotated %d pages → %s", pages_written, debug_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise detected consilium boundaries")
    parser.add_argument("--volume", metavar="NAME",
                        help="Process only this volume (e.g. Baldo_Consilia_v1)")
    parser.add_argument("--pages-only", action="store_true",
                        help="Write only the coverage report, skip image annotation")
    args = parser.parse_args()
    run(args.volume, args.pages_only)


if __name__ == "__main__":
    main()
