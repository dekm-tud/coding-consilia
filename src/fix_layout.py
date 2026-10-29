"""
fix_layout.py — Detect and fix tall-initial layout errors in Transkribus PageXML.

Error pattern (mainly page 100+):
  After a summary, a new consilium body starts with a tall first line (large initial
  letter). Transkribus gets the layout wrong:

    L  — truncated line: right part of real line 1, starts far from the left margin
    H  — merged tall line: [big initial text] + [line 2 text] fused into one TextLine

  L and H overlap vertically; L appears before H in reading order.

Fix: split H into H_left (big initial) and H_right (line 2 text).
  New reading order: ... [H_left] [L] [H_right] ...

Usage:
  python src/fix_layout.py                         # dry-run: report only
  python src/fix_layout.py --page 0101             # single page, dry-run
  python src/fix_layout.py --from-page 100         # pages >= 100, dry-run
  python src/fix_layout.py --fix                   # apply to all pages
  python src/fix_layout.py --fix --page 0101       # apply to one page
  python src/fix_layout.py --fix --from-page 100   # apply from page 100
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
ET.register_namespace("", NS)

# ── body-opener patterns that identify a consilium body start ──────────────
_BODY_OPENERS = re.compile(
    r"^(In Christi nomine|In nomine (Christi|Domini|Dei|Iesu)|IN (CHRISTI|DEI|NOMINE))",
    re.IGNORECASE,
)

# ── thresholds ──────────────────────────────────────────────────────────────
# L must start at least this fraction of the region width from the left margin
_TRUNC_RATIO_MIN = 0.15
# H must start within this fraction of the region width from the left margin
_MARGIN_RATIO_MAX = 0.10
# Minimum y-overlap in pixels for a pair to qualify
_MIN_Y_OVERLAP = 5
# a merged line without the opener formula must be this much taller than the region's median line
_TALL_RATIO = 1.6
_BIG_WORD = re.compile(r"^[†¶]?\s*[A-Z][A-Za-z]{2,}")


# ── XML helpers ─────────────────────────────────────────────────────────────

def _tag(local: str) -> str:
    return f"{{{NS}}}{local}"


def _parse_points(s: str) -> list[tuple[int, int]]:
    pts = []
    for tok in s.strip().split():
        x, y = tok.split(",")
        pts.append((int(x), int(y)))
    return pts


def _fmt_points(pts: list[tuple[int, int]]) -> str:
    return " ".join(f"{x},{y}" for x, y in pts)


def _get_coords(elem: ET.Element) -> list[tuple[int, int]]:
    c = elem.find(_tag("Coords"))
    return _parse_points(c.get("points", "")) if c is not None else []


def _set_coords(elem: ET.Element, pts: list[tuple[int, int]]) -> None:
    c = elem.find(_tag("Coords"))
    if c is not None:
        c.set("points", _fmt_points(pts))


def _get_baseline(tl: ET.Element) -> list[tuple[int, int]]:
    bl = tl.find(_tag("Baseline"))
    return _parse_points(bl.get("points", "")) if bl is not None else []


def _set_baseline(tl: ET.Element, pts: list[tuple[int, int]]) -> None:
    bl = tl.find(_tag("Baseline"))
    if bl is not None and pts:
        bl.set("points", _fmt_points(pts))


def _get_text(tl: ET.Element) -> str:
    te = tl.find(_tag("TextEquiv"))
    if te is None:
        return ""
    u = te.find(_tag("Unicode"))
    return (u.text or "").strip() if u is not None else ""


def _set_text(tl: ET.Element, text: str) -> None:
    te = tl.find(_tag("TextEquiv"))
    if te is None:
        te = ET.SubElement(tl, _tag("TextEquiv"))
    u = te.find(_tag("Unicode"))
    if u is None:
        u = ET.SubElement(te, _tag("Unicode"))
    u.text = text


def _get_ro(tl: ET.Element) -> int:
    m = re.search(r"readingOrder\s*\{[^}]*index:\s*(\d+)", tl.get("custom", ""))
    return int(m.group(1)) if m else 0


def _set_ro(tl: ET.Element, idx: int) -> None:
    custom = tl.get("custom", "")
    new = f"readingOrder {{index:{idx};}}"
    if "readingOrder" in custom:
        custom = re.sub(r"readingOrder\s*\{[^}]*\}", new, custom)
    else:
        custom = new + (" " + custom).rstrip() if custom else new
    tl.set("custom", custom)


# ── geometry helpers ─────────────────────────────────────────────────────────

def _x_range(pts: list[tuple[int, int]]) -> tuple[int, int]:
    xs = [p[0] for p in pts]
    return min(xs), max(xs)


def _y_range(pts: list[tuple[int, int]]) -> tuple[int, int]:
    ys = [p[1] for p in pts]
    return min(ys), max(ys)


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    """Return (x_min, y_min, x_max, y_max)."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _clip_poly_left(pts: list[tuple[int, int]], split_x: int) -> list[tuple[int, int]]:
    """Bounding-box clip: return a rectangle covering pts with x <= split_x."""
    left = [(x, y) for x, y in pts if x <= split_x]
    # Add interpolated intersections with x = split_x
    for i in range(len(pts)):
        p1, p2 = pts[i], pts[(i + 1) % len(pts)]
        if (p1[0] < split_x < p2[0]) or (p2[0] < split_x < p1[0]):
            t = (split_x - p1[0]) / (p2[0] - p1[0])
            left.append((split_x, int(p1[1] + t * (p2[1] - p1[1]))))
    if not left:
        y_min, y_max = _y_range(pts)
        x_min, _ = _x_range(pts)
        return [(x_min, y_min), (split_x, y_min), (split_x, y_max), (x_min, y_max)]
    xn, yn, xx, yx = _bbox(left)
    return [(xn, yn), (split_x, yn), (split_x, yx), (xn, yx)]


def _clip_poly_right(pts: list[tuple[int, int]], split_x: int) -> list[tuple[int, int]]:
    """Bounding-box clip: return a rectangle covering pts with x >= split_x."""
    right = [(x, y) for x, y in pts if x >= split_x]
    for i in range(len(pts)):
        p1, p2 = pts[i], pts[(i + 1) % len(pts)]
        if (p1[0] < split_x < p2[0]) or (p2[0] < split_x < p1[0]):
            t = (split_x - p1[0]) / (p2[0] - p1[0])
            right.append((split_x, int(p1[1] + t * (p2[1] - p1[1]))))
    if not right:
        y_min, y_max = _y_range(pts)
        _, x_max = _x_range(pts)
        return [(split_x, y_min), (x_max, y_min), (x_max, y_max), (split_x, y_max)]
    xn, yn, xx, yx = _bbox(right)
    return [(split_x, yn), (xx, yn), (xx, yx), (split_x, yx)]


def _split_baseline(pts: list[tuple[int, int]], split_x: int):
    left = [(x, y) for x, y in pts if x <= split_x]
    right = [(x, y) for x, y in pts if x > split_x]
    # Interpolate at split_x
    for i in range(len(pts) - 1):
        p1, p2 = pts[i], pts[i + 1]
        if p1[0] <= split_x <= p2[0] and p1[0] != p2[0]:
            t = (split_x - p1[0]) / (p2[0] - p1[0])
            yi = int(p1[1] + t * (p2[1] - p1[1]))
            if split_x not in [p[0] for p in left]:
                left.append((split_x, yi))
            if split_x not in [p[0] for p in right]:
                right.insert(0, (split_x, yi))
    if not left:
        left = [pts[0]]
    if not right:
        right = [pts[-1]]
    return left, right


# ── text split ────────────────────────────────────────────────────────────────

# Fixed opening formulae that should always stay intact in H_left
_OPENERS = re.compile(
    r"^("
    r"In\s+Christi\s+Nomine\.?\s+Amen\.?"    # "In Christi Nomine Amen." (v4/v5)
    r"|In\s+(?:Dei|Nomine\s+Domini)\s+Amen\.?"  # "In Dei Nomine Amen." / "In Nomine Domini Amen."
    r"|IN\s+Nomine\s+[Dd]omini\.?\s*(?:Amen\.?)?"  # "IN Nomine domini Amen."
    r"|In\s+Christi\s+nomine\."              # "In Christi nomine." (v1)
    r"|In\s+nomine\s+(?:Christi|Domini|Dei|Iesu|DominnI|DominnL)\."
    r"|IN\s+CHRISTI\s+NOMINE\.|IN\s+NOMINE\s+DEI\."
    r")",
    re.IGNORECASE,
)


def _split_text(text: str, ratio: float) -> tuple[str, str]:
    """Split text at a word boundary closest to `ratio` through the string.

    If the text starts with a known body-opener formula, always snap the split
    to exactly after that formula so that the opener stays intact in H_left.
    """
    if not text:
        return "", ""
    # Snap to known opener if present
    m = _OPENERS.match(text)
    if m:
        opener_end = m.end()
        right = text[opener_end:].strip()
        return text[:opener_end].strip(), right
    # Fallback: split at word boundary near ratio
    ratio = max(0.05, min(0.95, ratio))
    target = int(len(text) * ratio)
    pos = text.rfind(" ", 0, target + 1)
    if pos == -1:
        pos = text.find(" ", target)
    if pos == -1:
        pos = target
    return text[:pos].strip(), text[pos:].strip()


# ── detection ────────────────────────────────────────────────────────────────

def _find_pairs(region_elem: ET.Element) -> list[tuple[ET.Element, ET.Element, int]]:
    """
    Return (L, H, split_x) triples for body-start layout errors in this region.

    L = truncated line (right part of real first body line)
    H = merged tall line (big initial + line 2 text)
    split_x = x-coordinate where H should be split
    """
    region_pts = _get_coords(region_elem)
    if not region_pts:
        return []
    r_x_min, r_x_max = _x_range(region_pts)
    region_w = r_x_max - r_x_min
    if region_w < 100:
        return []

    trunc_threshold = r_x_min + region_w * _TRUNC_RATIO_MIN
    margin_threshold = r_x_min + region_w * _MARGIN_RATIO_MAX

    # Sort TextLines by reading order
    tls = sorted(region_elem.findall(_tag("TextLine")), key=_get_ro)
    heights = sorted(h for h in ((_y_range(_get_coords(t)) or (0, 0)) for t in tls) for h in [h[1] - h[0]] if h > 0)
    med_h = heights[len(heights) // 2] if heights else 0

    pairs: list[tuple[ET.Element, ET.Element, int]] = []
    for i in range(len(tls) - 1):
        L = tls[i]
        H = tls[i + 1]

        L_pts = _get_coords(L)
        H_pts = _get_coords(H)
        if not L_pts or not H_pts:
            continue

        L_x_min = min(p[0] for p in L_pts)
        H_x_min = min(p[0] for p in H_pts)

        # L must start far from the left margin
        if L_x_min < trunc_threshold:
            continue

        # H must start near the left margin
        if H_x_min > margin_threshold:
            continue

        # L and H must overlap vertically
        L_y_min, L_y_max = _y_range(L_pts)
        H_y_min, H_y_max = _y_range(H_pts)
        overlap = min(L_y_max, H_y_max) - max(L_y_min, H_y_min)
        if overlap < _MIN_Y_OVERLAP:
            continue

        H_text = _get_text(H)

        # H must start with a recognisable body opener phrase.
        # Geometric body-opener detection (tall + wide) is left to the segmenter
        # which has the SUMMARY/BODY context; here we would have no way to tell
        # apart a body opener from a mid-body tall line.
        # Since 2026-10-18 also without the formula: Baldus's consilia often open with a large-type
        # word ("Super", "Tutor", "Pone", "Statuto"); the merged line is then clearly taller than
        # the region's lines and starts with a capitalised word.
        tall = med_h and (H_y_max - H_y_min) > _TALL_RATIO * med_h
        if not (_BODY_OPENERS.match(H_text) or (tall and _BIG_WORD.match(H_text))):
            continue
        # never split a heading: "CONSILIVM CCCCXXXVII." is tall and capitalised too, and the line
        # before it (a right-aligned "Ego Baldus.") looks like a truncated L
        if "cons_title" in (H.get("custom") or "") or re.search(r"\bCO(?:NS|S)S?ILI[VU]M\b", H_text, re.IGNORECASE):
            continue

        # L must have substantial text (empty or single-character lines are stray
        # artifacts, not truncated body lines)
        L_text = _get_text(L)
        if len(L_text.strip()) < 3:
            continue

        # L must not be a heading line — after a fix, the heading that preceded the
        # original L ends up adjacent to the body opener and looks like a new pair,
        # but it isn't one we should touch.
        if re.search(r"\bCO(?:NS|S)ILI[VU]M\b", L_text, re.IGNORECASE):
            continue

        pairs.append((L, H, L_x_min))

    return pairs


# ── display helpers ──────────────────────────────────────────────────────────

_BAR_W = 46   # width of the ASCII ruler (characters between the two |)


def _x_bar(
    line_x_min: int,
    line_x_max: int,
    region_x_min: int,
    region_x_max: int,
    split_x: int | None = None,
) -> str:
    """One-line ruler showing line extent inside the region, with optional split marker.

    Example (split at ~50 %):
      |·············┆██████████████████████████████|
    """
    region_w = max(1, region_x_max - region_x_min)

    def col(x: int) -> int:
        return max(0, min(_BAR_W - 1, round((x - region_x_min) / region_w * (_BAR_W - 1))))

    bar = ["·"] * _BAR_W
    l, r = col(line_x_min), col(line_x_max)
    for c in range(l, r + 1):
        bar[c] = "█"
    if split_x is not None:
        sc = col(split_x)
        bar[sc] = "┆"
    return "|" + "".join(bar) + "|"


def _fmt_ctx(tl: ET.Element, r_x_min: int, r_x_max: int) -> str:
    pts = _get_coords(tl)
    text = _get_text(tl)
    ro = _get_ro(tl)
    xn = min(p[0] for p in pts) if pts else r_x_min
    xx = max(p[0] for p in pts) if pts else r_x_max
    bar = _x_bar(xn, xx, r_x_min, r_x_max)
    return f"  [ro={ro:2d}]  {bar}  \"{text[:48]}\""


# ── fix a single region ──────────────────────────────────────────────────────

def _fix_region(region_elem: ET.Element, dry_run: bool, show: bool = False) -> list[str]:
    """Apply fixes to all body-start pairs in this region. Return descriptions."""
    pairs = _find_pairs(region_elem)
    if not pairs:
        return []

    region_pts = _get_coords(region_elem)
    r_x_min, r_x_max = _x_range(region_pts)
    rtype = re.search(r"type:\s*(\S+?)[\s;}]", region_elem.get("custom", ""))
    region_label = rtype.group(1) if rtype else region_elem.get("id", "?")

    # sorted lines — needed for context lookup
    all_sorted = sorted(region_elem.findall(_tag("TextLine")), key=_get_ro)
    ro_to_idx = {_get_ro(tl): i for i, tl in enumerate(all_sorted)}

    msgs: list[str] = []

    for L, H, split_x in pairs:
        H_id   = H.get("id", "")
        H_text = _get_text(H)
        L_text = _get_text(L)
        ro_L   = _get_ro(L)
        ro_H   = _get_ro(H)

        # Estimate split ratio from baseline
        H_bl = _get_baseline(H)
        if H_bl:
            bl_left = sum(1 for (x, _) in H_bl if x <= split_x)
            ratio = bl_left / len(H_bl)
        else:
            H_pts = _get_coords(H)
            x_min, x_max = _x_range(H_pts)
            ratio = (split_x - x_min) / (x_max - x_min) if x_max > x_min else 0.4

        H_left_text, H_right_text = _split_text(H_text, ratio)
        snap = bool(_OPENERS.match(H_text))

        if show:
            # ── visual debug output ──────────────────────────────────────────
            L_pts = _get_coords(L)
            H_pts = _get_coords(H)
            L_xn, L_xx = _x_range(L_pts)
            H_xn, H_xx = _x_range(H_pts)
            sep = "  " + "─" * 62

            lines = [sep, f"  [{region_label}]  ro={ro_L}-{ro_H}  split={'snap' if snap else f'ratio={ratio:.2f}'}"]

            # context: 1 line before L
            idx_L = ro_to_idx.get(ro_L, 0)
            if idx_L > 0:
                lines.append(_fmt_ctx(all_sorted[idx_L - 1], r_x_min, r_x_max))

            # the pair
            lines.append(
                f"  L [ro={ro_L:2d}]  "
                f"{_x_bar(L_xn, L_xx, r_x_min, r_x_max, split_x)}"
                f"  x={L_xn}-{L_xx}"
            )
            lines.append(f"           \"{L_text[:55]}\"")
            lines.append(
                f"  H [ro={ro_H:2d}]  "
                f"{_x_bar(H_xn, H_xx, r_x_min, r_x_max, split_x)}"
                f"  x={H_xn}-{H_xx}"
            )
            lines.append(f"           \"{H_text[:55]}\"")
            lines.append(f"           split_x={split_x}")
            lines.append(f"           ├ H_left  \"{H_left_text[:52]}\"")
            lines.append(f"           └ H_right \"{H_right_text[:52]}\"")

            # context: 1 line after H
            idx_H = ro_to_idx.get(ro_H, len(all_sorted) - 1)
            if idx_H + 1 < len(all_sorted):
                lines.append(_fmt_ctx(all_sorted[idx_H + 1], r_x_min, r_x_max))

            msgs.append("\n".join(lines))
        else:
            # ── compact one-pair output ──────────────────────────────────────
            msgs.append(
                f"  ro={ro_L}  L  x={split_x}-{_x_range(_get_coords(L))[1]}"
                f"  \"{L_text[:45]}\"\n"
                f"  ro={ro_H}  H  x={_x_range(_get_coords(H))[0]}-{_x_range(_get_coords(H))[1]}"
                f"  \"{H_text[:55]}\"\n"
                f"       split={split_x}  "
                f"→  \"{H_left_text[:35]}\"  |  \"{H_right_text[:35]}\""
            )

        if dry_run:
            continue

        # ── apply fix ───────────────────────────────────────────────────────

        H_pts = _get_coords(H)
        H_bl_pts = _get_baseline(H)
        all_tls = region_elem.findall(_tag("TextLine"))
        L_pos = all_tls.index(L)
        H_pos = all_tls.index(H)

        if H_right_text:
            # Full split: create H_left (opener) + modify H → H_right (line 2)
            H_left = deepcopy(H)
            H_left.set("id", H_id + "_left")
            _set_coords(H_left, _clip_poly_left(H_pts, split_x))
            if H_bl_pts:
                bl_l, bl_r = _split_baseline(H_bl_pts, split_x)
                _set_baseline(H_left, bl_l)
            _set_text(H_left, H_left_text)

            _set_coords(H, _clip_poly_right(H_pts, split_x))
            if H_bl_pts:
                _, bl_r = _split_baseline(H_bl_pts, split_x)
                _set_baseline(H, bl_r)
            _set_text(H, H_right_text)

            # Remove all TextLines, rebuild: ... H_left, L, H_right(=H), ...
            for tl in all_tls:
                region_elem.remove(tl)
            new_order: list[ET.Element] = []
            i = 0
            while i < len(all_tls):
                if i == L_pos:
                    new_order.append(H_left)
                    new_order.append(L)
                    if H_pos == i + 1:
                        i += 2
                        new_order.append(H)
                    else:
                        i += 1
                    continue
                if i == H_pos:
                    new_order.append(H)
                else:
                    new_order.append(all_tls[i])
                i += 1
        else:
            # H contains only the opener (no continuation text):
            # just reorder to put H before L — no split needed.
            for tl in all_tls:
                region_elem.remove(tl)
            new_order = []
            i = 0
            while i < len(all_tls):
                if i == L_pos:
                    new_order.append(H)    # H (opener) goes first
                    new_order.append(L)    # truncated right part follows
                    if H_pos == i + 1:
                        i += 2
                    else:
                        i += 1
                    continue
                if i == H_pos:
                    pass  # already placed above
                else:
                    new_order.append(all_tls[i])
                i += 1

        for idx, tl in enumerate(new_order):
            _set_ro(tl, idx)
            region_elem.append(tl)

    return msgs


# ── main entry point ─────────────────────────────────────────────────────────

def _process_file(xml_path: Path, dry_run: bool, verbose: bool, show: bool) -> int:
    """Process one XML file. Returns number of fixes found/applied."""
    ET.register_namespace("", NS)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    page_elem = root.find(_tag("Page"))
    if page_elem is None:
        return 0

    all_msgs: list[str] = []
    for region_elem in page_elem.findall(_tag("TextRegion")):
        custom = region_elem.get("custom", "")
        if "header" in custom or "page-number" in custom:
            continue
        msgs = _fix_region(region_elem, dry_run=dry_run, show=show)
        all_msgs.extend(msgs)

    if all_msgs:
        action = "found" if dry_run else "fixed"
        print(f"\n{xml_path.name}: {len(all_msgs)} pair(s) {action}")
        for m in all_msgs:
            print(m)
        if not dry_run:
            tree.write(xml_path, encoding="unicode", xml_declaration=True)

    elif verbose:
        print(f"{xml_path.name}: ok")

    return len(all_msgs)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Detect and fix tall-initial body-start layout errors in PageXML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--print", dest="print_id", default=None,
                    help="Limit to this print folder (default: all prints)")
    ap.add_argument("--page", default=None,
                    help="Process only this page, e.g. 0101")
    ap.add_argument("--from-page", type=int, default=None, metavar="N",
                    help="Process pages with number >= N")
    ap.add_argument("--fix", action="store_true",
                    help="Apply fixes in-place (default: dry-run / report only)")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Print a line for each page even if nothing found")
    ap.add_argument("--show", "-s", action="store_true",
                    help="Visual debug: show ASCII ruler + context lines for each pair")
    args = ap.parse_args()

    dry_run = not args.fix
    if dry_run:
        print("DRY RUN — reporting only. Use --fix to apply changes.")

    data_root = Path(__file__).parent.parent / "data"
    total = 0

    for author_dir in sorted(data_root.iterdir()):
        if not author_dir.is_dir():
            continue
        for print_dir in sorted(author_dir.iterdir()):
            page_dir = print_dir / "page"
            if not page_dir.is_dir():
                continue
            if args.print_id and print_dir.name != args.print_id:
                continue

            for xml_path in sorted(page_dir.glob("*.xml")):
                stem = xml_path.stem
                if args.page and stem != args.page:
                    continue
                # Accept both "0101" and "page_101" filename styles
                nr_str = re.sub(r"^page_0*", "", stem)
                try:
                    page_nr = int(nr_str)
                except ValueError:
                    continue
                if args.from_page and page_nr < args.from_page:
                    continue

                total += _process_file(xml_path, dry_run=dry_run, verbose=args.verbose, show=args.show)

    action = "detected" if dry_run else "applied"
    print(f"\nTotal: {total} fix(es) {action}.")
    if dry_run and total:
        print("Run with --fix to apply.")


if __name__ == "__main__":
    main()
