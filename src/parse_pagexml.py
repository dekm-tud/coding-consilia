"""Parse a single Transkribus PageXML file into structured Python objects."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
_SOFT_HYPHEN = "¬"  # ¬
_RE_STRUCTURE_TYPE = re.compile(r"structure\s*\{[^}]*type:\s*([A-Za-z_0-9]+)")


def _tag(local: str) -> str:
    return f"{{{NS}}}{local}"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Coords:
    points: list[tuple[int, int]]

    @property
    def x_min(self) -> int:
        return min(p[0] for p in self.points)

    @property
    def x_max(self) -> int:
        return max(p[0] for p in self.points)

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @classmethod
    def from_string(cls, points_str: str) -> "Coords":
        points = []
        for pair in points_str.strip().split():
            x, y = pair.split(",")
            points.append((int(x), int(y)))
        return cls(points=points)

    def to_list(self) -> list[list[int]]:
        return [[x, y] for x, y in self.points]


@dataclass
class TextLine:
    line_id: str
    text: str
    coords: Coords
    reading_order: int = 0
    has_soft_hyphen: bool = False  # whether OCR text ended with ¬
    line_type: str = ""  # structural annotation, e.g. "cons_title", "cons_argument", "cons_body"


@dataclass
class TextRegion:
    region_id: str
    region_type: str  # "header", "column_1", "column_2", …
    coords: Coords
    lines: list[TextLine] = field(default_factory=list)
    reading_order: int = 0


@dataclass
class PageData:
    filename: str        # e.g. "0010.xml"
    page_nr: int
    image_filename: str
    image_width: int
    image_height: int
    regions: list[TextRegion] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_coords(elem: ET.Element) -> Optional[Coords]:
    coords_elem = elem.find(_tag("Coords"))
    if coords_elem is None:
        return None
    points_str = coords_elem.get("points", "")
    if not points_str.strip():
        return None
    try:
        return Coords.from_string(points_str)
    except (ValueError, AttributeError):
        return None


def _parse_text(elem: ET.Element) -> str:
    text_equiv = elem.find(_tag("TextEquiv"))
    if text_equiv is None:
        return ""
    unicode_elem = text_equiv.find(_tag("Unicode"))
    if unicode_elem is None or unicode_elem.text is None:
        return ""
    return unicode_elem.text.strip()


def _reading_order_index(elem: ET.Element) -> int:
    """Extract readingOrder index from the 'custom' attribute string."""
    custom = elem.get("custom", "")
    for part in custom.split(";"):
        part = part.strip()
        if part.startswith("readingOrder"):
            try:
                return int(part.split("index:")[1].rstrip(";}").strip())
            except (IndexError, ValueError):
                pass
    return 0


_LINE_TYPE_ALIASES = {
    # v2/v3/v4/v5 use "cons_arg" for ~60% of argument lines (v1 uses
    # "cons_argument" exclusively) — same annotation tool, inconsistent tag
    # naming across sessions. Without this alias, AnnotatedConsiliumSegmenter's
    # type whitelist silently drops every "cons_arg" line as unannotated content.
    "cons_arg": "cons_argument",
}


def _line_type_from_custom(elem: ET.Element) -> str:
    """Extract the TextLine-level structure type (e.g. 'cons_title', 'cons_body')
    from the 'custom' attribute, e.g. 'readingOrder {index:0;} structure {type:cons_body;}'.

    Unlike _region_type_from_custom, this does not naively split on ';' (which
    breaks because 'structure {type:x;}' itself contains a ';') and does not
    fall back to an explicit 'type' element attribute (TextLine has none).
    """
    m = _RE_STRUCTURE_TYPE.search(elem.get("custom", ""))
    if not m:
        return ""
    line_type = m.group(1)
    return _LINE_TYPE_ALIASES.get(line_type, line_type)


def _region_type_from_custom(elem: ET.Element) -> str:
    """Extract structure type (e.g. 'column_1') from the 'custom' attribute."""
    # Also check the 'type' attribute on the element itself
    explicit = elem.get("type", "")
    if explicit:
        return explicit
    # 'structure {score:0.99; type:column_1;}' — the score field precedes the type, so the
    # old split-on-';' lookup returned "unknown" for every region with a score (QA 2026-10-17).
    m = _RE_STRUCTURE_TYPE.search(elem.get("custom", ""))
    if m:
        return m.group(1).strip()
    return "unknown"


# ---------------------------------------------------------------------------
# Public parser
# ---------------------------------------------------------------------------


def parse_pagexml(xml_path: Path) -> PageData:
    """Parse a PageXML file and return a PageData object."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Transkribus metadata → page number
    transkribus_meta = root.find(f".//{_tag('TranskribusMetadata')}")
    page_nr = int(transkribus_meta.get("pageNr", 0)) if transkribus_meta is not None else 0

    page_elem = root.find(_tag("Page"))
    if page_elem is None:
        return PageData(
            filename=xml_path.name,
            page_nr=page_nr,
            image_filename="",
            image_width=0,
            image_height=0,
        )

    image_filename = page_elem.get("imageFilename", "")
    image_width = int(page_elem.get("imageWidth", 0))
    image_height = int(page_elem.get("imageHeight", 0))

    # Build reading-order map: regionRef → index
    ro_map: dict[str, int] = {}
    ordered_group = page_elem.find(f".//{_tag('OrderedGroup')}")
    if ordered_group is not None:
        for ref_elem in ordered_group.findall(_tag("RegionRefIndexed")):
            ro_map[ref_elem.get("regionRef", "")] = int(ref_elem.get("index", 0))

    regions: list[TextRegion] = []
    for region_elem in page_elem.findall(_tag("TextRegion")):
        region_id = region_elem.get("id", "")
        region_type = _region_type_from_custom(region_elem)
        coords = _parse_coords(region_elem)
        if coords is None:
            continue

        lines: list[TextLine] = []
        for line_elem in region_elem.findall(_tag("TextLine")):
            line_id = line_elem.get("id", "")
            text = _parse_text(line_elem)
            line_coords = _parse_coords(line_elem)
            if not text or line_coords is None:
                continue
            ro = _reading_order_index(line_elem)
            lines.append(TextLine(
                line_id=line_id,
                text=text,
                coords=line_coords,
                reading_order=ro,
                has_soft_hyphen=text.rstrip().endswith(_SOFT_HYPHEN),
                line_type=_line_type_from_custom(line_elem),
            ))

        lines.sort(key=lambda ln: ln.reading_order)
        regions.append(TextRegion(
            region_id=region_id,
            region_type=region_type,
            coords=coords,
            lines=lines,
            reading_order=ro_map.get(region_id, 999),
        ))

    regions.sort(key=lambda r: r.reading_order)
    return PageData(
        filename=xml_path.name,
        page_nr=page_nr,
        image_filename=image_filename,
        image_width=image_width,
        image_height=image_height,
        regions=regions,
    )
