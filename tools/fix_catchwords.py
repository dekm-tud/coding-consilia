#!/usr/bin/env python3
"""
Catchwords (custodes) in the PageXML: the printer repeats the first word of the next page below the
last line of a page. The layout analysis kept them either as a one-word line at the end of the last
column or appended to the last text line; joined into the running text they produce doubled words
("… cum ius ciuile circa || circa iura sanguinis …").

A catchword is recognised when the last content line of a page ends with a word (or a hyphenated word
start) that the first content line of the next page begins with. Two repairs, both recorded in the
line's `comments` attribute:
  - a one-word line gets the structure type `catch-word` (the segmenter ignores lines whose type is
    not a content type), its text is kept;
  - a word appended to a longer line is removed from that line's first TextEquiv (the other readings
    stay as they are).

    python tools/fix_catchwords.py            # dry run: list what would change
    python tools/fix_catchwords.py --apply    # write the PageXML; then rebuild (python src/build.py --no-embed)
"""
from __future__ import annotations

import argparse
import datetime
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
T = "{" + NS + "}"
ET.register_namespace("", NS)
NON_CONTENT = {"header", "page_number", "page-number", "marginalia", "catch-word", "catchword"}


def text_of(tl) -> str:
    te = tl.find(T + "TextEquiv")
    u = te.find(T + "Unicode") if te is not None else None
    return (u.text or "").strip() if u is not None else ""


def rindex(el) -> int:
    m = re.search(r"readingOrder \{index:(\d+);", el.get("custom", ""))
    return int(m.group(1)) if m else 9999


def stype(el) -> str:
    m = re.search(r"structure \{[^}]*type:([A-Za-z_0-9-]+)", el.get("custom", ""))
    return m.group(1) if m else ""


def content_lines(root) -> list:
    out = []
    for r in sorted(root.iter(T + "TextRegion"), key=rindex):
        for tl in sorted(r.iter(T + "TextLine"), key=rindex):
            if stype(tl) in NON_CONTENT or not text_of(tl):
                continue
            out.append(tl)
    return out


def wkey(w: str) -> str:
    return re.sub(r"[^\wæęœ]", "", w.replace("¬", "")).lower()


def is_catchword(last: str, first: str) -> bool:
    lw = last.split()[-1] if last else ""
    fw = first.split()[0] if first else ""
    k1, k2 = wkey(lw), wkey(fw)
    if not k1 or len(k1) < 3:
        return False
    if k1 == k2:
        return True
    if "¬" in lw and k2.startswith(k1):              # hyphenated catchword "deposi¬" / "depositum"
        return True
    return len(k1) >= 4 and k2.startswith(k1) and len(k2) - len(k1) <= 4 and not fw.isupper()


def set_type(tl, new_type: str) -> None:
    custom = tl.get("custom", "")
    if re.search(r"structure \{[^}]*type:", custom):
        custom = re.sub(r"(structure \{[^}]*type:)[A-Za-z_0-9-]+", r"\g<1>" + new_type, custom)
    else:
        custom = (custom + " " if custom else "") + f"structure {{type:{new_type};}}"
    tl.set("custom", custom)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    args = ap.parse_args()
    today = datetime.date.today().isoformat()
    n_line = n_word = 0
    for page_dir in sorted(args.data.glob("*/*/page")):
        files = sorted(page_dir.glob("*.xml"))
        trees = []
        for f in files:
            try:
                trees.append((f, ET.parse(f)))
            except ET.ParseError:
                continue
        for (f, tree), (f2, tree2) in zip(trees, trees[1:]):
            prev = content_lines(tree.getroot())
            nxt = content_lines(tree2.getroot())
            if not prev or not nxt:
                continue
            tl = prev[-1]
            last, first = text_of(tl), text_of(nxt[0])
            if not is_catchword(last, first):
                continue
            words = last.split()
            if len(words) == 1:
                n_line += 1
                print(f"{f.relative_to(ROOT)}: line '{last}' -> structure type catch-word (next page starts '{first[:30]}')")
                if args.apply:
                    set_type(tl, "catch-word")
                    tl.set("comments", (tl.get("comments", "") + "; " if tl.get("comments") else "") + f"catchword {today}: one-word line repeats the first word of the next page, typed catch-word by fix_catchwords.py")
            else:
                n_word += 1
                new = " ".join(words[:-1])
                print(f"{f.relative_to(ROOT)}: '…{last[-35:]}' -> drop '{words[-1]}' (next page starts '{first[:30]}')")
                if args.apply:
                    u = tl.find(T + "TextEquiv").find(T + "Unicode")
                    u.text = new
                    tl.set("comments", (tl.get("comments", "") + "; " if tl.get("comments") else "") + f"catchword {today}: trailing '{words[-1]}' removed by fix_catchwords.py (repeats the first word of the next page)")
            if args.apply:
                # keep the region's own Unicode in step with its lines
                region = next((r for r in tree.getroot().iter(T + "TextRegion") if tl in list(r.iter(T + "TextLine"))), None)
                if region is not None:
                    ru = region.find(T + "TextEquiv")
                    if ru is not None and ru.find(T + "Unicode") is not None:
                        ru.find(T + "Unicode").text = "\n".join(text_of(x) for x in sorted(region.iter(T + "TextLine"), key=rindex) if stype(x) not in NON_CONTENT)
                tree.write(f, encoding="unicode", xml_declaration=True)
    print(f"{n_line} one-word catchword lines, {n_word} catchwords appended to a line{' (written)' if args.apply else ' (dry run, nothing written)'}")


if __name__ == "__main__":
    main()
