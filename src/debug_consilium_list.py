"""
Create a CSV overview of all consilia for debugging.

Columns: print, n, roman, title, has_body, body_start, pages, first_page

Usage:
    python3 src/debug_consilium_list.py [--out debug_consilia.csv]
    python3 src/debug_consilium_list.py --print v4 --out debug_v4.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output"


def iter_prints(print_filter: str | None = None):
    """Yield (author_viaf, print_id, json_path) for every per-print JSON."""
    for author_dir in sorted(OUTPUT_DIR.iterdir()):
        if not author_dir.is_dir():
            continue
        for json_path in sorted(author_dir.glob("*.json")):
            if json_path.name in ("consilia.json",):
                continue
            if "_backup" in json_path.stem:
                continue
            print_id = json_path.stem
            if print_filter and print_filter not in print_id:
                continue
            yield author_dir.name, print_id, json_path


def shorten(text: str, n: int = 80) -> str:
    text = text.strip().replace("\n", " ")
    return text[:n] + "…" if len(text) > n else text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", dest="print_filter", metavar="PATTERN",
                        help="Only include prints whose ID contains PATTERN (e.g. v4)")
    parser.add_argument("--out", default="debug_consilia.csv",
                        help="Output CSV path (default: debug_consilia.csv)")
    args = parser.parse_args()

    out_path = Path(args.out)
    rows = []

    for author_viaf, print_id, json_path in iter_prints(args.print_filter):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        consilia = data.get("consilia", {})

        for cid, c in consilia.items():
            n = c.get("n", 0)
            roman = c.get("roman", "")
            title = shorten(c.get("title", ""), 70)
            body = c.get("body", [])
            has_body = 1 if body else 0
            # Skip trivially short leading sections (isolated OCR initials like "v")
            body_text = next((s for s in body if len(s.strip()) > 5), body[0] if body else "")
            body_start = shorten(body_text, 80) if body else ""
            sources = c.get("sources", [])
            pages = [s["page"] for s in sources]
            first_page = pages[0] if pages else ""
            last_page = pages[-1] if pages else ""
            all_pages = "; ".join(pages)

            rows.append({
                "print": print_id,
                "n": n,
                "roman": roman,
                "title": title,
                "has_body": has_body,
                "body_start": body_start,
                "first_page": first_page,
                "last_page": last_page,
                "all_pages": all_pages,
            })

    rows.sort(key=lambda r: (r["print"], r["n"]))

    fieldnames = ["print", "n", "roman", "title", "has_body",
                  "body_start", "first_page", "last_page", "all_pages"]

    with out_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
