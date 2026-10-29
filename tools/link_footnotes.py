#!/usr/bin/env python3
"""
Link the footnotes of the essay pages (introduction.html, baldus.html, panormitanus.html) to the bibliography.

Every footnote (<li id="n…"> in English, <li id="dn…"> in German) that cites entries of site-data/bibliography.json
(after src/codebooks/bibliography_citations.json) gets a short reference list before its back arrow, for example
"Bibliography: Dean 2007, Ascheri 2003", each name linking to bibliography.html#<entry id>. The span is replaced on
every run, so the script can be repeated after the bibliography changes.

    python tools/link_footnotes.py
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = ("introduction.html", "baldus.html", "panormitanus.html")
LABEL = {"en": "Bibliography", "de": "Bibliographie"}
SPAN = re.compile(r' ?<span class="fn-bib">.*?</span>', re.S)


def short(e: dict) -> str:
    people = e.get("authors") or e.get("editors") or []
    if people:
        p = people[0]
        name = p.get("last") or p.get("name", "")
        if len(people) == 2:
            q = people[1]; name += " / " + (q.get("last") or q.get("name", ""))
        elif len(people) > 2:
            name += " et al."
    else:
        name = e.get("archiveLocation") or e["title"].split(":")[0][:40]
    return f"{name} {e.get('dateText') or e.get('year') or ''}".strip()


def main() -> None:
    bib = json.loads((ROOT / "site-data" / "bibliography.json").read_text(encoding="utf-8"))
    entries = {e["id"]: e for e in bib["entries"]}
    merge = json.loads((ROOT / "src" / "codebooks" / "bibliography_overrides.json").read_text(encoding="utf-8")).get("merge", {})
    merge = {**merge, **bib.get("merged_duplicates", {})}          # duplicates found by build_bibliography.py
    cites = json.loads((ROOT / "src" / "codebooks" / "bibliography_citations.json").read_text(encoding="utf-8"))["citations"]
    per_note: dict[tuple[str, str], list[str]] = {}
    for c in cites:
        k = merge.get(c.get("key"), c.get("key"))
        k = k.replace("new:", "new-", 1) if k else k            # ids of works not in Zotero, as written by build_bibliography.py
        if k in entries:
            lst = per_note.setdefault((c["page"], c["note_id"]), [])
            if k not in lst:
                lst.append(k)
    total = 0
    for page in PAGES:
        p = ROOT / page
        s = p.read_text(encoding="utf-8")

        def fix(m: re.Match) -> str:
            nonlocal total
            note_id, body = m.group(1), SPAN.sub("", m.group(2))
            keys = per_note.get((page, note_id), [])
            if not keys:
                return f'<li id="{note_id}">{body}</li>'
            lang = "de" if note_id.startswith("d") else "en"
            links = ", ".join(f'<a href="bibliography.html#{k}">{html.escape(short(entries[k]))}</a>' for k in keys)
            span = f' <span class="fn-bib">{LABEL[lang]}: {links}</span>'
            total += 1
            back = body.rfind('<a href="#')
            body = body[:back].rstrip() + span + " " + body[back:] if back >= 0 else body + span
            return f'<li id="{note_id}">{body}</li>'

        s = re.sub(r'<li id="(d?n[0-9a-z]+)">(.*?)</li>', fix, s, flags=re.S)
        p.write_text(s, encoding="utf-8")
    print(f"{total} footnotes linked to the bibliography")


if __name__ == "__main__":
    main()
