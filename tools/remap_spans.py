#!/usr/bin/env python3
"""
Move span offsets to a changed unit text when only the spelling changed (e.g. "&" -> "et", "æ" -> "ae"): the old
and the new text of every unit are aligned character by character (difflib), and each span's start and end are
mapped through that alignment; the span text is taken from the new unit text. Spans whose ends fall into a
replaced stretch are mapped to its edges. Units whose text did not change are copied as they are.

    python tools/remap_spans.py OLD_UNITS NEW_UNITS SPANS.jsonl [SPANS.jsonl ...]     # rewrites the span files (backup .bak)
"""
from __future__ import annotations

import difflib
import json
import shutil
import sys
from pathlib import Path


def mapping(old: str, new: str) -> list[int]:
    """position in old -> position in new, for every index 0..len(old)"""
    m = [0] * (len(old) + 1)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        for k in range(i1, i2 + 1):
            if tag == "equal":
                m[k] = j1 + (k - i1)
            else:                                    # replaced or deleted stretch: map proportionally to its edges
                m[k] = j1 if k == i1 else j2 if k == i2 else j1 + round((k - i1) * (j2 - j1) / max(i2 - i1, 1))
    return m


def main() -> None:
    old_units = {json.loads(l)["unit"]: json.loads(l)["la"] for l in open(sys.argv[1], encoding="utf-8")}
    new_units = {json.loads(l)["unit"]: json.loads(l)["la"] for l in open(sys.argv[2], encoding="utf-8")}
    for f in map(Path, sys.argv[3:]):
        shutil.copy(f, f.with_suffix(f.suffix + ".bak"))
        out, moved, checked = [], 0, 0
        for line in f.open(encoding="utf-8"):
            r = json.loads(line)
            u = r.get("unit")
            if u in old_units and u in new_units and old_units[u] != new_units[u]:
                m, new = mapping(old_units[u], new_units[u]), new_units[u]
                for s in r.get("spans") or []:
                    if "start" not in s or s["start"] is None:
                        continue
                    a, b = m[min(s["start"], len(m) - 1)], m[min(s["end"], len(m) - 1)]
                    s["start"], s["end"], s["text"] = a, b, new[a:b]
                    moved += 1
                checked += 1
            out.append(json.dumps(r, ensure_ascii=False))
        f.write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"{f}: {checked} units with changed text, {moved} spans moved")


if __name__ == "__main__":
    main()
