#!/usr/bin/env python3
"""
Restore translations that a forced rebuild dropped although the Latin text did not change.

The prebuilt import (Panormitanus) writes its own `translations` block, so the carry-over of derived
layers does not fill a language that the import lacks (the English body made later by the GPU run).
This compares every consilium with a reference commit (default HEAD): where a language's body
translation is empty now, the reference has one, and the Latin body has the same number of sections,
the reference translation of that language is copied back. Same for `casus` and `argument` pieces.

    python tools/restore_translations.py [--ref HEAD] [--dry-run]

Writes output/consilia.json and the per-author and per-print copies.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git_json(ref: str, rel: str) -> dict | None:
    r = subprocess.run(["git", "show", f"{ref}:{rel}"], cwd=ROOT, capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    old = git_json(a.ref, "output/consilia.json")["consilia"]
    paths = [ROOT / "output" / "consilia.json"] + sorted((ROOT / "output").glob("*/consilia.json")) + \
            sorted(p for p in (ROOT / "output").glob("*/*_v*.json") if "backup" not in p.name)
    fixes: dict[str, dict] = {}
    new = json.loads(paths[0].read_text(encoding="utf-8"))["consilia"]
    for cid, c in new.items():
        o = old.get(cid)
        if not o or o.get("body") != c.get("body"):
            continue
        for lang in ("en", "de"):
            now = (c.get("translations") or {}).get(lang) or {}
            ref = (o.get("translations") or {}).get(lang) or {}
            piece = {}
            if not any(s and s.strip() for s in now.get("body") or []) and len(ref.get("body") or []) == len(c["body"]) and c["body"]:
                piece["body"] = ref["body"]
                for k in ("retranslated", "stale"):
                    if k in ref:
                        piece[k] = ref[k]
            if not (now.get("argument") or "").strip() and (ref.get("argument") or "").strip():
                piece["argument"] = ref["argument"]
            if c.get("casus") and o.get("casus") == c.get("casus") and not any(s and s.strip() for s in now.get("casus") or []) and len(ref.get("casus") or []) == len(c["casus"]):
                piece["casus"] = ref["casus"]
            if piece:
                fixes.setdefault(cid, {})[lang] = piece
    n_lang = sum(len(v) for v in fixes.values())
    print(f"{len(fixes)} consilia, {n_lang} language blocks to restore from {a.ref}")
    if a.dry_run or not fixes:
        return
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8")); cons = data.get("consilia", {}); n = 0
        for cid, langs in fixes.items():
            c = cons.get(cid)
            if not c:
                continue
            for lang, piece in langs.items():
                c.setdefault("translations", {}).setdefault(lang, {}).update(piece)
            n += 1
        if n:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{path.relative_to(ROOT)}: {n} consilia")


if __name__ == "__main__":
    main()
