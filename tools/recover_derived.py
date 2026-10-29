#!/usr/bin/env python3
"""
Recover derived layers (translations, keywords) that a rebuild lost because consilium ids changed.

src/build.py carries the layers from the previous per-print output by id. When a heading is read
differently the id changes (v4-unknown-12 -> v4-243, v1-17-2 -> v1-18) and the layers stay behind;
a second rebuild then no longer finds them. This takes the layers from a reference state of the
per-print outputs (by default the last commit) and gives each consilium that lacks them the layers
of its counterpart there: same id, else the old consilium whose body text is most similar (ratio of
at least 0.85 on the loosely normalised text, each old consilium used once).

    python tools/recover_derived.py [--ref HEAD] [--dry-run]

Writes output/<author>/<print>.json in place and output/debug/id_map.json (old id -> new id for the
consilia found by text, to carry the annotation layers along); afterwards run src/build.py (merges only, no reprocess)
and src/build_site_data.py.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from carry_derived import DERIVED_KEYS  # noqa: E402

AUTHOR = "Baldo_29618397"


def norm(c: dict) -> str:
    t = " ".join((c.get("casus") or []) + (c.get("body") or []))
    return re.sub(r"[^a-z]", "", t.lower().replace("v", "u").replace("j", "i"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    total = {"by id": 0, "by text": 0, "not found": 0}
    id_map = {}
    for f in sorted((ROOT / "output" / AUTHOR).glob("Baldo_Cons_Print_*.json")):
        rel = f.relative_to(ROOT).as_posix()
        if "backup" in rel:
            continue
        shown = subprocess.run(["git", "show", f"{a.ref}:{rel}"], cwd=ROOT, capture_output=True, text=True)
        if shown.returncode != 0:          # not in the reference commit (local working copy)
            continue
        old = json.loads(shown.stdout)["consilia"]
        data = json.loads(f.read_text(encoding="utf-8")); new = data["consilia"]
        lacking = [cid for cid, c in new.items() if not c.get("translations") or not c.get("keywords_final")]
        used = {cid for cid in new if cid in old and cid not in lacking}
        onorm = {cid: norm(c) for cid, c in old.items()}
        for cid in lacking:
            c = new[cid]; src = None; how = None
            if cid in old and cid not in used:
                src, how = cid, "by id"
            else:
                t = norm(c)
                best, score = None, 0.0
                for ocid, ot in onorm.items():
                    if ocid in used or not ot or not t:
                        continue
                    if abs(len(ot) - len(t)) > 0.3 * max(len(ot), len(t)):
                        continue
                    r = difflib.SequenceMatcher(None, t[:3000], ot[:3000], autojunk=False).quick_ratio()
                    if r > score:
                        r = difflib.SequenceMatcher(None, t[:3000], ot[:3000], autojunk=False).ratio()
                        if r > score:
                            best, score = ocid, r
                if best and score >= 0.85:
                    src, how = best, "by text"
            if not src:
                total["not found"] += 1
                print(f"  not found: {cid} ({c.get('title', '')[:30]})")
                continue
            used.add(src); total[how] += 1
            for k in DERIVED_KEYS:
                if k in old[src] and not c.get(k):
                    c[k] = old[src][k]
            if how == "by text":
                print(f"  {cid} <- {src}")
                id_map[src] = cid
        if not a.dry_run:
            f.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "output" / "debug" / "id_map.json").write_text(json.dumps(id_map, indent=1), encoding="utf-8")
    print(total, f"; {len(id_map)} renamed ids in output/debug/id_map.json")


if __name__ == "__main__":
    main()
