#!/usr/bin/env python3
"""Merge span runs of the same model on the same units: later files override earlier records per unit when
the later record is ok and has no more unmatched spans than the earlier one (batch run + batch retry +
synchronous rerun of the units with unmatched spans).

    python tools/merge_span_runs.py OUT.jsonl base.jsonl retry.jsonl sync.jsonl
"""
import json
import sys
from pathlib import Path


def main(out: str, files: list[str]) -> None:
    recs: dict[str, dict] = {}
    for f in files:
        for l in Path(f).open(encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            old = recs.get(r["unit"])
            if old is None or (r.get("ok") and (not old.get("ok") or len(r.get("unmatched") or []) <= len(old.get("unmatched") or []))):
                recs[r["unit"]] = r
    with Path(out).open("w", encoding="utf-8") as fh:
        for r in recs.values():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = sum(1 for r in recs.values() if r.get("ok"))
    unm = sum(1 for r in recs.values() if r.get("ok") and r.get("unmatched"))
    print(f"{out}: {len(recs)} units, {ok} ok, {len(recs) - ok} failed, {unm} with unmatched spans, {sum(len(r.get('spans') or []) for r in recs.values())} spans")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
