"""Remap the gold sets to the unit ids of the casus-corrected segmentation.

The gold records are unit-level on the OLD segmentation. After the casus correction some
units changed: the tail of the previous consilium's last section is now the casus of the
next consilium (new unit ids `...#c<i>w<j>`), and units that were entirely casus changed
document. Nothing is discarded. For every gold record:

  1. unchanged      unit id exists in the new units with identical Latin;
  2. evidence       each annotation is assigned to the unit (of the old document or the
                    next one) whose Latin contains its `evidence_la` (normalised);
                    annotations that ended up in different units become separate records;
  3. text           negative units follow their Latin (same id if truncated, otherwise the
                    unit with identical Latin). A negative label never transfers to a unit
                    that also holds text nobody reviewed.
Anything that cannot be mapped is listed and makes the script fail (exit 2) WITHOUT writing;
--allow-unmapped writes the rest and keeps the unmapped records under "unmapped" in the log.

This script is the only thing that may modify eval/gold/ (the gold sets are model-annotated
reference data; nothing else edits them).

The source is the pristine pre-correction copy in archive/, so the script is repeatable and
never remaps twice.

Usage:
    python src/eval/remap_gold.py --allow-unmapped   # old units + gold from the pre-correction snapshot
    python src/eval/remap_gold.py --dry-run
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
GOLD = ROOT / "eval" / "gold"
SOURCE_GOLD = ROOT / "archive" / "pre_casus_correction_2026-09-23" / "gold"
OLD_UNITS = ROOT / "archive" / "pre_casus_correction_2026-09-23" / "units.jsonl"
NEW_UNITS = ROOT / "output" / "units.jsonl"
NEW_CONSILIA = ROOT / "output" / "consilia.json"
CASUS_MOVES = ROOT / "output" / "debug" / "casus_moves.json"

MAIN_GOLD = "intimate_body_gold.jsonl"
RARE_GOLD = "intimate_body_rare_gold.jsonl"
SAMPLES = ["intimate_body_sample.jsonl", "intimate_body_rare_sample.jsonl"]
SPLIT = "intimate_body_split.json"
SUBJECTS = "subjects_gold.jsonl"


def norm(s: str) -> str:
    s = s.lower().replace("æ", "e").replace("ę", "e").replace("ae", "e").replace("j", "i").replace("v", "u")
    s = s.replace("¬", "")
    return re.sub(r"\s+", " ", s).strip()


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def write_jsonl(p: Path, rows: list[dict]) -> None:
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


class Units:
    def __init__(self, rows: list[dict]):
        self.by_id = {u["unit"]: u for u in rows}
        for u in rows:
            u["_n"] = norm(u["la"])
        self.by_doc: dict[str, list[dict]] = defaultdict(list)
        for u in rows:
            self.by_doc[u["doc"]].append(u)


def doc_of(unit_id: str) -> str:
    return unit_id.split("#")[0]


def remap_record(rec: dict, old: Units, new: Units, next_doc: dict[str, str]):
    """-> (list of new records, method) or (None, reason)."""
    uid = rec["unit"]
    old_u = old.by_id.get(uid)
    new_u = new.by_id.get(uid)
    if new_u and (old_u is None or new_u["_n"] == old_u["_n"]):
        return [rec], "unchanged"
    if old_u is None and new_u is None:
        return None, "unit id unknown in both old and new units"

    doc = doc_of(uid)
    cand = list(new.by_doc.get(doc, [])) + list(new.by_doc.get(next_doc.get(doc, ""), []))
    old_text = old_u["_n"] if old_u else None
    anns = rec.get("annotations") or []

    if anns:
        groups: dict[str, list[dict]] = defaultdict(list)
        for a in anns:
            ev = norm(a["evidence_la"])
            hits = [u for u in cand if ev in u["_n"]]
            if not hits:
                return None, f"evidence not found in the new units of {doc} or its successor: {a['evidence_la'][:60]!r}"
            # the same phrase can occur elsewhere in the neighbour (formulae): prefer the unit
            # that also carries the text AROUND the evidence in the old unit
            ctx_hits = hits
            if old_text and ev in old_text:
                pos = old_text.index(ev)
                ctx = old_text[max(0, pos - 40): pos + len(ev) + 40]
                ctx_hits = [u for u in hits if ctx in u["_n"]] or hits
            same = [u for u in ctx_hits if u["unit"] == uid]
            groups[(same or ctx_hits)[0]["unit"]].append(a)
        out = []
        for nid, alist in groups.items():
            r = copy.deepcopy(rec)
            r["unit"], r["annotations"] = nid, alist
            r["remapped_from"] = uid
            out.append(r)
        # the old id still exists with shorter Latin but received no annotation: keep it as
        # a negative record (it was annotated exhaustively, so what is left had none)
        if new_u and uid not in groups and (old_text is None or new_u["_n"] in old_text):
            r = copy.deepcopy(rec)
            r["annotations"], r["relevant"] = [], False
            r["note"] = (rec.get("note", "") + " [annotations moved to the casus unit by remap_gold.py]").strip()
            out.append(r)
        return out, "evidence"

    # negative unit: follows its Latin
    if new_u and old_text is not None and new_u["_n"] in old_text:
        return [rec], "truncated"
    if old_text is not None:
        same = [u for u in cand if u["_n"] == old_text]
        if same:
            r = copy.deepcopy(rec)
            r["unit"], r["remapped_from"] = same[0]["unit"], uid
            return [r], "text"
        # A negative label only transfers to a unit whose text it fully covered: a new unit
        # that also contains text nobody reviewed cannot inherit "no relevant content".
        inside = [u for u in cand if u["_n"] and u["_n"] in old_text and len(u["_n"]) >= 0.9 * len(old_text)]
        if len(inside) == 1:
            r = copy.deepcopy(rec)
            r["unit"], r["remapped_from"] = inside[0]["unit"], uid
            return [r], "text"
    return None, "negative unit: its Latin is no longer a unit of its own (re-windowed after the casus move)"


def check_evidence(rows: list[dict], new: Units) -> list[str]:
    bad = []
    for r in rows:
        u = new.by_id.get(r["unit"])
        if not u:
            bad.append(f"{r['unit']}: missing unit")
            continue
        for a in r.get("annotations") or []:
            if norm(a["evidence_la"]) not in u["_n"]:
                bad.append(f"{r['unit']}: evidence not in unit: {a['evidence_la'][:50]!r}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--old-units", type=Path, default=OLD_UNITS)
    ap.add_argument("--new-units", type=Path, default=NEW_UNITS)
    ap.add_argument("--source-gold", type=Path, default=SOURCE_GOLD,
                    help="pristine gold to remap (default: the pre-correction snapshot, so the "
                         "script is repeatable); results are written to eval/gold/")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-unmapped", action="store_true")
    args = ap.parse_args()

    src = args.source_gold
    old, new = Units(read_jsonl(args.old_units)), Units(read_jsonl(args.new_units))
    consilia = json.loads(NEW_CONSILIA.read_text(encoding="utf-8"))["consilia"]
    order = {}
    for cid, c in consilia.items():
        order.setdefault(c["volume"], []).append(cid)
    next_doc = {}
    for ids in order.values():
        ids = sorted(ids, key=lambda i: (consilia[i]["n"], i))
        next_doc.update(zip(ids, ids[1:]))

    log = {"methods": defaultdict(int), "unmapped": [], "evidence_failures": [], "files": {}}
    id_map: dict[str, list[str]] = defaultdict(list)
    outputs: dict[str, list[dict]] = {}

    for fname in (MAIN_GOLD, RARE_GOLD):
        rows_out, n_in = [], 0
        for rec in read_jsonl(src / fname):
            n_in += 1
            res, how = remap_record(rec, old, new, next_doc)
            if res is None:
                log["unmapped"].append({"file": fname, "unit": rec["unit"], "reason": how})
                continue
            log["methods"][how] += 1
            rows_out += res
            for r in res:
                id_map[rec["unit"]].append(r["unit"])
        log["evidence_failures"] += [f"{fname}: {m}" for m in check_evidence(rows_out, new)]
        outputs[fname] = rows_out
        log["files"][fname] = {"in": n_in, "out": len(rows_out)}

    if log["unmapped"] and not args.allow_unmapped:
        print(f"{len(log['unmapped'])} gold records could not be mapped; nothing written:", file=sys.stderr)
        for u in log["unmapped"]:
            print("  ", u, file=sys.stderr)
        sys.exit(2)

    # samples and split follow the id map (a unit that split keeps its split/stratum)
    for fname in SAMPLES:
        rows = []
        for r in read_jsonl(src / fname):
            for nid in id_map.get(r["unit"], [r["unit"]] if r["unit"] in new.by_id else []):
                rr = dict(r)
                rr["unit"] = nid
                rows.append(rr)
        outputs[fname] = rows
    split = json.loads((src / SPLIT).read_text(encoding="utf-8"))
    new_split = {}
    for uid, s in split.items():
        for nid in id_map.get(uid, [uid] if uid in new.by_id else []):
            new_split[nid] = s
    outputs[SPLIT] = new_split

    # subjects gold is document-level, annotated from the FULL English translation, which
    # contained the misplaced casus: flag the documents whose text changed
    affected: set[str] = set()
    if CASUS_MOVES.exists():
        data = json.loads(CASUS_MOVES.read_text(encoding="utf-8"))
        affected = set(data.get("_stale_keywords", []))
    subj = read_jsonl(src / SUBJECTS)
    for r in subj:
        if r["doc"] in affected:
            r["casus_affected"] = True
    outputs[SUBJECTS] = subj
    log["subjects_casus_affected"] = [r["doc"] for r in subj if r.get("casus_affected")]
    log["methods"] = dict(log["methods"])

    unmapped_rows = []
    for u in log["unmapped"]:
        rec = next(r for r in read_jsonl(src / u["file"]) if r["unit"] == u["unit"])
        rec = dict(rec)
        rec["unmapped_reason"] = u["reason"]
        rec["old_la"] = old.by_id[u["unit"]]["la"] if u["unit"] in old.by_id else None
        unmapped_rows.append(rec)
    print(json.dumps({k: v for k, v in log.items() if k != "evidence_failures"}, indent=1, ensure_ascii=False))
    if log["evidence_failures"]:
        print("EVIDENCE FAILURES:", *log["evidence_failures"], sep="\n  ", file=sys.stderr)
    if args.dry_run:
        return
    for fname, data in outputs.items():
        if fname.endswith(".jsonl"):
            write_jsonl(GOLD / fname, data)
        else:
            (GOLD / fname).write_text(json.dumps(data, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    if unmapped_rows:
        write_jsonl(GOLD / "remap_unmapped.jsonl", unmapped_rows)
    (GOLD / "remap_log.json").write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
    print("gold remapped; log at eval/gold/remap_log.json")


if __name__ == "__main__":
    main()
