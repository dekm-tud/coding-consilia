"""
Evaluate classification runs against the gold sets.

  body      (unit, code) pairs: precision/recall/F1 per code and overall, relevance P/R/F1, role
            accuracy on matched pairs, evidence validity rate; restricted to a split (dev/test/rare).
  subjects  per consilium: precision/recall of central and secondary, "every gold central is
            predicted (central or secondary)" rate, facet-level agreement.

Several prediction files may be given; a line per file plus, for `body`, an AGREEMENT line (pairs
predicted by every file) when 2+ files are given. Few-shot examples are excluded automatically
(their ids are not in the prediction files).

    python src/annotate/evaluate.py --task body --split test output/annotations/body_*_p1.jsonl
    python src/annotate/evaluate.py --task subjects output/annotations/subjects_*_p1.jsonl
Writes eval/results/<task>_<split>.md and .json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "eval" / "gold"
RES = ROOT / "eval" / "results"
COUNTED = {"topic", "argument", "case_fact", "analogy"}     # citation_only / formulaic are recorded, not counted


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def load_preds(path: Path, key: str) -> dict:
    out = {}
    for l in path.open(encoding="utf-8"):
        rec = json.loads(l)
        if rec.get("ok"):
            out[rec[key]] = rec
    return out


def eval_body(pred_files: list[Path], split: str) -> tuple[list[str], dict]:
    sp = json.loads((GOLD / "intimate_body_split.json").read_text())
    gold = [json.loads(l) for l in (GOLD / "intimate_body_gold.jsonl").open(encoding="utf-8")]
    gold += [json.loads(l) for l in (GOLD / "intimate_body_rare_gold.jsonl").open(encoding="utf-8")]
    gold = {g["unit"]: g for g in gold if split == "all" or sp.get(g["unit"]) == split}
    runs = {p.stem: load_preds(p, "unit") for p in pred_files}
    if len(runs) >= 2:
        common = set.intersection(*(set(r) for r in runs.values()))
        agree = {}
        for u in common:
            sets = [{(a["code"], a["role"] in COUNTED) for a in r[u]["result"]["annotations"]} for r in runs.values()]
            inter = set.intersection(*sets)
            agree[u] = {"result": {"annotations": [{"code": c, "role": "topic" if cnt else "citation_only", "evidence_valid": True} for c, cnt in inter]}}
        runs["AGREEMENT"] = agree
    lines = [f"# Layer B evaluation — split {split}, {len(gold)} gold units", "",
             "| run | units | relevance P/R/F1 | pair P/R/F1 (counted roles) | role acc | evidence valid | per code F1 (support) |", "|---|---|---|---|---|---|---|"]
    out = {}
    cb = yaml.safe_load((ROOT / "src" / "codebooks" / "intimate_body.yaml").read_text(encoding="utf-8"))
    for name, preds in runs.items():
        units = [u for u in gold if u in preds]
        rtp = rfp = rfn = 0
        ptp = pfp = pfn = 0
        role_ok = role_n = 0
        ev_ok = ev_n = 0
        per = defaultdict(lambda: [0, 0, 0])
        for u in units:
            g = gold[u]
            p = preds[u]["result"]
            g_pairs = {a["code"]: a["role"] for a in g["annotations"] if a["role"] in COUNTED}
            p_pairs = {a["code"]: a["role"] for a in p.get("annotations", []) if a.get("role") in COUNTED}
            g_rel, p_rel = bool(g_pairs), bool(p_pairs)
            rtp += g_rel and p_rel; rfp += (not g_rel) and p_rel; rfn += g_rel and not p_rel
            for c in set(g_pairs) | set(p_pairs):
                if c in g_pairs and c in p_pairs:
                    ptp += 1; per[c][0] += 1; role_n += 1; role_ok += g_pairs[c] == p_pairs[c]
                elif c in p_pairs:
                    pfp += 1; per[c][1] += 1
                else:
                    pfn += 1; per[c][2] += 1
            for a in p.get("annotations", []):
                ev_n += 1; ev_ok += bool(a.get("evidence_valid", True))
        rp, rr, rf = prf(rtp, rfp, rfn)
        pp, pr, pf = prf(ptp, pfp, pfn)
        codes = sorted(per.items(), key=lambda kv: -(kv[1][0] + kv[1][2]))
        percode = ", ".join(f"{c} {prf(*v)[2]:.2f} ({v[0] + v[2]})" for c, v in codes if v[0] + v[2] > 0)
        lines.append(f"| {name} | {len(units)} | {rp:.2f}/{rr:.2f}/{rf:.2f} | {pp:.2f}/{pr:.2f}/{pf:.2f} | "
                     f"{(role_ok / role_n if role_n else 0):.2f} | {(ev_ok / ev_n if ev_n else 1):.2f} | {percode} |")
        out[name] = {"units": len(units), "relevance": [rp, rr, rf], "pairs": [pp, pr, pf], "role_acc": role_ok / role_n if role_n else None,
                     "evidence_valid": ev_ok / ev_n if ev_n else None, "per_code": {c: {"tp": v[0], "fp": v[1], "fn": v[2]} for c, v in per.items()}}
    lines += ["", f"Codebook v{cb['version']}; gold annotated by a model (Claude Opus 5.5) on the editor's behalf. Counted roles: {', '.join(sorted(COUNTED))}."]
    return lines, out


def eval_subjects(pred_files: list[Path]) -> tuple[list[str], dict]:
    gold = {json.loads(l)["doc"]: json.loads(l) for l in (GOLD / "subjects_gold.jsonl").open(encoding="utf-8")}
    sc = yaml.safe_load((ROOT / "src" / "codebooks" / "subjects.yaml").read_text(encoding="utf-8"))
    facet = {c["id"]: c["facet"] for c in sc["concepts"]}
    lines = [f"# Layer A evaluation — {len(gold)} gold consilia", "",
             "| run | docs | central P/R/F1 | any P/R/F1 (central+secondary) | gold central covered | facet-level any F1 |", "|---|---|---|---|---|---|"]
    out = {}
    for p in pred_files:
        preds = load_preds(p, "id")
        docs = [d for d in gold if d in preds]
        c_tp = c_fp = c_fn = a_tp = a_fp = a_fn = f_tp = f_fp = f_fn = 0
        covered = 0
        for d in docs:
            g, r = gold[d], preds[d]["result"]
            gc, pc = set(g["central"]), set(r.get("central", []))
            ga, pa = gc | set(g["secondary"]), pc | set(r.get("secondary", []))
            c_tp += len(gc & pc); c_fp += len(pc - gc); c_fn += len(gc - pc)
            a_tp += len(ga & pa); a_fp += len(pa - ga); a_fn += len(ga - pa)
            gf, pf_ = {facet.get(x) for x in ga}, {facet.get(x) for x in pa}
            f_tp += len(gf & pf_); f_fp += len(pf_ - gf); f_fn += len(gf - pf_)
            covered += gc <= pa
        cp, cr, cf = prf(c_tp, c_fp, c_fn); ap_, ar, af = prf(a_tp, a_fp, a_fn); _, _, ff = prf(f_tp, f_fp, f_fn)
        lines.append(f"| {p.stem} | {len(docs)} | {cp:.2f}/{cr:.2f}/{cf:.2f} | {ap_:.2f}/{ar:.2f}/{af:.2f} | {(covered / len(docs) if docs else 0):.2f} | {ff:.2f} |")
        out[p.stem] = {"docs": len(docs), "central": [cp, cr, cf], "any": [ap_, ar, af], "central_covered": covered / len(docs) if docs else None, "facet_any_f1": ff}
    lines += ["", f"Scheme v{sc['version']}; gold annotated by a model (Claude Opus 5.5) on the editor's behalf. Target before the index switches to Layer A: central precision >= 0.8."]
    return lines, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", choices=["body", "subjects"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("preds", nargs="+", type=Path)
    args = ap.parse_args()
    lines, out = eval_body(args.preds, args.split) if args.task == "body" else eval_subjects(args.preds)
    RES.mkdir(parents=True, exist_ok=True)
    name = f"{args.task}_{args.split}" if args.task == "body" else "subjects"
    (RES / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RES / f"{name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {RES / (name + '.md')}")


if __name__ == "__main__":
    main()
