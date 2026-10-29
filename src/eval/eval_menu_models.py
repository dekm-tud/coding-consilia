"""Evaluate the browser-menu embedding models on the Layer B gold.

Why not the old keyword ground truth: it comes from the English translation (favours English
embeddings), and R@K is capped when there are more relevant documents than K. Here the
relevance judgements are the verified passage annotations of eval/gold (independent of the
keyword pipeline), and the metrics are rank-based (AP, nDCG@20, first hit).

Task: for each of the 27 codes of src/codebooks/intimate_body.yaml, a query (three variants:
short English label, English definition, Latin label + cues) must rank the passages annotated
with that code (roles topic/argument/case_fact) above the rest.

Two pools, both reported:
  corpus  every section of the corpus is a candidate; only annotated positives count as relevant
          (a lower bound: unannotated true positives are counted as errors)
  judged  only the sections that were judged in the gold (positives and negatives, incl. the
          homonym traps): how well does the model separate known positives from known negatives

Rows of a section (Latin and English chunk) are pooled by max score, as in the search page.
Gold is model-annotated (Claude), see the About text; numbers are for comparing models, not absolute.

    python src/eval/eval_menu_models.py --tags bge-m3 etp-bgem3-ft qwen3-0.6b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parent.parent.parent
GOLD = ROOT / "eval" / "gold"
COUNTED_ROLES = {"topic", "argument", "case_fact"}


def load_gold() -> dict[str, set[tuple[str, str, int]]]:
    """code -> set of sections (doc id, kind, index) annotated with it; plus the judged pool."""
    pos: dict[str, set] = {}
    judged: set = set()
    for fname in ("intimate_body_gold.jsonl", "intimate_body_rare_gold.jsonl"):
        for line in (GOLD / fname).open(encoding="utf-8"):
            r = json.loads(line)
            doc, rest = r["unit"].split("#")
            kind, idx = ("casus" if rest[0] == "c" else "body"), int(rest[1:].split("w")[0])
            sec = (doc, kind, idx)
            judged.add(sec)
            for a in r["annotations"]:
                if a["role"] in COUNTED_ROLES:
                    pos.setdefault(a["code"], set()).add(sec)
    return pos, judged


def queries(codes: list[dict]) -> dict[str, dict[str, str]]:
    out = {}
    for c in codes:
        out[c["id"]] = {
            "label": c["label_en"].split("(")[0].strip(),
            "definition": c.get("definition") or c["label_en"],
            "latin": (c.get("label_la") or "") + "; " + "; ".join((c.get("latin_cues") or [])[:3]),
        }
    return out


def ap_ndcg(order: list[int], rel: set[int], k: int = 20) -> tuple[float, float, float]:
    """average precision, nDCG@k, reciprocal rank of the first relevant"""
    hits, ap, dcg, first = 0, 0.0, 0.0, 0.0
    for rank, i in enumerate(order, 1):
        if i in rel:
            hits += 1
            ap += hits / rank
            first = first or 1.0 / rank
            if rank <= k:
                dcg += 1.0 / np.log2(rank + 1)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(rel), k) + 1))
    return ap / len(rel), (dcg / idcg if idcg else 0.0), first


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tags", nargs="+", default=["bge-m3", "etp-bgem3-ft", "qwen3-0.6b"])
    ap.add_argument("--update-models", action="store_true", help="write the numbers into site-data/models.json")
    ap.add_argument("--baseline", default="bge-m3")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results" / "menu_models_gold.json")
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    models = {m["tag"]: m for m in json.loads((ROOT / "site-data" / "models.json").read_text())["models"]}
    index = json.loads((ROOT / "site-data" / "index.json").read_text())
    id_of = {e["i"]: e["id"] for e in index}
    codes = yaml.safe_load((ROOT / "src" / "codebooks" / "intimate_body.yaml").read_text())["codes"]
    Q = queries(codes)
    pos, judged = load_gold()
    active = {c: s for c, s in pos.items() if s}
    print(f"{len(active)} codes with positives; {len(judged)} judged sections", file=sys.stderr)

    results = {}
    for tag in args.tags:
        cfg = models[tag]
        meta = json.loads((ROOT / "hf-export" / tag / "chunks.json").read_text())
        E = np.fromfile(ROOT / "hf-export" / tag / "chunks.f16.bin", dtype="<f2").astype(np.float32).reshape(meta["n"], meta["dims"])
        sec_of_row = [(id_of[d], "casus" if k else "body", c) for d, k, c in zip(meta["doc"], meta["kind"], meta["chunk"])]
        secs = sorted(set(sec_of_row))
        sec_idx = {s: i for i, s in enumerate(secs)}
        row_sec = np.array([sec_idx[s] for s in sec_of_row])
        judged_idx = np.array([sec_idx[s] for s in judged if s in sec_idx])
        st = SentenceTransformer(cfg["model_id"], device="cuda",
                                 model_kwargs={"torch_dtype": torch.bfloat16 if "qwen" in tag else torch.float16},
                                 tokenizer_kwargs={"padding_side": "left"} if "qwen" in tag else {})
        per = {}
        for variant in ("label", "definition", "latin"):
            qs = [cfg.get("query_prefix", "") + Q[c][variant] for c in active]
            qv = st.encode(qs, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            S = qv @ E.T                                            # codes x rows
            sec_scores = np.full((len(qs), len(secs)), -1e9, dtype=np.float32)
            # max-pool rows to sections
            order = np.argsort(row_sec, kind="stable")
            bounds = np.flatnonzero(np.diff(row_sec[order])) + 1
            for j, chunk in enumerate(np.split(order, bounds)):
                sec_scores[:, row_sec[chunk[0]]] = S[:, chunk].max(axis=1)
            for pool, cand in (("corpus", np.arange(len(secs))), ("judged", judged_idx)):
                m = {"ap": [], "ndcg20": [], "mrr": []}
                for qi, code in enumerate(active):
                    rel = {sec_idx[s] for s in active[code] if s in sec_idx} & set(cand.tolist())
                    if not rel:
                        continue
                    ranked = cand[np.argsort(-sec_scores[qi, cand])].tolist()
                    a, n, f = ap_ndcg(ranked, rel)
                    m["ap"].append(a); m["ndcg20"].append(n); m["mrr"].append(f)
                per[f"{variant}/{pool}"] = {k: round(float(np.mean(v)), 4) for k, v in m.items() if k != "codes_ap"} | {
                    "codes": len(m["ap"]), "per_code_ap": [round(x, 4) for x in m["ap"]]}
        results[tag] = per
        del st
        torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    cols = list(next(iter(results.values())))
    print(f"{'model':14}", *(f"{k:>17}" for k in cols))
    for tag, per in results.items():
        print(f"{tag:14}", *(f"AP {v['ap']:.3f} n{v['ndcg20']:.2f}".rjust(17) for v in per.values()))
    base = args.baseline if args.baseline in results else next(iter(results))
    rng = np.random.default_rng(20260923)
    print(f"\npaired bootstrap over codes, mean AP difference vs {base} (95% CI; * = CI excludes 0)")
    for tag in results:
        if tag == base:
            continue
        cells = []
        for c in cols:
            d = np.array(results[tag][c]["per_code_ap"]) - np.array(results[base][c]["per_code_ap"])
            bs = np.array([rng.choice(d, len(d)).mean() for _ in range(5000)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            cells.append(f"{d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]{'*' if lo > 0 or hi < 0 else ' '}")
        print(f"{tag:14}", *(x.rjust(26) for x in cells))
        results[tag]["_vs_" + base] = dict(zip(cols, cells))
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    if args.update_models:
        mp = ROOT / "site-data" / "models.json"
        data = json.loads(mp.read_text(encoding="utf-8"))
        for m in data["models"]:
            if m["tag"] in results:
                r = results[m["tag"]]
                m["eval"] = {
                    "gold": "Layer B gold (model-annotated), 25 codes; mean average precision",
                    "judged_pool": {v: r[f"{v}/judged"]["ap"] for v in ("label", "definition", "latin")},
                    "corpus": {v: r[f"{v}/corpus"]["ap"] for v in ("label", "definition", "latin")},
                }
        mp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        print("updated site-data/models.json (eval)")


if __name__ == "__main__":
    main()
