"""Evaluate the menu models (and their shipped ONNX variants) on ETP-Eval26.

ETP-Eval26 (mschonhardt/ETP-Eval26, CC BY 4.0): 2,051 chapters of Burchard's Decretum, 24,654
synthetic queries (paraphrase / keyword / persona; Latin, German, English), one target chapter per
query. Corpus side here: the Latin chapter text. Cross-lingual by design (German/English queries
against Latin passages).

Two questions:
  1. model comparison (domain: medieval canon law, NOT consilia). Caveat: mschonhardt/etp-* models
     were trained in the same project; if the fine-tune saw this domain or these queries the numbers
     are in-domain for it, not held-out. `leakage_flagged` queries are excluded.
  2. quantisation: does the shipped ONNX query encoder (int8 / other) lose retrieval quality against
     the fp32 query encoder, with the corpus embedded once by the reference model (as in production)?
     That is the criterion that matters; top-20 overlap (tools/parity_ref.py) is a stricter proxy.

Metrics: MRR@10, nDCG@10, Recall@10 (one relevant chapter per query => nDCG = 1/log2(rank+1)).
Queries: stratified sample, N per (language, style), fixed seed.

    python src/eval/eval_etp26.py --tags bge-m3 etp-bgem3-ft qwen3-0.6b
    python src/eval/eval_etp26.py --tags etp-bgem3-ft --onnx etp-bgem3-ft=int8:hf-models/etp-bgem3-contrastive-ft/onnx/model_quantized.onnx
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "hf-models" / "_eval" / "ETP-Eval26"


def load(n_per_cell: int, seed: int = 20260923):
    corpus = [json.loads(l) for l in open(DATA / "corpus.jsonl", encoding="utf-8")]
    qrels = {}
    for l in open(DATA / "qrels.jsonl", encoding="utf-8"):
        r = json.loads(l)
        qrels[r["query-id"]] = r["corpus-id"]
    cells = collections.defaultdict(list)
    for l in open(DATA / "queries.jsonl", encoding="utf-8"):
        q = json.loads(l)
        if q.get("leakage_flagged"):
            continue
        cells[(q["lang"], q["style"])].append(q)
    rnd = random.Random(seed)
    queries = []
    for k in sorted(cells):
        queries += rnd.sample(cells[k], min(n_per_cell, len(cells[k])))
    return corpus, queries, qrels


def metrics(S: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    """S: queries x docs scores; target: index of the relevant doc per query."""
    rank = 1 + (S > S[np.arange(len(S)), target][:, None]).sum(axis=1)
    return {"mrr10": np.where(rank <= 10, 1.0 / rank, 0.0),
            "ndcg10": np.where(rank <= 10, 1.0 / np.log2(rank + 1), 0.0),
            "r10": (rank <= 10).astype(float)}


def summarise(m: dict[str, np.ndarray], queries: list[dict]) -> dict:
    out = {k: round(float(v.mean()), 4) for k, v in m.items()}
    for by in ("lang", "style"):
        for val in sorted({q[by] for q in queries}):
            idx = [i for i, q in enumerate(queries) if q[by] == val]
            out[f"{by}={val}/ndcg10"] = round(float(m["ndcg10"][idx].mean()), 4)
    return out


def onnx_encoder(path: str, cfg: dict, tokenizer_dir: str):
    import onnxruntime as ort
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}

    def enc(texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            e = tok(cfg.get("query_prefix", "") + t, return_tensors="np", truncation=True, max_length=512)
            h = sess.run(None, {k: v.astype(np.int64) for k, v in e.items() if k in names})[0][0]
            v = h[0] if cfg["pooling"] == "cls" else h[-1]
            out.append(v / np.linalg.norm(v))
        return np.array(out, dtype=np.float32)
    return enc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tags", nargs="+", default=["bge-m3", "etp-bgem3-ft", "qwen3-0.6b"])
    ap.add_argument("--per-cell", type=int, default=60, help="queries per (language, style)")
    ap.add_argument("--onnx", nargs="*", default=[], help="tag=label:path (queries via onnxruntime CPU, corpus stays fp32)")
    ap.add_argument("--tokenizer-dir", action="append", default=[], help="tag=dir with tokenizer files for --onnx")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results" / "etp26.json")
    a = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    models = {m["tag"]: m for m in json.loads((ROOT / "site-data" / "models.json").read_text())["models"]}
    corpus, queries, qrels = load(a.per_cell)
    doc_ids = [c["_id"] for c in corpus]
    pos = {d: i for i, d in enumerate(doc_ids)}
    target = np.array([pos[qrels[q["_id"]]] for q in queries])
    texts = [c["text_lat"] for c in corpus]
    print(f"{len(corpus)} chapters, {len(queries)} queries", file=sys.stderr)
    tdirs = dict(x.split("=", 1) for x in a.tokenizer_dir)
    onnx = collections.defaultdict(list)
    for spec in a.onnx:
        tag, rest = spec.split("=", 1)
        label, path = rest.split(":", 1)
        onnx[tag].append((label, path))

    results = {}
    for tag in a.tags:
        cfg = models[tag]
        qwen = "qwen" in tag
        st = SentenceTransformer(cfg["model_id"], device="cuda",
                                 model_kwargs={"torch_dtype": torch.bfloat16 if qwen else torch.float16},
                                 tokenizer_kwargs={"padding_side": "left"} if qwen else {})
        st.max_seq_length = 512
        D = st.encode(texts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True,
                      show_progress_bar=False).astype(np.float32)
        qtexts = [q["text"] for q in queries]
        Q = st.encode([cfg.get("query_prefix", "") + t for t in qtexts], batch_size=32, normalize_embeddings=True,
                      convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        results[f"{tag}/reference"] = summarise(metrics(Q @ D.T, target), queries)
        del st
        torch.cuda.empty_cache()
        for label, path in onnx.get(tag, []):
            enc = onnx_encoder(path, cfg, tdirs.get(tag, str(Path(path).parent.parent)))
            Qo = enc(qtexts)
            results[f"{tag}/onnx:{label}"] = summarise(metrics(Qo @ D.T, target), queries) | {
                "mean_cos_to_reference": round(float((Qo * Q).sum(axis=1).mean()), 4)}
        print(tag, json.dumps({k: v for k, v in results.items() if k.startswith(tag)}, indent=None)[:400], file=sys.stderr)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"n_queries": len(queries), "per_cell": a.per_cell, "results": results}, indent=1), encoding="utf-8")
    keys = ["mrr10", "ndcg10", "r10", "lang=lat/ndcg10", "lang=deu/ndcg10", "lang=eng/ndcg10",
            "style=keyword/ndcg10", "style=paraphrase/ndcg10", "style=persona/ndcg10"]
    print(f"{'':32}" + "".join(f"{k.replace('/ndcg10', ''):>16}" for k in keys))
    for name, r in results.items():
        print(f"{name:32}" + "".join(f"{r[k]:16.3f}" for k in keys))


if __name__ == "__main__":
    main()
