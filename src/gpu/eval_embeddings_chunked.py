"""
Chunked embedding-model retrieval eval — run on the GPU server.

Compares candidate production embedding models on Recall@{10,20,50} against
keyword-tagged ground-truth clusters (adultery, rape, usury, dowry,
jurisdiction, incest). Includes the Qwen3-Embedding family at 8B/4B/0.6B,
since a smaller model that's "good enough" is cheaper to run/serve than a
larger one. Body text only, chunked at natural body-section
boundaries (already split at the original print's paragraph marks) — one
embedding per chunk, documents ranked by their best-matching chunk
(max-pool). Chunking matters here: whole-document embedding at full length
is impractically slow/memory-hungry for a 4B model (see max-seq-length
below), and per-section chunks match the production chunking plan anyway.

Usage:
    pip install sentence-transformers torch
    python src/gpu/eval_embeddings_chunked.py --input output/consilia.json

Not part of the main pipeline's requirements.txt — install on the GPU server
separately, same convention as qwen_process.py / name_clusters.py.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

CLUSTERS = {
    "adultery":     ("adulterium", "adultery"),
    "rape":         ("raptus", "rape"),
    "usury":        ("usura", "usury"),
    "dowry":        ("dos", "dowry"),
    "jurisdiction": ("iurisdictio", "jurisdiction"),
    "incest":       ("incestus", "incest"),
}

# Candidate models to compare. "lang" selects which chunk set (latin/english)
# to embed; prompt_name_doc/query select sentence-transformers' built-in
# asymmetric query/document prompting where the model defines one (None if
# the model has no such convention).
MODELS = [
    {"id": "Qwen/Qwen3-Embedding-8B", "lang": "latin",
     "prompt_doc": "document", "prompt_query": "query",
     "model_kwargs": {"dtype": torch.bfloat16}, "max_seq_length": 768},
    {"id": "Qwen/Qwen3-Embedding-8B", "lang": "english",
     "prompt_doc": "document", "prompt_query": "query",
     "model_kwargs": {"dtype": torch.bfloat16}, "max_seq_length": 768},
    {"id": "Qwen/Qwen3-Embedding-4B", "lang": "latin",
     "prompt_doc": "document", "prompt_query": "query",
     "model_kwargs": {"dtype": torch.bfloat16}, "max_seq_length": 768},
    {"id": "Qwen/Qwen3-Embedding-4B", "lang": "english",
     "prompt_doc": "document", "prompt_query": "query",
     "model_kwargs": {"dtype": torch.bfloat16}, "max_seq_length": 768},
    {"id": "Qwen/Qwen3-Embedding-0.6B", "lang": "latin",
     "prompt_doc": "document", "prompt_query": "query",
     "model_kwargs": {}, "max_seq_length": 768},
    {"id": "Qwen/Qwen3-Embedding-0.6B", "lang": "english",
     "prompt_doc": "document", "prompt_query": "query",
     "model_kwargs": {}, "max_seq_length": 768},
    {"id": "mschonhardt/etp-canonberta-contrastive-ft", "lang": "latin",
     "prompt_doc": None, "prompt_query": None,
     "model_kwargs": {}, "max_seq_length": None},
]


# Below this length a "body section" is almost always the formulaic opener
# ("In Christi Nomine Amen.", "Quaeritur,", …) rather than real content -- 485
# of 2504 consilia share that exact opener string, which would otherwise let
# unrelated consilia collide at ~1.0 cosine similarity on boilerplate alone.
MIN_CHUNK_CHARS = 50


def latin_chunks(cid: str, c: dict) -> list[tuple[str, str]]:
    return [(cid, sec) for sec in (c.get("body") or []) if len(sec.strip()) > MIN_CHUNK_CHARS]


def english_chunks(cid: str, c: dict) -> list[tuple[str, str]]:
    body = c.get("translations", {}).get("en", {}).get("body") or []
    return [(cid, sec) for sec in body if len(sec.strip()) > MIN_CHUNK_CHARS]


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    top = set(ranked_ids[:k])
    return len(top & relevant) / len(relevant)


def evaluate(model, chunk_pairs, query_fn, ground_truth, prompt_doc=None, prompt_query=None,
             label="", batch_size=32) -> dict:
    chunk_ids = [p[0] for p in chunk_pairs]
    chunk_texts = [p[1] for p in chunk_pairs]
    doc_ids_present = set(chunk_ids)

    enc_kwargs = dict(batch_size=batch_size, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True)
    if prompt_doc:
        enc_kwargs["prompt_name"] = prompt_doc
    chunk_emb = model.encode(chunk_texts, **enc_kwargs)

    results = {}
    for kw, (lat_q, en_q) in CLUSTERS.items():
        q = query_fn(lat_q, en_q)
        q_kwargs = dict(normalize_embeddings=True, convert_to_numpy=True)
        if prompt_query:
            q_kwargs["prompt_name"] = prompt_query
        q_emb = model.encode([q], **q_kwargs)[0]
        sims = chunk_emb @ q_emb

        doc_best: dict[str, float] = {}
        for cid, s in zip(chunk_ids, sims):
            if cid not in doc_best or s > doc_best[cid]:
                doc_best[cid] = s
        ranked_ids = [cid for cid, _ in sorted(doc_best.items(), key=lambda kv: -kv[1])]

        relevant = ground_truth[kw] & doc_ids_present
        row = {f"R@{k}": recall_at_k(ranked_ids, relevant, k) for k in (10, 20, 50)}
        row["n_relevant"] = len(relevant)
        results[kw] = row

    log.info("=== %s ===", label)
    for kw, row in results.items():
        vals = "  ".join(f"{k}={v:.2f}" if v is not None else f"{k}=NA"
                          for k, v in row.items() if k != "n_relevant")
        log.info("  %-14s n=%4d  %s", kw, row["n_relevant"], vals)
    return results


def unload(model) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="output/consilia.json")
    ap.add_argument("--output", default="output/eval_chunked_results.json")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    consilia = data["consilia"]

    ground_truth = {}
    for kw in CLUSTERS:
        ids = [cid for cid, c in consilia.items() if kw in (c.get("keywords_final") or [])]
        ground_truth[kw] = set(ids)
        log.info("%s: %d tagged consilia", kw, len(ids))

    latin_pairs = [p for cid, c in consilia.items() for p in latin_chunks(cid, c)]
    english_pairs = [p for cid, c in consilia.items() for p in english_chunks(cid, c)]
    log.info("%d latin chunks, %d english chunks", len(latin_pairs), len(english_pairs))

    all_results = {}
    for spec in MODELS:
        log.info("Loading %s ...", spec["id"])
        model = SentenceTransformer(spec["id"], device="cuda", model_kwargs=spec["model_kwargs"])
        if spec["max_seq_length"]:
            model.max_seq_length = spec["max_seq_length"]

        pairs = latin_pairs if spec["lang"] == "latin" else english_pairs
        query_fn = (lambda lat, en: lat) if spec["lang"] == "latin" else (lambda lat, en: en)
        label = f"{spec['id']} / {spec['lang']}"
        key = f"{spec['id'].split('/')[-1]} {spec['lang']}"

        all_results[key] = evaluate(
            model, pairs, query_fn, ground_truth,
            prompt_doc=spec["prompt_doc"], prompt_query=spec["prompt_query"],
            label=label, batch_size=args.batch_size,
        )
        unload(model)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    log.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
