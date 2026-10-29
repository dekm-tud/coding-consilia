"""
Generate chunked embeddings for candidate production models — run on the GPU
server, then copy the output files back.

Each consilium's body is split into its natural body-section chunks (already
split at the original print's paragraph marks). One embedding is stored per
chunk, not per consilium — retrieval ranks documents by their single
best-matching chunk (max-pool over a document's chunks), which is what
eval_embeddings_chunked.py measured and is why this script does not average
chunks into one per-document vector.

Writes one {safetensors, meta.json} pair per model/language below, so all
three can be compared/chosen between later without re-running the GPU job:
  - Qwen3-Embedding-8B, English
  - Qwen3-Embedding-4B, English
  - etp-canonberta-contrastive-ft, Latin

Usage:
    pip install sentence-transformers torch safetensors
    python src/gpu/embed_consilia_chunked.py --input output/consilia.json

Note for whichever model gets wired into search later: the Qwen models embed
documents with prompt_name="document" (done here) and expect queries encoded
with prompt_name="query" at retrieval time -- etp-canonberta-contrastive-ft
uses no such prompt for either side.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

MODELS = [
    {"id": "Qwen/Qwen3-Embedding-8B", "lang": "english", "tag": "qwen8b_en",
     "prompt_doc": "document", "model_kwargs": {"dtype": torch.bfloat16}, "max_seq_length": 768},
    {"id": "Qwen/Qwen3-Embedding-4B", "lang": "english", "tag": "qwen4b_en",
     "prompt_doc": "document", "model_kwargs": {"dtype": torch.bfloat16}, "max_seq_length": 768},
    {"id": "mschonhardt/etp-canonberta-contrastive-ft", "lang": "latin", "tag": "canonberta_lat",
     "prompt_doc": None, "model_kwargs": {}, "max_seq_length": None},
]

# Below this length a "body section" is almost always the formulaic opener
# ("In Christi Nomine Amen.", "Quaeritur,", …) rather than real content -- 485
# of 2504 consilia share that exact opener string. Embedding it as its own
# chunk makes unrelated consilia collide at ~1.0 cosine similarity on that
# boilerplate alone. Real content chunks jump straight from ~28 to ~159+
# characters in this corpus, so 50 cleanly separates the two.
MIN_CHUNK_CHARS = 50


def latin_chunks(cid: str, c: dict) -> list[tuple[str, str]]:
    # casus (statement of facts, printed before the heading) belongs to this consilium
    secs = list(c.get("body") or []) + list(c.get("casus") or [])
    return [(cid, sec) for sec in secs if len(sec.strip()) > MIN_CHUNK_CHARS]


def english_chunks(cid: str, c: dict) -> list[tuple[str, str]]:
    tr = c.get("translations", {}).get("en", {})
    secs = list(tr.get("body") or []) + list(tr.get("casus") or [])
    return [(cid, sec) for sec in secs if len(sec.strip()) > MIN_CHUNK_CHARS]


def unload(model) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="output/consilia.json")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    consilia = data["consilia"]

    latin_pairs = [p for cid, c in consilia.items() for p in latin_chunks(cid, c)]
    english_pairs = [p for cid, c in consilia.items() for p in english_chunks(cid, c)]
    log.info("%d latin chunks, %d english chunks", len(latin_pairs), len(english_pairs))

    for spec in MODELS:
        log.info("Loading %s ...", spec["id"])
        model = SentenceTransformer(spec["id"], device="cuda", model_kwargs=spec["model_kwargs"])
        if spec["max_seq_length"]:
            model.max_seq_length = spec["max_seq_length"]

        pairs = latin_pairs if spec["lang"] == "latin" else english_pairs
        chunk_ids = [p[0] for p in pairs]
        chunk_texts = [p[1] for p in pairs]

        enc_kwargs = dict(batch_size=args.batch_size, show_progress_bar=True,
                           normalize_embeddings=True, convert_to_numpy=True)
        if spec["prompt_doc"]:
            enc_kwargs["prompt_name"] = spec["prompt_doc"]

        log.info("Embedding %d %s chunks with %s ...", len(chunk_texts), spec["lang"], spec["id"])
        embeddings = model.encode(chunk_texts, **enc_kwargs).astype(np.float32)

        # per-chunk index within its own consilium, and author/volume for
        # filtering later -- matches embeddings_meta.json's existing shape
        chunk_index: list[int] = []
        seen: dict[str, int] = {}
        for cid in chunk_ids:
            chunk_index.append(seen.get(cid, 0))
            seen[cid] = seen.get(cid, 0) + 1

        st_path = out_dir / f"embeddings_{spec['tag']}.safetensors"
        meta_path = out_dir / f"embeddings_{spec['tag']}_meta.json"
        save_file({"embeddings": embeddings}, st_path)
        meta = {
            "model": spec["id"],
            "lang": spec["lang"],
            "dims": int(embeddings.shape[1]),
            "n_chunks": int(embeddings.shape[0]),
            "chunk_consilium_ids": chunk_ids,
            "chunk_index": chunk_index,
            "author_viafs": [consilia[cid].get("author_viaf", "") for cid in chunk_ids],
            "volumes": [consilia[cid].get("volume", "") for cid in chunk_ids],
        }
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        log.info("Saved %s (%.1f MB) and %s", st_path, st_path.stat().st_size / 1e6, meta_path)

        unload(model)

    log.info("Done.")


if __name__ == "__main__":
    main()
