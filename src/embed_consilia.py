"""
Generate BGE-M3 embeddings for all consilia and save as SafeTensors.

Usage:
    python src/embed_consilia.py
    python src/embed_consilia.py --input output/consilia.json --output output/embeddings.safetensors

One embedding per body-section or casus-section chunk (already split at the original print's
paragraph marks) -- both the Latin original and its English translation, so
a query in either language can match directly. Retrieval ranks a consilium
by its single best-matching chunk (max-pool over that document's rows, done
at query time), not by a whole-document vector: evaluation on this corpus
showed chunking recovers rare/short mentions that get diluted when a whole
consilium (title+summary+body) is embedded as one long passage.

BGE-M3 (1024-dim, multilingual, 8192-token context) is used so that queries
in Latin, German, or English all map to the same vector space.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

MODEL_ID = "BAAI/bge-m3"

# Below this length a "body section" is almost always the formulaic opener
# ("In Christi Nomine Amen.", "Quaeritur,", …) rather than real content -- 485
# of 2504 consilia share that exact opener string. Embedding it as its own
# chunk makes unrelated consilia collide at ~1.0 cosine similarity on that
# boilerplate alone, drowning out genuine similarity. Real content chunks
# jump straight from ~28 to ~159+ characters in this corpus, so 50 cleanly
# separates the two.
MIN_CHUNK_CHARS = 50


def build_chunks(consilia: list[dict]) -> tuple[list[str], list[str], list[str], list[int], list[str]]:
    """One row per body-section or casus-section chunk: all Latin chunks first, then all
    English (translated) chunks. Returns (ids, langs, texts, chunk_index, kinds) — parallel
    lists. `kinds` is "body" or "casus"; chunk_index is the index within that list.

    The casus (statement of facts, set before the heading in the print) belongs to the
    consilium it introduces, see casus.py."""
    ids: list[str] = []
    langs: list[str] = []
    texts: list[str] = []
    chunk_index: list[int] = []
    kinds: list[str] = []

    for lang in ("la", "en"):
        for c in consilia:
            for kind in ("body", "casus"):
                secs = (c.get(kind) or []) if lang == "la" else \
                    (c.get("translations", {}).get("en", {}).get(kind) or [])
                for i, sec in enumerate(secs):
                    if len(sec.strip()) <= MIN_CHUNK_CHARS:
                        continue
                    ids.append(c["id"])
                    langs.append(lang)
                    texts.append(sec)
                    chunk_index.append(i)
                    kinds.append(kind)

    return ids, langs, texts, chunk_index, kinds


def embed(
    input_path: str | Path = "output/consilia.json",
    output_path: str | Path = "output/embeddings.safetensors",
    meta_path: str | Path = "output/embeddings_meta.json",
    batch_size: int = 32,
    device: str | None = None,
    fp16: bool = False,
    max_seq_length: int = 512,
) -> None:
    """Generate chunked embeddings for all consilia in *input_path* and save to *output_path*."""
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    consilia = list(data["consilia"].values())
    consilia.sort(key=lambda c: c["id"])

    ids, langs, texts, chunk_index, kinds = build_chunks(consilia)
    log.info("%d chunks (%d latin, %d english) from %d consilia",
              len(texts), langs.count("la"), langs.count("en"), len(consilia))

    log.info("Loading %s …", MODEL_ID)
    model = SentenceTransformer(MODEL_ID, device=device)
    model.max_seq_length = max_seq_length
    if fp16:
        model.half()
        log.info("Using fp16 (halved GPU memory).")
    log.info("Device: %s  |  max_seq_length: %d", model.device, model.max_seq_length)

    log.info("Embedding %d chunks (batch_size=%d) …", len(texts), batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32)
    log.info("Embedding shape: %s", embeddings.shape)

    out = Path(output_path)
    save_file({"embeddings": embeddings}, out)
    log.info("Saved %s (%.1f MB)", out, out.stat().st_size / 1e6)

    meta = {
        "model": MODEL_ID,
        "dims": int(embeddings.shape[1]),
        "n": int(embeddings.shape[0]),
        "ids": ids,             # per-row consilium id -- NOT unique, one row per chunk
        "langs": langs,         # per-row "la" or "en"
        "chunk_index": chunk_index,
        "kinds": kinds,         # per-row "body" or "casus" (chunk_index counts within that list)
    }
    Path(meta_path).write_text(json.dumps(meta), encoding="utf-8")
    log.info("Saved %s", meta_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="output/consilia.json")
    ap.add_argument("--output", default="output/embeddings.safetensors")
    ap.add_argument("--meta",   default="output/embeddings_meta.json")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None,
                    help="Force device: 'cuda', 'cpu', 'cuda:0', etc.")
    ap.add_argument("--fp16", action="store_true",
                    help="Load model in float16 — halves GPU memory (~1 GB vs ~2 GB).")
    ap.add_argument("--max-seq-length", type=int, default=512,
                    help="Token budget per chunk (default 512, comfortably covers "
                         "the vast majority of body-section chunks). BGE-M3 supports "
                         "up to 8192 if a print has unusually long sections.")
    args = ap.parse_args()
    embed(
        input_path=args.input,
        output_path=args.output,
        meta_path=args.meta,
        batch_size=args.batch_size,
        device=args.device,
        fp16=args.fp16,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
