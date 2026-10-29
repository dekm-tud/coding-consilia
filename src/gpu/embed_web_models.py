"""
Embed the corpus with every model of the browser menu and write the web format directly
:  hf-export/<tag>/{chunks.f16.bin, chunks.json[, chunks.i8.bin]}

Menu (tag -> model): bge-m3, etp-bgem3-ft, qwen3-0.6b, qwen3-4b.

The SAME chunks are embedded for every model: body and casus sections longer than
MIN_CHUNK_CHARS, Latin and English translation, in the order of src/embed_consilia.build_chunks
(so a chunk index means the same thing in every model directory).

  * BGE-M3 and etp-bgem3-ft: no prefix, CLS pooling, L2-normalised (sentence-transformers config).
  * Qwen3-Embedding: documents get NO instruction; last-token pooling with left padding.
    Queries (in the browser) get  "Instruct: <task>\\nQuery:<q>"  -- the task string lives in
    site-data/models.json (`query_prefix`) and tools/parity_ref.py, never here.

Runs on this machine (12 GB: bge-m3, etp-bgem3-ft, qwen3-0.6b) or on the GPU server (all four).
Needs on the machine: output/consilia.json, site-data/index.json, site-data/manifest.json
(run src/build_site_data.py first), plus src/embed_consilia.py and src/export_embeddings_web.py.

    python src/gpu/embed_web_models.py --tags etp-bgem3-ft qwen3-0.6b
    python src/gpu/embed_web_models.py --tags qwen3-4b --batch-size 16     # GPU server
    python src/gpu/embed_web_models.py --tags bge-m3 --int8
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from embed_consilia import build_chunks  # noqa: E402
from export_embeddings_web import write_export  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

MODELS = {
    "bge-m3":       {"id": "BAAI/bge-m3", "dtype": torch.float16, "left_pad": False},
    "etp-bgem3-ft": {"id": "mschonhardt/etp-bgem3-contrastive-ft", "dtype": torch.float16, "left_pad": False},
    "qwen3-0.6b":   {"id": "Qwen/Qwen3-Embedding-0.6B", "dtype": torch.bfloat16, "left_pad": True},
    "qwen3-4b":     {"id": "Qwen/Qwen3-Embedding-4B", "dtype": torch.bfloat16, "left_pad": True},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tags", nargs="+", choices=list(MODELS), default=list(MODELS))
    ap.add_argument("--input", type=Path, default=ROOT / "output" / "consilia.json")
    ap.add_argument("--index", type=Path, default=ROOT / "site-data" / "index.json")
    ap.add_argument("--out-root", type=Path, default=ROOT / "hf-export")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq-length", type=int, default=512,
                    help="token budget per chunk; identical for all models so the chunks are comparable")
    ap.add_argument("--revision", default="v0.1")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--limit", type=int, help="embed only the first N chunks (smoke test, writes nothing)")
    args = ap.parse_args()

    consilia = sorted(json.loads(args.input.read_text(encoding="utf-8"))["consilia"].values(), key=lambda c: c["id"])
    ids, langs, texts, chunk_index, kinds = build_chunks(consilia)
    log.info("%d chunks (%d latin, %d english; %d casus)", len(texts), langs.count("la"), langs.count("en"), kinds.count("casus"))
    meta = {"ids": ids, "langs": langs, "chunk_index": chunk_index, "kinds": kinds}

    for tag in args.tags:
        spec = MODELS[tag]
        log.info("== %s (%s)", tag, spec["id"])
        t0 = time.time()
        model = SentenceTransformer(
            spec["id"], device="cuda",
            model_kwargs={"torch_dtype": spec["dtype"]},
            tokenizer_kwargs={"padding_side": "left"} if spec["left_pad"] else {},
        )
        model.max_seq_length = args.max_seq_length
        log.info("pooling/normalize modules: %s", [type(m).__name__ for m in model])
        sel = texts[: args.limit] if args.limit else texts
        E = model.encode(sel, batch_size=args.batch_size, show_progress_bar=True,
                         normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        log.info("%s: %s in %.0f s", tag, E.shape, time.time() - t0)
        if args.limit:
            log.info("smoke test: first vector norm %.4f, not written", float(np.linalg.norm(E[0])))
        else:
            write_export(E, meta, args.index, tag, args.out_root / tag, model_id=spec["id"],
                         revision=args.revision, int8=args.int8)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    log.info("done")


if __name__ == "__main__":
    main()
