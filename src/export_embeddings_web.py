"""
Convert chunk embeddings (safetensors + meta) to the compact format the browser fetches from
the Hugging Face dataset.

    <out>/chunks.f16.bin   raw little-endian float16, row-major N x D
    <out>/chunks.i8.bin    optional "lite" file: int8 rows + float32 per-row scale  (--int8)
    <out>/chunks.json      {model, tag, revision, dims, n, dtype, doc:[int], chunk:[int],
                            lang:[0=la|1=en], kind:[0=body|1=casus], created, data_revision}
                           `doc` indexes into site-data/index.json (replaces 45-char string ids)

Recall check (always run): top-20 overlap between the float32 and the float16 matrix over 50
query vectors. The queries are 50 randomly chosen chunk vectors (a stand-in for real queries that
needs no model); the export fails if the mean overlap is below --min-overlap (default 0.98).

Usage:
    python src/export_embeddings_web.py                                    # BGE-M3 -> hf-export/bge-m3/
    python src/export_embeddings_web.py --tag bge-m3 --int8
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).parent.parent


def topk_overlap(E32: np.ndarray, E16: np.ndarray, n_queries: int = 50, k: int = 20, seed: int = 20260923) -> float:
    rng = np.random.default_rng(seed)
    q = E32[rng.choice(len(E32), n_queries, replace=False)]
    ref = np.argsort(-(q @ E32.T), axis=1)[:, :k]
    got = np.argsort(-(q @ E16.astype(np.float32).T), axis=1)[:, :k]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(ref, got)]))


def write_export(E: np.ndarray, meta: dict, index_path: Path, tag: str, out: Path, *,
                 model_id: str | None = None, revision: str = "v0.1", int8: bool = False,
                 min_overlap: float = 0.98, extra: dict | None = None) -> dict:
    """Write chunks.f16.bin (+ optional chunks.i8.bin) and chunks.json for one model.

    E: float32 N x D; meta: {ids, chunk_index, langs, kinds?} as written by embed_consilia.py."""
    out.mkdir(parents=True, exist_ok=True)
    E = E.astype(np.float32)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    site_manifest = json.loads((index_path.parent / "manifest.json").read_text(encoding="utf-8"))
    doc_of = {e["id"]: e["i"] for e in index}
    if len(meta["ids"]) != len(E):
        sys.exit(f"meta has {len(meta['ids'])} rows but the matrix has {len(E)}")
    missing = sorted({i for i in meta["ids"] if i not in doc_of})
    if missing:
        sys.exit(f"{len(missing)} embedding ids are not in site-data/index.json (rebuild it): {missing[:3]}")

    E16 = E.astype("<f2")
    overlap = topk_overlap(E, E16)
    print(f"[{tag}] top-20 overlap float32 vs float16: {overlap:.4f}")
    if overlap < min_overlap:
        sys.exit(f"overlap {overlap:.4f} below {min_overlap}: not exporting")
    E16.tofile(out / "chunks.f16.bin")

    dtype = ["f16"]
    if int8:
        scale = np.abs(E).max(axis=1) / 127.0
        scale[scale == 0] = 1.0
        q = np.clip(np.rint(E / scale[:, None]), -127, 127).astype(np.int8)
        (out / "chunks.i8.bin").write_bytes(q.tobytes() + scale.astype("<f4").tobytes())
        print(f"[{tag}] top-20 overlap float32 vs int8: {topk_overlap(E, q.astype(np.float32) * scale[:, None]):.4f}")
        dtype.append("i8")

    kinds = meta.get("kinds") or ["body"] * len(E)
    chunks = {
        "model": model_id or meta.get("model"), "tag": tag, "revision": revision,
        "dims": int(E.shape[1]), "n": int(E.shape[0]), "dtype": dtype,
        "doc": [doc_of[i] for i in meta["ids"]],
        "chunk": meta["chunk_index"],
        "lang": [0 if l == "la" else 1 for l in meta["langs"]],
        "kind": [0 if k == "body" else 1 for k in kinds],
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "data_revision": site_manifest["revision"],
        "status": "work in progress: data and embeddings change without notice until v1.0",
        **(extra or {}),
    }
    (out / "chunks.json").write_text(json.dumps(chunks, separators=(",", ":")), encoding="utf-8")
    for p in sorted(out.iterdir()):
        print(f"  {p.stat().st_size / 1e6:7.2f} MB  {p.relative_to(ROOT)}")
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", default="bge-m3")
    ap.add_argument("--safetensors", type=Path, default=ROOT / "output" / "embeddings.safetensors")
    ap.add_argument("--meta", type=Path, default=ROOT / "output" / "embeddings_meta.json")
    ap.add_argument("--index", type=Path, default=ROOT / "site-data" / "index.json")
    ap.add_argument("--out", type=Path, default=None, help="default: hf-export/<tag>")
    ap.add_argument("--revision", default="v0.1")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--min-overlap", type=float, default=0.98)
    args = ap.parse_args()
    E = load_file(args.safetensors)["embeddings"]
    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    write_export(E, meta, args.index, args.tag, args.out or ROOT / "hf-export" / args.tag,
                 revision=args.revision, int8=args.int8, min_overlap=args.min_overlap)


if __name__ == "__main__":
    main()
