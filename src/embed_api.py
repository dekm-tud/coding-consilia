"""
Embed the corpus passages with an OpenAI-compatible embeddings API and write the browser format
. Same chunks as the other models (embed_consilia.build_chunks: body and casus
sections > 50 chars, Latin then English), output hf-export/<tag>/{chunks.f16.bin, chunks.json}.

Example (Blablador, Qwen3-Embedding-8B behind `alias-embeddings`):
    export BLABLADOR_API_KEY=...
    python src/embed_api.py --tag qwen3-8b-api --model alias-embeddings \\
        --base-url https://api.blablador.fz-juelich.de/v1 --key-env BLABLADOR_API_KEY --batch 16

Resumable: partial results are kept in output/embed_api_<tag>.npy every --save-every batches.
Afterwards add the model to site-data/models.json with a `query_api` block, e.g.
    {"tag": "qwen3-8b-api", "label": "Qwen3-Embedding-8B (API)", "model_id": "Qwen/Qwen3-Embedding-8B",
     "dims": 4096, "pooling": null, "normalize": true,
     "query_prefix": "Instruct: Given a query about medieval legal opinions, retrieve relevant passages\\nQuery:",
     "query_api": {"base_url": "https://api.blablador.fz-juelich.de/v1", "model": "alias-embeddings", "label": "Blablador"},
     "embeddings": ["hf-export/qwen3-8b-api/"], "notes": "..."}
The search page then embeds queries through that API with the key the user enters (browser only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from embed_consilia import build_chunks  # noqa: E402
from export_embeddings_web import write_export  # noqa: E402


def post(url: str, payload: dict, key: str, timeout: int = 180) -> dict:
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")[:300]
            if e.code in (429, 500, 502, 503, 504) and attempt < 7:
                wait = min(120, 5 * 2 ** attempt)
                print(f"  HTTP {e.code}, waiting {wait}s ({body[:80]})")
                time.sleep(wait)
                continue
            raise SystemExit(f"HTTP {e.code}: {body}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < 7:
                time.sleep(min(120, 5 * 2 ** attempt))
                continue
            raise SystemExit(str(e))
    raise SystemExit("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True, help="model name at the API, e.g. alias-embeddings")
    ap.add_argument("--model-id", help="canonical model id for chunks.json (default: --model)")
    ap.add_argument("--base-url", default="https://api.blablador.fz-juelich.de/v1")
    ap.add_argument("--key-env", default="BLABLADOR_API_KEY")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--max-chars", type=int, default=12000, help="truncate very long sections (API token limits)")
    ap.add_argument("--input", type=Path, default=ROOT / "output" / "consilia.json")
    ap.add_argument("--index", type=Path, default=ROOT / "site-data" / "index.json")
    ap.add_argument("--out-root", type=Path, default=ROOT / "hf-export")
    ap.add_argument("--revision", default="v0.1")
    args = ap.parse_args()
    key = os.environ.get(args.key_env, "")
    if not key:
        sys.exit(f"set {args.key_env}")

    consilia = list(json.loads(args.input.read_text(encoding="utf-8"))["consilia"].values())
    ids, langs, texts, chunk_index, kinds = build_chunks(consilia)
    n = len(texts)
    part = ROOT / "output" / f"embed_api_{args.tag}.npy"
    E = None
    done = 0
    if part.exists():
        E = np.load(part)
        done = int((np.abs(E).sum(axis=1) > 0).sum())
        if E.shape[0] != n:
            print("chunk count changed since the partial file was written; starting over")
            E, done = None, 0
    print(f"{n} chunks ({sum(1 for l in langs if l == 'la')} la / {sum(1 for l in langs if l == 'en')} en), {done} already embedded")
    url = args.base_url.rstrip("/") + "/embeddings"
    t0 = time.time()
    b = 0
    for start in range(done, n, args.batch):
        batch = [t[:args.max_chars] for t in texts[start:start + args.batch]]
        r = post(url, {"model": args.model, "input": batch}, key)
        vecs = np.array([d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])], dtype=np.float32)
        if E is None:
            E = np.zeros((n, vecs.shape[1]), dtype=np.float32)
        E[start:start + len(batch)] = vecs
        b += 1
        if b % args.save_every == 0 or start + args.batch >= n:
            np.save(part, E)
            el = time.time() - t0
            did = start + len(batch) - done
            print(f"  {start + len(batch)}/{n}  {did / el:.1f} chunks/s  eta {((n - start - len(batch)) / max(did / el, 1e-6)) / 60:.0f} min")
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    meta = {"ids": ids, "langs": langs, "chunk_index": chunk_index, "kinds": kinds, "model": args.model_id or args.model, "dims": int(E.shape[1]), "n": n}
    rep = write_export(E, meta, args.index, args.tag, args.out_root / args.tag, model_id=args.model_id or args.model, revision=args.revision,
                       extra={"source": "api", "api_base_url": args.base_url, "api_model": args.model})
    print(json.dumps(rep, indent=1)[:600])
    print(f"wrote {args.out_root / args.tag}; now add the model to site-data/models.json (see the docstring)")


if __name__ == "__main__":
    main()
