#!/usr/bin/env python3
"""Query-encoder parity.

A quantised query encoder against an fp32 corpus is the most likely SILENT failure mode of in-browser
search, so every shipped model must pass parity:  mean cosine >= 0.98 and mean top-20 overlap >= 0.9
between the reference query vectors (fp32, the pipeline that embedded the corpus) and the vectors of the
shipped ONNX model, measured against the corpus vectors (hf-export/<tag>/chunks.f16.bin).

  1. write reference vectors:      python tools/parity_ref.py ref  --tag etp-bgem3-ft
       -> tools/parity_ref/<tag>.json   (50 fixed queries in tools/parity_queries.json; Qwen3 queries get
          the instruction from site-data/models.json, exactly as the browser does)
  2. check an ONNX model in Python: python tools/parity_ref.py onnx --tag etp-bgem3-ft \\
          --onnx hf-models/etp-bgem3-contrastive-ft/onnx/model_quantized.onnx
  3. check the real runtime (transformers.js in the browser): open tools/parity_check.html
     (python -m http.server, then /tools/parity_check.html?model=<tag>) after step 1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
MIN_COS, MIN_OVERLAP = 0.98, 0.9


def cfg_of(tag: str) -> dict:
    models = json.loads((ROOT / "site-data" / "models.json").read_text(encoding="utf-8"))["models"]
    return next(m for m in models if m["tag"] == tag)


def queries(cfg: dict) -> list[str]:
    qs = json.loads((ROOT / "tools" / "parity_queries.json").read_text(encoding="utf-8"))
    return [cfg.get("query_prefix", "") + q["text"] for q in qs]


def corpus(tag: str) -> np.ndarray:
    meta = json.loads((ROOT / "hf-export" / tag / "chunks.json").read_text(encoding="utf-8"))
    E = np.fromfile(ROOT / "hf-export" / tag / "chunks.f16.bin", dtype="<f2").astype(np.float32)
    return E.reshape(meta["n"], meta["dims"])


def compare(ref: np.ndarray, got: np.ndarray, E: np.ndarray, k: int = 20) -> dict:
    cos = np.sum(ref * got, axis=1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1))
    a = np.argsort(-(ref @ E.T), axis=1)[:, :k]
    b = np.argsort(-(got @ E.T), axis=1)[:, :k]
    ov = np.array([len(set(x) & set(y)) / k for x, y in zip(a, b)])
    return {"mean_cos": float(cos.mean()), "min_cos": float(cos.min()), "mean_top20_overlap": float(ov.mean()),
            "min_top20_overlap": float(ov.min()),
            "pass": bool(cos.mean() >= MIN_COS and ov.mean() >= MIN_OVERLAP)}


def cmd_ref(a) -> None:
    from sentence_transformers import SentenceTransformer
    import torch
    cfg = cfg_of(a.tag)
    kw = {"torch_dtype": torch.bfloat16} if "qwen" in a.tag else {}
    tk = {"padding_side": "left"} if "qwen" in a.tag else {}
    m = SentenceTransformer(cfg["model_id"], device=a.device, model_kwargs=kw, tokenizer_kwargs=tk)
    v = m.encode(queries(cfg), normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    out = ROOT / "tools" / "parity_ref" / f"{a.tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"tag": a.tag, "model": cfg["model_id"], "dims": int(v.shape[1]),
                               "vectors": np.round(v, 6).tolist()}), encoding="utf-8")
    print("wrote", out, v.shape)


def cmd_onnx(a) -> None:
    import onnxruntime as ort
    from transformers import AutoTokenizer
    cfg = cfg_of(a.tag)
    ref = np.array(json.loads((ROOT / "tools" / "parity_ref" / f"{a.tag}.json").read_text())["vectors"], dtype=np.float32)
    onnx_path = Path(a.onnx)
    tok = AutoTokenizer.from_pretrained(a.tokenizer or onnx_path.parent.parent)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    out = []
    for q in queries(cfg):
        enc = tok(q, return_tensors="np", truncation=True, max_length=512)
        feed = {k: v.astype(np.int64) for k, v in enc.items() if k in names}
        h = sess.run(None, feed)[0][0]                    # (seq, dim) last_hidden_state
        v = h[0] if cfg["pooling"] == "cls" else h[-1]     # cls | last_token (no padding for a single query)
        out.append(v / np.linalg.norm(v))
    res = compare(ref, np.array(out, dtype=np.float32), corpus(a.tag))
    res.update({"tag": a.tag, "onnx": str(onnx_path)})
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["pass"] else 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("ref"); r.add_argument("--tag", required=True); r.add_argument("--device", default="cuda")
    o = sub.add_parser("onnx"); o.add_argument("--tag", required=True); o.add_argument("--onnx", required=True)
    o.add_argument("--tokenizer")
    a = ap.parse_args()
    {"ref": cmd_ref, "onnx": cmd_onnx}[a.cmd](a)


if __name__ == "__main__":
    main()
