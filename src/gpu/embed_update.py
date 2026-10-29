"""
Re-embed only the chunks whose text changed since the last full run and write them into the existing web
exports (hf-export/<tag>/chunks.f16.bin), then stamp the current site-data revision. For small text
corrections; the chunk list (ids, order, count) must be unchanged, otherwise run embed_web_models.py.

    python src/gpu/embed_update.py --before output/merge/recheck_2026-10-29/consilia.before.json --tags bge-m3 etp-bgem3-ft qwen3-0.6b qwen3-4b
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "gpu"))
from embed_consilia import build_chunks  # noqa: E402
from embed_web_models import MODELS, SentenceTransformer, torch  # noqa: E402


def chunks(path: Path):
    cons = sorted(json.loads(path.read_text(encoding="utf-8"))["consilia"].values(), key=lambda c: c["id"])
    return build_chunks(cons)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--before", type=Path, required=True, help="output/consilia.json as it was at the last full embedding run")
    ap.add_argument("--input", type=Path, default=ROOT / "output" / "consilia.json")
    ap.add_argument("--tags", nargs="+", choices=list(MODELS), default=list(MODELS))
    ap.add_argument("--out-root", type=Path, default=ROOT / "hf-export")
    ap.add_argument("--max-seq-length", type=int, default=512)
    a = ap.parse_args()
    old, new = chunks(a.before), chunks(a.input)
    if old[0] != new[0] or old[1] != new[1] or old[3] != new[3]:
        sys.exit("the chunk list changed (ids, languages or order): run src/gpu/embed_web_models.py")
    idx = [i for i, (x, y) in enumerate(zip(old[2], new[2])) if x != y]
    print(f"{len(new[2])} chunks, {len(idx)} with changed text")
    rev = json.loads((ROOT / "site-data" / "manifest.json").read_text(encoding="utf-8"))["revision"]
    for tag in a.tags:
        d = a.out_root / tag
        meta = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        if meta.get("dtype") != ["f16"]:
            sys.exit(f"{tag}: only f16 exports are handled ({meta.get('dtype')})")
        E = np.fromfile(d / "chunks.f16.bin", dtype=np.float16).reshape(meta["n"], meta["dims"])
        if meta["n"] != len(new[2]):
            sys.exit(f"{tag}: export has {meta['n']} chunks, the corpus {len(new[2])}")
        if idx:
            spec = MODELS[tag]
            model = SentenceTransformer(spec["id"], device="cuda", model_kwargs={"torch_dtype": spec["dtype"]},
                                        tokenizer_kwargs={"padding_side": "left"} if spec["left_pad"] else {})
            model.max_seq_length = a.max_seq_length
            V = model.encode([new[2][i] for i in idx], batch_size=8, normalize_embeddings=True, convert_to_numpy=True)
            E[idx] = V.astype(np.float16)
            E.tofile(d / "chunks.f16.bin")
            del model; torch.cuda.empty_cache()
        meta.update(data_revision=rev, updated=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    updated_chunks=len(idx))
        (d / "chunks.json").write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"{tag}: {len(idx)} vectors replaced, revision {rev}")


if __name__ == "__main__":
    main()
