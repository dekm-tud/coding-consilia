#!/usr/bin/env python3
"""Export an encoder to the ONNX layout transformers.js expects.

    python tools/export_onnx.py mschonhardt/etp-bgem3-contrastive-ft --out hf-models/etp-bgem3-contrastive-ft

Result (upload the whole folder into the model repo, next to model.safetensors):

    <out>/config.json, tokenizer.json, tokenizer_config.json, special_tokens_map.json
    <out>/onnx/model.onnx (+ model.onnx_data)      float32
    <out>/onnx/model_quantized.onnx                dynamic int8 (transformers.js dtype "q8", the WASM default)
    <out>/onnx/model_fp16.onnx                     float16 (dtype "fp16", for WebGPU)
    <out>/onnx/model_q4.onnx / model_q4f16.onnx    optional 4-bit (--q4 / --q4f16), for large models

Steps: optimum export (task feature-extraction, plain transformers, output last_hidden_state; pooling
and normalisation are done by transformers.js), onnxruntime dynamic quantisation (per-channel, last layers
kept in fp32), native fp16 export.
Whether the quantised encoder is good enough is decided by tools/parity_ref.py, not here.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-fp16", action="store_true")
    ap.add_argument("--q4", action="store_true",
                    help="also write model_q4.onnx (4-bit weights, fp32 activations; runs on WASM and WebGPU)")
    ap.add_argument("--q4f16", action="store_true",
                    help="also write model_q4f16.onnx (4-bit weights, fp16 activations; WebGPU only). Needs the "
                         "native fp16 export, i.e. a CUDA GPU. Use this for Qwen3-Embedding-4B.")
    ap.add_argument("--skip-int8", action="store_true", help="skip model_quantized.onnx (int8)")
    ap.add_argument("--keep-fp32-layers", type=int, default=6,
                    help="keep the MatMuls of the last N transformer layers in fp32 (default 6; measured on "
                         "etp-bgem3-ft: plain dynamic int8 fails parity, this passes with margin)")
    a = ap.parse_args()

    raw = a.out.with_name(a.out.name + "-raw")
    if not (raw / "model.onnx").exists():
        subprocess.run([str(Path(sys.executable).parent / "optimum-cli"), "export", "onnx", "--model", a.model,
                        "--task", "feature-extraction", "--library-name", "transformers", "--device", "cpu", str(raw)],
                       check=True)
    onnx_dir = a.out / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (raw / f).exists():
            shutil.copyfile(raw / f, a.out / f)
    for f in ("model.onnx", "model.onnx_data"):
        if (raw / f).exists() and not (onnx_dir / f).exists():
            shutil.copyfile(raw / f, onnx_dir / f)

    from onnxruntime.quantization import QuantType, quantize_dynamic
    q = onnx_dir / "model_quantized.onnx"
    if not q.exists() and not a.skip_int8:
        import re
        import onnx
        g = onnx.load(str(onnx_dir / "model.onnx"), load_external_data=False)
        pat = r"/layers?\.(\d+)/"      # encoders: /encoder/layer.N/  decoders: /model/layers.N/
        n_layers = 1 + max(int(x) for n in g.graph.node for x in re.findall(pat, n.name))
        keep = set(range(n_layers - a.keep_fp32_layers, n_layers))
        exclude = [n.name for n in g.graph.node
                   if any(int(x) in keep for x in re.findall(pat, n.name))]
        print(f"quantising: {n_layers} layers, last {len(keep)} kept in fp32 ({len(exclude)} nodes excluded)")
        # per-channel weights: plain per-tensor dynamic int8 misses the parity gate for this model
        quantize_dynamic(str(onnx_dir / "model.onnx"), str(q), weight_type=QuantType.QUInt8,
                         per_channel=True, nodes_to_exclude=exclude)   # < 2 GB: one file, as transformers.js expects
    if not a.skip_fp16 and not (onnx_dir / "model_fp16.onnx").exists():
        # Native fp16 export (needs a CUDA GPU). Converting the fp32 graph afterwards produced a graph
        # with mixed float/float16 operands that onnxruntime refuses to load.
        raw16 = a.out.with_name(a.out.name + "-raw-fp16")
        if not (raw16 / "model.onnx").exists():
            subprocess.run([str(Path(sys.executable).parent / "optimum-cli"), "export", "onnx", "--model", a.model,
                            "--task", "feature-extraction", "--library-name", "transformers", "--dtype", "fp16",
                            "--device", "cuda", str(raw16)], check=True)
        shutil.copyfile(raw16 / "model.onnx", onnx_dir / "model_fp16.onnx")   # ~1.1 GB: single file
    def matmul_nbits(src: Path, dst: Path) -> None:
        # block-wise 4-bit weights (MatMulNBits), the scheme of transformers.js' q4 / q4f16 dtypes
        import onnx
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
        m = onnx.load(str(src))
        qz = MatMulNBitsQuantizer(m, block_size=32, is_symmetric=True, bits=4)
        qz.process()
        qz.model.save_model_to_file(str(dst), use_external_data_format=False)

    if a.q4 and not (onnx_dir / "model_q4.onnx").exists():
        matmul_nbits(onnx_dir / "model.onnx", onnx_dir / "model_q4.onnx")
    if a.q4f16 and not (onnx_dir / "model_q4f16.onnx").exists():
        matmul_nbits(onnx_dir / "model_fp16.onnx", onnx_dir / "model_q4f16.onnx")
    for p in sorted(a.out.rglob("*")):
        if p.is_file():
            print(f"{p.stat().st_size / 1e6:9.1f} MB  {p.relative_to(a.out)}")


if __name__ == "__main__":
    main()
