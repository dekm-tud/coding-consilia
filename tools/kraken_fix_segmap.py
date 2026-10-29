#!/usr/bin/env python3
"""
Repair the class mapping of a kraken 7 segmentation model trained with `ketos segtrain --resize new`.

kraken 7.1.1 stores two class mappings in the weights: `_full_class_mapping` (the indices the network was
actually trained with) and `class_mapping` (a "canonical" one-to-one mapping used at inference and by
`ketos segtest`). With --resize new the second one is renumbered and no longer matches the network, so
`kraken segment` labels every line with the wrong class (e.g. cons_body as cons_argument, a header
baseline as a "text" region) and `ketos segtest` fails with a KeyError. This script copies the weights
with `class_mapping` set to `_full_class_mapping`.

    python tools/kraken_fix_segmap.py best_0.2467.safetensors consilia_seg.safetensors
    ketos convert consilia_seg.safetensors -o consilia_seg.mlmodel --weights-format coreml
"""
import json
import sys

from safetensors import safe_open
from safetensors.torch import save_file


def main(src: str, dst: str) -> None:
    f = safe_open(src, "pt")
    md = dict(f.metadata())
    tensors = {k: f.get_tensor(k) for k in f.keys()}
    meta = json.loads(md["kraken_meta"])
    key = next(iter(meta))
    full = meta[key].get("_full_class_mapping")
    if not full:
        sys.exit(f"{src}: no _full_class_mapping in the metadata, nothing to fix")
    idx = [i for grp in full.values() for i in grp.values()]
    if len(idx) != len(set(idx)):
        sys.exit(f"{src}: _full_class_mapping is many-to-one, cannot be used as is: {full}")
    before = meta[key].get("class_mapping")
    meta[key]["class_mapping"] = full
    md["kraken_meta"] = json.dumps(meta)
    save_file(tensors, dst, metadata=md)
    print(f"{dst}: class_mapping {'unchanged' if before == full else 'replaced'}: {json.dumps(full)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
