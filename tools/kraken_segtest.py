#!/usr/bin/env python3
"""
`ketos segtest` with a workaround for a kraken 7.1.1 bug: the per-class pixel table indexes
`class_iu` / `class_pixel_accuracy` (which are stored by position in `test_pixel_idxs`) with the raw
class index and dies with IndexError before printing the baseline metrics. This wrapper expands
the two tensors to full class-index length after the test epoch and then runs the normal CLI.

    .venv-kraken/bin/python tools/kraken_segtest.py -d cuda:0 segtest -f page -m model.safetensors -e test.txt
"""
import sys

import torch
from kraken.train import blla

_orig = blla.BLLASegmentationModel.on_test_epoch_end


def _patched(self):
    _orig(self)
    tm = self.test_metrics
    idxs = torch.tensor(self.test_pixel_idxs)
    n = int(idxs.max()) + 1
    vals = {}
    for name in ("class_iu", "class_pixel_accuracy"):
        v = getattr(tm, name)
        if v is None or len(v) != len(idxs):
            continue
        full = torch.zeros(n, dtype=v.dtype)
        full[idxs] = v.detach().cpu()
        vals[name] = full
    try:
        for k, v in vals.items():
            setattr(tm, k, v)
    except AttributeError:            # immutable container (NamedTuple / frozen dataclass)
        if hasattr(tm, "_replace"):
            self.test_metrics = tm._replace(**vals)
        else:
            import dataclasses
            self.test_metrics = dataclasses.replace(tm, **vals)


blla.BLLASegmentationModel.on_test_epoch_end = _patched

if __name__ == "__main__":
    from kraken.ketos import cli
    sys.argv = ["ketos"] + sys.argv[1:]
    cli()
