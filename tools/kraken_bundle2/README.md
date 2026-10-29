# Kraken training bundle 2 (Consilia, 2026-10-26)

Three kraken 7.1.1 models for one GPU session:

| job | data | base | result |
|---|---|---|---|
| `rec` | `baldus/rec/<volume>/*.xml` (final edition text, 1,207/98/71 pages; lists `baldus/rec_*.txt`, compiled to `.arrow` on the server) | `base_McCATMuS_nfd_nofix_V1.mlmodel` | `model/rec/consilia_rec.mlmodel` |
| `seg_baldus` | `baldus/pages/<volume>/*.xml` + jpg, lists `baldus/seg_*.txt` (1,221/100/71 pages) | kraken `blla.mlmodel` | `model/seg_baldus/consilia_seg.mlmodel` |
| `seg_stg` | `stgeorgen48/pages/*.xml` + jpg, lists `stgeorgen48/seg_*.txt` (331/26/22 pages) | kraken `blla.mlmodel` | `model/seg_stg/stgeorgen48_seg.mlmodel` |

    tar -xf kraken-train2.tar && cd kraken-train2 && ./train_kraken2.sh     # setup all pack
    ./train_kraken2.sh all pack                                             # after an interruption: resumes

`train_kraken2.sh` creates `.venv-kraken` (pip, kraken==7.1.1), compiles the recognition lines and runs the three trainings in parallel, then
`ketos test` / `ketos segtest` on the held-out pages, `ketos convert` to CoreML, and packs
`kraken-models2.tar.gz` (best checkpoints, repaired safetensors, mlmodels, logs). `kraken_fix_segmap.py`
repairs the class mapping that kraken 7.1.1 writes wrongly with `--resize new`.

Copy `kraken-models2.tar.gz` back to the repository (`output/kraken_server2/`); the model cards in
`dist/hf-*-kraken-*/` are filled from `logs/`.
