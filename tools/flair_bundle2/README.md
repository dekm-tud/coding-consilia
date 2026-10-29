# Flair tagger v2 bundle (Consilia, 2026-10-28)

Retrains the citation and entity tagger (MediBerta fine-tuned, stacked with the Latin legal Flair language
models, CRF) on the Gemini 4 Argon annotation of the whole corpus and tags all units with it. The first
tagger (21.10.) was trained on the Qwen run; its citations alone were wrong in 80 % of the cases where Argon
did not agree (sample check of 150 consilia, 28.10.).

    tar -xf flair-train2.tar && cd flair-train2 && ./train_flair2.sh      # setup train predict pack
    # or with the GPU queue: run_flair2.sh (see the script header)

Contents: `output/flair/{train,dev,test}.txt` (BIO corpus, split by consilium, built by
`src/annotate/export_flair.py`), `output/units.jsonl` (10,322 units, text as of 29.10.),
`output/annotations/spans_gemini-4-argon_merged.jsonl` (the silver annotation), `tools/train_flair_ner.py`,
`src/annotate/export_flair.py`, `requirements.txt`. Same recipe as v1: 20 epochs, batch 16, lr 5e-5.

Speed (29.10.): the mini-batch goes through the GPU in one pass (`CHUNK=0`; v1 cut it into chunks of 4 for a 12 GB
card), with mixed precision (`AMP=1`) and TF32; prediction tags 200 units per call in batches of 128 and halves the
batch on an out-of-memory error. Checked on 33 units with the v1 model: mixed precision gives the same 388 spans as
full precision in half the time. `CHUNK=4 AMP=0 ./train_flair2.sh` restores the v1 behaviour.

Results: `flair-results2.tar.gz` (predictions `spans_flair_v2.jsonl`, training log, test scores per class) and
`flair-model2.tar.gz` (best-model.pt). Back in the repo: unpack the results in the repo root, then

    python src/annotate/export.py --spans output/annotations/spans_qwen-qwen3-8-27b-fp8_p1.jsonl output/annotations/spans_flair.jsonl \
        --patch output/annotations/spans_gemini-4-argon_merged.jsonl output/annotations/spans_flair_v2.jsonl \
        --patch-units output/units_all_20261029.jsonl
    python src/build_site_data.py && python src/qa/check_data.py
