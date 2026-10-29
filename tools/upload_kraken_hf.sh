#!/usr/bin/env bash
# Prints the commands that publish the PageXML dataset and the two kraken models to Hugging Face.
# It uploads NOTHING itself: uploading is publishing, so you run the lines you want.
#
#   bash tools/upload_kraken_hf.sh
#
# Before: python tools/make_pagexml_dataset.py (dataset folder), ./tools/kraken_train_final.sh (models),
# then copy the model files into the card folders (see below) and fill the [[...]] fields of the cards.
set -eu
cd "$(dirname "$0")/.."
DS=dist/hf-consilia-baldus-venice1575-pagexml
SEG=dist/hf-consilia-baldus-venice1575-kraken-seg
REC=dist/hf-consilia-baldus-venice1575-kraken-rec
K=output/kraken_final
echo "# model files into the card folders (after training):"
echo "# (done 2026-10-27 from the server run dist/kraken_run_gpu: models, training.log, test.log, train_kraken2.sh are in the card folders)"
echo
echo "# open placeholders in the cards:"
grep -l '\[\[' $DS/README.md $SEG/README.md $REC/README.md 2>/dev/null || echo "none"
echo
echo "# publish (hf auth login once; the dataset is ~5 GB, the upload takes a while):"
echo "hf repo create mschonhardt/consilia-baldus-venice1575-pagexml --type dataset"
echo "hf upload mschonhardt/consilia-baldus-venice1575-pagexml $DS . --repo-type dataset --commit-message 'v1: 1,392 pages, 194,654 lines, images linked (Internet Archive)'"
echo "hf repo create mschonhardt/consilia-baldus-venice1575-kraken-seg --type model"
echo "hf upload mschonhardt/consilia-baldus-venice1575-kraken-seg $SEG . --commit-message 'v1'"
echo "hf repo create mschonhardt/consilia-baldus-venice1575-kraken-rec --type model"
echo "hf upload mschonhardt/consilia-baldus-venice1575-kraken-rec $REC . --commit-message 'v2'"

# --- Karlsruhe, BLB, Cod. St. Georgen 48 (layout only, no images) ---
DS2=dist/hf-blb-st-georgen-48-pagexml
SEG2=dist/hf-blb-st-georgen-48-kraken-seg
K2=output/kraken_stgeorgen48
echo
echo "# St. Georgen 48: model files into the card folder (after ./tools/kraken_train_stgeorgen48.sh):"
echo "# model files already copied (dist/kraken_run_gpu)"
echo "# open placeholders:"; grep -l '\[\[' $DS2/README.md $SEG2/README.md 2>/dev/null || echo "none"
echo "hf repo create mschonhardt/blb-st-georgen-48-pagexml --type dataset"
echo "hf upload mschonhardt/blb-st-georgen-48-pagexml $DS2 . --repo-type dataset --commit-message 'v1: 505 pages, 34,364 lines, no images'"
echo "hf repo create mschonhardt/blb-st-georgen-48-kraken-seg --type model"
echo "hf upload mschonhardt/blb-st-georgen-48-kraken-seg $SEG2 . --commit-message 'v1'"
