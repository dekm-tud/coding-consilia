#!/usr/bin/env bash
# Segment and recognise the Panormitanus volume with the Baldus kraken models (published on Hugging Face as
# mschonhardt/consilia-baldus-venice1575-kraken-seg / -rec), for the comparison with the Gemini transcription
# (tools/compare_panormitanus.py). Output: output/kraken_panormitanus/pages/<page>.xml (PAGE 2019, next to links to
# the images). Pages already done are skipped; run again after an interruption.
#   bash tools/kraken_panormitanus.sh [DEVICE]          # default cuda:0
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD
D=$ROOT/data/Panormitanus_32108367/Panormitanus_Cons_Print_Venice_1571_v1
O=$ROOT/output/kraken_panormitanus/pages
SEG=$ROOT/dist/hf-consilia-baldus-venice1575-kraken-seg/consilia_seg.mlmodel
REC=$ROOT/dist/hf-consilia-baldus-venice1575-kraken-rec/consilia_rec.mlmodel
mkdir -p "$O"; cd "$O"
todo=()
for img in "$D"/*.jpg; do
  p=$(basename "$img" .jpg)
  [ -s "$p.xml" ] && continue
  ln -sf "$img" "$p.jpg"; todo+=("-i" "$p.jpg" "$p.xml")
done
echo "$(( ${#todo[@]} / 3 )) pages to do"
[ ${#todo[@]} -eq 0 ] && exit 0
export COLUMNS=200
"$ROOT/.venv-kraken/bin/kraken" -d "${1:-cuda:0}" "${todo[@]}" -f image -x segment -bl -i "$SEG" ocr -m "$REC"
