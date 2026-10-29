#!/usr/bin/env bash
# Train the two kraken models of the Consilia edition on the final Baldus data, locally (RTX 3060, 12 GB).
#
#   python tools/make_pagexml_dataset.py           # first: dataset + lists in output/kraken_final/
#   ./tools/kraken_train_final.sh [compile|seg|rec|test|convert|all]      default: all
#
#   compile  binary line datasets from the recognition PageXML (CPU, ~15 min, ~16 GB in output/kraken_final/)
#   seg      fine-tunes kraken's blla.mlmodel on the layout (line and region classes of the edition)
#   rec      fine-tunes McCATMuS on the final text (polygon lines, a baselines-type model; the v1 model is bbox-type)
#   test     ketos test / segtest on the held-out test pages (never seen in training or validation)
#   convert  best checkpoints -> output/kraken_final/{seg,rec}/*.mlmodel (CoreML container for kraken < 7 / eScriptorium)
#   all      compile and seg in parallel (CPU / GPU), then rec, test, convert
# Logs: output/kraken_final/logs/. Checkpoints: output/kraken_final/{seg,rec}/.
set -eu
cd "$(dirname "$0")/.."
K=output/kraken_final
V=.venv-kraken/bin
DEV="${KRAKEN_DEVICE:-cuda:0}"
BLLA=$($V/python -c "import kraken,os;print(os.path.join(os.path.dirname(kraken.__file__),'blla.mlmodel'))")
REC_BASE="${REC_BASE:-output/trained_kraken/base_McCATMuS_nfd_nofix_V1.mlmodel}"
mkdir -p $K/logs $K/seg $K/rec
log() { echo "== $* $(date +%F' '%T)" | tee -a $K/logs/run.log; }

compile() {
  for s in train val test; do
    [ -s $K/rec_$s.arrow ] && { log "compile $s exists"; continue; }
    log "compile $s"
    $V/ketos --workers 12 compile -f page -F $K/rec_$s.txt -o $K/rec_$s.arrow 2>&1 | tee -a $K/logs/compile.log | tail -2
    echo "$PWD/$K/rec_$s.arrow" > $K/rec_$s.lst          # ketos train/test want a manifest, not the arrow file itself
  done
}

seg() {
  log "segtrain"
  # blla fine-tuned to the edition's classes (--resize new: output layer = exactly the training data's classes),
  # cosine schedule, early stop after 10 evaluations without improvement, augmentation on
  $V/ketos -d "$DEV" --workers 8 --precision 16-mixed segtrain -f page \
    -i "$BLLA" --resize new -t $K/seg_train.txt -e $K/seg_val.txt \
    -N 60 --min-epochs 15 -q early --lag 10 -r 0.0002 --schedule cosine --cos-max 60 --augment \
    -o $K/seg/consilia_seg 2>&1 | tee -a $K/logs/segtrain.log | grep -E "^stage|val_|Error|error" || true
}

rec() {
  log "train (recognition)"
  # from McCATMuS (baselines model, NFD), codec extended by the final text's characters,
  # batch 32, lr 5e-4 cosine over 16 epochs, early stop after 4 evaluations without improvement
  $V/ketos -d "$DEV" --workers 8 --precision 16-mixed train -f binary \
    -t $K/rec_train.lst -e $K/rec_val.lst -i "$REC_BASE" --resize union -u NFD \
    -B 32 -r 0.0005 --schedule cosine --cos-max 16 -N 16 --min-epochs 8 -q early --lag 4 --min-delta 0.0005 \
    --augment --pad 16 -F 1 -o $K/rec/consilia_rec 2>&1 | tee -a $K/logs/train.log | grep -E "^stage|val_|Error|error" || true
}

best() { ls -t $K/$1/$2/best_0*.safetensors 2>/dev/null | head -1; }   # ketos writes best_<score>.safetensors into the -o directory

# kraken 7.1.1 + --resize new stores a wrong inference class mapping (lines mislabelled, segtest KeyError);
# tools/kraken_fix_segmap.py writes the trained mapping into a copy used for test, convert and publication
segw() {
  B=$(best seg consilia_seg); [ -n "$B" ] || return 0
  [ $K/seg/consilia_seg.safetensors -nt "$B" ] || $V/python tools/kraken_fix_segmap.py "$B" $K/seg/consilia_seg.safetensors >&2
  echo $K/seg/consilia_seg.safetensors
}

test_models() {
  R=$(best rec consilia_rec); S=$(segw)
  log "test rec $R"
  [ -n "$R" ] && $V/ketos -d "$DEV" --workers 8 test -f binary -m "$R" -e $K/rec_test.lst -B 32 --pad 16 -u NFD 2>&1 | tee $K/logs/test_rec.log | tail -25
  log "segtest $S"
  [ -n "$S" ] && $V/ketos -d "$DEV" --workers 8 segtest -f page -m "$S" -e $K/seg_test.txt 2>&1 | tee $K/logs/segtest.log | tail -40
}

convert() {
  for m in rec seg; do
    if [ $m = seg ]; then B=$(segw); else B=$(best $m consilia_$m); fi; [ -n "$B" ] || continue
    log "convert $B"
    rm -f $K/$m/consilia_$m.mlmodel   # ketos convert refuses to overwrite
    $V/ketos convert "$B" -o $K/$m/consilia_$m.mlmodel --weights-format coreml 2>&1 | tail -2 || true
  done
}

case "${1:-all}" in
  compile) compile ;;
  seg) seg ;;
  rec) rec ;;
  test) test_models ;;
  convert) convert ;;
  all)
    compile > $K/logs/compile.out 2>&1 &
    CP=$!
    seg
    wait $CP
    rec
    test_models
    convert
    log "all done" ;;
esac
