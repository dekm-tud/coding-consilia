#!/usr/bin/env bash
# Train, test and convert the kraken layout model for Karlsruhe, BLB, Cod. St. Georgen 48.
#
#   python tools/make_stgeorgen48_dataset.py        # once: dataset + output/kraken_stgeorgen48/{pages,seg_*.txt}
#   ./tools/kraken_train_stgeorgen48.sh all          # seg -> segtest -> convert (logs in output/kraken_stgeorgen48/logs/)
#   ./tools/kraken_train_stgeorgen48.sh seg|test|convert
#
# Same recipe as tools/kraken_train_final.sh (Baldus): blla fine-tune with a new output layer for the
# region classes column_1, column_2, column_single, marginalia, page-number, quire-number, paragraph and
# the line classes body, marginalia, page-number, quire-number. 331 training / 26 validation pages.
set -euo pipefail
cd "$(dirname "$0")/.."
K=output/kraken_stgeorgen48
V=.venv-kraken/bin
BLLA=$($V/python -c "import kraken, os; print(os.path.join(os.path.dirname(kraken.__file__), 'blla.mlmodel'))")
mkdir -p $K/logs $K/seg
log() { echo "== $* $(date '+%F %T')" | tee -a $K/logs/run.log; }
best() { ls -t $K/seg/stgeorgen48_seg/best_0*.safetensors 2>/dev/null | head -1; }
# kraken 7.1.1 + --resize new stores a wrong inference class mapping; use a repaired copy (tools/kraken_fix_segmap.py)
segw() {
  B=$(best); [ -n "$B" ] || return 0
  [ $K/seg/stgeorgen48_seg.safetensors -nt "$B" ] || $V/python tools/kraken_fix_segmap.py "$B" $K/seg/stgeorgen48_seg.safetensors >&2
  echo $K/seg/stgeorgen48_seg.safetensors
}

seg() {
  log segtrain
  $V/ketos -d cuda:0 --workers 8 --precision 16-mixed segtrain -f page -i "$BLLA" --resize new \
    -t $K/seg_train.txt -e $K/seg_val.txt \
    -N 80 --min-epochs 20 -q early --lag 12 -r 0.0002 --schedule cosine --cos-max 80 --augment \
    -o $K/seg/stgeorgen48_seg 2>&1 | tee $K/logs/segtrain.log
  log "segtrain done, best: $(best)"
}

test_model() {
  log segtest
  $V/ketos segtest -f page -m "$(segw)" -e $K/seg_test.txt 2>&1 | tee $K/logs/segtest.log
}

convert() {
  log convert
  rm -f $K/seg/stgeorgen48_seg.mlmodel
  $V/ketos convert "$(segw)" -o $K/seg/stgeorgen48_seg.mlmodel --weights-format coreml 2>&1 | tee -a $K/logs/run.log
  log "all done"
}

case "${1:-all}" in
  seg) seg ;;
  test) test_model ;;
  convert) convert ;;
  all) seg; test_model; convert ;;
  *) echo "usage: $0 [all|seg|test|convert]"; exit 1 ;;
esac
