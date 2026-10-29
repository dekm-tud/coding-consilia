#!/usr/bin/env bash
# Consilia kraken training bundle 2 (2026-10-26): three models on one GPU (RTX PRO 6000 Blackwell, linglit-gpu).
#   ./train_kraken2.sh                      # setup all pack
#   ./train_kraken2.sh setup|all|rec|seg_baldus|seg_stg|test|convert|pack
# Jobs (all resumable from the newest checkpoint in model/<job>/, a finished job is marked in state/):
#   rec         Baldus recognition: McCATMuS fine-tuned on the final edition text (baldus/rec/<volume>/*.xml, compiled first)
#   seg_baldus  Baldus layout: blla with a new output layer (lines cons_body/cons_argument/cons_title/header/
#               page_number/catch-word; regions column_1/column_2/header/page-number/footer)
#   seg_stg     St. Georgen 48 layout: blla, lines body/marginalia/page-number/quire-number; regions column_1/
#               column_2/column_single/marginalia/page-number/quire-number/paragraph
# `all` runs the three trainings in parallel (together < 30 GB GPU memory), then test, convert, pack.
# The class-mapping bug of kraken 7.1.1 (--resize new stores an inference mapping that does not match the
# network) is repaired by kraken_fix_segmap.py before segtest and convert: model/<job>/<name>.safetensors is
# the repaired copy, <name>.mlmodel the CoreML container made from it.
set -u
cd "$(dirname "$0")"
DEV="${KRAKEN_DEVICE:-cuda:0}"; W="${KRAKEN_WORKERS:-8}"
export COLUMNS="${COLUMNS:-240}"   # rich progress bars: without a TTY kraken assumes 80 columns and truncates the validation values
V=.venv-kraken/bin
mkdir -p logs state model/rec model/seg_baldus model/seg_stg
log() { echo "== $* $(date '+%F %T')" | tee -a logs/train_kraken2.log; }
best() { ls -t model/$1/*/best_0*.safetensors 2>/dev/null | head -1; }
ckpt() { ls -t model/$1/*/*.ckpt 2>/dev/null | head -1; }
filt() { grep --line-buffered -E "^stage|^== |val_|Error|error|Converting" || true; }

setup() {
  [ -x $V/ketos ] || { python3 -m venv .venv-kraken && $V/pip install -q --upgrade pip && $V/pip install -q "kraken==7.1.1"; }
  $V/kraken --version | tee -a logs/train_kraken2.log
  $V/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))" | tee -a logs/train_kraken2.log
  touch state/setup.done
}
blla() { $V/python -c "import kraken,os;print(os.path.join(os.path.dirname(kraken.__file__),'blla.mlmodel'))"; }

compile() {   # binary line datasets from the recognition PageXML (CPU, ~15 min with 12 workers, ~15 GB)
  for s in train val test; do
    [ -s baldus/rec_$s.arrow ] && continue
    log "compile $s"
    $V/ketos --workers "${KRAKEN_COMPILE_WORKERS:-12}" compile -f page -F baldus/rec_$s.txt -o baldus/rec_$s.arrow 2>&1 | tee -a logs/compile.log | tail -2
    echo "$PWD/baldus/rec_$s.arrow" > baldus/rec_$s.lst   # ketos train/test want a manifest, not the arrow file
  done
}

rec() {
  [ -f state/rec.done ] && { log "rec already done"; return 0; }
  compile
  CK=$(ckpt rec)
  if [ -n "$CK" ]; then log "rec resume $CK"; ARGS="--resume $CK"
  else log "rec from base_McCATMuS_nfd_nofix_V1.mlmodel"; ARGS="-i base_McCATMuS_nfd_nofix_V1.mlmodel --resize union -u NFD"; fi
  # batch 32, lr 5e-4 cosine over 16 epochs, early stop after 4 evaluations without improvement, codec extended
  $V/ketos -d "$DEV" --workers $W --precision 16-mixed train -f binary \
    -t baldus/rec_train.lst -e baldus/rec_val.lst $ARGS \
    -B 32 -r 0.0005 --schedule cosine --cos-max 16 -N 16 --min-epochs 8 -q early --lag 4 --min-delta 0.0005 \
    --augment --pad 16 -F 1 -o model/rec/consilia_rec 2>&1 | tee -a logs/rec_train.log | filt
  [ -n "$(best rec)" ] && { touch state/rec.done; log "rec done, best $(best rec)"; } || log "rec stopped without a best model"
}

segjob() {   # $1 job, $2 data dir, $3 output name, $4 max epochs, $5 min epochs, $6 lag
  J=$1; D=$2; N=$3
  [ -f state/$J.done ] && { log "$J already done"; return 0; }
  CK=$(ckpt $J)
  if [ -n "$CK" ]; then log "$J resume $CK"; ARGS="--resume $CK"
  else log "$J from blla.mlmodel"; ARGS="-i $(blla) --resize new"; fi
  $V/ketos -d "$DEV" --workers $W --precision 16-mixed segtrain -f page \
    -t $D/seg_train.txt -e $D/seg_val.txt $ARGS \
    -N $4 --min-epochs $5 -q early --lag $6 -r 0.0002 --schedule cosine --cos-max $4 --augment \
    -o model/$J/$N 2>&1 | tee -a logs/${J}_train.log | filt
  [ -n "$(best $J)" ] && { touch state/$J.done; log "$J done, best $(best $J)"; } || log "$J stopped without a best model"
}
seg_baldus() { segjob seg_baldus baldus consilia_seg 60 15 10; }
seg_stg()    { segjob seg_stg stgeorgen48 stgeorgen48_seg 80 20 12; }

segw() {   # repaired weights for a seg job: model/<job>/<name>.safetensors
  B=$(best $1); [ -n "$B" ] || return 0
  OUT=model/$1/$2.safetensors
  [ "$OUT" -nt "$B" ] || $V/python kraken_fix_segmap.py "$B" "$OUT" >&2
  echo "$OUT"
}

test_models() {
  R=$(best rec)
  [ -n "$R" ] && { log "test rec $R"; $V/ketos -d "$DEV" --workers $W test -f binary -m "$R" -e baldus/rec_test.lst -B 32 --pad 16 -u NFD 2>&1 | tee logs/rec_test.log | tail -25; }
  S=$(segw seg_baldus consilia_seg)
  [ -n "$S" ] && { log "segtest baldus $S"; $V/ketos -d "$DEV" --workers $W segtest -f page -m "$S" -e baldus/seg_test.txt 2>&1 | tee logs/seg_baldus_test.log | tail -40; }
  S=$(segw seg_stg stgeorgen48_seg)
  [ -n "$S" ] && { log "segtest stgeorgen48 $S"; $V/ketos -d "$DEV" --workers $W segtest -f page -m "$S" -e stgeorgen48/seg_test.txt 2>&1 | tee logs/seg_stg_test.log | tail -40; }
}

convert() {
  for spec in "rec consilia_rec $(best rec)" "seg_baldus consilia_seg $(segw seg_baldus consilia_seg)" "seg_stg stgeorgen48_seg $(segw seg_stg stgeorgen48_seg)"; do
    set -- $spec; [ -n "${3:-}" ] || continue
    log "convert $3"
    rm -f model/$1/$2.mlmodel   # ketos convert refuses to overwrite
    $V/ketos convert "$3" -o model/$1/$2.mlmodel --weights-format coreml 2>&1 | tail -2
  done
}

pack() {
  log pack
  F=$( { for j in rec seg_baldus seg_stg; do best $j; done; ls model/*/*.safetensors model/*/*.mlmodel 2>/dev/null; } | sort -u )
  tar -czf kraken-models2.tar.gz $F logs state 2>/dev/null || tar -czf kraken-models2.tar.gz model logs state
  log "packed kraken-models2.tar.gz: $(du -h kraken-models2.tar.gz | cut -f1)"
}

all() {
  rec & P1=$!
  seg_baldus & P2=$!
  seg_stg & P3=$!
  wait $P1 $P2 $P3
  test_models; convert
}

for st in ${*:-setup all pack}; do
  case $st in
    setup) setup ;;
    all) [ -f state/setup.done ] || setup; all ;;
    rec|seg_baldus|seg_stg) [ -f state/setup.done ] || setup; $st ;;
    test) test_models ;;
    convert) convert ;;
    pack) pack ;;
    *) echo "usage: $0 [setup|all|rec|seg_baldus|seg_stg|test|convert|pack]"; exit 1 ;;
  esac
done
