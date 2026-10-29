#!/usr/bin/env bash
# Fine-tune a kraken recognition model on the consensus lines of the Consilia prints (GPU server).
# Usage:  ./train_kraken.sh [setup|train|resume|pack]      default: setup train pack
#   setup   creates .venv-kraken and installs kraken (once)
#   train   fine-tunes base_McCATMuS_nfd_nofix_V1.mlmodel on train.txt / val.txt -> model/consilia_*.mlmodel
#   resume  continues from the newest checkpoint in model/consilia/
#   pack    writes kraken-model.tar.gz with the best model and the training log
# Lines: train/ and val/ hold line images (grayscale PNG) with the edition text in <name>.gt.txt
# (long s = s, & = et, abbreviations expanded, hyphen at the line end = ¬); the split is by page.
set -eu
cd "$(dirname "$0")"
STAGES="${*:-setup train pack}"
DEV="${KRAKEN_DEVICE:-cuda:0}"
mkdir -p model logs
for st in $STAGES; do
  echo "== $st $(date +%F' '%T)" | tee -a logs/train_kraken.log
  case $st in
    setup)
      [ -d .venv-kraken ] || python3 -m venv .venv-kraken
      .venv-kraken/bin/pip install -q --upgrade pip
      .venv-kraken/bin/pip install -q kraken
      .venv-kraken/bin/kraken --version | tee -a logs/train_kraken.log ;;
    train)
      # fine-tuning: codec extended by the edition's characters (--resize union), NFD like the base model,
      # cosine schedule over at most 40 epochs, early stop after 6 evaluations without improvement,
      # augmentation on, mixed precision. ~145k lines per epoch; expect 10-15 min per epoch on an A100.
      .venv-kraken/bin/ketos -d "$DEV" --workers 12 --precision 16-mixed train -f path \
        -t train.txt -e val.txt -i base_McCATMuS_nfd_nofix_V1.mlmodel --resize union -u NFD \
        -B 32 -r 0.0005 --schedule cosine --cos-max 40 -N 40 --min-epochs 8 -q early --lag 6 --min-delta 0.0005 \
        --augment --pad 16 -F 1 -o model/consilia 2>&1 | tee -a logs/train_kraken.log ;;
    resume)
      CK=$(ls -t model/consilia/*.ckpt | head -1)
      .venv-kraken/bin/ketos -d "$DEV" --workers 12 --precision 16-mixed train -f path \
        -t train.txt -e val.txt --resume "$CK" -o model/consilia 2>&1 | tee -a logs/train_kraken.log ;;
    pack)
      BEST=$(ls -t model/consilia_best.mlmodel model/consilia/*.mlmodel 2>/dev/null | head -1 || true)
      tar -czf kraken-model.tar.gz model/*.mlmodel model/consilia/*.ckpt logs/ 2>/dev/null || tar -czf kraken-model.tar.gz model logs
      echo "packed: kraken-model.tar.gz (best: ${BEST:-none yet})" | tee -a logs/train_kraken.log ;;
  esac
done
