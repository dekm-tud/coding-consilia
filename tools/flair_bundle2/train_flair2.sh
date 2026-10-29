#!/usr/bin/env bash
# Consilia Flair tagger v2 (2026-10-28): retrain the citation/entity tagger on the Gemini 4 Argon annotation of the
# whole corpus (silver), then tag all units with it.
#   ./train_flair2.sh                     # setup train predict pack
#   ./train_flair2.sh setup|train|predict|pack
# The BIO corpus (output/flair/{train,dev,test}.txt, split by consilium) is built in the repo and shipped;
# output/units.jsonl and output/annotations/spans_gemini-4-argon_merged.jsonl are the source, for reference and
# for the prediction. Training resumes from the last epoch checkpoint (output/flair/model/model_epoch_<k>.pt).
set -u
cd "$(dirname "$0")"
V=.venv/bin
mkdir -p logs state
log() { echo "== $* $(date '+%F %T')" | tee -a logs/train_flair2.log; }
for st in ${*:-setup train predict pack}; do
  case $st in
    setup)
      [ -f state/setup.done ] && continue
      log setup
      [ -x $V/python ] || python3 -m venv .venv
      $V/pip install -q --upgrade pip && $V/pip install -q -r requirements.txt
      $V/python -c "import flair,torch;print('flair',flair.__version__,'torch',torch.__version__,'cuda',torch.cuda.is_available())" | tee -a logs/train_flair2.log
      touch state/setup.done ;;
    train)
      [ -f state/train.done ] && { log "train already done"; continue; }
      # speed on a large GPU (RTX PRO 6000, 96 GB): the mini-batch in one pass (CHUNK=0) and mixed precision (AMP=1);
      # CHUNK=4 AMP=0 is the v1 setting for small cards. Batch size and learning rate stay those of v1 (same recipe).
      log "train (stack: MediBerta fine-tuned + Latin legal Flair LMs, ${EPOCHS:-20} epochs, batch ${BATCH:-16}, chunk ${CHUNK:-0}, amp ${AMP:-1})"
      AMPF=""; [ "${AMP:-1}" = 1 ] && AMPF="--amp"
      $V/python tools/train_flair_ner.py --mode stack --epochs "${EPOCHS:-20}" --batch "${BATCH:-16}" --chunk "${CHUNK:-0}" $AMPF 2>&1 | tee -a logs/train.log
      [ -f output/flair/model/final-model.pt ] && touch state/train.done ;;
    predict)
      [ -f state/predict.done ] && { log "predict already done"; continue; }
      log predict
      AMPF=""; [ "${AMP:-1}" = 1 ] && AMPF="--amp"
      $V/python tools/train_flair_ner.py --predict --units output/units.jsonl --predict-out output/annotations/spans_flair_v2.jsonl --predict-batch 128 $AMPF 2>&1 | tee -a logs/predict.log
      [ -s output/annotations/spans_flair_v2.jsonl ] && touch state/predict.done ;;
    pack)
      log pack
      M=output/flair/model
      B=$( [ -f $M/best-model.pt ] && echo $M/best-model.pt || echo $M/final-model.pt )
      tar -czf flair-results2.tar.gz output/annotations/spans_flair_v2.jsonl $M/training.log $M/loss.tsv $M/test.tsv $M/dev.tsv logs state 2>/dev/null
      [ -f "$B" ] && tar -czf flair-model2.tar.gz "$B" $M/training.log
      ls -la flair-results2.tar.gz flair-model2.tar.gz 2>/dev/null | tee -a logs/train_flair2.log ;;
    *) echo "usage: $0 [setup|train|predict|pack]"; exit 1 ;;
  esac
done
