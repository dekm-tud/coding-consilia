#!/usr/bin/env bash
# GPU queue entry point (gpq) for the kraken training: register this script, it runs when the slot is ours.
#   cp run_kraken.sh ~/.gpq/session/run_kraken.sh && chmod +x ~/.gpq/session/run_kraken.sh
# Expects kraken-train.tar in $DIR (or the folder already unpacked as $DIR/kraken-train). Resumable: if
# the session ends mid-training, register the script again and it continues from the newest checkpoint
# (one checkpoint per epoch); a finished training is marked in state/ and only the packing runs again.
# Logs: $DIR/kraken-train/logs/gpq_<timestamp>.log. Result: $DIR/kraken-train/kraken-model.tar.gz
set -uo pipefail
DIR="${DIR:-$HOME/consilia_new}"
END_CMD="${GPQ_END_CMD:-}"                               # command that releases the GPU session, if the queue needs one
cd "$DIR" || { echo "no $DIR"; exit 1; }
# unpack once
if [ ! -x kraken-train/train_kraken.sh ]; then
  tar -xf kraken-train.tar || { echo "kraken-train.tar missing in $DIR"; exit 1; }
fi
cd kraken-train || exit 1
mkdir -p logs state model
LOG="logs/gpq_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "== run_kraken.sh start $(date '+%F %T') on $(hostname), dir $DIR/kraken-train"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

finish() {
  echo "== finishing $(date '+%F %T')"
  ./train_kraken.sh pack 2>&1 | tail -2 || true           # model tarball, always, even after an interruption
  [ -n "$END_CMD" ] && { echo "== releasing the session: $END_CMD"; eval "$END_CMD"; }
  echo "== run_kraken.sh end $(date '+%F %T')"
}
trap finish EXIT

[ -f state/setup.done ] || { ./train_kraken.sh setup && touch state/setup.done; }
if [ -f state/train.done ]; then
  echo "== training already done"
elif ls model/consilia/*.ckpt >/dev/null 2>&1; then
  echo "== resuming from $(ls -t model/consilia/*.ckpt | head -1) $(date '+%F %T')"
  if ./train_kraken.sh resume; then touch state/train.done; else echo "== training stopped without finishing; re-register the script to continue"; exit 1; fi
else
  echo "== training from the base model $(date '+%F %T')"
  if ./train_kraken.sh train; then touch state/train.done; else echo "== training stopped without finishing; re-register the script to continue"; exit 1; fi
fi
echo "== all done"
