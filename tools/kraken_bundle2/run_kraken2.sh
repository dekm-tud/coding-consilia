#!/usr/bin/env bash
# GPU queue entry point (gpq) for the kraken training bundle 2 (three models, 2026-10-26):
#   cp run_kraken2.sh ~/.gpq/session/run_kraken2.sh && chmod +x ~/.gpq/session/run_kraken2.sh
# Expects kraken-train2.tar in $DIR (or the folder already unpacked as $DIR/kraken-train2). Resumable: if the
# session ends mid-training, register the script again; each job continues from its newest checkpoint and
# finished jobs are skipped (state/). Result: $DIR/kraken-train2/kraken-models2.tar.gz
set -uo pipefail
DIR="${DIR:-$HOME}"
END_CMD="${GPQ_END_CMD:-}"
cd "$DIR" || { echo "no $DIR"; exit 1; }
if [ ! -x kraken-train2/train_kraken2.sh ]; then
  tar -xf kraken-train2.tar || { echo "kraken-train2.tar missing in $DIR"; exit 1; }
fi
cd kraken-train2 || exit 1
mkdir -p logs
LOG="logs/gpq_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "== run_kraken2.sh start $(date '+%F %T') on $(hostname), dir $DIR/kraken-train2"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
finish() {
  echo "== finishing $(date '+%F %T')"
  ./train_kraken2.sh pack 2>&1 | tail -2 || true
  [ -n "$END_CMD" ] && { echo "== releasing the session: $END_CMD"; eval "$END_CMD"; }
  echo "== run_kraken2.sh end $(date '+%F %T')"
}
trap finish EXIT
./train_kraken2.sh setup all || { echo "== not all jobs finished; re-register the script to continue"; exit 1; }
echo "== all done"
