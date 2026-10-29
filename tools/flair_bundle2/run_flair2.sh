#!/usr/bin/env bash
# GPU queue entry point (gpq) for the Flair tagger v2:
#   cp run_flair2.sh ~/.gpq/session/run_flair2.sh && chmod +x ~/.gpq/session/run_flair2.sh
# Expects flair-train2.tar in $DIR (or the folder unpacked as $DIR/flair-train2). Resumable: register again after an
# interruption; finished stages are skipped, training continues from the last epoch checkpoint.
# Results: $DIR/flair-train2/flair-results2.tar.gz (predictions, logs) and flair-model2.tar.gz (model, ~2.7 GB).
set -uo pipefail
DIR="${DIR:-$HOME}"
END_CMD="${GPQ_END_CMD:-}"
cd "$DIR" || { echo "no $DIR"; exit 1; }
[ -x flair-train2/train_flair2.sh ] || tar -xf flair-train2.tar || { echo "flair-train2.tar missing in $DIR"; exit 1; }
cd flair-train2 || exit 1
mkdir -p logs
exec > >(tee -a "logs/gpq_$(date +%Y%m%d-%H%M%S).log") 2>&1
echo "== run_flair2.sh start $(date '+%F %T') on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
finish() { ./train_flair2.sh pack 2>&1 | tail -3; [ -n "$END_CMD" ] && eval "$END_CMD"; echo "== end $(date '+%F %T')"; }
trap finish EXIT
./train_flair2.sh setup train predict
