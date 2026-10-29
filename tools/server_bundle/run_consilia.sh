#!/usr/bin/env bash
# GPU queue entry point (gpq): register this script, it runs when the slot is ours.
#   cp run_consilia.sh ~/.gpq/session/run_consilia.sh && chmod +x ~/.gpq/session/run_consilia.sh
# Expects the bundle in $DIR (consilia-server-bundle.tar.gz or already unpacked). Every stage is
# resumable: if the session ends before the work is done, register the script again and it continues
# where it stopped (finished stages are marked in $DIR/state/, classify/spans skip finished items;
# the Flair training resumes from its last epoch checkpoint). Logs: $DIR/logs/gpq_<timestamp>.log.
set -uo pipefail
DIR="${DIR:-$HOME/consilia_new}"
STAGES="${STAGES:-full spans flair body2}"               # gold already done locally; pack runs always at the end
END_CMD="${GPQ_END_CMD:-}"                               # command that releases the GPU session, if the queue needs one
cd "$DIR" || { echo "no $DIR"; exit 1; }
mkdir -p logs state
LOG="logs/gpq_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "== run_consilia.sh start $(date '+%F %T') on $(hostname), dir $DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

finish() {
  echo "== finishing $(date '+%F %T')"
  ./run.sh pack 2>&1 | tail -3 || true                   # results tarball, always, even after an interruption
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 3
  [ -n "$END_CMD" ] && { echo "== releasing the session: $END_CMD"; eval "$END_CMD"; }
  echo "== run_consilia.sh end $(date '+%F %T')"
}
trap finish EXIT

# unpack once
if [ ! -x run.sh ]; then
  tar xzf consilia-server-bundle.tar.gz --strip-components=1 || { echo "bundle missing in $DIR"; exit 1; }
fi
# setup once (venvs, vLLM, Flair); run.sh setup is itself idempotent but slow
[ -f state/setup.done ] || { ./run.sh setup && touch state/setup.done; }
# vLLM lives for this session only; run.sh starts it when a stage needs it and it is not answering
for s in $STAGES; do
  if [ -f "state/$s.done" ]; then echo "== stage $s already done"; continue; fi
  echo "== stage $s $(date '+%F %T')"
  if ./run.sh "$s"; then touch "state/$s.done"; else echo "== stage $s stopped without finishing; re-register the script to continue"; exit 1; fi
done
echo "== all stages done"
