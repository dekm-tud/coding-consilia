#!/usr/bin/env bash
# Consilia annotation bundle: run on the GPU server.
#   ./run.sh                 # all stages in order
#   ./run.sh gold            # one or more stages: setup serve gold full spans flair body2 pack
# Settings via environment: MODEL (default Qwen/Qwen3.8-27B-FP8), BASE_URL, WORKERS, THINKING (--thinking|--no-thinking),
# GPU_UTIL (0.90), ATTN_BACKEND (TRITON_ATTN), FLASHINFER_SAMPLER (0), ENFORCE_EAGER, EXTRA_VLLM_ARGS.
# Two virtualenvs: .venv (annotation scripts + Flair) and .venv-vllm (vLLM, its own torch).
# Everything is resumable: a stopped stage continues where it left off when run again.
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_ENABLE_HF_TRANSFER=0
MODEL="${MODEL:-Qwen/Qwen3.8-27B-FP8}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
WORKERS="${WORKERS:-32}"
THINKING="${THINKING:---thinking}"
PY="${PY:-.venv/bin/python}"
CLS="src/annotate/classify.py --backend openai --base-url $BASE_URL --model $MODEL $THINKING"
SLUG=$(echo "$MODEL" | tr 'A-Z' 'a-z' | sed 's/[^a-z0-9]\+/-/g; s/^-//; s/-$//')
mkdir -p logs eval/results output/annotations
log() { echo "[$(date +%H:%M:%S)] $*"; }

stage_setup() {
  log "setup: .venv (annotation + Flair) and .venv-vllm (vLLM from PyPI); first run downloads several GB"
  [ -d .venv ] || python3 -m venv .venv
  .venv/bin/pip install -q -U pip
  .venv/bin/pip install -q -r requirements.txt
  if [ -z "${VLLM_BIN:-}" ]; then
    [ -d .venv-vllm ] || python3 -m venv .venv-vllm
    .venv-vllm/bin/pip install -q -U pip
    .venv-vllm/bin/pip install -q "vllm>=0.17" "transformers>=5.8"
  fi
}

stage_serve() {
  # Start vLLM if nothing answers at BASE_URL. Adjust VLLM_BIN if vLLM lives in another environment.
  if curl -sf "$BASE_URL/models" >/dev/null 2>&1; then log "serve: $BASE_URL already answers"; return; fi
  VLLM_BIN="${VLLM_BIN:-.venv-vllm/bin/vllm}"
  log "serve: starting $VLLM_BIN serve $MODEL (log logs/vllm.log; the first start downloads the weights, ~30 GB for the FP8 27B)"
  [ -x "$VLLM_BIN" ] || command -v "$VLLM_BIN" >/dev/null || { echo "vLLM binary $VLLM_BIN not found (setup failed?)"; exit 1; }
  # ENFORCE_EAGER=1 skips torch.compile and CUDA graph capture (slower, but robust); EXTRA_VLLM_ARGS passes anything else.
  # FlashInfer JIT builds (fp8 KV kernels, top-p/top-k sampler) failed on the Blackwell box (2026-10-17); Triton attention
  # and the torch sampler need no JIT. Set ATTN_BACKEND=FLASHINFER / FLASHINFER_SAMPLER=1 to try FlashInfer again.
  nohup env VLLM_ATTENTION_BACKEND="${ATTN_BACKEND:-TRITON_ATTN}" VLLM_USE_FLASHINFER_SAMPLER="${FLASHINFER_SAMPLER:-0}" "$VLLM_BIN" serve "$MODEL" --port 8000 --max-model-len 32768 \
      --gpu-memory-utilization "${GPU_UTIL:-0.90}" --reasoning-parser qwen3 --enable-prefix-caching \
      ${ENFORCE_EAGER:+--enforce-eager} ${EXTRA_VLLM_ARGS:-} > logs/vllm.log 2>&1 &
  VPID=$!
  for i in $(seq 1 360); do
    sleep 10
    curl -sf "$BASE_URL/models" >/dev/null 2>&1 && { log "serve: ready"; return; }
    if ! kill -0 "$VPID" 2>/dev/null; then
      echo "vLLM exited. Root cause (first error in logs/vllm.log):"
      grep -n -m 1 -A 60 "EngineCore failed to start\|EngineCore encountered" logs/vllm.log | grep -v "INFO\|^[0-9]*-\s*$" | cut -c1-240 || tail -n 40 logs/vllm.log
      echo "Retry with ATTN_BACKEND=TRITON_ATTN ./run.sh serve  or  ENFORCE_EAGER=1 ./run.sh serve"
      exit 1
    fi
    if [ $((i % 6)) -eq 0 ]; then log "serve: still starting ($((i / 6)) min). Last log line:"; tail -n 1 logs/vllm.log | cut -c1-200; fi
  done
  echo "vLLM did not come up within an hour; see logs/vllm.log"; exit 1
}

need_server() {
  # every annotation stage first makes sure vLLM answers; starts it if not (it dies with the shell that started it)
  curl -sf "$BASE_URL/models" >/dev/null 2>&1 || stage_serve
}

stage_gold() {
  need_server
  log "gold: Layer B on dev/test/rare, Layer A on gold (compare with eval/results/*gemini*.md)"
  for s in dev test rare; do $PY $CLS --task body --split $s --workers "$WORKERS" 2>&1 | tee -a logs/gold.log; done
  $PY $CLS --task subjects --split gold --workers "$WORKERS" 2>&1 | tee -a logs/gold.log
  for s in dev test rare; do $PY src/annotate/evaluate.py --task body --split $s "output/annotations/body_${SLUG}_p1.jsonl" 2>&1 | tee -a logs/gold.log; done
  $PY src/annotate/evaluate.py --task subjects "output/annotations/subjects_${SLUG}_p1.jsonl" 2>&1 | tee -a logs/gold.log
  log "gold: done. Gemini pilot for comparison: Layer B pair F1 test 0.81 / rare 0.86, Layer A central F1 0.78"
}

stage_full() {
  need_server
  log "full: Layer B pass 1 (all units), Layer A (all consilia)"
  $PY $CLS --task body --workers "$WORKERS" 2>&1 | tee -a logs/full.log
  $PY $CLS --task subjects --workers "$WORKERS" 2>&1 | tee -a logs/full.log
}

stage_body2() {
  need_server
  log "body2: Layer B pass 2 (second reading with other few-shots; the export keeps codes both passes agree on)"
  $PY $CLS --task body --pass 2 --workers "$WORKERS" 2>&1 | tee -a logs/full.log
}

stage_spans() {
  need_server
  log "spans: Layer C over all units (silver corpus); evaluation against the gold if present"
  $PY src/annotate/spans.py --backend openai --base-url "$BASE_URL" --model "$MODEL" $THINKING --workers "$WORKERS" 2>&1 | tee -a logs/spans.log
  if [ -f eval/gold/spans_gold.jsonl ]; then
    $PY src/annotate/spans.py --eval eval/gold/spans_gold.jsonl "output/annotations/spans_${SLUG}_p1.jsonl" 2>&1 | tee -a logs/spans.log
  else
    log "spans: no eval/gold/spans_gold.jsonl yet (correct output/qa/spans_review.md and save the gold); skipping the evaluation"
  fi
}

stage_flair() {
  log "flair: export silver (+gold) corpus, train the stacked tagger (MediBerta + Latin legal Flair LMs), tag all units"
  GOLD=""; [ -f eval/gold/spans_gold.jsonl ] && GOLD="--gold eval/gold/spans_gold.jsonl"
  $PY src/annotate/export_flair.py output/annotations/spans_${SLUG}_p1.jsonl $GOLD 2>&1 | tee -a logs/flair.log
  $PY tools/train_flair_ner.py --mode stack --epochs "${EPOCHS:-20}" --batch "${BATCH:-16}" 2>&1 | tee -a logs/flair.log
  $PY tools/train_flair_ner.py --predict 2>&1 | tee -a logs/flair.log
  if [ -f eval/gold/spans_gold.jsonl ]; then
    $PY src/annotate/spans.py --eval eval/gold/spans_gold.jsonl output/annotations/spans_flair.jsonl "output/annotations/spans_${SLUG}_p1.jsonl" 2>&1 | tee -a logs/flair.log
  fi
}

stage_pack() {
  log "pack: results for the repo (annotations, evaluation, logs) and the tagger model separately"
  tar czf consilia-results.tar.gz output/annotations eval/results output/flair/*.txt output/flair/labels.json logs
  [ -f output/flair/model/final-model.pt ] && tar czf consilia-flair-model.tar.gz -C output/flair model
  ls -la consilia-results.tar.gz consilia-flair-model.tar.gz 2>/dev/null || true
  log "pack: copy consilia-results.tar.gz back and unpack it in the repo root; then export.py / build_site_data.py"
}

STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=(setup serve gold full spans flair body2 pack)
for s in "${STAGES[@]}"; do "stage_$s"; done
log "all requested stages done"
