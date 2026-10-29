#!/usr/bin/env bash
# Second transcription of the Baldus prints with Gemini (Batch API) for quality control, and the
# comparison with the PageXML text. Needs GEMINI_API_KEY and the page JPGs in data/<author>/<print>/.
#
#   tools/gemini_recheck.sh crop            # column crops for all five volumes (data/<print>/crops_gemini, gitignored)
#   tools/gemini_recheck.sh submit          # batch jobs (half price; results within 24 h)
#   tools/gemini_recheck.sh collect         # fetch results -> output/ocr_gemini/<print>/<page>_c<n>.txt; resubmit what failed
#   tools/gemini_recheck.sh compare         # src/qa/compare_ocr.py per volume -> output/qa/ocr_*.md
#   tools/gemini_recheck.sh ocr 123         # live run of single pages (e.g. a test), same output directory
# PRINTS="Baldo_Cons_Print_Venice_1575_v1" tools/gemini_recheck.sh submit   # one volume only
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
PIPE=src/gemini_ocr/pipeline.py
PROMPT=src/gemini_ocr/prompt_baldus.txt
PRINTS=${PRINTS:-"Baldo_Cons_Print_Venice_1575_v1 Baldo_Cons_Print_Venice_1575_v2 Baldo_Cons_Print_Venice_1575_v3 Baldo_Cons_Print_Venice_1575_v4 Baldo_Cons_Print_Venice_1575_v5"}
cmd=${1:-}; shift || true
for p in $PRINTS; do
  D=data/Baldo_29618397/$p
  OUT=../../../output/ocr_gemini/$p          # relative to $D (pipeline.py joins it with --data)
  mkdir -p output/ocr_gemini/$p
  echo "== $p"
  case "$cmd" in
    crop)    $PY $PIPE crop --data $D --crops crops_gemini "$@" ;;
    submit)  $PY $PIPE batch-submit --data $D --crops crops_gemini --out $OUT --prompt $PROMPT "$@" ;;
    collect) $PY $PIPE batch-collect --data $D --crops crops_gemini --out $OUT --prompt $PROMPT "$@" ;;
    ocr)     $PY $PIPE ocr --data $D --crops crops_gemini --out $OUT --prompt $PROMPT --pages "$@" ;;
    compare) $PY src/qa/compare_ocr.py --print $p --ocr output/ocr_gemini/$p ;;
    *) echo "usage: $0 crop|submit|collect|compare|ocr <pages>"; exit 1 ;;
  esac
done
