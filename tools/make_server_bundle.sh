#!/usr/bin/env bash
# Build dist/consilia-server-bundle.tar.gz: everything the GPU server needs for the annotation layers
# (A/B classification, C spans, Flair tagger). See tools/server_bundle/README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
B=dist/consilia-server-bundle
rm -rf "$B"; mkdir -p "$B"/src/annotate "$B"/src/codebooks "$B"/tools "$B"/output/annotations "$B"/eval/gold "$B"/eval/results "$B"/output/qa
cp src/annotate/{classify,evaluate,spans,export_flair}.py "$B"/src/annotate/
cp src/codebooks/*.yaml "$B"/src/codebooks/
cp tools/train_flair_ner.py "$B"/tools/
cp output/units.jsonl output/consilia.json "$B"/output/
cp eval/gold/*.json eval/gold/*.jsonl "$B"/eval/gold/
cp eval/results/*.md "$B"/eval/results/ 2>/dev/null || true          # Gemini pilot numbers for comparison
cp output/qa/spans_review.md "$B"/output/qa/ 2>/dev/null || true
cp tools/server_bundle/{run.sh,run_consilia.sh,requirements.txt,README.md} "$B"/
chmod +x "$B"/run.sh "$B"/run_consilia.sh
( cd dist && tar czf consilia-server-bundle.tar.gz consilia-server-bundle )
rm -rf "$B"
ls -la dist/consilia-server-bundle.tar.gz
