#!/usr/bin/env bash
# Build dist/flair-train2.tar (+ run_flair2.sh on the Windows desktop) for tools/flair_bundle2/.
set -eu
cd "$(dirname "$0")/.."
S=dist/flair-train2
rm -rf "$S"; mkdir -p "$S"/src/annotate "$S"/tools "$S"/output/annotations "$S"/output/flair
cp tools/flair_bundle2/{train_flair2.sh,run_flair2.sh,README.md} "$S"/
cp tools/server_bundle/requirements.txt "$S"/
cp tools/train_flair_ner.py "$S"/tools/
cp src/annotate/export_flair.py "$S"/src/annotate/
cp output/units_all_20261029.jsonl "$S"/output/units.jsonl
cp output/annotations/spans_gemini-4-argon_merged.jsonl "$S"/output/annotations/
.venv/bin/python src/annotate/export_flair.py output/annotations/spans_gemini-4-argon_merged.jsonl \
    --units output/units_all_20261029.jsonl --out "$S"/output/flair
chmod +x "$S"/*.sh
( cd dist && tar -cf flair-train2.tar flair-train2 )
rm -rf "$S"
ls -la dist/flair-train2.tar
D="${COPY_TO:-}"          # optional: a folder to copy the bundle to (e.g. a desktop shared with the upload machine)
[ -n "$D" ] && [ -d "$D" ] && cp dist/flair-train2.tar tools/flair_bundle2/run_flair2.sh "$D"/ && echo "copied to $D" || true
