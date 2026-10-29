#!/usr/bin/env bash
# Build dist/kraken-train2.tar (and a copy on the Windows desktop) for tools/kraken_bundle2/ (GPU server training).
#   bash tools/make_kraken_bundle2.sh [staging dir]
# Needs output/kraken_final/rec (recognition PageXML, compiled on the server), output/kraken_stgeorgen48/pages and the Baldus
# PageXML dataset in dist/hf-consilia-baldus-venice1575-pagexml/.
set -eu
cd "$(dirname "$0")/.."
S="${1:-/tmp/claude-1000/-home-micha-github-Consilia/f0bf4aa9-62fa-4347-8d5c-fb10df6c8303/scratchpad/kraken-train2}"
rm -rf "$S"; mkdir -p "$S/baldus/pages" "$S/stgeorgen48"
cp tools/kraken_bundle2/train_kraken2.sh tools/kraken_bundle2/run_kraken2.sh tools/kraken_bundle2/README.md tools/kraken_fix_segmap.py "$S/"
ln -s "$PWD/output/trained_kraken/base_McCATMuS_nfd_nofix_V1.mlmodel" "$S/"
for s in train val test; do
  sed "s#^$PWD/output/kraken_final/rec/#baldus/rec/#" output/kraken_final/rec_$s.txt > "$S/baldus/rec_$s.txt"
  sed "s#^$PWD/dist/hf-consilia-baldus-venice1575-pagexml/#baldus/pages/#" output/kraken_final/seg_$s.txt > "$S/baldus/seg_$s.txt"
  sed "s#^$PWD/output/kraken_stgeorgen48/pages/#stgeorgen48/pages/#" output/kraken_stgeorgen48/seg_$s.txt > "$S/stgeorgen48/seg_$s.txt"
done
for v in dist/hf-consilia-baldus-venice1575-pagexml/Baldo_Cons_Print_Venice_1575_v?; do ln -s "$PWD/$v" "$S/baldus/pages/"; done
ln -s "$PWD/output/kraken_stgeorgen48/pages" "$S/stgeorgen48/pages"
ln -s "$PWD/output/kraken_final/rec" "$S/baldus/rec"   # recognition PageXML, images hard-linked to pages/ (stored once by tar)
# every listed page must exist (with its image)
( cd "$S" && cat baldus/seg_*.txt baldus/rec_*.txt stgeorgen48/seg_*.txt | while read -r f; do [ -f "$f" ] && [ -f "${f%.xml}.jpg" ] || echo "missing: $f"; done | head -5 )
echo "staged in $S"; du -shL "$S"
OUT=dist/kraken-train2.tar
tar -h -C "$(dirname "$S")" -cf "$OUT" kraken-train2
ls -la "$OUT"
D="${COPY_TO:-}"          # optional: a folder to copy the bundle to (e.g. a desktop shared with the upload machine)
[ -n "$D" ] && [ -d "$D" ] && { cp "$OUT" "$D/"; cp tools/kraken_bundle2/run_kraken2.sh "$D/"; echo "copied to $D: kraken-train2.tar, run_kraken2.sh"; } || true
