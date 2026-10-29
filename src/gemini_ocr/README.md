# Gemini column transcription (Panormitanus)

`pipeline.py` transcribes a two-column print column by column with a vision language model and
exports the result in the corpus schema. It was written for the Panormitanus volume
(Transkribus document 18098597: layout only, no text lines) and lives here so the main build can
use its output. Working directory is the print folder; the JPGs there are gitignored.

```
D=data/Panormitanus_32108367/Panormitanus_Cons_Print_Venice_1571_v1
export GEMINI_API_KEY=...
python src/gemini_ocr/pipeline.py crop         --data $D            # column crops from page/*.xml -> crops/
python src/gemini_ocr/pipeline.py ocr          --data $D            # transcription -> transkription/<page>_c<n>.txt (resumable)
python src/gemini_ocr/pipeline.py batch-submit --data $D            # same via the Batch API (half price), then batch-collect
python src/gemini_ocr/pipeline.py export-json  --data $D            # -> $D/prebuilt.json (segmentation into consilia, IIIF offset)
python src/gemini_ocr/pipeline.py translate-submit --data $D        # German translation (batch), then translate-collect
python src/gemini_ocr/pipeline.py export-json  --data $D            # again: attaches the translations
python src/build.py --print Panormitanus_Cons_Print_Venice_1571_v1 --no-embed   # import into output/, site-data follows
```

Prompts: `prompt_hauptspalte.txt` (transcription rules: abbreviations expanded in brackets,
`{S}` summarium lines, `{n}` marginal numbers), `prompt_aufloesung.txt` (second pass for residual
abbreviations), `prompt_uebersetzung.txt` (German translation). `prebuilt.json` is read by
`src/import_prebuilt.py` (see its docstring for the schema mapping). State as of 2026-10-17:
all 552 pages transcribed, 226 consilia in two partes, 3 consilia translated.
