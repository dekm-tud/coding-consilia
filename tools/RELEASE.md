# Release pipeline: from corrected PageXML to a citable data revision

Order matters: the data revision (`site-data/manifest.json`) is a hash of the index, so every step
that changes texts, ids, keywords or translations changes it, and the published embeddings must be
rebuilt afterwards (the search page refuses vectors of another revision). Do everything below, then
embed once.

## 1. Rebuild the texts

Corrections already in the PageXML (catchwords typed `catch-word` or trimmed, `tools/fix_catchwords.py`)
and in the segmenter (misprinted headings `CONISLIVM`, `CONSILIM`, prefixes `Super eodem.`, `Se summat.`)
take effect with a rebuild:

    python src/build.py --no-embed --force          # all prints; ~7 min per volume with the Flair line-break model
    python tools/recover_derived.py                 # translations and keywords of the renumbered consilia, matched by text
    python tools/restore_translations.py            # translations a forced rebuild dropped although the Latin did not change (prebuilt prints)
    python src/build.py --no-embed                  # merge only (nothing stale)
    python src/qa/check_data.py

Expected: the nine ids `…-unknown-N` become numbered consilia (v2-267, v4-142, v4-301, v4-311, v4-324,
v4-335, v5-70, v5-490; `v5-unknown-2`, a heading without a numeral on page_223, stays), and the doubled
words at page breaks disappear. Derived layers (translations, keywords) are carried over by id
(`src/carry_derived.py`); the renumbered consilia get theirs back from the last commit through
`tools/recover_derived.py` (text similarity), so step 2 only has to fill what never existed.

## 2. Fill the gaps (language model, once)

    python tools/fill_gaps.py plan --price-in <USD/M input> --price-out <USD/M output>

As of 2026-10-24 the gaps are: English argumentum for Panormitanus (225) and 38 Baldus consilia, English
and German casus for 104 Panormitanus consilia, keywords for Panormitanus (226) and 25 Baldus consilia,
Layer A subjects for 94 consilia. About 1.1 M input and 0.1 M output tokens in all; with a Flash-class
Gemini model well under 5 USD, with Qwen on the GPU server free.

    export GEMINI_API_KEY=…
    python tools/fill_gaps.py translate --backend gemini --model gemini-3.8-flash --workers 4
    python tools/fill_gaps.py keywords  --backend gemini --model gemini-3.8-flash --workers 4
    python tools/fill_gaps.py apply                 # writes consilia.json and the per-author/per-print copies

or with Qwen (vLLM, OpenAI-compatible): `--backend openai --base-url http://<gpu>:8000/v1 --model Qwen/Qwen3.6-27B --thinking --workers 16`.

Subjects (Layer A) use the existing classifier; `plan` prints the exact commands (ids in
`output/gaps/subjects_missing_ids.txt`, then `export.py --subjects` on the concatenated runs). The three
few-shot examples of the classifier are never classified; their gold assignments are in
`output/annotations/subjects_gold_shots.jsonl` and go into the concatenation as well.

Done on 2026-10-24 with gemini-3.8-flash: 272 translation requests, 257 keyword sets, 92 subject
assignments; about 2.6 M input and 0.7 M output tokens in all (thinking included), far below 5 USD.

Body layer (Layer B) and citations (Layer C) exist for both prints already. After a rebuild the
citation layer is re-exported from the stored runs (nothing to pay):

    python src/annotate/export.py --spans output/annotations/spans_qwen-qwen3-8-27b-fp8_p1.jsonl output/annotations/spans_flair.jsonl

## 3. Site data, embeddings, TEI

    python src/build_site_data.py
    python src/qa/check_data.py --quiet
    python src/gpu/embed_web_models.py --tags bge-m3 etp-bgem3-ft qwen3-0.6b       # local 12 GB GPU, ~7 min per model
    python src/gpu/embed_web_models.py --tags qwen3-4b --batch-size 8               # GPU server (or try locally)
    python tools/upload_embeddings_hf.py --tag v0.2 --bump-models                   # needs `hf auth login`; publishes
    python src/export_tei.py && python tools/make_packages.py                        # dist/tei, dist/zenodo
    npx -y -p @playwright/test playwright test -c tests/playwright.config.js        # smoke tests

Validate the TEI against tei_all before a deposit (lxml, ~6 s per file):

    curl -sLo /tmp/tei_all.rng https://www.tei-c.org/release/xml/tei/custom/schema/relaxng/tei_all.rng
    python -c "from lxml import etree,glob; r=etree.RelaxNG(etree.parse('/tmp/tei_all.rng')); [print(f, r.validate(etree.parse(f))) for f in glob.glob('dist/tei/*.xml')]"

## 4. Zenodo

Deposit `dist/zenodo/` (TEI files, README, CITATION.cff, checksums) as a dataset, CC BY 4.0. Reserve the
DOI before publishing, run `python src/export_tei.py --doi 10.5281/zenodo.NNNN && python tools/make_packages.py`,
replace the files in the draft, publish. Put the concept DOI into `CITATION.cff` (`doi:`) and into
`site-data/editions.json` → `portal.doi`. Every later data revision is a new version of the same concept DOI.

## Known limits that stay

- Citation resolution reaches about 53 % of the marked citations (58 % in vol. 1). Half of the rest are bare
  lex references ("l. si filius.", "l. fi.") that name no title; resolving them needs an incipit table of the
  Digest and Codex leges, not a model. The other half are OCR-damaged sigla and Decretum/Authenticum forms.
- Translations are machine-made reading aids; the TEI deposit does not include them (`--translations` would).
- Doubled words that stand in the print ("domini domini", "qualis qualis", "culpa culpa") are kept as printed.

## 5. Ground truth and kraken models (Hugging Face)

The Baldus PageXML (layout and final text) is published as one dataset, and two kraken models are trained on
it; the Panormitanus volume has no line-level ground truth and is not part of it.

    python tools/ia_image_map.py                    # output/ia_images.json: page -> original file in the Internet
                                                    # Archive item (checked by size; the files are the same bytes)
    python tools/make_pagexml_dataset.py            # dist/hf-consilia-baldus-venice1575-pagexml/ (clean PAGE XML,
                                                    # lines.parquet, splits, pages.csv with image URL + SHA-1,
                                                    # fetch_images.py; no images: Google scans are not re-hosted)
                                                    # + output/kraken_final/ lists (images hard-linked there)
    bash tools/make_kraken_bundle2.sh               # dist/kraken-train2.tar (~6 GB) + run_kraken2.sh for a GPU server
                                                    # with a job queue: trains rec, seg_baldus, seg_stg in parallel,
                                                    # tests, converts, packs kraken-models2.tar.gz (tools/kraken_bundle2/)
    bash tools/upload_kraken_hf.sh                  # prints the `hf upload` commands

Done 2026-10-27: the server run is in `dist/kraken_run_gpu/` (models, logs); model files, logs and the training
script are copied into the card folders and the cards are filled. `tools/kraken_train_final.sh` is the local
(RTX 3060) equivalent; the local recognition training crashed twice with a CUDA error under WSL.

Two kraken 7.1.1 problems to keep in mind: (1) `--resize new` stores an inference class mapping that does not
match the network; `tools/kraken_fix_segmap.py` writes the trained mapping into a copy (the published
safetensors/mlmodel are repaired copies). (2) `ketos segtest` dies in its per-class table (IndexError);
`tools/kraken_segtest.py` wraps it. Also: kraken reads the PAGE `@type` of a region before the structure
type, so the dataset builders give the columns no `@type` (otherwise both columns train as one class and
are predicted as one block).

The v1 recognition model (output/kraken/model/consilia_e31.mlmodel, Zenodo package in dist/zenodo-kraken/) is a
bounding-box model trained on dilated crops; on kraken's polygon lines it reads at 29 % and must not be used
in a `segment -bl … ocr` pipeline. The v2 model is a baselines model for exactly that.

## 6. St. Georgen 48 layout ground truth and model (Hugging Face, no images)

Source: the Transkribus export in `dist/hf-censilia-med-1466/` (not for publication: it contains the BLB images and
uncorrected HTR text). `python tools/make_stgeorgen48_dataset.py` builds `dist/hf-blb-st-georgen-48-pagexml/` (PageXML
without images and text, line types inherited from regions) and `output/kraken_stgeorgen48/` (training pages with images,
region-only pages excluded). The layout model was trained on the GPU server in the same bundle (job `seg_stg`;
`./tools/kraken_train_stgeorgen48.sh all` is the local equivalent). Card and model files are in
`dist/hf-blb-st-georgen-48-kraken-seg/`. The user decided to publish dataset and model only after the second hand
(images 394 to 486) is line-annotated; date "15th century" and the scribe Widmann are confirmed.
