# Annotation layers, embeddings and exports

Everything in this document is added to the edition text after the build (see [pipeline.md](pipeline.md)). The
layers are machine-made, and the website labels them as such. Gold sets in `eval/gold/` were annotated by a
language model on the editor's behalf, which is why the published evaluation numbers measure agreement with a
model and not with a human annotator.

## Translations and keywords

The English and German translations and the keywords come from a pipeline that runs a Qwen3 model on a GPU server
with [vLLM](https://github.com/vllm-project/vllm), while the preparation and merging steps run locally.

```bash
python src/prepare_llm_input.py                 # output/llm_input.jsonl, one record per consilium
python src/gpu/qwen_process.py --input llm_input.jsonl --output llm_output.jsonl     # GPU server
python src/merge_llm_output.py                  # translations and keywords_raw into the edition data
python src/refine_keywords.py                   # cluster the raw keywords with BGE-M3
python src/gpu/name_clusters.py --input keyword_clusters.json --output keyword_clusters_named.json   # GPU server
python src/apply_keyword_mapping.py             # canonical labels per consilium
python src/build_vocabulary.py                  # output/vocabulary.json
python src/prepare_keyword_reduction_input.py   # candidates for the final selection
python src/apply_keyword_reduction.py           # keywords_final, after src/gpu/reduce_keywords.py on the server
```

The translations are not shown in the reader, because their quality is not sufficient for reading alongside the Latin, but the search and the Ask page use them. They can be added to the TEI export with `--translations`. `qwen_process.py` translates first and then extracts keywords from its own English translation, because the model
reads clean English more reliably than sixteenth-century legal Latin. It writes after every batch, so an interrupted
run continues where it stopped. The raw keywords are merged into one vocabulary by clustering their embeddings and
letting the model name each cluster.

The vocabulary is curated by exclusion and not by frequency. Every candidate label was read and either kept or
listed in `src/vocabulary_exclusions.json` (names of cited jurists, places and source titles, vague words, Latin
fragments, near-duplicates). A frequency threshold had been tried first, but it dropped rare and meaningful topics
such as *impotence* or *incest*, which occur in few consilia but are real legal questions. The vocabulary has
3,463 terms. When new volumes are added, `python src/build_vocabulary.py --list-candidates` shows new labels that
contain names, and the remaining new labels need another reading by hand.

`keywords_final` holds about ten keywords per consilium. `reduce_keywords.py` selects them from the consilium's own
candidates in two passes, one for its central themes and one for distinctive minor topics, and every label it
returns must come from that candidate list. A selection from the whole vocabulary by embedding similarity was
tested and dropped, because it missed topics that are mentioned only briefly in a long text.

Translations that no longer match their Latin paragraphs after a text correction can be redone with
`tools/retranslate_stale.py`.

## Subject index and body codebook

Two closed schemes complement the free machine keywords. The subject index (`src/codebooks/subjects.yaml`) assigns
concepts in facets to each consilium, and the codebook on body and sexuality (`src/codebooks/intimate_body.yaml`)
marks text units with a code, a role and a Latin evidence quote. The labels exist in English, German and Latin, and
the prompts use the English and Latin ones.

```bash
python src/prepare_units.py                     # output/units.jsonl, one unit per section (long sections in windows)
python src/annotate/classify.py --task body --backend openai --base-url http://<server>:8000/v1 --model Qwen/Qwen3.6-27B --thinking
python src/annotate/classify.py --task subjects --backend gemini --model gemini-3.8-flash --split dev
python src/annotate/evaluate.py --task body --split test output/annotations/body_*_p1.jsonl
python src/annotate/export.py --body output/annotations/body_<model>_p1.jsonl --subjects output/annotations/subjects_<model>_p1.jsonl
python src/build_site_data.py
```

The backends are any OpenAI-compatible chat API (for example a vLLM server) or the Gemini API, both with JSON
schema output. Evidence quotes are checked against the unit text, and a second pass (`--pass 2`) marks the
assignments on which two runs agree.

## Citations and entities

`src/codebooks/spans.yaml` defines spans for citations of the Corpus iuris civilis, the Corpus iuris canonici and
the Libri feudorum, and for jurists, persons, places, institutions, statutes and dates. `src/annotate/spans.py` asks
a language model to mark them as verbatim quotes and locates them in the text by character offsets.

The current layer was made in two steps. Gemini 4 Argon marked all 10,322 text units through the Batch API
(`tools/gemini_batch_spans.py`, runs merged by `tools/merge_span_runs.py`), and a Flair tagger trained on that
annotation gives the second vote (`tools/train_flair_ner.py`, MediBerta fine-tuned and stacked with Latin legal
Flair language models). The export keeps spans on which both agree as confirmed, keeps spans found only by Gemini
as unconfirmed, and drops citations found only by the tagger, because a sample check showed most of them to be
wrong.

```bash
python src/annotate/export_flair.py output/annotations/spans_gemini-4-argon_merged.jsonl --units output/units_all_<date>.jsonl
python tools/train_flair_ner.py                 # GPU, see tools/flair_bundle2/ for the server bundle
python tools/train_flair_ner.py --predict --units output/units_all_<date>.jsonl --predict-out output/annotations/spans_flair_<date>.jsonl
python src/annotate/export.py --spans <older runs> --patch output/annotations/spans_gemini-4-argon_merged.jsonl \
    output/annotations/spans_flair_<date>.jsonl --patch-units output/units_all_<date>.jsonl
python src/build_site_data.py
```

The loci of the citations (for example `D. 28.2.29`) come from the rule-based resolver
`src/annotate/resolve_citations.py`, which uses title tables and the conventions of the allegations and lets a
citation of a lex or capitulum alone inherit the title of the citation directly before it. For citations the rules
cannot place, two language models propose a locus independently (`tools/resolve_llm.py`), and a proposal is kept
only when both agree (`eval/reviews/citation_resolutions.jsonl`).

## Embeddings

The search page embeds the query in the browser and compares it with precomputed vectors of every body and casus
section. The vectors cover the Latin text and the English translation. The same sections are embedded with every model, so a row means the same passage
in every model directory, and a consilium scores with the best of its sections, which keeps short mentions in long
texts findable.

```bash
python src/gpu/embed_web_models.py --tags bge-m3 etp-bgem3-ft qwen3-0.6b    # local GPU (12 GB)
python src/gpu/embed_web_models.py --tags qwen3-4b --batch-size 16          # larger GPU
python src/gpu/embed_update.py --before <consilia.json of the last full run>  # only the sections whose text changed
python tools/upload_embeddings_hf.py --bump-models                          # dataset release and site-data/models.json
```

Each model directory in `hf-export/` holds `chunks.f16.bin` (float16, one row per section) and `chunks.json`
(document index, language, kind and the data revision). The vectors must carry the data revision of
`site-data/manifest.json`, otherwise the search page refuses them, so they have to be refreshed after every change
of the text. `upload_embeddings_hf.py` publishes them to
[dekmATtud/consilia-embeddings](https://huggingface.co/datasets/dekmATtud/consilia-embeddings) under a release tag
and points `site-data/models.json` to that tag.

A new model is added in `MODELS` of `src/gpu/embed_web_models.py` (or embedded through an OpenAI-compatible API
with `src/embed_api.py`) and registered in `site-data/models.json` with its tag, label, dimensions, vector sources
and either a browser query model (`onnx_repo`, checked with `tools/parity_ref.py`) or a `query_api`.
`src/eval/eval_menu_models.py` computes the evaluation numbers shown in the model picker.

## Lemma search

The full-text search finds all forms of a word through a lemma list (`site-data/lemmas.json`, form to lemma for the
normalised word forms of the corpus). `tools/lemmatize_corpus.py` builds it with the
[Latin lemmatizer](https://huggingface.co/mschonhardt/latin-lemmatizer) of the project and applies a short list of
hand corrections (`OVERRIDES`), which can be applied to an existing list with `--overrides-only`. In the browser
every word is indexed with its form and a lemma key, a query word is searched as typed and by its lemma, and a word
in quotation marks is searched in that form only.

## TEI and archive packages

`python src/export_tei.py [--translations] [--doi …]` writes one TEI P5 file per print to `dist/tei/`. The header
records the editor, the responsibilities including the recognition models, the licence, the identifiers, the print
and its digitised copies (from `site-data/editions.json`), the editorial declarations and the classification
schemes. A `<facsimile>` lists every page with its IIIF canvas or image URL, and each consilium has its casus, head,
argument list, sections with marginal numbers, citations and entities as `ref` and `rs` elements, and notes with
keywords and subjects. `python tools/make_packages.py` builds archive packages per print (`dist/packages/`) and a
folder for a Zenodo deposit (`dist/zenodo/`). After the deposit the export is repeated with `--doi`, so that the
header carries the DOI.

## Bibliography

`bibliography.html` lists the research literature on legal and medical consilia and the sources cited on the essay
pages. `python tools/build_bibliography.py` reads the editor's Zotero collection "Consilia" through Zotero's local
API (or from a saved export with `--from-file`) and takes the titles tagged `consilia_legal`, `consilia_med` or
`consilia_general` together with every work cited in the footnotes. It writes `site-data/bibliography.json` and
`site-data/bibliography.ris`, and Zotero itself is not changed.

Three files in `src/codebooks/` complete the Zotero data. `bibliography_overrides.json` holds the fields that were
missing in Zotero (editors, publishers, places) and corrections of wrong records, each with the catalogue page that
confirms it, the duplicates to merge and the cited works that are not in Zotero. `bibliography_item_tags.json` gives
every title its subject tags from the closed scheme in `bibliography_tags.yaml`, which replaces the automatic Zotero
tags. `bibliography_citations.json` records which footnote cites which entry, and `python tools/link_footnotes.py`
adds the links from the footnotes to the entries.
