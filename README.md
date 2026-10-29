# Consilia

A digital edition of late medieval legal opinions (*consilia*) from early modern prints. This repository holds
the pipeline that turns page scans and PageXML into structured texts, the static website built on them, the
annotation layers, and the tools that produced the published datasets and models.

Website: <https://dekm-tud.github.io/coding-consilia/> · Licences: code MIT (`LICENSE`), texts and data
CC BY 4.0 (`LICENSE-DATA`) · How to cite: `CITATION.cff`

Developed in the project *Embedding the Past* (LOEWE-Exploration, TU Darmstadt, AdWL Mainz), funded by HMWK
Hessen. The edition is in beta, and its texts are machine transcriptions that were checked against the page
images in part but have not been proofread.

## Corpus

2,740 consilia in six volumes (the data revision is in `site-data/manifest.json`).

| Print | Pages | Consilia | Text | Images |
|---|---|---|---|---|
| Baldus de Ubaldis, *Consilia*, Venice 1575, vol. 1 to 5 (`Baldo_Cons_Print_Venice_1575_v1` … `_v5`) | 1,392 | 2,514 | PageXML, with each line decided from three readings (PyLaia, kraken, Gemini) | Internet Archive (copy of the Universidad Complutense de Madrid) |
| Panormitanus, *Consilia*, Venice 1571 (`Panormitanus_Cons_Print_Venice_1571_v1`) | 552 | 226 (2 partes) | Gemini column transcription (`prebuilt.json`) | Harvard, Houghton Library (IIIF) |

## Published data and models

| What | Where |
|---|---|
| Baldus PageXML ground truth (layout and text, 194,654 lines) | [mschonhardt/consilia-baldus-venice1575-pagexml](https://huggingface.co/datasets/mschonhardt/consilia-baldus-venice1575-pagexml), v1: [doi:10.57967/hf/11321](https://doi.org/10.57967/hf/11321) |
| kraken layout model | [mschonhardt/consilia-baldus-venice1575-kraken-seg](https://huggingface.co/mschonhardt/consilia-baldus-venice1575-kraken-seg) |
| kraken recognition model | [mschonhardt/consilia-baldus-venice1575-kraken-rec](https://huggingface.co/mschonhardt/consilia-baldus-venice1575-kraken-rec) |
| Passage embeddings of the search page | [dekmATtud/consilia-embeddings](https://huggingface.co/datasets/dekmATtud/consilia-embeddings) |
| Embedding models fine-tuned on Latin legal text | [etp-bgem3-contrastive-ft](https://huggingface.co/mschonhardt/etp-bgem3-contrastive-ft), [etp-canonberta-contrastive-ft](https://huggingface.co/mschonhardt/etp-canonberta-contrastive-ft) and others under [mschonhardt](https://huggingface.co/mschonhardt) |
| Lemmatizer used for the full-text search | [mschonhardt/latin-lemmatizer](https://huggingface.co/mschonhardt/latin-lemmatizer) |
| TEI P5, one file per volume | built by `src/export_tei.py`, with a Zenodo deposit prepared in `dist/zenodo/` |

## Quick start

You need Python 3.10 or later.

```bash
git clone https://github.com/dekm-tud/coding-consilia.git
cd coding-consilia
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/build.py --no-embed        # PageXML and prebuilt.json -> output/consilia.json (only changed prints)
python src/qa/check_data.py           # consistency checks, hard failures stop the CI
python src/build_site_data.py         # output/consilia.json -> site-data/ (what the website loads)
python -m http.server 8765            # then open http://localhost:8765/
```

The website needs no build step and no server code, because every page reads `site-data/`. Models used by the
build (the line-break detector, embedding models) are downloaded from Hugging Face on first use.

Browser tests (Playwright is not a project dependency):

```bash
npx -y -p @playwright/test playwright test -c tests/playwright.config.js
```

## Repository layout

```
data/<Author>_<VIAF>/author.json            author metadata
data/<Author>_<VIAF>/<print>/page/*.xml     PageXML (page images are not stored in the repository)
data/<Author>_<VIAF>/<print>/prebuilt.json  text segmented outside the pipeline (Panormitanus)
src/                                        build, segmentation, annotation, embeddings, TEI export, quality checks
src/qa/                                     audits of the PageXML and of the generated data
src/annotate/                               annotation layers (subjects, body codebook, citations and entities)
src/gemini_ocr/                             column transcription with Gemini
tools/                                      transcription decisions, reviews, kraken and Flair training, dataset
                                            builders, publishing helpers (see tools/RELEASE.md)
output/consilia.json                        the generated edition data, source of the website
site-data/                                  what the website loads: index, texts per print, translations,
                                            annotation layers, lemma list, model registry
index.html, search.html, chat.html, …       the website, with scripts, styles, fonts and Mirador in assets/
eval/                                       gold sets and evaluation reports of the annotation layers
tests/                                      pytest (segmentation invariants) and Playwright (website)
docs/                                       technical documentation
```

## Documentation

- [docs/pipeline.md](docs/pipeline.md): adding a print, the build, segmentation, text quality, website data,
  deployment, tests
- [docs/annotation.md](docs/annotation.md): translations, keywords, subject layers, citations and entities,
  embeddings, lemma search, TEI
- [docs/mcp.md](docs/mcp.md): the MCP server that gives AI assistants access to the corpus
- [tools/RELEASE.md](tools/RELEASE.md): the steps from corrected PageXML to a new data release

## Contributing a collection

The portal can grow with the help of others: anyone who has transcribed consilia, from a print or a manuscript,
can add them through GitHub. The pipeline turns PageXML into the structured, searchable edition, a pull request
brings the result into the portal, and GitHub Pages publishes it after review.

1. **Announce it.** Open an issue with the template *New collection of consilia* (author, witness, IIIF manifest,
   how the text was made, licence). This avoids double work and settles the identifiers before you start.
2. **Fork and branch.** Fork `dekm-tud/coding-consilia`, clone your fork, create a branch and set up the
   environment as under *Quick start*.
3. **Make the text.** Transcribe in Transkribus, eScriptorium or kraken and export PAGE XML. The edition's kraken
   models give a head start on early modern prints: the layout model tags the lines with the classes the pipeline
   needs (`cons_title`, `cons_argument`, `cons_body`, `header`, …), and the recognition model reads the type:

   ```bash
   kraken -i page.jpg page.xml -x segment -bl -i consilia_seg.mlmodel ocr -m consilia_rec.mlmodel
   ```

   Check layout and text by hand, then place the files as described in
   [docs/pipeline.md](docs/pipeline.md#adding-a-print): `data/<Name>_<VIAF>/author.json` and
   `data/<Name>_<VIAF>/<print>/page/*.xml`. Text segmented by other means can come in as `prebuilt.json`.
4. **Point to the images.** The repository stores no page images. Add the holding library's IIIF manifest to
   `output/iiif_manifests.json` (with `canvas_offset` or `canvas_map`, so that local page N finds its canvas) and
   describe the witness in `site-data/editions.json` (title, imprint, copy, catalogue link, a short author text).
5. **Build and check.**

   ```bash
   python src/build.py --no-embed           # PageXML -> output/consilia.json
   python src/qa/check_layout.py            # PageXML problems (regions, line classes, reading order)
   python src/qa/check_data.py              # the hard checks must pass
   python src/build_site_data.py            # output/ -> site-data/
   python -m http.server 8765               # read the new print at http://localhost:8765/index.html?vol=<print>
   ```

   Translations, keywords, subject layers, citations and embeddings need GPUs or API keys. They are optional for
   a contribution, and the maintainers can run them after the merge.
6. **Open a pull request** against `main` and fill in the checklist of the template. The workflow *Check data*
   rebuilds `site-data/` and runs the consistency checks and the size limits of the website on every pull
   request. After review and merge the portal is rebuilt and published.

Contributed transcriptions are published under CC BY 4.0 (`LICENSE-DATA`), so please add only text that you may
publish under this licence. Contributors are named in the edition's metadata (`site-data/editions.json`) and in `CITATION.cff`.

## Licence and citation

Code: MIT (`LICENSE`). Texts, translations, annotations and other data: CC BY 4.0 (`LICENSE-DATA`). The page
images belong to the holding libraries and are only linked. Please cite the edition as given in `CITATION.cff`
and name the data revision, which is recorded in `site-data/manifest.json` and in every TEI file.
