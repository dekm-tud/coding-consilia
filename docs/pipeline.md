# Pipeline: from page images to the edition text

This document describes how the texts are made, how a print is added, how the website data are built and
published, and how the result is tested. Annotation layers, embeddings and TEI are in
[annotation.md](annotation.md), and the order of a release is in [../tools/RELEASE.md](../tools/RELEASE.md).

## Adding a print

Prints are organised by author and print: `data/{Author}_{VIAF}/{print_id}/`. The VIAF number in the author
folder links the author to the [Virtual International Authority File](https://viaf.org/).

```
data/
  Baldo_29618397/
    author.json
    Baldo_Cons_Print_Venice_1575_v1/
      page/0001.xml … 0324.xml        (v4 and v5: page_001.xml …)
      mets.xml, metadata.xml
```

Both file name patterns sort correctly. Folders whose names end in `_backup` are skipped. A new author needs an
`author.json`:

```json
{ "viaf": "89597697", "name": "Bartolus de Saxoferrato", "dates": "1313–1357", "note": "Italian jurist, professor at Perugia" }
```

`viaf` (number only) and `name` are required, `dates` and `note` optional. The next `python src/build.py` finds
the new print and processes it.

**Prints segmented outside the pipeline.** If a print folder contains `prebuilt.json`
(`{"meta": …, "consilia": {id: record}}`, written by `src/gemini_ocr/pipeline.py export-json`), the build imports it
through `src/import_prebuilt.py` instead of segmenting PageXML lines. `page/` must still exist, because it gives the
page sizes and region boxes. The Panormitanus volume enters the corpus this way. A `corrections.json` next to it is applied after the spelling rules of the edition. For Panormitanus it is made
by `tools/panormitanus_corrections.py` from the comparison with the kraken reading of the same pages
(`tools/kraken_compare_panormitanus.py`): abbreviation marks left unexpanded, artefacts of the long s and single
words where only the kraken reading is a word of the Baldus text.

**Images.** The repository stores no page images. The website shows them from the holding library or the
Internet Archive through IIIF. `output/iiif_manifests.json` maps each print to its manifest, and `canvas_offset`
or `canvas_map` tells which canvas belongs to local page N. For Internet Archive items the reader builds the
manifest in the browser from the item's static page images, because the Archive's IIIF service is unreliable, and
takes the page sizes from `site-data/image_dims.json`.

## The build

`src/build.py` re-processes only the prints whose files changed and then writes the merged outputs.

```bash
python src/build.py --no-embed                                  # changed prints
python src/build.py --force --no-embed                          # everything
python src/build.py --print Baldo_Cons_Print_Venice_1575_v4 --no-embed
python src/pipeline.py --no-lb-model --no-embed                 # all prints, fast line joining (development)
```

With the Flair line-break model a volume takes about seven minutes on an RTX 3060, while `--no-lb-model` joins lines
by the hyphen sign only and takes seconds. The build keeps consilium ids stable: they carry the translations, keywords
and annotations, and `src/carry_derived.py` re-attaches these layers after every rebuild.

### Steps

1. **PageXML parsing** (`src/parse_pagexml.py`): text regions and lines with text, polygon, baseline and the
   hyphen flag `¬`. Running headers are skipped.
2. **Layout correction** (`src/fix_layout.py`): a tall printed initial made Transkribus merge the initial with the
   second line of the opening and leave the rest as a short separate line. The script finds such pairs
   geometrically (truncated line narrower than 15 % of the column, merged line within 10 % of the left margin,
   vertical overlap), splits the merged line at the right character and reorders the pieces. It edits the XML in
   place, and `--show` lists the pairs without writing. It was applied to v1 (309 pairs), v4 (30) and v5 (10).
3. **Segmentation** (`src/segment_consilia.py`): the Baldus PageXML carries a structure type on every line
   (`cons_title`, `cons_argument`, `cons_body`), so `AnnotatedConsiliumSegmenter` starts a consilium at each title
   line, groups the argument lines into numbered items and splits the body at every `†`. Prints without these
   types go through `ConsiliumSegmenter`, a state machine on heading patterns (CONSILIVM with OCR variants and
   Roman numerals), body openers (*In Christi nomine*, *Ad evidentiam*, `†` …) and geometry, with a repair of
   out-of-sequence numerals.
4. **Line joining** (`src/lb_detector.py`): the Flair model `mschonhardt/latin-contextual-lb-detector` decides for
   every line break whether the words join or not (labels `WB` for join, `NB` for space), and `--no-lb-model`
   joins only at `¬`. A word list (`src/codebooks/word_freq.json`) guards joins without a hyphen.
5. **Misplaced casus** (`src/casus.py`): the print sets the statement of facts (*CASVS. Punctus talis est …*)
   before the heading of the consilium it belongs to, after the closing signature of the previous one. The block
   is moved to the following consilium when its text is clearly closer to it (TF-IDF against both neighbours), and
   undecided cases are settled in `src/casus_overrides.json`. The ids do not change, because only text moves
   between neighbours. Translations are split at the translated casus label, and a piece that cannot be split
   safely is left empty and listed in `output/debug/casus_retranslate.json`.

### Output format

`output/consilia.json` holds one object per consilium, keyed by id:

```json
{
  "consilia": {
    "consilium-baldo_cons_print_venice_1575_v1-44": {
      "id": "consilium-baldo_cons_print_venice_1575_v1-44",
      "n": 44, "roman": "XLIIII",
      "volume": "Baldo_Cons_Print_Venice_1575_v1", "author_viaf": "Baldo_29618397",
      "title": "CONSILIVM XLIIII.",
      "summary": "1 Literae principis non sunt extensibiles…",
      "argument": [{"n": 1, "text": "Literae principis non sunt extensibiles…"}],
      "casus": [],
      "body": ["In Christi nomine. Dominus Bailardinus…", "† Secunda quaestio est…"],
      "body_meta": [{"marginal": null}],
      "sources": [{"page": "0030.xml", "regions": [{"region_id": "r1", "type": "column_1", "coords": [[0, 0]]}]}],
      "translations": {"en": {"argument": "…", "casus": [], "body": ["…"]}, "de": {}},
      "keywords_final": ["dowry", "…"]
    }
  }
}
```

| Field | Content |
|---|---|
| `id` | `consilium-{print id in lower case}-{n}` |
| `n`, `roman` | number as Arabic numeral and as printed |
| `title` | heading as printed |
| `argument` | the numbered summary items (`summary` is the same as one string) |
| `casus` | the statement of facts, split at `†`, or an empty list when there is none |
| `body` | the text, split at every `†`, while `body_meta` holds the printed marginal number per section |
| `sources` | pages and regions the consilium comes from |
| `translations` | machine translations (English, German), section by section |
| `keywords_final` | machine keywords (see [annotation.md](annotation.md)) |

## Text quality (Baldus)

The Baldus text was first recognised with a PyLaia model in Transkribus. Three further steps decided it line by
line:

1. **Second reading with Gemini.** `tools/gemini_recheck.sh crop|submit|collect|compare` transcribes every column
   with Gemini through the Batch API (prompt `src/gemini_ocr/prompt_baldus.txt`, results in
   `output/ocr_gemini/<print>/`), and `src/qa/compare_ocr.py` aligns it line by line with the PageXML.
   `src/qa/relayout_from_gemini.py` then rebuilt the line structure of every column from the Gemini lines (missing
   lines added with interpolated geometry, merged lines split), keeping the PageXML geometry.
2. **Third reading with kraken.** A kraken model trained on the lines where both readings agree
   (`tools/kraken_export.py`, `tools/kraken_read_all.py`) gives a third, independent reading.
3. **Decision.** `tools/final_text.py` takes, per line, the wording on which two of the three readings agree and
   keeps the earlier text where there is no majority, while hand decisions always take precedence. The rules were measured against a
   gold standard of 429 lines checked against the images (`tools/gold_server.py`, `tools/gold_eval.py`,
   `output/gold/`), where they made 73 lines better and none worse and reached a character error rate of 0.073 %.
   `tools/apply_final_text.py` writes the result into the PageXML, with the edition text as TextEquiv index 0 and
   the Gemini, PyLaia and kraken readings as indexes 1, 2 and 3.

Remaining disagreements were reviewed against the page images: lines differing by 10 % or more in a first round
(`tools/agent_review_batches.py`), lines differing by 5–10 % in a second (`tools/agent_review_final.py`, 317 lines
corrected). Every reviewed line carries `reviewed <date>` in its PAGE comments. `tools/review_server.py` is a local
review tool for the same lines.

The edition text expands abbreviations without brackets, writes `s` for long s and `et` for `&`, keeps `u/v`,
`i/j` and `ę` as printed, drops accents and marks a hyphenated line end with `¬`.

## Quality checks

| Script | Checks | Report |
|---|---|---|
| `src/qa/check_layout.py` | PageXML: regions, columns, overlaps, untyped lines, heading sequence | `output/qa/layout_summary.md` |
| `src/qa/check_lines.py` | line rhythm inside each column: gaps, merged or missing lines | `output/qa/lines_summary.md` |
| `src/qa/check_data.py` | schema and consistency of `output/` and `site-data/`, numbering, translation alignment, embedding revisions | `output/qa/data_report.md` (hard failures end with exit code 1) |

## Website data

The website never loads `output/consilia.json`. `src/build_site_data.py` writes what it needs to `site-data/`:

| File | Content |
|---|---|
| `index.json` | metadata per consilium as an array whose index `i` the embeddings refer to, pages and region boxes |
| `text/{print}.json` | Latin argument, casus and body, loaded per print |
| `tr/{en,de}/{print}.json` | machine translations, used by the search and the Ask page and not shown in the reader |
| `annotations/` | subject layers and citations (see [annotation.md](annotation.md)) |
| `lemmas.json` | form → lemma list of the full-text search |
| `manifest.json` | data revision (hash of `index.json`), counts, file sizes |
| `models.json` | embedding models and where their vectors are |
| `editions.json`, `image_sources.json`, `image_dims.json` | hand-maintained descriptions of the witnesses, image attribution, page sizes |

The embeddings must belong to the same data revision as `site-data/`, because the search page refuses vectors of
another revision. After a change of the text, run `src/build_site_data.py` and then refresh the vectors
(see [annotation.md](annotation.md#embeddings)).

### Pages of the website

Eight static pages without a backend. Shared code is in `assets/common.js`, styles in `assets/app.css` (font Source
Sans 3, self-hosted). Permalinks and citations use the canonical address in `site-data/editions.json`.

| Page | Scripts | Content |
|---|---|---|
| Reader `index.html` | `reader-app.js`, `viewer.js` | one print at a time: index, text with casus, argument, body and keywords, facsimile in Mirador 4 with the consilium's regions, permalink, citation, error report. Deep links `?id=`, `?vol=&n=`, `&q=` |
| Search `search.html` | `search-app.js`, `search-worker.js`, `embed-worker.js` | full text (word forms and lemmas), semantic search (query embedded in the browser), similar passages, subjects and citations, with all state kept in the URL |
| Ask `chat.html` | `chat-app.js` | an assistant driven by a chat API the reader provides (Anthropic, OpenAI, Gemini or any OpenAI-compatible endpoint), which calls search tools that run in the browser and answers with links to the consilia |
| `introduction.html`, `baldus.html`, `panormitanus.html` | `essay.js` | introduction to the genre and the authors, English and German |
| `bibliography.html` | `bibliography.js` | the bibliography with search, subject filters and download (see [annotation.md](annotation.md#bibliography)) |
| `about.html` | | method, status, citation, licences |

### Deployment

`.github/workflows/pages.yml` builds and publishes the site on every push to `main`, and `tools/build_pages_artifact.py`
collects the published files and stops when `site-data/` is stale or the artifact is too large.
`.github/workflows/check.yml` runs the same checks on pull requests without publishing.

While the portal is shown to colleagues before its release, every page asks for a password (`assets/gate.js`), and
`noindex` and `robots.txt` keep search engines out. `python tools/set_site_password.py` sets the password and writes
only a salted PBKDF2 hash to `assets/gate-config.js`, and `--off` removes the gate again. The gate keeps casual
visitors out, but it does not protect the data, which are public in this repository.

## Tests

```bash
pip install pytest
python -m pytest tests/test_casus.py
npx -y -p @playwright/test playwright test -c tests/playwright.config.js
```

`tests/test_casus.py` checks that the casus correction keeps ids and numbering, loses or duplicates no text and
keeps translations aligned. `tests/smoke.spec.js` covers the reader (deep links, casus, Latin text only, facsimile),
the search (full text with lemmas, exact forms in quotation marks, semantic and similar search, URL state), the
preview gate and the about and chat pages, and some of the tests need network access. `tests/models.spec.js` covers the
browser query models.

## Debugging

`src/debug_visualize.py --volume <print>` draws the detected headings and openings on the page images
(`output/debug/<print>/`). `src/debug_consilium_list.py --out list.csv` lists all consilia with their first and last
page for comparison with the print.
