<!-- For a new collection, see README.md, "Contributing a collection". -->

**What this adds or changes**

**Checklist for a new collection**
- [ ] `data/<Author>_<VIAF>/author.json` and `data/<Author>_<VIAF>/<print_id>/page/*.xml` (or `prebuilt.json`), no images in git
- [ ] `output/iiif_manifests.json` and `site-data/image_sources.json`: the IIIF manifest and the page mapping (`canvas_offset` / `canvas_map`)
- [ ] `site-data/editions.json` describes the witness (title, imprint, copy, catalogue link) and has an `authors` entry with a short text in English, optionally also in German
- [ ] `python src/build.py --no-embed`, `python src/qa/check_layout.py`, `python src/qa/check_data.py` and `python src/build_site_data.py` run without hard errors, and output and site-data are committed
- [ ] the reader shows the new consilia with the right page images (`python -m http.server`, then `index.html?vol=<print_id>`)
- [ ] the transcription can be published under CC BY 4.0, and the source of the text and its checking are described above
