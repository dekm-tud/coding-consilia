"""
Shard output/consilia.json into what the static site actually fetches.

    site-data/
      index.json               metadata only, one entry per consilium (array; `i` = stable doc index)
      text/{print}.json        {id: {"argument": [{n,text}], "casus": [...], "body": [...]}}   (Latin)
      tr/{lang}/{print}.json   {id: {"argument": "...", "casus": [...], "body": [...]}}         (machine translation)
      authors.json, vocabulary.json, iiif_manifests.json, image_sources.json  (copied / generated)
      manifest.json            data revision, counts, sizes

`keywords_raw` / `keywords` (pre-reduction) stay in output/consilia.json for transparency and are
not shipped to the browser. `i` is the position of the consilium in output/consilia.json; the
compact embedding files (export_embeddings_web.py) refer to it.

Usage:
    python src/build_site_data.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "output"
SITE = ROOT / "site-data"
ARG_SHORT_CHARS = 240



def display_title(c: dict) -> str:
    """Normalised heading for display: "Consilium 12" (arabic number; "pars N" added for prints with
    several partes) instead of the print's "CONSILIVM XII."; the print form stays in `title`. When the
    number could not be read from the print, the print heading is used unchanged."""
    n = c.get("n")
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        return (c.get("title") or "").strip()
    s = f"Consilium {n}"
    if c.get("pars"):
        s += f" (pars {c['pars']})"
    return s

def roman_print(n: int) -> str:
    """Roman numeral in the print's own manner: IIII for 4, IX for 9, XL, XC, CCCC for 400"""
    out = "M" * (n // 1000); n %= 1000
    h = n // 100; out += ("D" + "C" * (h - 5)) if h >= 5 else "C" * h; n %= 100
    out += ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"][n // 10]
    out += ["", "I", "II", "III", "IIII", "V", "VI", "VII", "VIII", "IX"][n % 10]
    return out


def uniform_title(c: dict) -> str:
    """The heading in one form for the site: "CONSILIVM CCXXV" (no periods, no reading residue, the
    numeral written the print's way); the print's own heading stays in output/consilia.json. Without
    a number the print heading is kept, only without periods."""
    try:
        n = int(c.get("n") or 0)
    except (TypeError, ValueError):
        n = 0
    if n > 0:
        return f"CONSILIVM {roman_print(n)}"
    return re.sub(r"\s+", " ", (c.get("title") or "").replace(".", " ")).strip()


def dump(path: Path, obj) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    path.write_bytes(data)
    return len(data)


def bbox(coords: list[list[int]]) -> list[int]:
    xs, ys = [p[0] for p in coords], [p[1] for p in coords]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def pages_of(c: dict, dims: dict) -> list[dict]:
    """Per source page: stem, image size (as declared by the PageXML, the coordinate space
    of the regions) and the region boxes [x, y, w, h]."""
    out = []
    for s in c.get("sources") or []:
        stem = s["page"].replace(".xml", "")
        d = dims.get(c["volume"], {}).get(stem, {})
        out.append({
            "page": stem,
            "w": d.get("xml_w"), "h": d.get("xml_h"),
            "regions": [bbox(r["coords"]) for r in s.get("regions", []) if r.get("coords")],
        })
    return out


_RE_IMG = re.compile(r'imageWidth="(\d+)"\s+imageHeight="(\d+)"')


def with_pagexml_dims(dims: dict, consilia: dict) -> dict:
    """Prints without image derivatives (IIIF prints) get their page sizes from the PageXML."""
    for c in consilia.values():
        v = c["volume"]
        if v in dims:
            continue
        page_dir = ROOT / "data" / c.get("author_viaf", "") / v / "page"
        if not page_dir.is_dir():
            continue
        d = {}
        for xml in page_dir.glob("*.xml"):
            m = _RE_IMG.search(xml.read_text(encoding="utf-8", errors="ignore")[:3000])
            if m:
                d[xml.stem] = {"xml_w": int(m.group(1)), "xml_h": int(m.group(2))}
        dims[v] = d
    return dims


def build(consilia: dict, dims: dict) -> dict:
    index, sizes = [], {}
    text: dict[str, dict] = {}
    tr: dict[str, dict[str, dict]] = {"en": {}, "de": {}}
    for i, (cid, c) in enumerate(consilia.items()):
        v = c["volume"]
        index.append({
            "i": i, "id": cid, "n": c.get("n"), "roman": c.get("roman", ""), "volume": v,
            "author_viaf": c.get("author_viaf", ""), "title": uniform_title(c),
            "title_display": display_title(c),
            "argument_short": (c.get("summary") or "")[:ARG_SHORT_CHARS],
            "keywords_final": c.get("keywords_final") or [],
            "has_casus": bool(c.get("casus")),
            **({"pars": c["pars"]} if c.get("pars") else {}),
            "pages": pages_of(c, dims),
        })
        rec = {"argument": c.get("argument") or [], "casus": c.get("casus") or [], "body": c.get("body") or []}
        marg = [m.get("marginal") for m in (c.get("body_meta") or [])]     # printed marginal numbers (Panormitanus)
        if any(x is not None for x in marg):
            rec["marg"] = marg
        text.setdefault(v, {})[cid] = rec
        for lang in ("en", "de"):
            t = (c.get("translations") or {}).get(lang) or {}
            body_tr = t.get("body") or []
            rec_tr = {"argument": t.get("argument") or "", "casus": t.get("casus") or [], "body": body_tr}
            if body_tr and len(body_tr) != len(c.get("body") or []):
                # the Latin paragraphs changed after the translation was made (text corrections); a
                # paragraph-by-paragraph pairing would show the wrong text, so the body translation is
                # withheld until the piece is retranslated (output/debug/casus_retranslate.json)
                rec_tr["body"] = []
                rec_tr["stale"] = True
            tr[lang].setdefault(v, {})[cid] = rec_tr
    sizes["index.json"] = dump(SITE / "index.json", index)
    for v, d in text.items():
        sizes[f"text/{v}.json"] = dump(SITE / "text" / f"{v}.json", d)
    for lang, by_v in tr.items():
        for v, d in by_v.items():
            sizes[f"tr/{lang}/{v}.json"] = dump(SITE / "tr" / lang / f"{v}.json", d)
    return sizes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.parse_args()
    consilia = json.loads((OUTPUT / "consilia.json").read_text(encoding="utf-8"))["consilia"]
    dims_path = SITE / "image_dims.json"
    dims = json.loads(dims_path.read_text(encoding="utf-8")) if dims_path.exists() else {}
    dims = with_pagexml_dims(dims, consilia)
    if SITE.exists():
        for sub in ("text", "tr"):
            shutil.rmtree(SITE / sub, ignore_errors=True)
    sizes = build(consilia, dims)

    if (SITE / "lemmas.json").exists():          # lemma dictionary of the full-text search (tools/lemmatize_corpus.py)
        sizes["lemmas.json"] = (SITE / "lemmas.json").stat().st_size
    for name in ("authors.json", "vocabulary.json", "iiif_manifests.json"):
        src = OUTPUT / name
        if src.exists():
            shutil.copyfile(src, SITE / name)
            sizes[name] = src.stat().st_size
    for extra in sorted((SITE / "annotations").rglob("*.json")) if (SITE / "annotations").is_dir() else []:
        sizes[str(extra.relative_to(SITE)).replace("\\", "/")] = extra.stat().st_size      # published layers (src/annotate/export.py), incl. references/<print>.json
    src_json = SITE / "image_sources.json"
    if not src_json.exists():
        print("note: site-data/image_sources.json missing (hand-maintained registry, see README)")

    rev = hashlib.sha256((SITE / "index.json").read_bytes()).hexdigest()[:12]
    manifest = {
        "revision": rev, "n_consilia": len(consilia), "status": "work in progress",
        "files": sizes, "total_bytes": sum(sizes.values()),
    }
    dump(SITE / "manifest.json", manifest)
    print(f"{len(consilia)} consilia; {len(sizes)} files; {manifest['total_bytes'] / 1e6:.1f} MB raw; revision {rev}")
    for k, v in sorted(sizes.items(), key=lambda kv: -kv[1])[:6]:
        print(f"  {v / 1e6:6.2f} MB  {k}")


if __name__ == "__main__":
    main()
