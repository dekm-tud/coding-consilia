"""
Import a print whose consilia were segmented outside the PageXML pipeline.

A print directory may contain `prebuilt.json` ({"meta": {...}, "consilia": {id: record}}) as
written by src/gemini_ocr/pipeline.py export-json (Gemini column transcription of the
Panormitanus print). pipeline.process_print() uses it instead of segmenting PageXML lines when
the file exists. This module normalises such records to the corpus schema (README "Output
format"):

  n                int (the Gemini export writes it as int already; strings are converted)
  pars             kept (part of a multi-part print; ids carry -p<pars>-<n>)
  casus            str | None  -> list of sections split at "†"
  summary          str         -> kept, plus `argument` = [{n, text}] parsed from "1 … 2 …"
  body_de/casus_de/summary_de  -> translations.de.{body, casus, argument}; missing sections ""
  translation_complete, roman_print  -> dropped / kept as `roman_print`
  sources          kept (regions may have coords=None; canvas/image/folio from the IIIF manifest
                   are kept so the reader can address the IIIF image directly)
  body_meta        kept (marginal number and first/last line id per section)
"""

from __future__ import annotations

import collections
import html
import json
import re
import unicodedata
from pathlib import Path

_ARG_ITEM = re.compile(r"(?:(?<=^)|(?<=\s))(\d{1,3})\s+(?=\S)")


def split_argument(summary: str) -> list[dict]:
    """'1 Sponsus … 2 Matrimonium …' -> [{n:1,text:…},{n:2,text:…}]; numbers must ascend."""
    s = (summary or "").strip()
    if not s:
        return []
    items, expected, last_end, cur_n = [], 1, None, None
    for m in _ARG_ITEM.finditer(s):
        n = int(m.group(1))
        if n != expected:
            continue
        if cur_n is not None:
            items.append({"n": cur_n, "text": s[last_end:m.start()].strip()})
        cur_n, last_end, expected = n, m.end(), n + 1
    if cur_n is None:
        return [{"n": 1, "text": s}]
    items.append({"n": cur_n, "text": s[last_end:].strip()})
    return items


def split_sections(text: str | None) -> list[str]:
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"\s*†\s*", text)]
    parts = [p for p in parts if p]
    return [parts[0]] + ["† " + p for p in parts[1:]] if parts else []


_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# The Latin text in the edition's form, as in the Baldus volumes (tools/final_text.py, gold_eval.clean): "et" for
# "&", "ae"/"oe" for the ligatures, no accents (ę stays), and the glyphs the column transcription used for long s
# (ű, Ź, ż, ž), the ct and st ligatures (ĉ, ţ) and dotless i (ı) replaced by plain letters. Abbreviations with
# a tilde (nõ, tamẽ) depend on the word and are expanded through the corrections of the print (apply_corrections).
_FORM = str.maketrans({"æ": "ae", "Æ": "Ae", "œ": "oe", "Œ": "Oe", "ű": "s", "Ź": "s", "ż": "s", "ž": "s",
                       "ĉ": "c", "ţ": "t", "ı": "i", "à": "a", "è": "e", "ì": "i", "ò": "o", "ù": "u",
                       "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "À": "A", "È": "E", "Ò": "O", "\u0301": "", "\u0300": "", "¶": "†"})


def edition_form(t: str) -> str:
    if not isinstance(t, str):
        return t
    t = re.sub(r"&#?\w+;", lambda m: html.unescape(m.group(0)), t)    # HTML entities the transcription wrote (&aelig;)
    t = t.translate(_FORM)
    return re.sub(r"&", "et", t)


def apply_corrections(t: str, corr: dict | None, places: list | None = None) -> str:
    """Corrections of a print (corrections.json next to prebuilt.json, see tools/panormitanus_corrections.py):
    word forms replaced wherever they stand, then single places given as old and new text."""
    if not isinstance(t, str) or not corr:
        return t
    words = corr.get("words") or {}
    if words:
        keys = sorted(words, key=len, reverse=True)
        rx = re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in keys) + r")(?!\w)")
        t = rx.sub(lambda m: words[m.group(1)], unicodedata.normalize("NFC", t))
    for p in places or []:
        if t.count(p["old"]) == 1:
            t = t.replace(p["old"], p["new"])
    return t


def _clean(v):
    """Strip control characters the OCR model occasionally emits (\\x00/\\x10 found in 3 consilia)."""
    if isinstance(v, str):
        return _CTRL.sub("", v)
    if isinstance(v, list):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    return v


def normalise(rec: dict, corr: dict | None = None, cid: str | None = None) -> dict:
    c = _clean(dict(rec))
    places = collections.defaultdict(list)
    for p in (corr or {}).get("places", []):
        if p["id"] == cid:
            places[(p["field"], p.get("index"))].append(p)
    for k in ("title", "summary", "casus"):
        c[k] = apply_corrections(edition_form(c.get(k)), corr, places[(k, None)])
    c["body"] = [apply_corrections(edition_form(t), corr, places[("body", i)]) for i, t in enumerate(c.get("body") or [])]
    c["n"] = int(str(c.get("n", 0)).strip() or 0)
    c["casus"] = split_sections(c.get("casus"))
    c.setdefault("summary", "")
    c["argument"] = split_argument(c.get("summary", ""))
    body = c.get("body") or []
    body_de = c.pop("body_de", None)
    casus_de = c.pop("casus_de", None)
    summary_de = c.pop("summary_de", None)
    c.pop("translation_complete", None)
    if body_de is not None or casus_de or summary_de:
        de = {
            "argument": summary_de or "",
            "casus": split_sections(casus_de) if casus_de else [],
            "body": [(t or "") for t in (body_de or [])] + [""] * max(0, len(body) - len(body_de or [])),
        }
        c["translations"] = {**(c.get("translations") or {}), "de": de}
    else:
        c.setdefault("translations", {})
    return c


def load_prebuilt(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    cf = path.with_name("corrections.json")
    corr = json.loads(cf.read_text(encoding="utf-8")) if cf.exists() else None
    out = {}
    for cid, rec in data["consilia"].items():
        c = normalise(rec, corr, cid)
        c["id"] = cid
        out[cid] = c
    return out


if __name__ == "__main__":  # quick look: python src/import_prebuilt.py data/<author>/<print>/prebuilt.json
    import sys
    d = load_prebuilt(Path(sys.argv[1]))
    first = next(iter(d.values()))
    print(len(d), "consilia;", json.dumps({k: (v if not isinstance(v, (list, dict)) else f"{type(v).__name__}[{len(v)}]") for k, v in first.items()}, ensure_ascii=False))
    print(first["argument"][:3])
