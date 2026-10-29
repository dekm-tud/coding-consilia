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

import json
import re
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


def _clean(v):
    """Strip control characters the OCR model occasionally emits (\\x00/\\x10 found in 3 consilia)."""
    if isinstance(v, str):
        return _CTRL.sub("", v)
    if isinstance(v, list):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    return v


def normalise(rec: dict) -> dict:
    c = _clean(dict(rec))
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
    out = {}
    for cid, rec in data["consilia"].items():
        c = normalise(rec)
        c["id"] = cid
        out[cid] = c
    return out


if __name__ == "__main__":  # quick look: python src/import_prebuilt.py data/<author>/<print>/prebuilt.json
    import sys
    d = load_prebuilt(Path(sys.argv[1]))
    first = next(iter(d.values()))
    print(len(d), "consilia;", json.dumps({k: (v if not isinstance(v, (list, dict)) else f"{type(v).__name__}[{len(v)}]") for k, v in first.items()}, ensure_ascii=False))
    print(first["argument"][:3])
