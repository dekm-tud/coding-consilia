"""
Build classification units for the keyword/annotation pipeline.

One unit = one body or casus section of a consilium (the `body` / `casus` lists, already split at the
print's paragraph marks), or a <= MAX_UNIT_CHARS window of a longer section, split
at sentence boundaries with one sentence of overlap. Sections of <= MIN_CHUNK_CHARS
are formulaic openers ("In Christi nomine. Amen.") and are skipped, exactly as in
embed_consilia.py.

The English translation of a section is attached when the translation has the same
number of sections (it is a reading aid for the LLM; decisions are made on the Latin).

Usage:
    python src/prepare_units.py                       # -> output/units.jsonl
    python src/prepare_units.py --output units.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
MIN_CHUNK_CHARS = 50        # same threshold as embed_consilia.py
MAX_UNIT_CHARS = 4000
CONTEXT_CHARS = 300

_SENT_END = re.compile(r"(?<=[.;:†])\s+")


def split_windows(text: str, max_chars: int = MAX_UNIT_CHARS) -> list[str]:
    """Split text into windows of <= max_chars at sentence boundaries, with the
    last sentence of each window repeated at the start of the next one."""
    if len(text) <= max_chars:
        return [text]
    sents = [s for s in _SENT_END.split(text) if s]
    windows, cur = [], []
    for s in sents:
        if cur and sum(len(x) + 1 for x in cur) + len(s) > max_chars:
            windows.append(" ".join(cur))
            cur = [cur[-1]] if len(cur[-1]) < max_chars // 2 else []
        cur.append(s)
    if cur:
        windows.append(" ".join(cur))
    # a single sentence longer than max_chars is hard-cut
    out = []
    for w in windows:
        out.extend(w[i:i + max_chars] for i in range(0, len(w), max_chars))
    return out


def argument_text(c: dict, lang: str | None = None) -> str:
    if lang:
        a = (c.get("translations", {}).get(lang, {}) or {}).get("argument") or ""
        return " ".join(a) if isinstance(a, list) else a
    return " ".join(f"{it.get('n', '')} {it.get('text', '')}".strip() for it in (c.get("argument") or [])) \
        or (c.get("summary") or "")


def build_units(consilia: dict) -> list[dict]:
    """Units of the body sections (`#s<i>w<j>`) and of the casus sections (`#c<i>w<j>`).

    The casus is the statement of facts set before the consilium's heading (see casus.py);
    it is classified like any other passage."""
    units = []
    for c in consilia.values():
        # reading aid: the English translation, or the German one where only that exists (Panormitanus)
        trs = c.get("translations", {}) or {}
        tr_lang = "en" if (trs.get("en") or {}).get("body") else "de" if (trs.get("de") or {}).get("body") else "en"
        tr_en = trs.get(tr_lang) or {}
        arg_la, arg_en = argument_text(c), argument_text(c, tr_lang)
        for kind, tag in (("body", "s"), ("casus", "c")):
            secs = c.get(kind) or []
            secs_en = tr_en.get(kind) or []
            en_aligned = len(secs_en) == len(secs)
            for i, sec in enumerate(secs):
                if len(sec.strip()) <= MIN_CHUNK_CHARS:
                    continue
                prev_la = secs[i - 1][-CONTEXT_CHARS:] if i > 0 else ""
                next_la = secs[i + 1][:CONTEXT_CHARS] if i + 1 < len(secs) else ""
                wins = split_windows(sec)
                for w, text in enumerate(wins):
                    units.append({
                        "unit": f"{c['id']}#{tag}{i}w{w}",
                        "doc": c["id"], "n": c.get("n"), "volume": c.get("volume"),
                        "kind": kind, "section": i, "window": w, "n_windows": len(wins),
                        "la": text,
                        "en": (secs_en[i] if en_aligned else "") if len(wins) == 1 else "",
                        "en_section": secs_en[i] if en_aligned else "",
                        "prev_la": prev_la, "next_la": next_la,
                        "title": c.get("title") or "",
                        "argumentum_la": arg_la, "argumentum_en": arg_en, "tr_lang": tr_lang,
                    })
    return units


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", type=Path, default=ROOT / "output" / "consilia.json")
    ap.add_argument("--output", type=Path, default=ROOT / "output" / "units.jsonl")
    args = ap.parse_args()
    consilia = json.loads(args.input.read_text(encoding="utf-8"))["consilia"]
    units = build_units(consilia)
    with args.output.open("w", encoding="utf-8") as fh:
        for u in units:
            fh.write(json.dumps(u, ensure_ascii=False) + "\n")
    print(f"{len(units)} units from {len(consilia)} consilia -> {args.output}")


if __name__ == "__main__":
    main()
