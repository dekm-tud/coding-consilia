#!/usr/bin/env python3
"""
Lexicon vote for the disputed lines below the agent threshold: of two readings, the one whose words
are known from the rest of the corpus wins.

    python tools/lexicon_pick.py report            # counts and samples, nothing written
    python tools/lexicon_pick.py apply --max-cer 0.10

Vocabulary: every word of every reading in the five volumes (both TextEquiv entries), counted; a word is
"known" if it occurs at least --min-freq times. Residual abbreviation marks (ũ ẽ ꝑ ꝙ …) count as unknown.
A line flips to the other reading only if that reading has no unknown words and the valid one has at
least one, or if it has at least two unknown words fewer. Flipped lines get the comment
"reviewed <date>: lexicon <choice>", so the review tool does not show them again.
"""
from __future__ import annotations

import argparse
import collections
import glob
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "src" / "qa"))
from review_server import collect, apply_decision  # noqa: E402

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"; T = "{%s}" % NS
ABBR = re.compile(r"[ãẽĩõũñÃẼĨÕŨꝑꝓꝗꝙꝰꝯ̃̄̅ʒȝ]|[a-z]̃")
TOK = re.compile(r"[A-Za-zſÀ-ɏ]+")


def words(t: str) -> list[str]:
    import unicodedata
    t = unicodedata.normalize("NFD", t.lower().replace("ſ", "s"))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return TOK.findall(t)


def vocabulary() -> collections.Counter:
    c = collections.Counter()
    for f in glob.glob(str(ROOT / "data" / "Baldo_29618397" / "*" / "page" / "*.xml")):
        for u in ET.parse(f).getroot().iter(T + "Unicode"):
            if u.text:
                c.update(words(u.text))
    return c


def unknown(t: str, voc, min_freq: int) -> list[str]:
    bad = [w for w in words(t) if len(w) > 1 and voc[w] < min_freq]
    bad += ["~" + m.group(0) for m in ABBR.finditer(t)]
    return bad


def decide(it, voc, min_freq):
    g, p = unknown(it["gemini"], voc, min_freq), unknown(it["pylaia"], voc, min_freq)
    valid, other = it["valid"], "pylaia" if it["valid"] == "gemini" else "gemini"
    uv, uo = (g, p) if valid == "gemini" else (p, g)
    if (not uo and uv) or (len(uv) - len(uo) >= 2):
        return other, uv, uo
    return None, uv, uo


def _one_letter(a: str, b: str) -> bool:
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if abs(len(a) - len(b)) == 1:
        s, l = (a, b) if len(a) < len(b) else (b, a)
        return any(l[:i] + l[i + 1:] == s for i in range(len(l)))
    return False


def decide_one_letter(it, voc):
    """Second rule: every differing word of the valid reading is a one-letter slip of a form that is at
    least ten times more frequent in the corpus (testameto/testamento, fibi/sibi, loan/Ioan)."""
    import difflib
    valid, other = it["valid"], "pylaia" if it["valid"] == "gemini" else "gemini"
    A, B = words(it[valid]), words(it[other])
    diffs = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=A, b=B, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace" and i2 - i1 == 1 and j2 - j1 == 1:
            a, b = A[i1], B[j1]
            if len(a) >= 4 and len(b) >= 4 and _one_letter(a, b) and voc[a] < 200 and voc[b] >= 10 * max(voc[a], 1):
                diffs.append((a, b)); continue
        return None, []
    return (other, diffs) if diffs else (None, [])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["report", "apply"])
    ap.add_argument("--max-cer", type=float, default=0.10)
    ap.add_argument("--min-freq", type=int, default=3)
    ap.add_argument("--samples", type=int, default=15)
    ap.add_argument("--one-letter", action="store_true", help="also apply the one-letter-slip rule")
    a = ap.parse_args()
    voc = vocabulary()
    items = [it for it in collect(None) if it["kind"] == "readings differ" and it["cer"] < a.max_cer]
    flips = []
    for it in items:
        ch, uv, uo = decide(it, voc, a.min_freq)
        if not ch and a.one_letter:
            ch, uv = decide_one_letter(it, voc); uo = []
        if ch:
            flips.append((it, ch, uv, uo))
    print(f"{len(items)} disputed lines below {a.max_cer:.2f}; lexicon flips {len(flips)}:",
          dict(collections.Counter(ch for _, ch, _, _ in flips)))
    if a.cmd == "report":
        import random; random.seed(7)
        for it, ch, uv, uo in random.sample(flips, min(a.samples, len(flips))):
            print(f"\n{it['vol'][-2:]} {it['page']} cer {it['cer']:.2f}  {it['valid']} -> {ch}   unknown valid {uv} other {uo}")
            print("  G:", it["gemini"][:110]); print("  P:", it["pylaia"][:110])
        return
    n = 0
    for it, ch, _, _ in flips:
        apply_decision(it, ch, "", who="lexicon" if not a.one_letter else "lexicon"); n += 1
    print("flipped", n)


if __name__ == "__main__":
    main()
