#!/usr/bin/env python3
"""
Evaluate the readings against the gold standard (output/gold/gold.jsonl from tools/gold_server.py).

Two comparison levels:
  letters  case, spacing and punctuation ignored: is the wording right?
  print    case, punctuation and word division kept: is the line right as it will appear in the edition?
The leading margin number is taken off before both comparisons and scored on its own (the line
polygons end at the text, so line recognisers cannot see the gutter). Both levels ignore accents (à, ò, è), the line-break sign ¬, ae/oe/æ/œ/ę against e, long s, u/v,
i/j and & against et. Tilde and macron are kept, since they mark an abbreviation left unexpanded.

Per reading (Gemini, PyLaia, kraken, the text currently online, a 2-of-3 line majority) it reports
line accuracy and character error rate, per stratum and weighted to the corpus, then by line type,
volume and flag, and the error classes of every wrong word (s/f, i/l, abbreviation mark left,
margin number, line end cut, other).

    python tools/gold_eval.py [--out output/gold/report.md]
"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "qa"))
from compare_ocr import dist  # noqa: E402

OUT = ROOT / "output" / "gold"
ABBR_MARK = re.compile(r"[ꝑꝓꝗꝙꝯꝰ⁊ãẽĩõũāēīōūȳ]|\bq[;:]|[a-z];[a-z]")


ACCENTS = {"\u0300", "\u0301", "\u0302", "\u0308"}     # grave, acute, circumflex, diaeresis; tilde and macron stay (abbreviation marks)
PUNCT = re.compile(r"[^\w\s§†]")


def clean(t: str) -> str:
    """edition cleaning applied to every reading and to the gold text before comparison:
    expansion brackets dropped (et[iam] -> etiam), q; -> que, accents removed, one space on each side
    of †, a space after every . , : ; ? ! that is followed by text, spaces collapsed"""
    t = (t or "").replace("[", "").replace("]", "")
    t = t.replace("q;", "que")
    t = re.sub(r";(?=[.,:])", "", t)                        # quodcunque;. -> quodcunque. (mark kept after expanding)
    t = re.sub(r"\b([lc§]|cap|col|lib|q)\. ?i\.", r"\1. 1.", t)        # l. i. -> l. 1., as the agreed lines write it
    t = unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", t) if c not in ACCENTS))
    t = re.sub(r"\s*†\s*", " † ", t)
    t = re.sub(r"([.,:;?!])(?=[^\s.,:;?!¬])", r"\1 ", t)
    return re.sub(r"\s+", " ", t).strip()


def fold(t: str, level: str = "print") -> str:
    """the comparison key: differences the evaluation does not count are folded away
    (accents, the line-break sign, ae/oe/e, long s, u/v, i/j, & = et); 'letters' also drops case,
    punctuation and spacing"""
    t = t.replace("¬", "").replace("ſ", "s").replace("&", "et").replace("[", "").replace("]", "")   # et[iam] = etiam
    t = re.sub(r"-\s*$", "", t.strip())
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if c not in ACCENTS)
    t = unicodedata.normalize("NFC", t)
    for a, b in (("æ", "e"), ("Æ", "E"), ("œ", "e"), ("Œ", "E"), ("ę", "e"), ("ae", "e"), ("oe", "e"), ("Ae", "E"), ("Oe", "E"), ("AE", "E"), ("OE", "E")):
        t = t.replace(a, b)
    t = t.replace("v", "u").replace("V", "U").replace("j", "i").replace("J", "I")
    t = re.sub(r"\b([lc§]|cap|col|lib|q)\. ?i\.", r"\1. 1.", t)        # l. i. = l. 1. (lex prima)
    if level == "letters":
        return PUNCT.sub("", t.lower()).replace(" ", "")
    return re.sub(r"\s+", " ", re.sub(r"\s*([,.:;])", r"\1", t)).strip()


MARGIN_NO = re.compile(r"^\s*(\d{1,2})\s+(?=\S)")


def has_abbreviation(t: str) -> bool:
    """an unexpanded abbreviation mark (tilde, macron, q;, special letters): such a reading is always wrong"""
    d = unicodedata.normalize("NFD", t)
    return bool(ABBR_MARK.search(t)) or any(c in d for c in "\u0303\u0304\u0305") or any(c in t for c in "ꝑꝓꝗꝙꝯꝰ⁊")


def options(s: dict) -> list[str]:
    """the readings a person has to choose between: abbreviated readings dropped (unless all are),
    readings equal under fold() merged"""
    R = [s[k] for k in ("gemini", "pylaia", "kraken") if (s.get(k) or "").strip()]
    clean = [t for t in R if not has_abbreviation(t)] or R
    out = {}
    for t in clean:
        out.setdefault(fold(t, "print"), t)
    return list(out.values())


def margin_number(t: str) -> str:
    m = MARGIN_NO.match(t)
    return m.group(1) if m else ""


def body(t: str) -> str:
    """the line without its leading margin number: line recognisers do not see the gutter"""
    return MARGIN_NO.sub("", t, count=1)


def fold_print(t: str) -> str:
    return fold(t, "print")


def words(t: str) -> list[str]:
    return re.findall(r"[a-z0-9§†]+", fold_print(t).lower())


def classify(hyp: str, gold: str) -> list[str]:
    """error classes of the word differences between a reading and the gold text"""
    out = []
    if ABBR_MARK.search(hyp):
        out.append("abbreviation mark left")
    H, G = words(hyp), words(gold)
    ops = difflib.SequenceMatcher(a=H, b=G, autojunk=False).get_opcodes()
    for n, (op, a1, a2, b1, b2) in enumerate(ops):
        if op == "equal":
            continue
        h, g = " ".join(H[a1:a2]), " ".join(G[b1:b2])
        if op == "delete" and re.fullmatch(r"[\d†\s]+", h) or op == "insert" and re.fullmatch(r"[\d†\s]+", g):
            out.append("margin number or †")
        elif op == "insert" and n == len(ops) - 1 or (op == "replace" and n == len(ops) - 1 and g.startswith(h)):
            out.append("line end cut")
        elif op == "replace" and len(h) == len(g) and all(x == y or {x, y} == {"s", "f"} for x, y in zip(h, g)):
            out.append("s/f")
        elif op == "replace" and len(h) == len(g) and all(x == y or {x, y} <= {"i", "l", "1"} for x, y in zip(h, g)):
            out.append("i/l/1")
        elif op == "replace" and (g.startswith(h) or h.startswith(g)) and a2 - a1 == b2 - b1 == 1:
            out.append("abbreviation not expanded or cut word")
        else:
            out.append("other word")
    return out


def majority(s: dict) -> str:
    """two of three readings agree (letters level), readings with an abbreviation mark do not vote;
    no majority: Gemini"""
    n = {k: fold(body(s[k]), "letters") for k in ("gemini", "pylaia", "kraken") if not has_abbreviation(s[k])}
    for a, b in (("pylaia", "kraken"), ("gemini", "kraken"), ("gemini", "pylaia")):
        if a in n and b in n and n[a] == n[b]:
            return s[a]
    return s["gemini"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT / "report.md")
    ap.add_argument("--raw", action="store_true", help="compare the readings as they are, without clean()")
    a = ap.parse_args()
    sample = {s["id"]: s for s in json.loads((OUT / "sample.json").read_text(encoding="utf-8"))}
    gold = {}
    for l in (OUT / "gold.jsonl").open(encoding="utf-8"):
        if l.strip():
            r = json.loads(l); gold[r["id"]] = r
    items = [dict(sample[i], gold=g["gold"], flags=g.get("flags", [])) for i, g in gold.items()]
    if not a.raw:
        for it in items:
            for k in ("gemini", "pylaia", "kraken", "gold"):
                it[k] = clean(it[k])
    for s in items:
        s["online"] = s["gemini"] if s["valid"] == "gemini" else s["pylaia"]
        s["majority"] = majority(s)
    SYS = ("gemini", "pylaia", "kraken", "online", "majority")

    def stats(sel, level):
        res = {}
        for k in SYS:
            ok = w = e = c = 0.0
            for s in sel:
                h, g = fold(body(s[k]), level), fold(body(s["gold"]), level)
                wt = s["weight"]
                ok += wt * (h == g); w += wt; e += wt * dist(h, g); c += wt * max(len(g), 1)
            res[k] = (ok / w * 100 if w else 0, e / c * 100 if c else 0)
        return res

    L = [f"# Gold standard evaluation\n", f"{len(items)} of {len(sample)} lines done.\n"]
    for level in ("letters", "print"):
        L += [f"## {level}: line accuracy / CER\n", "| subset | n | " + " | ".join(SYS) + " |", "|---|---|" + "---|" * len(SYS)]
        groups = [("corpus (weighted)", items)]
        groups += [(f"stratum {st}", [s for s in items if s["stratum"] == st]) for st in ("agreed", "disputed")]
        if level == "letters":
            groups += [(f"line type {t}", [s for s in items if s["line_type"] == t]) for t in sorted({s["line_type"] for s in items})]
            groups += [(f"volume {v[-2:]}", [s for s in items if s["print"] == v]) for v in sorted({s["print"] for s in items})]
            groups += [(f"flag {f}", [s for s in items if f in s["flags"]]) for f in sorted({f for s in items for f in s["flags"]})]
        for name, sel in groups:
            if not sel:
                continue
            st = stats(sel, level)
            L.append(f"| {name} | {len(sel)} | " + " | ".join(f"{st[k][0]:.1f} % / {st[k][1]:.2f}" for k in SYS) + " |")
        L.append("")
    L += ["## Margin numbers (separate from the text above)\n", "| | " + " | ".join(SYS) + " |", "|---|" + "---|" * len(SYS)]
    withno = [s for s in items if margin_number(s["gold"])]
    L.append(f"| gold lines with a margin number: {len(withno)}, right | " + " | ".join(str(sum(margin_number(s[k]) == margin_number(s["gold"]) for s in withno)) for k in SYS) + " |")
    L.append("| number read where gold has none | " + " | ".join(str(sum(bool(margin_number(s[k])) and not margin_number(s["gold"]) for s in items)) for k in SYS) + " |")
    L.append("")
    L += ["## Error classes (word differences against gold, unweighted counts)\n", "| class | " + " | ".join(SYS) + " |", "|---|" + "---|" * len(SYS)]
    cls = {k: collections.Counter() for k in SYS}
    for s in items:
        for k in SYS:
            if fold(body(s[k])) != fold(body(s["gold"])):
                cls[k].update(classify(body(s[k]), body(s["gold"])))
    for c in sorted({c for v in cls.values() for c in v}, key=lambda c: -sum(v[c] for v in cls.values())):
        L.append(f"| {c} | " + " | ".join(str(cls[k][c]) for k in SYS) + " |")
    L += ["\n## Lines no reading got right (letters)\n"]
    for s in items:
        if not any(fold(s[k], "letters") == fold(s["gold"], "letters") for k in ("gemini", "pylaia", "kraken")):
            L.append(f"- {s['id']} `{s['gold']}`  \n  G `{s['gemini']}`  \n  P `{s['pylaia']}`  \n  K `{s['kraken']}`")
    a.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
