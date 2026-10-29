#!/usr/bin/env python3
"""
Build the corrections of the Panormitanus text (Venice 1571) from the comparison with kraken.

The edition text of Panormitanus comes from the Gemini column transcription (prebuilt.json). The comparison with
the kraken reading of the same pages (tools/kraken_compare_panormitanus.py, image check of 100 disagreements in
output/qa/kraken_panormitanus_sample40.md) showed four kinds of errors in it, and this script turns them into
corrections that src/import_prebuilt.py applies after the spelling rules of the edition:

  words   every occurrence of a form is replaced
          - abbreviations with a tilde or macron left unexpanded (nõ, cũ, tamẽ, testamẽ to): the tilde is read
            as n or m and the reading that the Baldus text attests most often is taken; words split
            at the old line break (cõ sequitur) are joined; a short list of other abbreviations is fixed by hand
          - Gemini artefacts for the long s (sfed, efse, pofsit, sftatu): every reading of f and s is tried and the
            one most frequent in the Baldus text is taken
          - accents the print uses on single words (quâ, poenâ), which the edition drops
  places  single words where kraken and Gemini differ, kraken's reading is a word of the Baldus text (at least
          five times) and Gemini's reading is not (and occurs at most twice in Panormitanus), with at most two
          letters changed and words of more than three letters only; compared on the whole word, so that a word divided at the end of a line is judged as one
          word. Each place is stored
          with enough context to be found exactly once in its text field.

    python tools/panormitanus_corrections.py            # writes data/…/corrections.json and a report
"""
from __future__ import annotations

import collections
import difflib
import itertools
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from import_prebuilt import edition_form, apply_corrections  # noqa: E402

PRINT = ROOT / "data" / "Panormitanus_32108367" / "Panormitanus_Cons_Print_Venice_1571_v1"
COMPARE = ROOT / "output" / "qa" / "kraken_panormitanus_compare.jsonl"
REPORT = ROOT / "output" / "qa" / "panormitanus_corrections.md"
FIELDS = ("title", "summary", "casus", "body")

# abbreviations whose expansion is not a matter of n or m
HAND = {"põt": "potest", "rõne": "ratione", "rõ": "ratio", "qd̃": "quod", "qñ": "quando", "dñs": "dominus",
        "dñi": "domini", "ptãs": "potestas", "oĩbus": "omnibus", "oẽs": "omnes", "hẽt": "habet", "qõnem": "quaestionem",
        "s̃": "supra", "j̃": "infra", "p̃ dicta": "praedicta", "gē tium": "gentium", "cã": "causa", "dẽt": "debet",
        "procura tiõne": "procuratione", "quamquãm": "quamquam", "inchoädus": "inchoandus", "Innocẽ": "Innocen",
        "qdvod": "quod", "negociatīonum": "negociationum", "cūm": "cum", "cõi": "communi",
        "cõ-nsensu": "consensu", "cõ-nsanguineorum": "consanguineorum", "põderandus": "ponderandus",
        "cõndictionis": "condictionis", "cõnmemoratio": "commemoratio", "mõstrabitur": "monstrabitur",
        "praesentãdus": "praesentandus", "scãdalizentur": "scandalizentur", "unĩco": "unico", "depẽ dente": "dependente",
        "satisfaciẽ tis": "satisfacientis", "argumẽtatum": "argumentatum", "reprimẽdum": "reprimendum", "istã": "istam",
        "cõtuma": "contuma"}
# places where the reading of kraken is wrong although it is a word of the Baldus text
KEEP = {"Kalendas", "contulere", "transiisset", "faeminam", "actusque", "receptan"}
MARKS = "̃̄"                                   # combining tilde and macron
WORD = re.compile(r"[^\W\d_]+", re.U)


def decompose(w: str) -> str:
    return unicodedata.normalize("NFD", w)


def has_mark(w: str) -> bool:
    return any(c in MARKS for c in decompose(w)) or "ñ" in w


def lexicon() -> collections.Counter:
    d = json.loads((ROOT / "output" / "consilia.json").read_text(encoding="utf-8"))["consilia"]
    f = collections.Counter()
    for r in d.values():
        if "baldo" in r["id"]:
            for t in itertools.chain(r.get("body") or [], r.get("casus") or [], [a["text"] for a in r.get("argument") or []]):
                f.update(w.lower() for w in WORD.findall(t))
    return f


def expansions(w: str) -> list[str]:
    """Readings of a word with tildes or macrons: each mark becomes n, m or nothing; other accents are dropped."""
    parts = []
    for ch in decompose(w):
        if ch in MARKS:
            if not parts or parts[-1][0] not in "aeiouAEIOU":
                return []
            parts.append(["n", "m"])                  # a tilde always stands for a nasal
        elif unicodedata.combining(ch) and ch != "\u0328":   # accents go, the ogonek of ę stays
            continue
        else:
            parts.append([ch])
    return [unicodedata.normalize("NFC", "".join(p)) for p in itertools.product(*parts)]


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def match_case(src: str, new: str) -> str:
    return new[:1].upper() + new[1:] if src[:1].isupper() else new


def word_map(texts: list[str], lex: collections.Counter, pan: collections.Counter, log: list) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, val in HAND.items():
        out[key] = val
    tok = re.compile(r"(?<!\w)([^\W\d_]*[̃̄ãẽĩõũñāēīōūâêîôûäëïöü][^\W\d_]*)(?:([ -])([a-z]+))?", re.U)
    for t in texts:
        for m in tok.finditer(unicodedata.normalize("NFC", t)):
            w, sep, nxt = m.group(1), m.group(2), m.group(3)
            if w in out or (nxt and f"{w}{sep}{nxt}" in out):
                continue
            cands = expansions(w)
            if not cands:
                log.append(("unresolved", w)); continue
            best = max(cands, key=lambda c: lex[c.lower()])
            if nxt:                                       # a word split at the old line break: cõ sequitur
                joined = max((c + nxt for c in cands), key=lambda c: lex[c.lower()])
                if lex[joined.lower()] >= 2 and lex[joined.lower()] > lex[best.lower()]:
                    out[f"{w}{sep}{nxt}"] = match_case(w, joined); continue
            if lex[best.lower()] >= (1 if len(best) >= 6 else 3) or not has_mark(w):
                out[w] = match_case(w, best)
            else:
                log.append(("unresolved", w))
    for t in texts:                                       # long-s artefacts of the transcription
        for w in set(WORD.findall(t)):
            if w in out:
                continue
            if not (re.match(r"[Ss]f", w) or "fs" in w or "sft" in w) or lex[w.lower()] > 0:
                continue
            base = re.sub(r"^([Ss])f", r"\1", w).replace("sfs", "ss").replace("sft", "st")
            pos = [i for i, ch in enumerate(base) if ch in "fs"]
            cands = {base}
            if len(pos) <= 6:                             # every reading of f and s, the most frequent one wins
                for combo in itertools.product("fs", repeat=len(pos)):
                    b = list(base)
                    for i, ch in zip(pos, combo):
                        b[i] = ch
                    cands.add("".join(b))
            new = max(cands, key=lambda c: lex[c.lower()])
            if lex[new.lower()] >= 3:
                out[w] = new
    return out


def texts_of(rec: dict) -> dict[tuple[str, int | None], str]:
    out = {}
    for f in FIELDS:
        v = rec.get(f)
        if isinstance(v, list):
            for i, t in enumerate(v):
                out[(f, i)] = edition_form(t)
        elif isinstance(v, str):
            out[(f, None)] = edition_form(v)
    return out


def line_words(s: str | None, gemini: bool) -> list[str]:
    s = unicodedata.normalize("NFC", s or "")
    if gemini:
        s = re.sub(r"\{[^}]*\}", "", s)
        s = re.sub(r"\[([^\]]*)\]", r"\1", s).replace("ſ", "s").replace("&#182;", "")
    return WORD.findall(edition_form(s))


def main() -> None:
    pre = json.loads((PRINT / "prebuilt.json").read_text(encoding="utf-8"))
    lex = lexicon()
    recs = pre["consilia"]
    base = {cid: texts_of(r) for cid, r in recs.items()}
    pan = collections.Counter(w.lower() for tt in base.values() for t in tt.values() for w in WORD.findall(t))
    log: list = []
    words = word_map([t for tt in base.values() for t in tt.values()], lex, pan, log)
    corr = {"words": words, "places": []}
    cur = {cid: {k: apply_corrections(t, {"words": words}) for k, t in tt.items()} for cid, tt in base.items()}
    pan = collections.Counter(w.lower() for tt in cur.values() for t in tt.values() for w in WORD.findall(t))
    pages = {cid: {s["page"][:4] for s in r.get("sources") or []} for cid, r in recs.items()}

    stats = collections.Counter(); seen = set()
    for line in COMPARE.open(encoding="utf-8"):
        p = json.loads(line)
        if not p.get("cer") or p["gemini"] is None or p["kraken"] is None:
            continue
        g, k = line_words(p["gemini"], True), line_words(p["kraken"], False)
        sm = difflib.SequenceMatcher(None, [x.lower() for x in g], [x.lower() for x in k], autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op != "replace" or i2 - i1 != j2 - j1:
                continue
            for i, j in zip(range(i1, i2), range(j1, j2)):
                wg, wk = g[i], k[j]
                if wg.isdigit() or len(wg) < 2 or difflib.SequenceMatcher(None, wg.lower(), wk.lower()).ratio() < .6:
                    continue
                stats["candidates"] += 1
                prev = g[i - 1] if i > 0 else None
                nxt = g[i + 1] if i + 1 < len(g) else None
                left = rf"{re.escape(prev)}\W+" if prev else ""
                right = rf"\W+{re.escape(nxt)}" if nxt else ""
                head = r"(?<![^\W\d_])" if prev else ""
                pat = re.compile(rf"{left}{head}(?P<pre>[^\W\d_]*?)(?P<w>{re.escape(wg)})(?P<post>[^\W\d_]*){right}", re.U)
                hits = [(cid, key, m) for cid in recs if p["page"] in pages[cid]
                        for key, t in cur[cid].items() for m in pat.finditer(t)]
                if not prev and not nxt:
                    stats["no context"] += 1; continue
                if len(hits) != 1:
                    stats["not unique" if hits else "not found"] += 1; continue
                cid, key, m = hits[0]
                if not prev and m.group("pre") == "" and False:
                    pass
                full_old = m.group("pre") + m.group("w") + m.group("post")
                full_new = m.group("pre") + match_case(m.group("w"), wk) + m.group("post")
                if (lex[full_new.lower()] >= 5 and lex[full_old.lower()] == 0 and pan[full_old.lower()] <= 2
                        and full_old not in KEEP and len(full_old) > 3 and lev(full_old.lower(), full_new.lower()) <= 2):
                    t = cur[cid][key]
                    s, e = m.start("pre"), m.end("post")
                    a, b = max(0, s - 25), min(len(t), e + 25)
                    while t.count(t[a:b]) > 1 and (a > 0 or b < len(t)):
                        a, b = max(0, a - 10), min(len(t), b + 10)
                    old = t[a:b]; new = t[a:s] + full_new + t[e:b]
                    if (cid, key, s) in seen:
                        continue
                    seen.add((cid, key, s))
                    corr["places"].append({"id": cid, "field": key[0], "index": key[1], "old": old, "new": new,
                                           "word": [full_old, full_new], "page": p["page"], "col": p["col"]})
                    cur[cid][key] = t.replace(old, new)     # later places are found in the corrected text
                    stats["applied"] += 1
                else:
                    stats["kept (lexicon does not decide)"] += 1
    out = PRINT / "corrections.json"
    out.write_text(json.dumps(corr, ensure_ascii=False, indent=1), encoding="utf-8")
    pairs = collections.Counter(tuple(x["word"]) for x in corr["places"])
    rep = ["# Corrections of the Panormitanus text", "",
           f"- word forms replaced everywhere: {len(words)}",
           f"- single places corrected after kraken: {len(corr['places'])}",
           f"- word differences examined: {stats['candidates']}, not decided by the lexicon: {stats['kept (lexicon does not decide)']}, "
           f"not found or not unique: {stats['not found'] + stats['not unique']}, without context: {stats['no context']}",
           f"- forms left as they are: {', '.join(sorted({w for _, w in log}))}", "",
           "## Word forms", "", *(f"- {k} → {v}" for k, v in sorted(words.items())), "",
           "## Places (old → new, count)", "", *(f"- {a} → {b} ({n})" for (a, b), n in pairs.most_common())]
    REPORT.write_text("\n".join(rep) + "\n", encoding="utf-8")
    print(json.dumps({"words": len(words), "places": len(corr["places"]), **stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
