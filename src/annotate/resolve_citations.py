#!/usr/bin/env python3
"""
Rule-based resolution of abbreviated legal citations (Layer C spans) to a work and title:
"ff. de pac. l. tale pactum. §. fi." -> D. 2.14 (Digesta, De pactis); "extra de proba. c. cum causam." -> X 2.20.

Works are recognised by their sigla, titles by matching the abbreviated words of the citation as
prefixes of the words of the title names in src/codebooks/corpus_titles.json (Digest, Codex,
Institutes, Authenticum/Novels, Liber Extra, Sextus, Clementines, Libri feudorum, Decretum). A
citation resolves only when exactly one title fits best; ties are reported as ambiguous, no guess.

    python src/annotate/resolve_citations.py "ff. de ver. ob. l. 1."        # one citation
    python src/annotate/resolve_citations.py --stats                        # resolution rate over all published spans

API: resolve(text, prev=None, adjacent=False) -> {"work": "D", "key": "D. 45.1", "title": "De verborum obligationibus",
"status": "resolved"|"ambiguous"|"unresolved"|"no_title", "candidates": [...]} ; `prev` is the previous
resolved key in the same text for "eo. tit." / "d. l." style references; with `adjacent=True` (the
citation follows the previous one directly, "… l. 2. §. voluntatem. et l. dotem") a lex- or
capitulum-only citation inherits the previous title ("via": "adjacent").
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLE = json.loads((ROOT / "src" / "codebooks" / "corpus_titles.json").read_text(encoding="utf-8"))

STOP = {"de", "et", "ad", "in", "ex", "si", "ne", "ut", "vt", "an", "a", "ab", "pro", "per", "cum", "non", "quod", "qui", "quae", "que", "quo", "sub", "l", "ti", "tit", "titu", "rub", "rubr", "c", "ca", "cap", "capit", "capitulo", "§", "para", "s", "v", "ver", "vers", "versi", "fi", "fin", "fina", "prin", "princ", "ff", "d", "dicta", "dicto", "eo", "eod", "eodem", "ibi", "ibid", "ibidem", "et", "seq", "sequen", "col", "colla", "nu", "num", "extra", "ext", "x", "ff", "dig", "digest", "cod", "codi", "post", "ante", "princ", "princi", "principio", "glo", "gl", "glos", "gloss", "no", "not", "nota", "tex", "text", "dicit", "ar", "arg", "argu", "argum"}
ROMAN = re.compile(r"^[ivxlcdm]+$")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower().replace("ſ", "s"))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("æ", "ae").replace("œ", "oe").replace("v", "u").replace("j", "i").replace("ę", "e").replace("ae", "e").replace("oe", "e")
    s = s.replace("adqu", "acqu")        # adquirenda / acquirenda: the prints use both
    return s


@lru_cache(maxsize=None)
def title_index(work: str):
    """[(key, title, [content words of the title])]"""
    out = []
    w = TABLE[work]
    if work == "Nov":
        for coll, entries in w["collationes"].items():
            for n, e in entries.items():
                for name in {e["title"], e.get("rubric_1604", "")}:
                    if name:
                        out.append((f"Nov. {e['nov']}", e["title"], words(name)))
    else:
        for book, titles in w["books"].items():
            for n, name in titles.items():
                key = f"{work}{'.' if work in ('D', 'C', 'Inst', 'Clem') else ''} {book}.{n}"
                out.append((key, name, words(name)))
    return out


def words(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z]+", norm(s)) if t not in STOP and len(t) > 1]


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z§]+|\d+", norm(text))


def detect_work(text: str) -> tuple[str | None, int]:
    """(work, index of the first token after the siglum) from the printed sigla"""
    raw = text.strip()
    t = tokens(raw)
    low = norm(raw)
    if re.search(r"(^|[\s(,.])C\.", raw) and not re.search(r"(^|[\s(])ff\.", low):
        return "C", next((i + 1 for i, x in enumerate(re.findall(r"[a-zA-Z§]+|\d+", raw)) if x == "C"), 0)      # Codex, also when an Authentica is the locator ("auth. ingressi, C. de sac. san. ecc.")
    if re.search(r"(^|[\s(])ff\.?", low) or re.search(r"(^|[\s(])(dig|digest)\.", low):
        return "D", next((i + 1 for i, x in enumerate(t) if x in ("ff", "dig", "digest")), 0)      # before the Institutes: "ff. de haere. insti." is a Digest title
    if re.search(r"(^|[\s(])(in )?(auth|authen|authent|aut|nouel|nou)\.", low):
        return "Nov", next((i + 1 for i, x in enumerate(t) if x in ("auth", "authen", "authent", "aut", "nouel", "nou")), 0)
    if re.search(r"(^|[\s(])(insti|inst|instit|institu)\.", low):
        return "Inst", next((i + 1 for i, x in enumerate(t) if x.startswith("inst")), 0)
    if re.search(r"(^|[\s(])(in )?(usib|usi|vsib|usibus) ?feu", low) or re.search(r"(^|[\s(])(feu|feud|feudo|feudor)\.", low) or "de feu" in low:
        return "LF", 0
    if re.search(r"(^|[\s(])ff\.?", low) or re.search(r"(^|[\s(])(dig|digest)\.", low):
        return "D", next((i + 1 for i, x in enumerate(t) if x in ("ff", "dig", "digest")), 0)
    if re.search(r"(^|[\s(,])C\.", raw) or re.search(r"(^|[\s(])(cod|codi)\.", low):
        for i, x in enumerate(re.findall(r"[a-zA-Z§]+|\d+", raw)):
            if x == "C" or x.lower() in ("cod", "codi"):
                return "C", i + 1
        return "C", 0
    if re.search(r"(^|[\s(])(in )?(ui|vi)\.", low) or re.search(r"li(b)?\. ?(6|vi)\.", low) or "sexto" in low:
        return "VI", 0
    if re.search(r"(^|[\s(])clem", low):
        return "Clem", 0
    if re.search(r"(^|[\s(.])(dist|di|dis|distinct|distin)\.", low) or re.search(r"\bq\.", low) or re.search(r"(^|[\s(])(de pe|de poe|de cons)\.", low):
        return "Decretum", 0
    if re.search(r"(^|[\s(])(extra|ext|x)\.? ?de", low) or re.search(r"(^|[\s(])extra", low) or re.search(r"(^|[\s(])c\.", low):
        return "X", 0
    return None, 0


def title_phrase(text: str, start: int, canon: bool = False) -> list[str]:
    """content words of the title part: after the siglum, up to the lex/capitulum marker; in canon law
    the title often follows the capitulum ("c. ex conquestione ... de res. spo."): then the last "de …"
    segment is the title"""
    t = tokens(text)
    if canon and "de" in t[start:]:
        last = max(i for i, x in enumerate(t) if x == "de" and i >= start)
        # not "de" inside the capitulum name ("c. cum de appellatione"): take the last "de" that is followed by title-like words
        start = last
    seg = []
    i = start
    # a citation may name the lex first ("l. si quis emptionis. C. de praescrip."): words before the
    # siglum belong to the lex, those after it to the title
    prev = None
    for x in t[i:]:
        if x in ("l", "lex", "c", "ca", "cap", "capit", "§", "para", "auth", "authen", "post", "ante") and seg and prev != "ad":
            break          # the lex / capitulum / locator begins ("ad l. Fal." = ad legem Falcidiam is part of the title)
        if x in ("l", "lex", "c", "ca", "cap", "capit", "§", "para") and not seg and prev != "ad":
            break          # lex named first ("l. si duo. ff. de admini. tut."): the title comes after the siglum, handled by `start`
        if prev == "ad" and x in ("l", "le", "leg", "lege", "legem"):
            x = "legem"                      # "ad l. Fal." = ad legem Falcidiam, "ad l. Aquil." = ad legem Aquiliam
        seg.append(x); prev = x
    return [w for w in seg if w not in STOP and not w.isdigit() and not ROMAN.match(w) and not (w == "l" )]


STRICT = False      # set by match_title: first pass with exact prefixes only, then with the 5-letter approximation


def score(abbr: list[str], title_words: list[str]) -> float:
    """abbreviated words must appear in order as prefixes of title words; score = matched share minus
    a small penalty for skipped title words"""
    if not abbr:
        return 0.0
    j = 0; matched = 0; skipped = 0; missed = 0
    for a in abbr:
        k = j; found = False
        while k < len(title_words):
            if title_words[k].startswith(a) or (not STRICT and len(a) >= 5 and title_words[k].startswith(a[:-1])):     # furioso / furiosi (last letter differs)
                matched += 1; skipped += k - j; j = k + 1; found = True; break
            k += 1
        if not found:
            missed += 1          # one abbreviation may fail (misprint, unusual abbreviation) when three or more are given
    if missed > (1 if len(abbr) >= 3 else 0):
        # order-free fallback: every abbreviation matches a distinct title word ("de iure emphy." = De emphyteutico iure)
        used = set(); m2 = 0
        for a in abbr:
            for k, w in enumerate(title_words):
                if k not in used and (w.startswith(a) or (not STRICT and len(a) >= 5 and w.startswith(a[:-1]))):
                    used.add(k); m2 += 1; break
        if m2 == len(abbr):
            return round(m2 - 0.2 + (0.25 if m2 == len(title_words) else 0), 3)
        return 0.0
    covered = matched == len(title_words)       # every content word of the title is accounted for
    return round(matched - 0.3 * skipped - 0.3 * missed + (0.25 if covered else 0), 3)


LEX_ONLY = re.compile(r"^\s*(?:et |vel |ac |& |item )?(l|le|lex|§|para)\.", re.I)
CAP_ONLY = re.compile(r"^\s*(?:et |vel |ac |& |item )?(c|ca|cap|capit)\.", re.I)


def inherit(text: str, prev: str | None) -> dict | None:
    """A citation that names only a lex / paragraph ("et l. dotem", "§. voluntatem") or only a capitulum
    ("et c. auditis") directly after a resolved citation belongs to that citation's title. Only called
    when the caller has checked that the two citations are adjacent in the text (export.py)."""
    if not prev:
        return None
    if " de " in " " + norm(text) + " ":
        return None                      # names a title of its own
    w, _ = detect_work(text)
    if w not in (None, "X"):             # "X" is the fallback for a bare "c."
        return None
    pw = prev.split()[0].rstrip(".")
    if (LEX_ONLY.match(text) and pw in ("D", "C", "Inst", "Nov", "LF")) or (CAP_ONLY.match(text) and pw in ("X", "VI", "Clem")):
        return {"work": pw, "key": prev, "title": None, "status": "resolved", "candidates": [prev], "via": "adjacent"}
    return None


def resolve(text: str, prev: str | None = None, adjacent: bool = False) -> dict:
    low = norm(text)
    if adjacent and (r := inherit(text, prev)):
        return r
    if re.search(r"(^|[\s(])(eo|eod|eodem)\.? ?(ti|tit|titu)", low) or re.search(r"(^|[\s(])(d|dicta|dicto|dict)\.? ?(l|c|ca|cap)\.", low):
        if prev:
            return {"work": prev.split()[0].rstrip("."), "key": prev, "title": None, "status": "resolved", "candidates": [], "via": "previous"}
        return {"work": None, "key": None, "title": None, "status": "no_title", "candidates": [], "via": "previous"}
    work, start = detect_work(text)
    if work == "Decretum":
        low = text.lower().replace("j", "i")
        m = re.search(r"(\d+|[ivxlc]+)\.? ?(dist|di|dis|distinct|distin)\.", low) or re.search(r"(dist|di|dis|distinct|distin)\.? ?(\d+|[ivxlc]+)", low)
        if m:
            num = m.group(1) if m.group(1)[0].isdigit() or ROMAN.match(m.group(1)) else m.group(2)
            return {"work": "Decretum", "key": f"Decr. D. {roman_to_int(num)}", "title": None, "status": "resolved", "candidates": []}
        m = re.search(r"(\d+|[ivxlc]+)\.? ?q\.? ?(\d+|[ivxlc]+)", low)
        if m:
            return {"work": "Decretum", "key": f"Decr. C. {roman_to_int(m.group(1))} q. {roman_to_int(m.group(2))}", "title": None, "status": "resolved", "candidates": []}
        return {"work": "Decretum", "key": None, "title": None, "status": "no_title", "candidates": []}
    if work is None:
        # no siglum: try the Digest and Codex titles alone; accept only a unique strong match
        best = None
        for w in ("D", "C", "X"):
            r = match_title(text, w, 0)
            if r["status"] == "resolved" and r["score"] >= 2 - 0.01:      # at least two matched words
                if best is None:
                    best = r
                else:
                    return {"work": None, "key": None, "title": None, "status": "ambiguous", "candidates": [best["key"], r["key"]]}
        return best or {"work": None, "key": None, "title": None, "status": "unresolved", "candidates": []}
    return match_title(text, work, start)


# conventional abbreviations that a prefix match cannot settle (the medieval usage is fixed)
CONVENTIONS = {
    ("C", ("testa",)): "C. 6.23", ("C", ("test",)): "C. 6.23", ("C", ("testam",)): "C. 6.23",
    ("D", ("iud",)): "D. 5.1", ("D", ("iudi",)): "D. 5.1", ("D", ("iudic",)): "D. 5.1",
    ("D", ("testi",)): "D. 22.5", ("D", ("test",)): "D. 22.5", ("D", ("testib",)): "D. 22.5",
    ("X", ("testi",)): "X 2.20", ("X", ("test",)): "X 2.20", ("X", ("testib",)): "X 2.20",
    ("D", ("reg", "iur")): "D. 50.17", ("D", ("re", "iu")): "D. 50.17", ("VI", ("reg", "iur")): "VI 5.13", ("VI", ("re", "iu")): "VI 5.13",
    ("D", ("ver", "sig")): "D. 50.16", ("D", ("verb", "sig")): "D. 50.16", ("D", ("ver", "signi")): "D. 50.16",
    ("D", ("ver", "ob")): "D. 45.1", ("D", ("verb", "ob")): "D. 45.1", ("D", ("ver", "obl")): "D. 45.1",
    ("D", ("sol", "mat")): "D. 24.3", ("D", ("solu", "mat")): "D. 24.3",
    ("D", ("act",)): "D. 44.7", ("D", ("acti",)): "D. 44.7", ("D", ("actio",)): "D. 44.7",
    ("D", ("rei", "uen")): "D. 6.1", ("C", ("rei", "uen")): "C. 3.32",
    ("D", ("cond", "ind")): "D. 12.6", ("D", ("condi", "indeb")): "D. 12.6",
    ("D", ("pact",)): "D. 2.14", ("D", ("pac",)): "D. 2.14", ("C", ("pac",)): "C. 2.3", ("C", ("pact",)): "C. 2.3",
    ("D", ("leg",)): "D. 30.1", ("D", ("lega",)): "D. 30.1",
    # one title is cited far more often than its namesakes; the printers' usage follows the first
    ("D", ("app",)): "D. 49.1", ("D", ("appe",)): "D. 49.1", ("D", ("appel",)): "D. 49.1", ("D", ("appell",)): "D. 49.1",
    ("D", ("flu",)): "D. 43.12", ("D", ("flum",)): "D. 43.12",
    ("D", ("contr", "emp")): "D. 18.1", ("D", ("contra", "emp")): "D. 18.1", ("D", ("contrah", "emp")): "D. 18.1", ("D", ("contrata",)): "D. 18.1",
    ("D", ("an", "le")): "D. 33.1", ("D", ("ann", "le")): "D. 33.1", ("D", ("an", "leg")): "D. 33.1", ("D", ("annu", "leg")): "D. 33.1",
}


def match_title(text: str, work: str, start: int) -> dict:
    abbr = title_phrase(text, start, canon=work in ("X", "VI", "Clem"))
    if not abbr and work in ("X", "VI", "Clem"):
        abbr = title_phrase(text, start)
    if work == "D" and len(abbr) == 1 and abbr[0] in ("le", "leg", "lega", "legat"):
        m = re.search(r"\ble\w*\.? ?(1|2|3|i|ii|iii)\b", norm(text))      # de legatis I/II/III = D. 30/31/32
        if m:
            n = {"1": 30, "i": 30, "2": 31, "ii": 31, "3": 32, "iii": 32}[m.group(1)]
            return {"work": work, "key": f"D. {n}.1", "title": "De legatis et fideicommissis", "status": "resolved", "candidates": [f"D. {n}.1"], "score": 1.0, "abbr": abbr}
    conv = CONVENTIONS.get((work, tuple(abbr)))
    if conv:
        return {"work": work, "key": conv, "title": None, "status": "resolved", "candidates": [conv], "score": 9.0, "abbr": abbr, "via": "convention"}
    if not abbr:
        return {"work": work, "key": None, "title": None, "status": "no_title", "candidates": [], "score": 0.0}
    global STRICT
    scored = []
    for strict in (True, False):        # exact prefixes first ("publicia." = Publiciana, not publicanis / publicis)
        STRICT = strict
        scored = [(s, key, title) for key, title, tw in title_index(work) if (s := score(abbr, tw)) > 0]
        STRICT = False
        if scored:
            top = [x for x in scored if x[0] >= max(scored)[0] - 0.01]
            if len({k for _, k, _ in top}) == 1 or not strict:
                break
    if not scored:
        return {"work": work, "key": None, "title": None, "status": "unresolved", "candidates": [], "score": 0.0, "abbr": abbr}
    scored.sort(reverse=True)
    top = [x for x in scored if x[0] >= scored[0][0] - 0.01]
    keys = sorted({k for _, k, _ in top})
    if work == "D" and len(abbr) == 1 and abbr[0] in ("le", "leg", "lega", "legat"):
        m = re.search(r"\ble\w*\.? ?(1|2|3|i|ii|iii)\b", norm(text))      # de legatis I/II/III = D. 30/31/32
        if m:
            n = {"1": 30, "i": 30, "2": 31, "ii": 31, "3": 32, "iii": 32}[m.group(1)]
            return {"work": work, "key": f"D. {n}.1", "title": "De legatis et fideicommissis", "status": "resolved", "candidates": [f"D. {n}.1"], "score": 1.0, "abbr": abbr}
    if len(keys) == 1:
        return {"work": work, "key": keys[0], "title": top[0][2], "status": "resolved", "candidates": keys, "score": scored[0][0], "abbr": abbr}
    return {"work": work, "key": None, "title": None, "status": "ambiguous", "candidates": keys[:6], "score": scored[0][0], "abbr": abbr}


def roman_to_int(s: str) -> int:
    if s.isdigit():
        return int(s)
    vals = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    for i, ch in enumerate(s):
        v = vals.get(ch, 0)
        total += -v if i + 1 < len(s) and vals.get(s[i + 1], 0) > v else v
    return total


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--stats":
        import collections, glob
        c = collections.Counter(); ex = collections.defaultdict(list)
        for f in glob.glob(str(ROOT / "site-data" / "annotations" / "references" / "*.json")):
            for doc, spans in json.loads(Path(f).read_text(encoding="utf-8")).items():
                prev = None
                for sp in spans:
                    if not sp["l"].startswith("CIT"):
                        continue
                    r = resolve(sp["t"], prev)
                    c[(sp["l"], r["status"])] += 1
                    if r["status"] == "resolved":
                        prev = r["key"]
                    if len(ex[r["status"]]) < 6:
                        ex[r["status"]].append((sp["t"], r.get("key") or r.get("candidates")))
        for k, v in sorted(c.items()):
            print(f"{k[0]:10} {k[1]:11} {v}")
        for st, xs in ex.items():
            print("--", st); [print("   ", x) for x in xs]
        return
    for a in sys.argv[1:]:
        print(json.dumps(resolve(a), ensure_ascii=False))


if __name__ == "__main__":
    main()
