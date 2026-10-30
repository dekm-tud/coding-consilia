"""
Full-text search over the Latin text of the consilia, with the query syntax and the lemma logic of the search page
(assets/common.js), for the MCP server and other Python tools.

    words            all of them in the same consilium (feudum vasallus)
    OR, |            either word (femina OR mulier)
    -word            consilia without that word (feudum -vasallus)
    "word"           that form only, without the other forms of its lemma ("heredem")
    a, b             several terms compared: each is searched on its own

A word is normalised as on the website (lower case, j = i, v = u, ae and ę = e) and matches every corpus form that
begins with it and every form of its lemma (site-data/lemmas.json). Scores follow BM25 per field with the boosts of
the search page (title 3, argument and casus 1.5, text 1), and forms found only by their beginning count less, as in
MiniSearch. Words of more than four letters also match forms one or more letters off (15 % of the length, rounded),
as the fuzzy matching of the browser does (vasallus also finds vassallus).

    from fulltext import FullText
    ft = FullText(consilia)                       # {id: record} as in output/consilia.json
    ft.search("feudum -vasallus")                 # [(id, score, fields)]
    ft.profile(["feudum", "vasallus"])            # counts per print and along the prints
"""
from __future__ import annotations

import bisect
import json
import math
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {"title": 3.0, "summary": 1.5, "casus": 1.5, "body": 1.0}
TOKEN = re.compile(r"[^\W_]+", re.U)          # runs of letters and digits, as MiniSearch splits at spaces and punctuation
QUOTES = "\"“”„«»"
K1, B, D = 1.2, 0.7, 0.5                       # BM25+ parameters of MiniSearch
PREFIX_WEIGHT = 0.375
FUZZY_WEIGHT, FUZZY = 0.45, 0.15


def normalise(t: str) -> str:
    return t.lower().replace("æ", "e").replace("ę", "e").replace("ae", "e").replace("j", "i").replace("v", "u")


def split_groups(raw: str) -> list[str]:
    """Terms to compare, separated by commas or semicolons outside quotation marks."""
    out, cur, quoted = [], "", False
    for ch in raw:
        if ch in QUOTES:
            quoted = not quoted
        if not quoted and ch in ",;":
            if cur.strip():
                out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def parse_group(raw: str) -> tuple[list[list[dict]], list[dict]]:
    """-> (clauses, negated words): the clauses are joined by AND, the words within a clause by OR."""
    toks = []
    for m in re.finditer(r'(-?)["“”„«»]([^"“”„«»]*)["“”„«»]?|(\S+)', raw):
        if m.group(3) is not None:
            w = m.group(3)
            if w in ("OR", "|"):
                toks.append({"or": True})
                continue
            neg = len(w) > 1 and w[0] == "-"
            w = re.sub(r"^[\W_]+|[\W_]+$", "", w[1:] if neg else w)
            if w:
                toks.append({"w": w, "exact": False, "neg": neg})
        else:
            toks += [{"w": w, "exact": True, "neg": m.group(1) == "-"} for w in m.group(2).split()]
    clauses: list[list[dict]] = []
    neg: list[dict] = []
    join = False
    for t in toks:
        if t.get("or"):
            join = bool(clauses)
            continue
        if t["neg"]:
            neg.append(t)
        elif join:
            clauses[-1].append(t)
        else:
            clauses.append([t])
        join = False
    return clauses, neg


class FullText:
    def __init__(self, consilia: dict[str, dict], lemmas_path: Path = ROOT / "site-data" / "lemmas.json"):
        self.docs = consilia
        self.post: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))   # form -> field -> id -> tf
        self.flen: dict[str, dict[str, int]] = {f: {} for f in FIELDS}
        self.words: dict[str, int] = {}
        for cid, c in consilia.items():
            n = 0
            for f in FIELDS:
                toks = [normalise(t) for t in TOKEN.findall(self._field(c, f))]
                n += len(toks)
                self.flen[f][cid] = len(set(toks))
                for t in toks:
                    p = self.post[t][f]
                    p[cid] = p.get(cid, 0) + 1
            self.words[cid] = n
        self.avg = {f: (sum(v.values()) / max(len(v), 1)) for f, v in self.flen.items()}
        self.vocab = sorted(self.post)
        self.lemma_of: dict[str, str] = {}
        self.forms_of: dict[str, list[str]] = defaultdict(list)
        if lemmas_path.exists():
            forms = json.loads(lemmas_path.read_text(encoding="utf-8")).get("forms", {})
            for form, lemma in forms.items():
                self.lemma_of[form] = lemma
                self.forms_of[lemma].append(form)
            self.lemma_set = set(forms.values())
        else:
            self.lemma_set = set()

    @staticmethod
    def _field(c: dict, f: str) -> str:
        v = c.get(f) or ""
        if f == "summary" and not v:
            v = " ".join(a.get("text", "") for a in c.get("argument") or [])
        return " ".join(v) if isinstance(v, list) else v

    def expand(self, t: dict) -> dict[str, float]:
        """corpus forms a query word stands for, with their weight (1 for the form and its lemma, less for a prefix)"""
        f = normalise(t["w"])
        if t["exact"]:
            return {f: 1.0} if f in self.post else {}
        out: dict[str, float] = {}
        i = bisect.bisect_left(self.vocab, f)
        while i < len(self.vocab) and self.vocab[i].startswith(f):
            v = self.vocab[i]
            out[v] = 1.0 if v == f else PREFIX_WEIGHT * len(v) / (len(v) + 0.3 * (len(v) - len(f)))
            i += 1
        k = round(len(f) * FUZZY) if len(f) > 4 else 0
        if k:
            from rapidfuzz.distance import Levenshtein
            for v in self.vocab:
                if abs(len(v) - len(f)) <= k and v not in out:
                    d = Levenshtein.distance(f, v, score_cutoff=k)
                    if 0 < d <= k:
                        out[v] = FUZZY_WEIGHT * len(v) / (len(v) + d)
        lemma = self.lemma_of.get(f) or (f if f in self.lemma_set else None)
        if lemma:
            for v in [lemma, *self.forms_of.get(lemma, [])]:
                if v in self.post:
                    out[v] = 1.0
        return out

    def _word_scores(self, t: dict) -> dict[str, tuple[float, set]]:
        res: dict[str, tuple[float, set]] = {}
        n_docs = len(self.docs)
        for form, w in self.expand(t).items():
            for f, boost in FIELDS.items():
                p = self.post[form].get(f)
                if not p:
                    continue
                idf = math.log(1 + (n_docs - len(p) + 0.5) / (len(p) + 0.5))
                for cid, tf in p.items():
                    norm = 1 - B + B * self.flen[f][cid] / (self.avg[f] or 1)
                    s = w * boost * idf * (D + tf * (K1 + 1) / (tf + K1 * norm))
                    old = res.get(cid)
                    res[cid] = (s + (old[0] if old else 0), (old[1] if old else set()) | {f})
        return res

    def search_group(self, raw: str) -> dict[str, tuple[float, set]]:
        clauses, neg = parse_group(raw)
        if not clauses:
            return {}
        result: dict[str, tuple[float, set]] | None = None
        for clause in clauses:
            merged: dict[str, tuple[float, set]] = {}
            for t in clause:
                for cid, (s, fs) in self._word_scores(t).items():
                    o = merged.get(cid)
                    merged[cid] = (s + (o[0] if o else 0), fs | (o[1] if o else set()))
            if result is None:
                result = merged
            else:
                result = {cid: (result[cid][0] + s, result[cid][1] | fs) for cid, (s, fs) in merged.items() if cid in result}
        for t in neg:
            for cid in self._word_scores(t):
                result.pop(cid, None)
        return result or {}

    def search(self, raw: str) -> list[tuple[str, float, list[str]]]:
        """All matching consilia, best first, with the fields that matched; compared terms are joined by OR."""
        merged: dict[str, tuple[float, set]] = {}
        for g in split_groups(raw):
            for cid, (s, fs) in self.search_group(g).items():
                o = merged.get(cid)
                merged[cid] = (max(s, o[0]) if o else s, fs | (o[1] if o else set()))
        return [(cid, s, sorted(fs)) for cid, (s, fs) in sorted(merged.items(), key=lambda kv: -kv[1][0])]

    def occurrences(self, raw: str, ids=None) -> dict[str, int]:
        """how often the searched words of a term occur in each consilium (all fields), for the given ids"""
        clauses, _ = parse_group(raw)
        forms: set[str] = set()
        for clause in clauses:
            for t in clause:
                forms |= set(self.expand(t))
        occ: dict[str, int] = defaultdict(int)
        for form in forms:
            for p in self.post[form].values():
                for cid, tf in p.items():
                    if ids is None or cid in ids:
                        occ[cid] += tf
        return dict(occ)

    def snippet(self, raw: str, cid: str, width: int = 300) -> str:
        """a stretch of the text around the first searched word"""
        forms: set[str] = set()
        for g in split_groups(raw):
            for clause in parse_group(g)[0]:
                for t in clause:
                    forms |= set(self.expand(t))
        c = self.docs[cid]
        for f in ("body", "casus", "summary", "title"):
            text = self._field(c, f)
            for m in TOKEN.finditer(text):
                if normalise(m.group(0)) in forms:
                    a = max(0, m.start() - width // 2)
                    return ("…" if a else "") + text[a:a + width] + ("…" if a + width < len(text) else "")
        return self._field(c, "body")[:width]

    def profile(self, terms: list[str], ids: list[str] | None = None, section: int | None = None) -> dict:
        """Frequency of each term per print and along the prints, in sections of consecutive consilia of each print:
        consilia containing the term (count and share) and occurrences per 10,000 words."""
        pool = [cid for cid in (ids or self.docs)]
        by_vol: dict[str, list[str]] = defaultdict(list)
        for cid in pool:
            by_vol[self.docs[cid].get("volume", "")].append(cid)
        size = section or max(10, math.ceil(len(pool) / 40))
        pool_set = set(pool)
        hits = []
        for t in terms:
            found = {cid for cid in self.search_group(t) if cid in pool_set}
            occ = self.occurrences(t, found)
            hits.append({cid: occ.get(cid, 0) for cid in found})
        out = {"terms": [], "prints": [], "sections": [], "section_size": size}
        for t, occ in zip(terms, hits):
            words = sum(self.words[c] for c in pool)
            out["terms"].append({"term": t, "consilia": len(occ), "share_pct": round(100 * len(occ) / max(len(pool), 1), 2),
                                 "occurrences": sum(occ.values()), "per_10k_words": round(1e4 * sum(occ.values()) / max(words, 1), 2)})
        for v in sorted(by_vol):
            lst = sorted(by_vol[v], key=lambda c: (self.docs[c].get("pars") or 0, self.docs[c].get("n") or 0))
            words = sum(self.words[c] for c in lst)
            out["prints"].append({"print": v, "consilia": len(lst), "words": words, "terms": [
                {"term": t, "consilia": sum(1 for c in lst if c in occ), "share_pct": round(100 * sum(1 for c in lst if c in occ) / len(lst), 2),
                 "per_10k_words": round(1e4 * sum(occ.get(c, 0) for c in lst) / max(words, 1), 2)} for t, occ in zip(terms, hits)]})
            nb = max(1, round(len(lst) / size))
            for b in range(nb):
                part = lst[b * len(lst) // nb:(b + 1) * len(lst) // nb]
                if not part:
                    continue
                words = sum(self.words[c] for c in part)
                first, last = self.docs[part[0]], self.docs[part[-1]]
                label = lambda c: f"{c['pars']}.{c['n']}" if c.get("pars") else str(c.get("n"))
                out["sections"].append({"print": v, "from_n": label(first), "to_n": label(last), "consilia": len(part),
                                        "share_pct": [round(100 * sum(1 for c in part if c in occ) / len(part), 1) for occ in hits],
                                        "per_10k_words": [round(1e4 * sum(occ.get(c, 0) for c in part) / max(words, 1), 1) for occ in hits]})
        return out
