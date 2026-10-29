#!/usr/bin/env python3
"""
Decide the s/f disagreements between the three readings of a line (PyLaia, Gemini, kraken).

Line recognisers confuse long s and f, so a majority of readings does not settle these words. The
edition writes round s throughout, so every disputed word has exactly two candidates: the s-form and
the f-form. Three tiers decide, in this order:

  1 lexicon      one form is at least 20 times more frequent in the agreed text (lines on which all
                 three readings agree) and the other occurs at most three times there
  2 abbreviation fi./si, fin./sin, feu./seu: a period after the word in at least two readings marks the
                 abbreviation (f-form), no period in at least two readings the word (s-form)
  3 MediBerta    the masked language model (mschonhardt/MediBerta) scores both candidates in the
                 context of the line and its neighbours (pseudo-log-likelihood over the word and four
                 tokens on each side, each masked in turn); the higher score wins if the margin is at
                 least --margin, otherwise the site is left open

Before the run, each tier is measured on words with a known answer: occurrences of the same word
pairs in agreed lines, scored as if they were disputed.

    python tools/resolve_sf.py [--in output/qa/kraken_all_e31.jsonl] [--model ~/github/MediBerta]

Writes output/qa/sf_resolution.jsonl (one record per disputed word) and output/qa/sf_resolution.md.
Nothing is written to the PageXML.
"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "qa"))
from compare_ocr import normalise  # noqa: E402

WORD = re.compile(r"[^\W\d_]+|\d+|§", re.UNICODE)
ABBR = {("si", "fi"), ("sin", "fin"), ("seu", "feu")}


def tokens(text: str) -> list[tuple[str, bool]]:
    """(lower-case word with long s as s and diacritics removed, followed by a period)"""
    t = text.replace("ſ", "s").replace("¬", "")
    out = []
    for m in WORD.finditer(t):
        w = unicodedata.normalize("NFD", m.group(0).lower())
        w = "".join(c for c in w if not unicodedata.combining(c))
        out.append((w, t[m.end():m.end() + 1] == "."))
    return out


def hyphenated(text: str) -> bool:
    return text.rstrip().endswith(("¬", "-"))


def fold(w: str) -> str:
    return w.replace("v", "u").replace("j", "i")


def sf_pair(a: str, b: str) -> tuple[str, str] | None:
    """(s-form, f-form) if a and b differ only by s/f substitutions"""
    if len(a) != len(b) or a == b:
        return None
    if not all(x == y or {x, y} == {"s", "f"} for x, y in zip(a, b)):
        return None
    return (a, b) if a.count("s") > b.count("s") else (b, a)


class Scorer:
    def __init__(self, path: str):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.m = AutoModelForMaskedLM.from_pretrained(path, dtype=torch.float16).cuda().eval()

    def pll(self, words: list[str], i: int, window: int = 4) -> float:
        tok, torch = self.tok, self.torch
        text = " ".join(words)
        enc = tok(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=512)
        ids = enc["input_ids"][0]; offs = enc["offset_mapping"][0].tolist()
        start = sum(len(w) + 1 for w in words[:i]); end = start + len(words[i])
        tgt = [k for k, (a, b) in enumerate(offs) if b > a and a < end and b > start]
        if not tgt:
            return float("nan")
        lo = max(1, min(tgt) - window); hi = min(len(ids) - 1, max(tgt) + window + 1)
        pos = list(range(lo, hi))
        batch = ids.repeat(len(pos), 1)
        for r, k in enumerate(pos):
            batch[r, k] = tok.mask_token_id
        with torch.no_grad():
            lp = self.m(batch.cuda()).logits.float().log_softmax(-1)
        return sum(lp[r, k, ids[k]].item() for r, k in enumerate(pos))

    def decide(self, context: list[str], i: int, s_form: str, f_form: str) -> tuple[str, float]:
        a = list(context); a[i] = s_form
        b = list(context); b[i] = f_form
        sa, sb = self.pll(a, i), self.pll(b, i)
        return ("s" if sa >= sb else "f"), abs(sa - sb)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in", dest="inp", type=Path, default=ROOT / "output" / "qa" / "kraken_all_e31.jsonl")
    ap.add_argument("--model", default=str(Path.home() / "github" / "MediBerta"))
    ap.add_argument("--margin", type=float, default=None, help="MediBerta margin; default: calibrated on the test words")
    ap.add_argument("--test-per-pair", type=int, default=150)
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "qa" / "sf_resolution.jsonl")
    a = ap.parse_args()

    rows = [json.loads(l) for l in a.inp.open(encoding="utf-8")]
    # neighbours for context: records are in reading order within each region
    by_region = collections.defaultdict(list)
    for idx, r in enumerate(rows):
        by_region[(r["file"], r["region"])].append(idx)
    prev_next = {}
    for idxs in by_region.values():
        for j, idx in enumerate(idxs):
            prev_next[idx] = (idxs[j - 1] if j else None, idxs[j + 1] if j + 1 < len(idxs) else None)

    agreed = []
    vocab = collections.Counter()
    for idx, r in enumerate(rows):
        if normalise(r["gemini"], True) == normalise(r["pylaia"], True) == normalise(r["kraken"], True):
            agreed.append(idx)
            vocab.update(fold(w) for w, _ in tokens(r["gemini"]))

    def context(idx: int, words: list[str]) -> tuple[list[str], int]:
        p, n = prev_next[idx]
        before = [w for w, _ in tokens(rows[p]["gemini"])][-25:] if p is not None else []
        after = [w for w, _ in tokens(rows[n]["gemini"])][:25] if n is not None else []
        return before + words + after, len(before)

    # character 7-grams of the agreed vocabulary (with word boundaries), for inflected and rare forms
    grams = collections.Counter()
    for w, n in vocab.items():
        ww = f"^^^{w}$$$"
        for k in range(len(ww) - 6):
            grams[ww[k:k + 7]] += n

    def full_word(idx: int, src: str, wi: int, form: str) -> tuple[str, int]:
        """the candidate as a whole word: joined with the next line's first word if this is the last word of a
        hyphenated line, or with the previous line's last word if this is the first word after a hyphen;
        returns (word, offset of the candidate inside it)"""
        r = rows[idx]
        words = [w for w, _ in tokens(r[src])]
        p, n = prev_next[idx]
        if wi == len(words) - 1 and hyphenated(r[src]) and n is not None:
            nxt = [w for w, _ in tokens(rows[n][src] or rows[n]["gemini"])]
            if nxt:
                return form + nxt[0], 0
        if wi == 0 and p is not None and hyphenated(rows[p][src] or rows[p]["gemini"]):
            prv = [w for w, _ in tokens(rows[p][src] or rows[p]["gemini"])]
            if prv:
                return prv[-1] + form, len(prv[-1])
        return form, 0

    # ---- disputed sites ------------------------------------------------------------------------
    sites = []
    SRC = {"G": "gemini", "P": "pylaia", "K": "kraken"}
    for idx, r in enumerate(rows):
        if normalise(r["gemini"], True) == normalise(r["pylaia"], True) == normalise(r["kraken"], True):
            continue
        T = {k: tokens(r[v]) for k, v in SRC.items()}
        W = {k: [w for w, _ in v] for k, v in T.items()}
        found = {}
        for x, y in (("G", "P"), ("G", "K"), ("P", "K")):
            for op, a1, a2, b1, b2 in difflib.SequenceMatcher(a=W[x], b=W[y], autojunk=False).get_opcodes():
                if op != "replace" or a2 - a1 != b2 - b1:
                    continue
                for o in range(a2 - a1):
                    pr = sf_pair(W[x][a1 + o], W[y][b1 + o])
                    if not pr:
                        continue
                    rel = (a1 + o) / max(len(W[x]), 1)
                    key = (pr, round(rel, 1))
                    pos = found.setdefault(key, {})
                    pos[x] = a1 + o; pos[y] = b1 + o
        for (pr, rel), pos in found.items():
            s_form, f_form = pr
            # the third reading: the occurrence of either form nearest to the same relative position
            for k in W:
                if k not in pos:
                    cands = [i for i, w in enumerate(W[k]) if w in pr]
                    if cands:
                        pos[k] = min(cands, key=lambda i: abs(i / max(len(W[k]), 1) - rel))
            votes = {k: (("s" if W[k][pos[k]] == s_form else "f") if k in pos else "-") for k in W}
            period = {k: T[k][pos[k]][1] for k in pos}
            prev = {k: (W[k][pos[k] - 1] if pos[k] > 0 else "^") for k in pos}
            anchor = "G" if "G" in pos else next(iter(pos))
            sites.append(dict(idx=idx, s=s_form, f=f_form, anchor=anchor, word_index=pos[anchor], votes=votes,
                              period=period, prev=prev))

    # ---- tiers ---------------------------------------------------------------------------------
    def lexicon(s_word: str, f_word: str):
        cs, cf = vocab[fold(s_word)], vocab[fold(f_word)]
        if cs >= 20 * max(cf, 1) and cf <= 3:
            return "s"
        if cf >= 20 * max(cs, 1) and cs <= 3:
            return "f"
        return None

    def ngram(s_word: str, f_word: str):
        """compare the 7-grams around each differing position"""
        sw, fw = f"^^^{fold(s_word)}$$$", f"^^^{fold(f_word)}$$$"
        diffs = [k for k, (x, y) in enumerate(zip(sw, fw)) if x != y]
        if not diffs:
            return None
        cs = min(grams[sw[max(0, k - 3):k + 4]] for k in diffs)
        cf = min(grams[fw[max(0, k - 3):k + 4]] for k in diffs)
        if cs >= 20 * max(cf, 1) and cf <= 3:
            return "s"
        if cf >= 20 * max(cs, 1) and cs <= 3:
            return "f"
        return None

    CITE = {"l", "c", "in", "leg", "lege", "cap", "ca"}

    def abbreviation(s_form: str, f_form: str, period: dict, prev: dict):
        if (s_form, f_form) not in ABBR:
            return None
        with_period = sum(1 for v in period.values() if v) * 2 > len(period)
        pv = collections.Counter(prev.values()).most_common(1)[0][0] if prev else ""
        if (s_form, f_form) == ("si", "fi"):
            if with_period and pv == "cum":
                return "s"                      # cum si. = cum similibus
            if with_period and (pv in CITE or pv == "§"):
                return "f"                      # l. fi., §. fi., c. fi., in fi.
            if not with_period:
                return "s"
            return None
        return "f" if with_period else "s"      # fin. / sin, feu. / seu

    scorer = Scorer(a.model)

    # ---- calibration on words with a known answer ------------------------------------------------
    pair_types = collections.Counter((st["s"], st["f"]) for st in sites)
    random.seed(23)
    test = []
    occ = collections.defaultdict(list)
    for idx in agreed:
        T = tokens(rows[idx]["gemini"])
        for wi, (w, pd) in enumerate(T):
            for (s_form, f_form) in pair_types:
                if w in (s_form, f_form):
                    occ[(s_form, f_form)].append((idx, wi, w, pd))
    for pr, lst in occ.items():
        for idx, wi, w, pd in random.sample(lst, min(a.test_per_pair, len(lst))):
            test.append(dict(idx=idx, wi=wi, s=pr[0], f=pr[1], truth="s" if w == pr[0] else "f", period=pd))
    # extra test words at line edges (fragments of hyphenated words), where an s/f swap gives the other form
    edge = []
    for idx in agreed:
        T = tokens(rows[idx]["gemini"])
        if T and hyphenated(rows[idx]["gemini"]):
            w = T[-1][0]
            if "s" in w or "f" in w:
                edge.append((idx, len(T) - 1, w))
    for idx, wi, w in random.sample(edge, min(600, len(edge))):
        k = random.choice([i for i, c in enumerate(w) if c in "sf"])
        other = w[:k] + ("f" if w[k] == "s" else "s") + w[k + 1:]
        s_form, f_form = (w, other) if w[k] == "s" else (other, w)
        test.append(dict(idx=idx, wi=wi, s=s_form, f=f_form, truth="s" if w == s_form else "f", period=False))
    calib = collections.defaultdict(list)
    for t in test:
        words = [w for w, _ in tokens(rows[t["idx"]]["gemini"])]
        sw, off_s = full_word(t["idx"], "gemini", t["wi"], t["s"])
        fw, _ = full_word(t["idx"], "gemini", t["wi"], t["f"])
        lx = lexicon(sw, fw) or ngram(sw, fw)
        if lx:
            calib["lexicon"].append(lx == t["truth"]); continue
        prevw = words[t["wi"] - 1] if t["wi"] else "^"
        ab = abbreviation(t["s"], t["f"], {"G": t["period"]}, {"G": prevw})
        if ab:
            calib["abbreviation"].append(ab == t["truth"]); continue
        ctx, off = context(t["idx"], words)
        choice, margin = scorer.decide(ctx, off + t["wi"], t["s"], t["f"])
        calib["mlm"].append((margin, choice == t["truth"]))
    # smallest margin at which MediBerta reaches 97 % precision on the test words
    mlm = sorted(calib["mlm"], key=lambda x: -x[0])
    thr = a.margin
    if thr is None:
        thr = float("inf")
        right = 0
        for n, (m, ok) in enumerate(mlm, 1):
            right += ok
            if n >= 30 and right / n >= 0.97:
                thr = m
        if thr == float("inf"):
            thr = 1e9
    mlm_acc_all = sum(ok for _, ok in mlm) / max(len(mlm), 1)
    mlm_kept = [(m, ok) for m, ok in mlm if m >= thr]

    # ---- decide the disputed sites ---------------------------------------------------------------
    out = []
    stats = collections.Counter()
    for st in sites:
        r = rows[st["idx"]]
        src = SRC[st["anchor"]]
        sw, _ = full_word(st["idx"], src, st["word_index"], st["s"])
        fw, _ = full_word(st["idx"], src, st["word_index"], st["f"])
        decision, tier, margin = lexicon(sw, fw), "lexicon", None
        if decision is None:
            decision, tier = ngram(sw, fw), "ngram"
        if decision is None:
            decision, tier = abbreviation(st["s"], st["f"], st["period"], st["prev"]), "abbreviation"
        if decision is None:
            words = [w for w, _ in tokens(r[src])]
            ctx, off = context(st["idx"], words)
            choice, margin = scorer.decide(ctx, off + st["word_index"], st["s"], st["f"])
            decision, tier = (choice, "mediberta") if margin >= thr else (None, "open")
        stats[(tier, decision)] += 1
        out.append({"file": r["file"], "line_id": r["line_id"], "region": r["region"], "s_form": st["s"], "f_form": st["f"], "s_word": sw, "f_word": fw,
                    "votes": st["votes"], "period": st["period"], "decision": decision, "word": (st["s"] if decision == "s" else st["f"] if decision == "f" else None),
                    "tier": tier, "margin": None if margin is None else round(margin, 2),
                    "gemini": r["gemini"], "pylaia": r["pylaia"], "kraken": r["kraken"]})
    with a.out.open("w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    # ---- report --------------------------------------------------------------------------------
    def pct(lst):
        return f"{sum(lst) / max(len(lst), 1) * 100:.1f} % of {len(lst)}"
    L = ["# s/f decisions\n", f"Disputed s/f words: {len(sites):,} in {len({s['idx'] for s in sites}):,} lines\n",
         "## Accuracy on words with a known answer (agreed lines)\n",
         f"- lexicon and 7-gram (whole words, joined across hyphens): {pct(calib['lexicon'])}",
         f"- abbreviation (period): {pct(calib['abbreviation'])}",
         f"- MediBerta, all margins: {mlm_acc_all * 100:.1f} % of {len(mlm)}",
         f"- MediBerta with margin >= {thr:.2f}: {pct([ok for _, ok in mlm_kept])} (the rest would stay open)\n",
         "## Decisions on the disputed words\n", "| tier | s | f | open |", "|---|---|---|---|"]
    for tier in ("lexicon", "ngram", "abbreviation", "mediberta", "open"):
        L.append(f"| {tier} | {stats[(tier, 's')]:,} | {stats[(tier, 'f')]:,} | {stats[(tier, None)]:,} |")
    agree = collections.Counter()
    for o in out:
        if o["decision"]:
            for k, v in o["votes"].items():
                agree[(k, v == o["decision"])] += 1
    L += ["\n## How often each reading had the decided form\n"]
    for k, name in (("G", "Gemini"), ("P", "PyLaia"), ("K", "kraken")):
        tot = agree[(k, True)] + agree[(k, False)]
        L.append(f"- {name}: {agree[(k, True)] / max(tot, 1) * 100:.1f} % of {tot:,}")
    L.append("\n## Samples\n")
    random.seed(5)
    for tier in ("lexicon", "ngram", "abbreviation", "mediberta", "open"):
        sel = [o for o in out if o["tier"] == tier]
        L.append(f"\n### {tier}\n")
        for o in random.sample(sel, min(10, len(sel))):
            L.append(f"- **{o['s_word']} / {o['f_word']}** -> {o['word'] or 'open'} (margin {o['margin']})  ")
            L.append(f"  G: `{o['gemini'][:100]}`  ")
            L.append(f"  P: `{o['pylaia'][:100]}`  ")
            L.append(f"  K: `{o['kraken'][:100]}`")
    a.out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:24]))


if __name__ == "__main__":
    main()
