#!/usr/bin/env python3
"""
The edition text of every line from the three readings (Gemini, PyLaia, kraken), decided by rules that
were measured against the gold standard (output/gold).

Per line, in this order:
  1  human decision: a gold or review answer (output/gold/**/gold.jsonl, not the automatic ones) or an
     earlier manual decision in the review tool (comment "reviewed <date>: gemini|pylaia|edited"): kept
  2  readings cleaned (gold_eval.clean: expansion brackets, q; -> que, accents, spacing around † and
     after punctuation); a reading with an abbreviation mark does not vote
  3  wording: two of three readings agree (letters level). The text of data/ stays as it is when its
     wording already is the majority's, when there is no majority, and when PyLaia and kraken outvote an
     earlier decision for Gemini (the two line recognisers share crops and errors)
  4  form of a changed line: Gemini's when Gemini has the winning wording (it has the better spacing and
     punctuation), else the majority reading closest in print form to the others
  5  marginal number: Gemini's when kraken reads the same; else the number data/ has; a missing number on
     a line with † from output/merge/margin_numbers.jsonl; otherwise none
  6  hyphen ¬ at the line end when a reading (or data/) has it and ends in the same word
Measured on all 429 hand-checked gold and review lines: 73 lines better, none worse (letters and print).

    python tools/final_text.py --check            # score the variants on the gold lines (rule 1 off)
    python tools/final_text.py --write            # write output/merge/final_text.jsonl

Nothing is written to the PageXML here.
"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "src" / "qa"))
from gold_eval import body, clean, fold, has_abbreviation, margin_number  # noqa: E402
from compare_ocr import dist  # noqa: E402

MERGE = ROOT / "output" / "merge"
GOLD = ROOT / "output" / "gold"
SRC = ("gemini", "pylaia", "kraken")
PUNCT = re.compile(r"^(.*?)([.,:;?!]*)$")


def toks(t):
    out = []
    for w in t.split():
        m = PUNCT.match(w)
        out.append((m.group(1), m.group(2)))
    return out


def vote_form(chosen: str, others: list[str]) -> str:
    """punctuation after each word and the capital of each word by majority of the readings that have
    the same word at that place; ties keep the chosen reading's form"""
    C = toks(chosen)
    key = lambda w: fold(w, "letters")
    votes = [collections.Counter([p]) for _, p in C]
    caps = [collections.Counter([w[:1].isupper()]) for w, _ in C]
    for o in others:
        O = toks(o)
        sm = difflib.SequenceMatcher(a=[key(w) for w, _ in C], b=[key(w) for w, _ in O], autojunk=False)
        for op, a1, a2, b1, b2 in sm.get_opcodes():
            if op == "equal":
                for d in range(a2 - a1):
                    votes[a1 + d][O[b1 + d][1]] += 1
                    caps[a1 + d][O[b1 + d][0][:1].isupper()] += 1
    out = []
    for (w, p), v, c in zip(C, votes, caps):
        best_p = v.most_common(1)[0]
        p2 = best_p[0] if best_p[1] >= 2 else p
        cap = c.most_common(1)[0]
        if cap[1] >= 2 and w[:1].isalpha():
            w = (w[:1].upper() if cap[0] else w[:1].lower()) + w[1:]
        out.append(w + p2)
    return " ".join(out)


class Picker:
    """MediBerta pseudo-log-likelihood over the whole line, for lines without a majority"""
    def __init__(self):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        p = str(Path.home() / "github" / "MediBerta")
        self.torch = torch; self.tok = AutoTokenizer.from_pretrained(p)
        self.m = AutoModelForMaskedLM.from_pretrained(p, dtype=torch.float16).cuda().eval()

    def score(self, t):
        import unicodedata
        t = unicodedata.normalize("NFD", body(t).lower().replace("¬", "")); t = "".join(c for c in t if not unicodedata.combining(c))
        t = " ".join(re.findall(r"[a-z]+", t.replace("j", "i")))
        ids = self.tok(t, return_tensors="pt", truncation=True, max_length=128)["input_ids"][0]
        if len(ids) < 3:
            return -1e9
        b = ids.repeat(len(ids) - 2, 1)
        for r in range(len(ids) - 2):
            b[r, r + 1] = self.tok.mask_token_id
        with self.torch.no_grad():
            lp = self.m(b.cuda()).logits.float().log_softmax(-1)
        return sum(lp[r, r + 1, ids[r + 1]].item() for r in range(len(ids) - 2))


def current(r: dict) -> str:
    """the text in data/ now (index 0)"""
    return clean(r[r["valid"]] if r.get("valid") in SRC else r["gemini"])


def decide(r: dict, number, fallback: str = "gemini", form: bool = True, picker=None, numbers: str = "rule",
           keep_gemini: bool = False) -> tuple[str, str]:
    R = {k: clean(r[k]) for k in SRC if (r.get(k) or "").strip()}
    voters = {k: v for k, v in R.items() if not has_abbreviation(v)} or R
    L = {k: fold(body(v), "letters") for k, v in voters.items()}
    groups = collections.defaultdict(list)
    for k, v in L.items():
        groups[v].append(k)
    best = max(groups.values(), key=len) if groups else []
    guarded = (keep_gemini and sorted(best) == ["kraken", "pylaia"] and "gemini" in L
               and r.get("valid") == "gemini" and fold(body(current(r)), "letters") == L["gemini"])
    if guarded:
        # PyLaia and kraken share line crops and errors; an earlier decision for Gemini against them stays
        chosen = None; text = current(r); how = "PyLaia+kraken against an earlier choice of Gemini: data/ kept"
    elif len(best) >= 2 and fold(body(current(r)), "letters") == next(iter(L[k] for k in best)):
        chosen = None; text = current(r); how = "majority agrees with data/: kept as it is"
    elif len(best) >= 2:
        how = "majority " + "+".join(best)
        if form == "gemini" and "gemini" in best:
            chosen = "gemini"            # same wording: Gemini sees the page and has the better spacing and punctuation
        elif form:
            # the reading of the winning group closest in print form to all readings
            chosen = min(best, key=lambda k: sum(dist(fold(body(R[k])), fold(body(R[o]))) for o in R if o != k))
        else:
            chosen = best[0]
        text = R[chosen]
    elif fallback == "data":
        chosen = None; text = current(r); how = "no majority: text of data/ kept"
    elif fallback == "mediberta" and picker and len(voters) > 1:
        chosen = max(voters, key=lambda k: picker.score(voters[k])); text = R[chosen]; how = "MediBerta " + chosen
    else:
        chosen = "gemini" if "gemini" in R else next(iter(R), "gemini"); text = R.get(chosen, ""); how = "fallback " + chosen
    t = body(text)
    if form is True and chosen:
        t = vote_form(t, [body(R[k]) for k in R if k != chosen])
    # hyphen at the line end
    if not t.endswith("¬"):
        last = fold((t.split() or [""])[-1], "letters")
        for k, v in list(R.items()) + [("data", current(r))]:
            if v.rstrip().endswith(("¬", "-")) and fold((v.split() or [""])[-1], "letters") == last:
                t = t.rstrip("-") + "¬"; break
    if numbers == "careful":
        g, k2, now = margin_number(R.get("gemini", "")), margin_number(R.get("kraken", "")), margin_number(current(r))
        cross = any("†" in v for v in R.values())
        if g and g == k2:
            n = g
        elif now:
            n = now
        elif cross and number not in (None, "open"):
            n = number
        else:
            n = None
    elif numbers == "agree":
        # a number changes only when Gemini and kraken read the same one; otherwise the number of data/ stays
        g, k2, now = margin_number(R.get("gemini", "")), margin_number(R.get("kraken", "")), margin_number(current(r))
        n = g if g and g == k2 else (now or None)
    else:
        n = number if number not in (None, "open") else (margin_number(R.get("gemini", "")) or None)
    return ((f"{n} " if n else "") + t).strip(), how


def heading_number(t: str) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from segment_consilia import ConsiliumSegmenter
    ok, _, n, _ = ConsiliumSegmenter._is_heading(t.strip())
    return n if ok else 0


def fix_roman(t: str) -> str:
    """kraken writes U for V in Roman numerals: CUI -> CVI"""
    return re.sub(r"\b[IVXLCDMU]+\b", lambda m: m.group(0).replace("U", "V"), t)


def guard_title(new: str, old: str) -> tuple[str, str | None]:
    """a heading line keeps its text unless the old one yields no consilium number and the new one does:
    the numbers decide the ids that translations, keywords, annotations and links hang on"""
    new, old = fix_roman(new), fix_roman(old)
    if heading_number(old) == 0 and heading_number(new) > 0:
        return new, None
    return old, "heading: kept (its number decides the consilium id)"


def line_types(root: Path) -> dict:
    import xml.etree.ElementTree as ET
    T = "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}"
    out = {}
    for f in root.glob("*/page/*.xml"):
        for tl in ET.parse(f).getroot().iter(T + "TextLine"):
            m = re.search(r"type:(\w+)", tl.get("custom") or "")
            out[(f.parts[-3], f.name, tl.get("id"))] = m.group(1) if m else ""
    return out


def human_answers() -> dict:
    """gold and review answers given by hand, by (print, page, line_id)"""
    out = {}
    for d in [GOLD / "round1_400", GOLD / "round2_300", GOLD, GOLD / "review"]:
        sp, ap = d / "sample.json", d / "gold.jsonl"
        if not sp.exists() or not ap.exists():
            continue
        S = {s["id"]: s for s in json.loads(sp.read_text(encoding="utf-8"))}
        last = {}
        for l in ap.open(encoding="utf-8"):
            a = json.loads(l); last[a["id"]] = a
        for i, a in last.items():
            if i in S and "auto" not in a.get("flags", []):
                s = S[i]; out[(s["print"], s["page"], s["line_id"])] = a["gold"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--fallback", default="gemini", choices=("gemini", "mediberta"))
    a = ap.parse_args()
    cur = {}
    for l in (MERGE / "kraken_current.jsonl").open(encoding="utf-8"):
        r = json.loads(l); cur[(r["print"], r["page"], r["line_id"])] = r
    MN = {(o["print"], o["page"], o["line_id"]): o["decision"] for o in map(json.loads, (MERGE / "margin_numbers.jsonl").open())}
    if a.check:
        picker = Picker()
        S = {s["id"]: s for s in json.loads((GOLD / "sample.json").read_text(encoding="utf-8"))}
        A = {}
        for l in (GOLD / "gold.jsonl").open(encoding="utf-8"):
            x = json.loads(l); A[x["id"]] = x
        variants = {
            "data/ now (valid reading)": lambda r, k: clean(r[r["valid"]] if r.get("valid") in SRC else r["gemini"]),
            "majority, Gemini fallback, no form vote": lambda r, k: decide(r, MN.get(k), "gemini", form=False)[0],
            "majority, Gemini fallback, form vote": lambda r, k: decide(r, MN.get(k), "gemini", form=True)[0],
            "majority, MediBerta fallback, form vote": lambda r, k: decide(r, MN.get(k), "mediberta", True, picker)[0],
            "majority, Gemini fallback, Gemini form": lambda r, k: decide(r, MN.get(k), "gemini", "gemini")[0],
            "majority, MediBerta fallback, Gemini form": lambda r, k: decide(r, MN.get(k), "mediberta", "gemini", picker)[0],
            "majority, data/ kept, Gemini form, number rule": lambda r, k: decide(r, MN.get(k), "data", "gemini")[0],
            "majority, data/ kept, Gemini form, numbers G=K": lambda r, k: decide(r, MN.get(k), "data", "gemini", numbers="agree")[0],
        }
        for name, f in variants.items():
            res = {}
            for lev in ("letters", "print"):
                ok = w = e = c = 0
                okd = nd = 0
                for i, x in A.items():
                    s = S[i]; k = (s["print"], s["page"], s["line_id"]); r = cur[k]; wt = s["weight"]
                    h, g = fold(f(r, k), lev), fold(clean(x["gold"]), lev)
                    ok += wt * (h == g); w += wt; e += wt * dist(h, g); c += wt * max(len(g), 1)
                    if s["stratum"] == "disputed":
                        nd += 1; okd += h == g
                res[lev] = f"{ok / w * 100:.1f} % / CER {e / c * 100:.3f} % (disputed {okd}/{nd})"
            print(f"{name:42} letters {res['letters']}   print {res['print']}")
    if a.write:
        picker = Picker() if a.fallback == "mediberta" else None
        human = human_answers()
        types = line_types(MERGE / "Baldo_29618397")
        stats = collections.Counter()
        with (MERGE / "final_text.jsonl").open("w", encoding="utf-8") as f:
            for k, r in cur.items():
                cm = r.get("comment") or ""
                if k in human:
                    t, how = clean(human[k]), "human (gold or review answer)"
                elif re.search(r"reviewed [0-9-]+: (gemini|pylaia|edited)\b", cm):
                    t, how = clean(r[r["valid"]] if r.get("valid") in SRC else r["gemini"]), "human (review tool)"
                else:
                    t, how = decide(r, MN.get(k), "data", "gemini", numbers="careful", keep_gemini=True)
                if types.get(k) == "cons_title":
                    if how.startswith("human"):
                        t = fix_roman(t)
                    else:
                        t, why = guard_title(t, (r[r["valid"]] if r.get("valid") in SRC else r["gemini"]).strip())
                        how = why or how
                stats[how.split()[0] + (" " + how.split()[1] if how.startswith("human") else "")] += 1
                f.write(json.dumps(dict(print=k[0], page=k[1], line_id=k[2], text=t, how=how), ensure_ascii=False) + "\n")
        print(dict(stats))
        # headings in sequence: within a print the consilia are numbered 1, 2, 3 ...; a heading whose number
        # fits neither neighbour, while both neighbours ask for the same other number and another reading
        # of the line gives it, takes that reading (neighbours counted with their own numbers, not chained)
        recs = [json.loads(l) for l in (MERGE / "final_text.jsonl").open(encoding="utf-8")]
        heads = collections.defaultdict(list)
        for i, o in enumerate(recs):
            if types.get((o["print"], o["page"], o["line_id"])) == "cons_title":
                heads[o["print"]].append(i)
        fixed = 0
        for prt, idx in heads.items():
            nums = [heading_number(recs[i]["text"]) for i in idx]
            for j, i in enumerate(idx):
                n = nums[j]
                prev = next(((nums[q], j - q) for q in range(j - 1, -1, -1) if nums[q]), None)
                nxt = next(((nums[q], q - j) for q in range(j + 1, len(idx)) if nums[q]), None)
                if not prev or not nxt:
                    continue
                e_prev, e_next = prev[0] + prev[1], nxt[0] - nxt[1]
                if n in (e_prev, e_next) or e_prev != e_next:
                    continue
                o = recs[i]; r = cur[(o["print"], o["page"], o["line_id"])]
                for alt in [current(r)] + [clean(r[x]) for x in SRC]:
                    alt = fix_roman(alt)
                    if heading_number(alt) == e_prev:
                        o["text"], o["how"] = alt, f"heading: reading that fits both neighbours ({e_prev})"
                        fixed += 1
                        break
        with (MERGE / "final_text.jsonl").open("w", encoding="utf-8") as f:
            for o in recs:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        print("headings set to the reading that fits both neighbours:", fixed)


if __name__ == "__main__":
    main()
