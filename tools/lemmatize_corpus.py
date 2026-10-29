#!/usr/bin/env python3
"""
Lemma dictionary for the full-text search: site-data/lemmas.json, {form: lemma} for the normalised word forms of
the Latin text (title, argumentum, casus, body of every consilium).

Two engines:
  --engine flair  (default) mschonhardt/latin-lemmatizer, a Flair seq2seq lemmatizer trained on ~1.6M sentences of
                  medieval Latin (Latin Text Archive); handles the e-for-ae spelling of the prints (heredem -> heres,
                  hereditatem -> hereditas, statuta -> statutum), where LatinCy fails. Runs in .venv (Flair).
  --engine spacy  LatinCy (https://huggingface.co/latincy, la_core_web_trf on a GPU), in .venv-latincy.

    .venv/bin/python tools/lemmatize_corpus.py                                   # Flair, GPU
    .venv-latincy/bin/python tools/lemmatize_corpus.py --engine spacy --model la_core_web_trf

Forms and lemmas are normalised exactly as the site normalises search terms (assets/common.js normalise: lower
case, ae/æ/ę -> e, j -> i, v -> u). A form gets the lemma the model assigns to it most often; forms whose lemma
equals the form, numbers, single letters and abbreviations with a dot are left out (the site falls back to the
form itself). The site loads the file in the search worker and the reader: indexing adds the lemma of every word,
a query word is replaced by its lemma (see assets/common.js). Rebuild after the text changes (stale entries are
harmless: unknown forms are searched as forms).

Setup (once):  python3 -m venv .venv-latincy && .venv-latincy/bin/pip install "spacy[cuda12x]" \
    https://huggingface.co/latincy/la_core_web_trf/resolve/main/la_core_web_trf-3.9.9-py3-none-any.whl
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site-data"
WORD = re.compile(r"^[a-z]+$")
# hand corrections of frequent legal terms the model gets wrong (checked against the corpus); applied last
OVERRIDES = {
    "heredem": "heres",                                             # model: heresis
    **{f: "legatum" for f in ("legata", "legati", "legato", "legatis", "legatorum", "legatos", "legatum")},   # model splits legatum / legatus / lego
}


def normalise(s: str) -> str:
    return s.lower().replace("æ", "e").replace("ę", "e").replace("ae", "e").replace("j", "i").replace("v", "u")


def texts():
    """(volume, text) for every section of every consilium, in the order of the shards"""
    for f in sorted((SITE / "text").glob("*.json")):
        for c in json.loads(f.read_text(encoding="utf-8")).values():
            for key in ("title", "argument", "casus", "body"):
                v = c.get(key) or []
                for s in (v if isinstance(v, list) else [v]):
                    s = s if isinstance(s, str) else (s.get("la") or "")
                    if s.strip():
                        yield s


def add(counts, form: str, lemma: str, next_is_dot: bool) -> None:
    f = normalise(form)
    if not WORD.match(f) or len(f) < 2:
        return
    if next_is_dot and len(form) <= 5:
        return                                                      # abbreviation ("ff.", "tit.", "gl.")
    l = normalise(lemma or "")
    if WORD.match(l):
        counts[f][l] += 1


def run_spacy(a, pieces, counts, t0):
    import spacy
    gpu = spacy.prefer_gpu()
    name = a.model or "la_core_web_trf"
    nlp = spacy.load(name)
    disable = [p for p in nlp.pipe_names if p in ("parser", "ner", "trf_vectors")]      # the normalisers and enclitic splitter are needed for good lemmas
    print(f"spaCy {name} ({nlp.meta.get('version')}), gpu {gpu}, disabled {disable}")
    with nlp.select_pipes(disable=disable):
        for k, doc in enumerate(nlp.pipe(pieces, batch_size=a.batch), 1):
            for t in doc:
                if t.is_punct:
                    continue
                nxt = doc[t.i + 1] if t.i + 1 < len(doc) else None
                add(counts, t.text, t.lemma_, nxt is not None and nxt.text == ".")
                # the enclitic splitter turns "uxorque" into "uxor" + "que": the site indexes the joined word
                if nxt is not None and not t.whitespace_ and normalise(nxt.text) in ("que", "ne", "ue") and len(t.text) >= 3:
                    add(counts, t.text + nxt.text, t.lemma_, False)
            if k % 2000 == 0:
                print(f"  {k}/{len(pieces)}  {time.time() - t0:.0f}s", flush=True)
    return name, nlp.meta.get("version")


TOK = re.compile(r"[^\W\d_]+|\d+|[^\w\s]")
SENT_END = {".", ";", ":", "?", "!"}


def run_flair(a, pieces, counts, t0):
    import torch
    import torch.nn.utils.rnn as rnn
    _pack = rnn.pack_padded_sequence

    def pack(input, lengths, *args, **kw):              # Flair 0.15 passes the lengths on the GPU; torch wants them on the CPU
        return _pack(input, lengths.cpu() if torch.is_tensor(lengths) else lengths, *args, **kw)
    rnn.pack_padded_sequence = pack
    import flair
    from flair.data import Sentence
    from flair.models import Lemmatizer
    from huggingface_hub import hf_hub_download
    flair.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = a.model or "mschonhardt/latin-lemmatizer"
    m = Lemmatizer.load(hf_hub_download(name, "best-model.pt") if "/" in name and not Path(name).exists() else name)
    print(f"Flair {name} on {flair.device}")
    sents = []
    for p in pieces:                                     # sentences of at most 60 tokens, punctuation as own tokens
        cur = []
        for tok in TOK.findall(p):
            cur.append(tok)
            if (tok in SENT_END and len(cur) >= 4) or len(cur) >= 60:
                sents.append(cur); cur = []
        if cur:
            sents.append(cur)
    print(f"{len(sents)} sentences")
    B = 2000
    for i in range(0, len(sents), B):
        batch = [Sentence(toks) for toks in sents[i:i + B]]
        m.predict(batch, mini_batch_size=a.batch, label_name="lemma")
        for s, toks in zip(batch, sents[i:i + B]):
            for j, t in enumerate(s):
                nxt = toks[j + 1] if j + 1 < len(toks) else None
                add(counts, t.text, t.get_label("lemma").value, nxt == ".")
        print(f"  {min(i + B, len(sents))}/{len(sents)}  {time.time() - t0:.0f}s", flush=True)
    return name, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--engine", choices=["flair", "spacy"], default="flair")
    ap.add_argument("--model", default=None, help="spaCy model (default la_core_web_trf) or Flair model (default mschonhardt/latin-lemmatizer)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-chars", type=int, default=4000, help="longer sections are cut at sentence ends into pieces of this size")
    ap.add_argument("--out", type=Path, default=SITE / "lemmas.json")
    ap.add_argument("--limit", type=int, help="first N sections only (test)")
    ap.add_argument("--overrides-only", action="store_true", help="apply OVERRIDES to an existing --out file, no lemmatisation")
    a = ap.parse_args()
    if a.overrides_only:
        d = json.loads(a.out.read_text(encoding="utf-8"))
        d["forms"] = dict(sorted({**d["forms"], **OVERRIDES}.items()))
        d["_meta"].update(n_forms=len(d["forms"]), n_lemmas=len(set(d["forms"].values())), overrides=len(OVERRIDES))
        a.out.write_text(json.dumps(d, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"{a.out}: {len(OVERRIDES)} overrides applied")
        return
    pieces = []
    for s in texts():
        while len(s) > a.max_chars:
            cut = max(s.rfind(". ", 0, a.max_chars), s.rfind(": ", 0, a.max_chars), s.rfind("; ", 0, a.max_chars))
            cut = cut + 1 if cut > a.max_chars // 2 else a.max_chars
            pieces.append(s[:cut]); s = s[cut:]
        pieces.append(s)
        if a.limit and len(pieces) >= a.limit:
            break
    print(f"{len(pieces)} pieces, {sum(map(len, pieces)) / 1e6:.1f} M chars")
    counts: dict[str, Counter] = defaultdict(Counter)
    t0 = time.time()
    if a.engine == "spacy":
        model_name, model_version = run_spacy(a, pieces, counts, t0)
    else:
        model_name, model_version = run_flair(a, pieces, counts, t0)
    forms = {}
    for f, c in counts.items():
        l, _ = c.most_common(1)[0]
        if l != f:
            forms[f] = l
    forms.update(OVERRIDES)
    meta = {"engine": a.engine, "model": model_name, "model_version": model_version, "n_forms_seen": len(counts), "n_forms": len(forms),
            "n_lemmas": len(set(forms.values())), "made": time.strftime("%Y-%m-%d"), "overrides": len(OVERRIDES),
            "note": "form -> lemma, normalised as assets/common.js normalise(); machine lemmatisation, see tools/lemmatize_corpus.py"}
    a.out.write_text(json.dumps({"_meta": meta, "forms": dict(sorted(forms.items()))}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{a.out}: {meta['n_forms']} forms -> {meta['n_lemmas']} lemmas ({a.out.stat().st_size / 1e6:.1f} MB), {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
