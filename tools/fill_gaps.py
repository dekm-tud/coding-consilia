#!/usr/bin/env python3
"""
Bring every print to the same level of derived layers: list what is missing, fill it with a language
model, and merge the results back into output/consilia.json (and the per-author and per-print copies).

What counts as a gap (per consilium):
  translation   English argumentum (when the print has one), English casus, German casus, and the English and
                German text itself when a consilium has no body translation at all (renumbered or newly cut ones)
  keywords      keywords_final empty (Panormitanus as a whole, a few Baldus consilia)
  subjects      no entry in site-data/annotations/subjects.json  -> run src/annotate/classify.py (printed by `plan`)

    python tools/fill_gaps.py plan                                   # counts, token volume, the commands to run
    python tools/fill_gaps.py translate --backend gemini --model gemini-3.8-flash
    python tools/fill_gaps.py keywords  --backend gemini --model gemini-3.8-flash
    python tools/fill_gaps.py translate --backend openai --base-url http://gpu:8000/v1 --model Qwen/Qwen3.6-27B --thinking --workers 16
    python tools/fill_gaps.py apply                                  # merge output/gaps/*.jsonl into consilia.json
Then: python src/build_site_data.py, re-embed (data revision changes), python src/export_tei.py.

Backends and the JSON-schema call are those of src/annotate/classify.py (Gemini REST with GEMINI_API_KEY,
or any OpenAI-compatible endpoint: vLLM, Blablador, OpenRouter; key from --key-env). Resumable: records
already in the output file are skipped. Nothing is written into consilia.json before `apply`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "annotate"))
from classify import GeminiBackend, OpenAIBackend  # noqa: E402

CONSILIA = ROOT / "output" / "consilia.json"
SUBJECTS = ROOT / "site-data" / "annotations" / "subjects.json"
VOCAB = ROOT / "site-data" / "vocabulary.json"
OUT = ROOT / "output" / "gaps"

TRANSLATE_SYSTEM = """You translate legal opinions (consilia) of the medieval ius commune from Latin into English and German. The source is a normalised reading text of a sixteenth-century print, made by text recognition and not proof-read.
Rules: scholarly and close to the text, in readable prose; nothing is shortened, paraphrased or added. Render the terminology of the learned law precisely and consistently (statutum: statute / Statut; beneficium: benefice / Pfründe; dos: dowry / Mitgift; exceptio: exception / Einrede). Citations of legal sources and of jurists stay untranslated exactly as printed ("ff. de pac. l. tale pactum", "C. de iud.", "Bar. in l. Lucius", "Inn. in c. cum olim"); only the words that introduce them are translated. A † at the start of a passage is kept at the start of its translation. An obviously garbled passage is translated by its sense and followed by [?]. Personal names keep the form of the source; place names take the usual English or German form.
The argumentum is a numbered list; keep its numbers and give one line per item, "1. ...", "2. ...". Return only the fields asked for, as JSON."""

KEYWORD_SYSTEM = """You index legal opinions (consilia) of the medieval ius commune with English keywords, as a legal historian would: the legal institutions, concepts, procedural devices, persons' roles and matters at issue. 8 to 12 keywords per opinion, lowercase, singular, two or three words at most, no proper names, no citations, no sentences. Prefer terms from the existing vocabulary of the edition when they fit; coin a new term only when the vocabulary has no adequate one. Order the keywords from the most central to the least. Return JSON only."""

TRANSLATE_SCHEMA = {"type": "object", "properties": {
    "argument_en": {"type": ["string", "null"], "description": "English argumentum, numbered lines, or null when none was asked for"},
    "casus_en": {"type": ["array", "null"], "items": {"type": "string"}, "description": "English casus, one string per Latin paragraph, same count and order"},
    "casus_de": {"type": ["array", "null"], "items": {"type": "string"}, "description": "German casus, one string per Latin paragraph, same count and order"},
    "body_en": {"type": ["array", "null"], "items": {"type": "string"}, "description": "English text, one string per Latin paragraph, same count and order"},
    "body_de": {"type": ["array", "null"], "items": {"type": "string"}, "description": "German text, one string per Latin paragraph, same count and order"}},
    "required": ["argument_en", "casus_en", "casus_de", "body_en", "body_de"], "additionalProperties": False}
KEYWORD_SCHEMA = {"type": "object", "properties": {"keywords": {"type": "array", "items": {"type": "string"}}},
                  "required": ["keywords"], "additionalProperties": False}


def load_consilia() -> dict:
    return json.loads(CONSILIA.read_text(encoding="utf-8"))["consilia"]


def tr(c: dict, lang: str, key: str):
    return ((c.get("translations") or {}).get(lang) or {}).get(key)


def translation_gaps(c: dict) -> dict:
    """which translated pieces are missing: {'argument_en', 'casus_en', 'casus_de'} subset"""
    g = {}
    if c.get("argument") and not (tr(c, "en", "argument") or "").strip():
        g["argument_en"] = True
    if c.get("casus"):
        for lang in ("en", "de"):
            have = tr(c, lang, "casus") or []
            if len([s for s in have if s and s.strip()]) < len(c["casus"]):
                g[f"casus_{lang}"] = True
    if c.get("body"):
        for lang in ("en", "de"):
            have = tr(c, lang, "body") or []
            if len(have) != len(c["body"]) or not any(s and s.strip() for s in have):
                g[f"body_{lang}"] = True
    return g


def keyword_gap(c: dict) -> bool:
    return not c.get("keywords_final") and bool((tr(c, "en", "body") or c.get("body")))


def subject_gaps(cons: dict) -> list[str]:
    docs = json.loads(SUBJECTS.read_text(encoding="utf-8"))["docs"] if SUBJECTS.exists() else {}
    return [cid for cid in cons if cid not in docs]


def est_tokens(chars: int) -> int:
    return chars // 4 + 1


# ── plan ──────────────────────────────────────────────────────────────────
def cmd_plan(args) -> None:
    cons = load_consilia()
    by_print: dict[str, dict] = {}
    chars = {"translate": 0, "keywords": 0, "subjects": 0}
    for cid, c in cons.items():
        p = by_print.setdefault(c["volume"], {"n": 0, "argument_en": 0, "casus_en": 0, "casus_de": 0, "body_en": 0, "body_de": 0, "keywords": 0, "subjects": 0})
        p["n"] += 1
        g = translation_gaps(c)
        for k in g:
            p[k] += 1
        if g:
            chars["translate"] += len(" ".join(a["text"] for a in c.get("argument") or [])) + sum(len(s) for s in c.get("casus") or []) + (sum(len(s) for s in c.get("body") or []) if "body_en" in g or "body_de" in g else 0)
        if keyword_gap(c):
            p["keywords"] += 1
            chars["keywords"] += min(sum(len(s) for s in (tr(c, "en", "body") or c.get("body") or [])), 16000) + 1500
    subj = subject_gaps(cons)
    for cid in subj:
        by_print[cons[cid]["volume"]]["subjects"] += 1
        chars["subjects"] += min(sum(len(s) for s in (tr(cons[cid], "en", "body") or [])), 24000) + 6000
    print(f"{'print':40} {'n':>5} {'arg_en':>7} {'cas_en':>7} {'cas_de':>7} {'body_en':>8} {'body_de':>8} {'keywords':>9} {'subjects':>9}")
    for v, p in sorted(by_print.items()):
        print(f"{v:40} {p['n']:5} {p['argument_en']:7} {p['casus_en']:7} {p['casus_de']:7} {p['body_en']:8} {p['body_de']:8} {p['keywords']:9} {p['subjects']:9}")
    tin = {k: est_tokens(v) for k, v in chars.items()}
    tout = {"translate": int(tin["translate"] * 1.3), "keywords": sum(1 for c in cons.values() if keyword_gap(c)) * 60, "subjects": len(subj) * 120}
    print("\nestimated tokens (input / output):")
    for k in tin:
        print(f"  {k:10} {tin[k]:>9,} / {tout[k]:>8,}")
    print(f"  {'total':10} {sum(tin.values()):>9,} / {sum(tout.values()):>8,}")
    if args.price_in or args.price_out:
        cost = sum(tin.values()) / 1e6 * args.price_in + sum(tout.values()) / 1e6 * args.price_out
        print(f"  at ${args.price_in}/M input and ${args.price_out}/M output: about ${cost:.2f} (thinking tokens not included)")
    else:
        print("  pass --price-in/--price-out (USD per million tokens of the chosen model) for a cost estimate")
    ids = ROOT / "output" / "gaps" / "subjects_missing_ids.txt"
    ids.parent.mkdir(parents=True, exist_ok=True)
    ids.write_text(",".join(subj), encoding="utf-8")
    print(f"\nsubjects: {len(subj)} consilia without a Layer A entry; ids in {ids.relative_to(ROOT)}. Run, with the backend of your choice:")
    print(f"  python src/annotate/classify.py --task subjects --backend gemini --model gemini-3.8-flash --split $(cat {ids.relative_to(ROOT)})")
    print("  then concatenate the new jsonl with the published run and re-export:")
    print("  cat output/annotations/subjects_qwen-qwen3-8-27b-fp8_p1.jsonl output/annotations/subjects_<model>_p1.jsonl > output/annotations/subjects_merged_p1.jsonl")
    print("  python src/annotate/export.py --subjects output/annotations/subjects_merged_p1.jsonl")


# ── backends ──────────────────────────────────────────────────────────────
def backend(args):
    ns = SimpleNamespace(model=args.model, temperature=args.temperature, max_tokens=args.max_tokens, timeout=args.timeout,
                         key_env=args.key_env or ("GEMINI_API_KEY" if args.backend == "gemini" else "OPENAI_API_KEY"),
                         base_url=args.base_url, thinking=args.thinking, thinking_level=args.thinking_level)
    return GeminiBackend(ns) if args.backend == "gemini" else OpenAIBackend(ns)


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(l)["id"] for l in path.open(encoding="utf-8") if l.strip()}


def run_jobs(args, jobs: list[tuple[str, str]], system: str, schema: dict, out_path: Path, post) -> None:
    """jobs: (id, user prompt). Each answer is validated by post(id, answer) -> record or None (retry once)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    have = done_ids(out_path)
    jobs = [j for j in jobs if j[0] not in have]
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"{len(jobs)} requests ({len(have)} already in {out_path.relative_to(ROOT)})")
    be = backend(args)
    t0 = time.time()
    n_ok = n_fail = 0
    with out_path.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=args.workers) as ex:
        def one(cid, user):
            last = None
            for attempt in range(2):
                ans, usage = be.complete(system, [], user, schema)
                rec = post(cid, ans)
                if rec is not None:
                    rec.update({"id": cid, "model": args.model, "backend": args.backend, "usage": usage, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
                    return rec
                last = ans
            raise ValueError(f"answer rejected twice: {json.dumps(last, ensure_ascii=False)[:200]}")
        futs = {ex.submit(one, cid, user): cid for cid, user in jobs}
        for f in as_completed(futs):
            cid = futs[f]
            try:
                rec = f.result()
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
                n_ok += 1
                with be.lock:
                    be.usage["prompt"] += rec["usage"]["prompt"]; be.usage["output"] += rec["usage"]["output"]; be.usage["calls"] += 1
            except Exception as e:
                n_fail += 1
                print(f"  {cid}: {e}", file=sys.stderr)
            if (n_ok + n_fail) % 20 == 0:
                print(f"  {n_ok + n_fail}/{len(jobs)} done, {n_fail} failed, {time.time() - t0:.0f} s, tokens in/out {be.usage['prompt']:,}/{be.usage['output']:,}")
    print(f"done: {n_ok} written, {n_fail} failed, {time.time() - t0:.0f} s, tokens in/out {be.usage['prompt']:,}/{be.usage['output']:,} -> {out_path.relative_to(ROOT)}")


# ── translate ─────────────────────────────────────────────────────────────
def translate_prompt(c: dict, gaps: dict) -> str:
    parts = [f"Consilium: {c.get('title_display') or c.get('title', '')} ({c['volume']})"]
    want = []
    if "argument_en" in gaps:
        parts.append("ARGUMENTUM (Latin, numbered):\n" + "\n".join(f"{a['n']}. {a['text']}" for a in c["argument"]))
        want.append("argument_en: the argumentum in English, numbered lines")
    else:
        parts.append("argument_en: null (not needed)")
    if "casus_en" in gaps or "casus_de" in gaps:
        parts.append("CASUS (Latin, one paragraph per line, keep the count):\n" + "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(c["casus"])))
        if "casus_en" in gaps:
            want.append(f"casus_en: the casus in English as a list of {len(c['casus'])} strings")
        if "casus_de" in gaps:
            want.append(f"casus_de: the casus in German as a list of {len(c['casus'])} strings")
    if "body_en" in gaps or "body_de" in gaps:
        parts.append("TEXT (Latin, one paragraph per line, keep the count; a leading † stays):\n" + "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(c["body"])))
        for lang, name in (("en", "English"), ("de", "German")):
            if f"body_{lang}" in gaps:
                want.append(f"body_{lang}: the text in {name} as a list of {len(c['body'])} strings")
    for k in ("casus_en", "casus_de", "body_en", "body_de"):
        if k not in gaps:
            parts.append(f"{k}: null (not needed)")
    # context for the translator: the English text when it exists
    en = tr(c, "en", "body") or []
    if en and "body_en" not in gaps:
        parts.append("CONTEXT (English translation of the opinion's text, for terminology only, do not return it):\n" + " ".join(en)[:3000])
    parts.append("Return: " + "; ".join(want) + ".")
    return "\n\n".join(parts)


def cmd_translate(args) -> None:
    cons = load_consilia()
    jobs, meta = [], {}
    for cid, c in cons.items():
        g = translation_gaps(c)
        if g:
            jobs.append((cid, translate_prompt(c, g))); meta[cid] = g
    def post(cid, ans):
        g = meta[cid]; c = cons[cid]; rec = {}
        if "argument_en" in g:
            if not isinstance(ans.get("argument_en"), str) or len(ans["argument_en"].strip()) < 10:
                return None
            rec["argument_en"] = ans["argument_en"].strip()
        for k in ("casus_en", "casus_de", "body_en", "body_de"):
            if k in g:
                v = ans.get(k); n = len(c["casus"] if k.startswith("casus") else c["body"])
                if not isinstance(v, list) or len(v) != n or not all(isinstance(s, str) and s.strip() for s in v):
                    return None
                rec[k] = [s.strip() for s in v]
        return rec
    run_jobs(args, jobs, TRANSLATE_SYSTEM, TRANSLATE_SCHEMA, OUT / f"translate_{slug(args.model)}.jsonl", post)


# ── keywords ──────────────────────────────────────────────────────────────
def vocabulary() -> list[dict]:
    return json.loads(VOCAB.read_text(encoding="utf-8"))["terms"] if VOCAB.exists() else []


def keyword_prompt(c: dict, vocab_top: list[str]) -> str:
    en = tr(c, "en", "body") or []
    text = " ".join(en) if en else " ".join(c.get("body") or [])
    arg = tr(c, "en", "argument") or "\n".join(f"{a['n']}. {a['text']}" for a in c.get("argument") or [])
    return (f"Consilium: {c.get('title_display') or c.get('title', '')} ({c['volume']})\n\n"
            f"ARGUMENTUM:\n{arg}\n\n{'TEXT (English translation)' if en else 'TEXT (Latin)'}:\n{text[:16000]}\n\n"
            "EXISTING VOCABULARY (most frequent terms of the edition, use these spellings when they fit):\n" + ", ".join(vocab_top) +
            "\n\nReturn {\"keywords\": [...]} with 8 to 12 keywords.")


def normalise_kw(k: str) -> str:
    return re.sub(r"\s+", " ", k.lower().strip().strip(".,;:"))


def cmd_keywords(args) -> None:
    cons = load_consilia()
    vocab = vocabulary()
    labels = {normalise_kw(t["label"]): t["label"] for t in vocab}
    sing = {}
    for lab in labels:        # crude plural folding so that "statutes" maps to "statute"
        sing.setdefault(lab[:-1] if lab.endswith("s") and not lab.endswith("ss") else lab, labels[lab])
    top = [t["label"] for t in vocab[:args.vocab_top]]
    jobs = [(cid, keyword_prompt(c, top)) for cid, c in cons.items() if keyword_gap(c)]
    def post(cid, ans):
        kws = ans.get("keywords")
        if not isinstance(kws, list) or len(kws) < 4:
            return None
        final, new = [], []
        for k in kws:
            if not isinstance(k, str):
                continue
            n = normalise_kw(k)
            lab = labels.get(n) or sing.get(n[:-1] if n.endswith("s") and not n.endswith("ss") else n)
            if lab:
                if lab not in final:
                    final.append(lab)
            elif n and n not in new:
                new.append(n)
        return {"keywords_final": (final + new)[:12], "in_vocabulary": final, "new_terms": new}
    run_jobs(args, jobs, KEYWORD_SYSTEM, KEYWORD_SCHEMA, OUT / f"keywords_{slug(args.model)}.jsonl", post)


# ── apply ─────────────────────────────────────────────────────────────────
def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def cmd_apply(args) -> None:
    updates: dict[str, dict] = {}
    for f in sorted(OUT.glob("translate_*.jsonl")) + sorted(OUT.glob("keywords_*.jsonl")):
        for l in f.open(encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l); u = updates.setdefault(r["id"], {})
            if "argument_en" in r:
                u.setdefault("translations", {}).setdefault("en", {})["argument"] = r["argument_en"]
            for k in ("casus_en", "casus_de", "body_en", "body_de"):
                if k in r:
                    u.setdefault("translations", {}).setdefault(k[-2:], {})[k[:-3]] = r[k]
            if "keywords_final" in r:
                u["keywords_final"] = r["keywords_final"]
                u["keywords_source"] = f"{r.get('model', '')} via tools/fill_gaps.py"
    if not updates:
        print("nothing to apply (no files in output/gaps/)"); return
    paths = [CONSILIA] + sorted((ROOT / "output").glob("*/consilia.json")) + sorted(p for p in (ROOT / "output").glob("*/*_v*.json") if "backup" not in p.name)
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8")); cons = data.get("consilia", {}); n = 0
        for cid, u in updates.items():
            c = cons.get(cid)
            if not c:
                continue
            for lang, piece in (u.get("translations") or {}).items():
                c.setdefault("translations", {}).setdefault(lang, {}).update(piece)
            for k in ("keywords_final", "keywords_source"):
                if k in u:
                    c[k] = u[k]
            n += 1
        if n:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{path.relative_to(ROOT)}: {n} consilia updated")
    print("now: python src/build_site_data.py && python src/qa/check_data.py; then re-embed (the data revision changes) and export the TEI")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--price-in", type=float, default=0.0); p.add_argument("--price-out", type=float, default=0.0)
    for name in ("translate", "keywords"):
        q = sub.add_parser(name)
        q.add_argument("--backend", choices=["gemini", "openai"], required=True)
        q.add_argument("--model", required=True)
        q.add_argument("--base-url", default="http://localhost:8000/v1")
        q.add_argument("--key-env", default=None)
        q.add_argument("--workers", type=int, default=4)
        q.add_argument("--limit", type=int)
        q.add_argument("--temperature", type=float, default=0.1)
        q.add_argument("--max-tokens", type=int, default=6000)
        q.add_argument("--timeout", type=int, default=240)
        q.add_argument("--thinking", dest="thinking", action="store_true", default=None)
        q.add_argument("--no-thinking", dest="thinking", action="store_false")
        q.add_argument("--thinking-level", default="", help="gemini: low|medium|high; empty = off")
        if name == "keywords":
            q.add_argument("--vocab-top", type=int, default=300)
    sub.add_parser("apply")
    args = ap.parse_args()
    {"plan": cmd_plan, "translate": cmd_translate, "keywords": cmd_keywords, "apply": cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    main()
