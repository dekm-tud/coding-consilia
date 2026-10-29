#!/usr/bin/env python3
"""Layer C: citation and entity spans in the Latin text (src/codebooks/spans.yaml), annotated by an LLM
with exact substrings, located in the unit text as character offsets, for training a Flair sequence
tagger (src/annotate/export_flair.py) and for the reference index.

  python src/annotate/spans.py --backend gemini --model gemini-3.8-flash --split sample     # pilot on the sample
  python src/annotate/spans.py --backend openai --base-url http://localhost:8000/v1 --model Qwen/Qwen3.6-27B --thinking
  python src/annotate/spans.py --review output/annotations/spans_<model>_p1.jsonl             # bracketed text to read
  python src/annotate/spans.py --from-review output/qa/spans_review.md                       # corrected text -> gold
  python src/annotate/spans.py --eval eval/gold/spans_gold.jsonl output/annotations/spans_<model>_p1.jsonl

Units: output/units.jsonl (same as Layer B). The model returns verbatim substrings with a label; a span
that cannot be found in the unit text is reported back once and, if still unmatched, dropped and listed
under `unmatched`. Output: output/annotations/spans_<model-slug>_p1.jsonl, one record per unit:
  {unit, doc, spans: [{start, end, text, label, ref}], unmatched: [...], usage, ok}
Gold (eval/gold/spans_gold.jsonl) uses the same record shape; `--eval` scores exact span matches
(start, end, label) and a relaxed overlap match per label.
Same backends and conventions as classify.py (resumable JSONL, JSON-schema output).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import GeminiBackend, OpenAIBackend, slug  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
UNITS = ROOT / "output" / "units.jsonl"
SCHEME = ROOT / "src" / "codebooks" / "spans.yaml"
GOLD_DIR = ROOT / "eval" / "gold"
OUT_DIR = ROOT / "output" / "annotations"

SYSTEM = """You mark citations and named entities in passages of late-medieval Latin legal opinions (consilia, Venice prints of 1571/1575, transcribed as printed with abbreviations). Mark spans ONLY of the label set below. Copy every span VERBATIM from the UNIT TEXT, character for character (same abbreviations, dots, ligatures, u/v, line fragments). Spans do not overlap and do not nest; the longer unit wins (a jurist's name followed by his citation = JURIST then CIT_*). Mark every occurrence, including repeated ones; if the same string occurs more than once, give `occurrence` (1-based, in reading order). Do not mark roles without names (testator, filius, iudex), legal terms, or titles of law used as words. For CIT_* give `ref` as a normalised modern reference only when you are confident (D. 28.2.29, C. 6.28.4, Inst. 2.18, Nov. 118, X 4.1.3, VI 1.3.11, Clem. 2.1.2, LF 2.26), else null. Output JSON only.

LABELS
{labels}

EXAMPLE
UNIT TEXT:
† quia certe hoc non prouenit iudicio aui: sed casu fortuito: vt ff. de l. 3. sed si sic. et ideo prodicta nepte concludo, quod quia pręterita, potest rumpere testamentum. licet Bar. in quodam loco dicat, quod in linea materna non requiritur, quod relinquatur iure institutionis: vt sic iura codicis in linea materna remaneant incorrecta: quia mater non dicitur propriae praeterire: vt insti. de inoffi. te. §. mater. Sed contrarium est verius per §. aliud quinque capitulum. in Auth. vt cum de appe. cogno. quia modo etiam hodie dicitur pręterire: et redditur testamentum nullum. Ego Bal.
ANSWER:
{{"spans": [{{"text": "ff. de l. 3. sed si sic.", "label": "CIT_CIV", "occurrence": 1, "ref": null}}, {{"text": "Bar.", "label": "JURIST", "occurrence": 1, "ref": null}}, {{"text": "insti. de inoffi. te. §. mater", "label": "CIT_CIV", "occurrence": 1, "ref": "Inst. 2.18.3"}}, {{"text": "§. aliud quinque capitulum. in Auth. vt cum de appe. cogno.", "label": "CIT_CIV", "occurrence": 1, "ref": "Nov. 115.3"}}, {{"text": "Ego Bal.", "label": "JURIST", "occurrence": 1, "ref": null}}]}}"""

SCHEMA = {
    "type": "object",
    "properties": {"spans": {"type": "array", "items": {
        "type": "object",
        "properties": {"text": {"type": "string"}, "label": {"type": "string", "enum": []},
                       "occurrence": {"type": "integer"}, "ref": {"type": ["string", "null"]}},
        "required": ["text", "label", "occurrence", "ref"], "additionalProperties": False}}},
    "required": ["spans"], "additionalProperties": False,
}


def render_labels(sc: dict) -> str:
    out = []
    for l in sc["labels"]:
        ex = "; ".join(l.get("examples", []))
        out.append(f"- {l['id']}: {l['en']}. {' '.join(l.get('scope', '').split())} Examples: {ex}")
    return "\n".join(out)


def user_prompt(u: dict) -> str:
    return f"UNIT {u['unit']}\n\nUNIT TEXT:\n{u['la']}\n\nMark the spans of the UNIT TEXT."


# ── locating spans ─────────────────────────────────────────────────────────
def _flex(s: str) -> str:
    """Regex that tolerates whitespace and line-break differences inside a quoted span."""
    return r"\s*".join(re.escape(ch) for ch in s.split()) if s.strip() else "(?!x)x"


def locate(text: str, spans: list[dict]) -> tuple[list[dict], list[dict]]:
    found, unmatched, taken = [], [], []
    for s in spans:
        q = (s.get("text") or "").strip()
        if not q:
            continue
        occ = max(1, int(s.get("occurrence") or 1))
        hits = [m.span() for m in re.finditer(re.escape(q), text)] or [m.span() for m in re.finditer(_flex(q), text)]
        hits = [h for h in hits if not any(h[0] < b and a < h[1] for a, b in taken)]   # no overlap with accepted spans
        if not hits:
            unmatched.append(s)
            continue
        a, b = hits[min(occ, len(hits)) - 1]
        taken.append((a, b))
        found.append({"start": a, "end": b, "text": text[a:b], "label": s["label"], "ref": s.get("ref")})
    found.sort(key=lambda x: x["start"])
    return found, unmatched


# ── review and evaluation ──────────────────────────────────────────────────
def bracketed(text: str, spans: list[dict]) -> str:
    out, pos = [], 0
    for s in sorted(spans, key=lambda x: x["start"]):
        out.append(text[pos:s["start"]]); out.append(f"[{text[s['start']:s['end']]}|{s['label']}]"); pos = s["end"]
    out.append(text[pos:])
    return "".join(out)


def review(files: list[Path], out: Path) -> None:
    units = {json.loads(l)["unit"]: json.loads(l) for l in UNITS.open(encoding="utf-8")}
    lines = ["# Span annotations for review", "", "Spans in brackets as `[text|LABEL]`; `ref` after the unit where given.", ""]
    for f in files:
        for l in f.open(encoding="utf-8"):
            r = json.loads(l)
            if not r.get("ok"):
                lines += [f"## {r['unit']}  (error: {r.get('error', '')[:120]})", ""]
                continue
            u = units.get(r["unit"])
            if not u:
                continue
            lines += [f"## {r['unit']}", "", bracketed(u["la"], r["spans"]), ""]
            refs = [f"{s['text']} → {s['ref']}" for s in r["spans"] if s.get("ref")]
            if refs:
                lines += ["refs: " + "; ".join(refs), ""]
            if r.get("unmatched"):
                lines += ["unmatched: " + "; ".join(f"{s.get('text')} ({s.get('label')})" for s in r["unmatched"]), ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out.relative_to(ROOT))


def from_review(md: Path, out: Path) -> None:
    """Read a (hand-corrected) review file back into gold records: brackets `[text|LABEL]` become spans."""
    units = {json.loads(l)["unit"]: json.loads(l) for l in UNITS.open(encoding="utf-8")}
    labels = {l["id"] for l in yaml.safe_load(SCHEME.read_text(encoding="utf-8"))["labels"]}
    recs, unit, buf = [], None, []
    def flush():
        if not unit or unit not in units:
            return
        marked = "\n".join(buf).strip()
        plain, spans, pos, i = [], [], 0, 0
        pat = re.compile(r"\[([^\[\]]+?)\|([A-Z_]+)\]")
        for m in pat.finditer(marked):
            plain.append(marked[i:m.start()]); start = sum(map(len, plain))
            plain.append(m.group(1)); i = m.end()
            if m.group(2) not in labels:
                print(f"  {unit}: unknown label {m.group(2)}"); continue
            spans.append({"start": start, "end": start + len(m.group(1)), "text": m.group(1), "label": m.group(2), "ref": None})
        plain.append(marked[i:]); text = "".join(plain)
        if text != units[unit]["la"]:
            print(f"  {unit}: text differs from the unit after removing brackets (edited words?); offsets recomputed by search")
            fixed = []
            for sp in spans:
                k = units[unit]["la"].find(sp["text"])
                if k >= 0:
                    fixed.append({**sp, "start": k, "end": k + len(sp["text"])})
            spans = fixed
        recs.append({"unit": unit, "doc": units[unit]["doc"], "task": "spans", "model": "editor", "spans": spans, "unmatched": [], "ok": True})
    for line in md.read_text(encoding="utf-8").split("\n"):
        if line.startswith("## "):
            flush(); unit = line[3:].split()[0]; buf = []
        elif unit and not line.startswith(("refs:", "unmatched:", "# ", "Spans in brackets")):
            buf.append(line)
    flush()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}: {len(recs)} units, {sum(len(r['spans']) for r in recs)} spans")


def evaluate(gold_file: Path, pred_files: list[Path]) -> None:
    gold = {json.loads(l)["unit"]: json.loads(l) for l in gold_file.open(encoding="utf-8")}
    for pf in pred_files:
        preds = {json.loads(l)["unit"]: json.loads(l) for l in pf.open(encoding="utf-8")}
        common = [u for u in gold if u in preds and preds[u].get("ok")]
        tp = fp = fn = 0; per = defaultdict(lambda: [0, 0, 0]); otp = ofp = ofn = 0
        for u in common:
            g = {(s["start"], s["end"], s["label"]) for s in gold[u]["spans"]}
            p = {(s["start"], s["end"], s["label"]) for s in preds[u]["spans"]}
            tp += len(g & p); fp += len(p - g); fn += len(g - p)
            for s in g | p:
                per[s[2]][0] += s in g and s in p; per[s[2]][1] += s in p and s not in g; per[s[2]][2] += s in g and s not in p
            # relaxed: same label, overlapping offsets
            def ov(x, ys): return any(x[2] == y[2] and x[0] < y[1] and y[0] < x[1] for y in ys)
            otp += sum(ov(x, g) for x in p); ofp += sum(not ov(x, g) for x in p); ofn += sum(not ov(x, p) for x in g)
        def prf(a, b, c):
            P = a / (a + b) if a + b else 0.0; R = a / (a + c) if a + c else 0.0
            return P, R, (2 * P * R / (P + R) if P + R else 0.0)
        P, R, F = prf(tp, fp, fn); oP, oR, oF = prf(otp, ofp, ofn)
        print(f"{pf.name}: {len(common)} units; exact P/R/F1 {P:.2f}/{R:.2f}/{F:.2f}; overlap P/R/F1 {oP:.2f}/{oR:.2f}/{oF:.2f}")
        for lab, (a, b, c) in sorted(per.items(), key=lambda kv: -(kv[1][0] + kv[1][2])):
            p_, r_, f_ = prf(a, b, c)
            print(f"   {lab:9s} F1 {f_:.2f} (gold {a + c}, pred {a + b})")


# ── driver ─────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["openai", "gemini"])
    ap.add_argument("--model")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--key-env", default=None)
    ap.add_argument("--split", default="all", help="all | sample (eval/gold/spans_sample.json) | comma-separated unit ids")
    ap.add_argument("--pass", dest="pass_no", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=8000, help="answer budget incl. thinking")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--thinking", dest="thinking", action="store_true", default=None)
    ap.add_argument("--no-thinking", dest="thinking", action="store_false")
    ap.add_argument("--thinking-level", default="low")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--units", type=Path, default=UNITS, help="units file (default output/units.jsonl)")
    ap.add_argument("--make-sample", type=int, metavar="N", help="write eval/gold/spans_sample.json with N units (stratified by print) and exit")
    ap.add_argument("--review", nargs="+", type=Path, metavar="JSONL", help="write output/qa/spans_review.md from these runs and exit")
    ap.add_argument("--eval", nargs="+", type=Path, metavar="FILE", help="GOLD PRED... : score runs against a gold file and exit")
    ap.add_argument("--from-review", type=Path, metavar="MD", help="turn a corrected review file into eval/gold/spans_gold.jsonl and exit")
    args = ap.parse_args()
    if args.key_env is None:
        import os
        args.key_env = "GEMINI_API_KEY" if args.backend == "gemini" else ("BLABLADOR_API_KEY" if os.environ.get("BLABLADOR_API_KEY") else "OPENAI_API_KEY")

    if args.review:
        review(args.review, ROOT / "output" / "qa" / "spans_review.md"); return
    if args.from_review:
        from_review(args.from_review, GOLD_DIR / "spans_gold.jsonl"); return
    if args.eval:
        evaluate(args.eval[0], args.eval[1:]); return

    units = [json.loads(l) for l in args.units.open(encoding="utf-8")]
    if args.make_sample:
        rng = random.Random(20261017)
        by_vol = defaultdict(list)
        for u in units:
            if 200 <= len(u["la"]) <= 2500:
                by_vol[u["volume"]].append(u["unit"])
        k = max(1, args.make_sample // len(by_vol))
        sample = sorted(x for v in by_vol.values() for x in rng.sample(v, min(k, len(v))))
        (GOLD_DIR / "spans_sample.json").write_text(json.dumps(sample, indent=1), encoding="utf-8")
        print(f"wrote eval/gold/spans_sample.json: {len(sample)} units from {len(by_vol)} prints"); return
    if not args.backend or not args.model:
        ap.error("--backend and --model are required for an annotation run")

    sc = yaml.safe_load(SCHEME.read_text(encoding="utf-8"))
    labels = [l["id"] for l in sc["labels"]]
    system = SYSTEM.format(labels=render_labels(sc))
    schema = json.loads(json.dumps(SCHEMA)); schema["properties"]["spans"]["items"]["properties"]["label"]["enum"] = labels
    prompt_hash = hashlib.sha256(system.encode()).hexdigest()[:12]

    if args.split == "sample":
        want = set(json.loads((GOLD_DIR / "spans_sample.json").read_text(encoding="utf-8")))
        items = [u for u in units if u["unit"] in want]
    elif args.split != "all":
        want = set(args.split.split(",")); items = [u for u in units if u["unit"] in want]
    else:
        items = units
    out = args.out or OUT_DIR / f"spans_{slug(args.model)}_p{args.pass_no}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.open(encoding="utf-8"):
            try:
                rec = json.loads(l)
                if rec.get("ok", True) and rec.get("prompt_hash") == prompt_hash:   # failed items and old prompts are redone
                    done.add(rec["unit"])
            except Exception:
                pass
    todo = [u for u in items if u["unit"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"spans: {len(items)} units, {len(done)} done, {len(todo)} to do -> {out}  (model {args.model}, prompt {prompt_hash})")
    if not todo:
        return
    be = (OpenAIBackend if args.backend == "openai" else GeminiBackend)(args)
    lock = threading.Lock(); fh = out.open("a", encoding="utf-8"); t0 = time.time(); n_ok = n_err = 0; stats = Counter()

    def work(u: dict) -> dict:
        rec = {"unit": u["unit"], "doc": u["doc"], "task": "spans", "model": args.model, "backend": args.backend, "pass": args.pass_no,
               "prompt_hash": prompt_hash, "scheme_version": sc["version"], "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            res, usage = be.complete(system, [], user_prompt(u), schema)
            spans = [s for s in res.get("spans", []) if s.get("label") in labels]
            found, unmatched = locate(u["la"], spans)
            if unmatched:
                retry = user_prompt(u) + "\n\nThese spans of your previous answer are NOT verbatim substrings of the UNIT TEXT: " + \
                    json.dumps([s.get("text") for s in unmatched], ensure_ascii=False) + ". Answer again with every span copied exactly."
                res2, usage2 = be.complete(system, [], retry, schema)
                usage = {k: usage[k] + usage2[k] for k in usage}
                found2, unmatched2 = locate(u["la"], [s for s in res2.get("spans", []) if s.get("label") in labels])
                if len(unmatched2) <= len(unmatched):
                    found, unmatched = found2, unmatched2
            rec.update(spans=found, unmatched=[{"text": s.get("text"), "label": s.get("label")} for s in unmatched], usage=usage, ok=True)
        except Exception as e:  # noqa: BLE001
            rec.update(ok=False, error=str(e)[:500])
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, u) for u in todo]
        for i, f in enumerate(as_completed(futs), 1):
            rec = f.result()
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
                if rec["ok"]:
                    n_ok += 1; be.usage["prompt"] += rec["usage"]["prompt"]; be.usage["output"] += rec["usage"]["output"]
                    stats.update(s["label"] for s in rec["spans"]); stats["_unmatched"] += len(rec["unmatched"])
                else:
                    n_err += 1
                    if n_err >= 20 and n_ok == 0:
                        print(f"stopping: the first {n_err} items all failed, last error: {rec.get('error')}\n"
                              f"(is the model server at the base URL running? failed items are retried on the next run)")
                        os._exit(2)
                if i % 20 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"  {i}/{len(todo)} ok {n_ok} err {n_err}  {el / i:.1f}s/unit  tokens prompt {be.usage['prompt']} output {be.usage['output']}")
    fh.close()
    print("labels:", dict(stats.most_common()))
    if n_err:
        print(f"{n_err} units failed; rerun the same command to retry them (failed records are kept in the file; delete them first)")


if __name__ == "__main__":
    main()
