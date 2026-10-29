#!/usr/bin/env python3
"""
Loci for the citations the rules cannot resolve, proposed by language models and accepted only when two
models agree (same convention as the other annotation layers: two passes -> verified_by_model).

  python tools/resolve_llm.py --tasks output/qa/cit_tasks.jsonl [--docs output/qa/pilot_docs.json]
        collect the unresolved / ambiguous citation spans (site-data/annotations/references/*.json) with
        their sentence context, one task per (consilium, printed citation)
  python tools/resolve_llm.py --run output/qa/cit_tasks.jsonl --backend gemini --model gemini-3.8-flash
        -> output/qa/cit_proposals_<model>.jsonl  ({id, key|null, note})   [a Claude agent writes the same shape]
  python tools/resolve_llm.py --merge output/qa/cit_proposals_a.jsonl output/qa/cit_proposals_b.jsonl
        -> eval/reviews/citation_resolutions.jsonl (verified_by_model when both give the same valid key;
           disputed otherwise, not applied); then src/annotate/export.py --spans … and build_site_data.py

Keys are title-level, as in the index: D. 24.3, C. 6.23, Inst. 2.18, Nov. 118, X 2.20, VI 5.13, Clem. 3.10,
LF 2.26, Decr. D. 12, Decr. C. 1 q. 1. A key is valid only when the work matches the printed siglum (ff. -> D,
C. -> C, extra -> X, …) and the title exists in src/codebooks/corpus_titles.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "annotate"))
from resolve_citations import TABLE, detect_work, resolve  # noqa: E402

KEY = re.compile(r"^(?:(D|C|Inst|Clem)\. (\d+)\.(\d+)|(X|VI|LF) (\d+)\.(\d+)|Nov\. (\d+)|Decr\. D\. \d+|Decr\. C\. \d+ q\. \d+)$")
NOV = {str(e["nov"]) for coll in TABLE["Nov"]["collationes"].values() for e in coll.values()}

SYSTEM = """You identify the place cited by abbreviated legal citations in Latin legal opinions (consilia) of the fourteenth and fifteenth centuries, printed in Venice 1571/1575. The citations use the medieval system: a work siglum (ff. = Digest, C. = Codex, Inst. = Institutes, Auth. / in Auth. = Authenticum (Novels), extra / X = Liber Extra, in vi. / lib. 6 = Liber Sextus, Clem. = Clementinae, in usib. feud. = Libri feudorum, dist. / q. = Decretum Gratiani), a title named by its first words after "de", and a lex or capitulum named by its incipit after "l." / "c." / "§.". Often only the lex or capitulum incipit is given ("l. dotem", "c. si una", "l. quod nepos"); then you must know which title that lex or chapter belongs to.

For every task give the TITLE-LEVEL key of the cited place in exactly this format: "D. 24.3" (Digest book.title), "C. 6.23", "Inst. 2.18", "Nov. 118", "X 2.20", "VI 5.13", "Clem. 3.10", "LF 2.26", "Decr. D. 12", "Decr. C. 1 q. 1". Never give the lex number. Use the context (the sentence around the citation and the previous citation's key) to decide. If you are not confident which title is meant, answer null; a wrong key is worse than none. Give a short note (the lex / chapter you identified, e.g. "D. 24.3.3 Dotem quae"). Output JSON only."""

SCHEMA = {"type": "object", "properties": {"answers": {"type": "array", "items": {"type": "object", "properties": {
    "id": {"type": "string"}, "key": {"type": ["string", "null"]}, "note": {"type": "string"}},
    "required": ["id", "key", "note"], "additionalProperties": False}}}, "required": ["answers"], "additionalProperties": False}


def normalise(key: str | None) -> str | None:
    """model spellings -> index keys: "D. 31" / "D.31" / "X 2.20.3" -> "D. 31.1" / "X 2.20" (title level)"""
    if not key:
        return None
    k = re.sub(r"\s+", " ", key.strip()).replace("Dig. ", "D. ").replace("Cod. ", "C. ")
    k = re.sub(r"^(Clem|Inst|Nov|Decr|D|C)\.?\s*", lambda m: m.group(1) + ". ", k)
    k = re.sub(r"^(X|VI|LF)\.?\s*", lambda m: m.group(1) + " ", k)
    m = re.match(r"^(D\.|C\.|Inst\.|Clem\.|X|VI|LF) (\d+)\.(\d+)(?:\.\d+)*(?:\s.*)?$", k)      # drop lex / paragraph numbers
    if m:
        return f"{m.group(1)} {m.group(2)}.{m.group(3)}"
    m = re.match(r"^D\. (30|31|32)$", k)                                                          # single-title books
    if m:
        return f"D. {m.group(1)}.1"
    m = re.match(r"^Nov\. (\d+)", k)
    if m:
        return f"Nov. {m.group(1)}"
    return k


def valid(key: str | None, text: str) -> bool:
    if not key or not KEY.match(key):
        return False
    m = KEY.match(key)
    if m.group(1):
        return m.group(3) in TABLE[m.group(1)]["books"].get(m.group(2), {})
    if m.group(4):
        return m.group(6) in TABLE[m.group(4)]["books"].get(m.group(5), {})
    if m.group(7):
        return m.group(7) in NOV
    return True


def consistent(key: str, text: str) -> bool:
    """the work of the key must match the printed siglum when there is one"""
    w, _ = detect_work(text)
    kw = key.split()[0].rstrip(".")
    if kw == "Decr":
        kw = "Decretum"
    if w is None or w == "X" and not re.search(r"(^|[\s(])(extra|ext|x)\b", text.lower()):   # bare "c." may be any canon-law work
        return kw in ("X", "VI", "Clem", "Decretum") if w == "X" else True
    return w == kw


def make_tasks(out: Path, docs_file: Path | None) -> None:
    import glob
    want = set(json.load(docs_file.open(encoding="utf-8"))) if docs_file else None
    tasks = {}
    for f in sorted(glob.glob(str(ROOT / "site-data" / "annotations" / "references" / "*.json"))):
        vol = Path(f).stem
        text = json.loads((ROOT / "site-data" / "text" / f"{vol}.json").read_text(encoding="utf-8"))
        for doc, spans in json.loads(Path(f).read_text(encoding="utf-8")).items():
            if want and doc not in want:
                continue
            secs = text.get(doc) or {}
            prev = None
            for sp in sorted((s for s in spans if s["l"].startswith("CIT")), key=lambda s: (s["k"], s["s"], s["a"])):
                if sp.get("rk"):
                    prev = sp["rk"]; continue
                if (doc, sp["t"]) in tasks:
                    continue
                sec = (secs.get(sp["k"]) or [])
                sec = sec[sp["s"]] if sp["s"] < len(sec) else ""
                sec = sec if isinstance(sec, str) else (sec.get("la") or "")
                a, b = sp["o"] + sp["a"], sp["o"] + sp["b"]
                ctx = sec[max(0, a - 220):a] + "[[" + sec[a:b] + "]]" + sec[b:b + 160]
                r = resolve(sp["t"], prev)
                tasks[(doc, sp["t"])] = {"id": f"{doc}|{len(tasks)}", "doc": doc, "text": sp["t"], "label": sp["l"],
                                         "context": ctx.replace("\n", " "), "prev_key": prev,
                                         "rule_status": r["status"], "candidates": r.get("candidates") or []}
    with out.open("w", encoding="utf-8") as fh:
        for t in tasks.values():
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"{len(tasks)} tasks -> {out}")


def run(tasks_file: Path, args) -> None:
    from classify import GeminiBackend, OpenAIBackend, slug
    be = (GeminiBackend if args.backend == "gemini" else OpenAIBackend)(args)
    tasks = [json.loads(l) for l in tasks_file.open(encoding="utf-8")]
    out = args.out or ROOT / "output" / "qa" / f"cit_proposals_{slug(args.model)}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.open(encoding="utf-8") if l.strip()}
    todo = [t for t in tasks if t["id"] not in done]
    print(f"{len(tasks)} tasks, {len(done)} done, {len(todo)} to do -> {out}")
    B = args.batch
    with out.open("a", encoding="utf-8") as fh:
        for i in range(0, len(todo), B):
            batch = todo[i:i + B]
            user = "TASKS:\n" + "\n".join(json.dumps({k: t[k] for k in ("id", "text", "context", "prev_key", "candidates")}, ensure_ascii=False) for t in batch) + \
                "\n\nAnswer with one entry per task id."
            try:
                res, usage = be.complete(SYSTEM, [], user, SCHEMA)
                got = {a["id"]: a for a in res.get("answers", [])}
            except Exception as e:  # noqa: BLE001
                print("batch failed:", str(e)[:200]); time.sleep(5); continue
            for t in batch:
                a = got.get(t["id"]) or {"key": None, "note": "no answer"}
                fh.write(json.dumps({"id": t["id"], "doc": t["doc"], "text": t["text"], "key": a.get("key"), "note": a.get("note", ""), "model": args.model}, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  {min(i + B, len(todo))}/{len(todo)}", flush=True)


def merge(files: list[Path], out: Path) -> None:
    props = {}
    for f in files:
        for l in f.open(encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                r["key"] = normalise(r.get("key"))
                props.setdefault(r["id"], []).append(r)
    n_ok = n_dis = n_none = 0
    with out.open("w", encoding="utf-8") as fh:
        for pid, rs in props.items():
            keys = [r["key"] for r in rs if valid(r.get("key"), r["text"]) and consistent(r["key"], r["text"])]
            doc, text = pid.rsplit("|", 1)[0], rs[0]["text"]       # id = "<doc>|<n>"
            if len(rs) >= 2 and len(keys) == len(rs) and len(set(keys)) == 1:
                st = "verified_by_model"; key = keys[0]; n_ok += 1
            elif keys:
                st = "disputed"; key = keys[0]; n_dis += 1
            else:
                st = "none"; key = None; n_none += 1
            fh.write(json.dumps({"doc": doc, "text": text, "key": key, "status": st, "models": [r.get("model") for r in rs],
                                 "keys": [r.get("key") for r in rs], "notes": [r.get("note", "") for r in rs]}, ensure_ascii=False) + "\n")
    print(f"{out}: {n_ok} verified_by_model, {n_dis} disputed, {n_none} without a valid key")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=Path, help="write the task file")
    ap.add_argument("--docs", type=Path, help="JSON list of consilium ids to restrict the tasks to")
    ap.add_argument("--run", type=Path, help="task file to send to a model")
    ap.add_argument("--merge", nargs="+", type=Path, help="proposal files to merge")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--backend", choices=["openai", "gemini"], default="gemini")
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--key-env", default=None)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--thinking", dest="thinking", action="store_true", default=None)
    ap.add_argument("--thinking-level", default="medium")
    args = ap.parse_args()
    if args.key_env is None:
        args.key_env = "GEMINI_API_KEY" if args.backend == "gemini" else "OPENAI_API_KEY"
    if args.tasks:
        make_tasks(args.tasks, args.docs)
    elif args.run:
        run(args.run, args)
    elif args.merge:
        merge(args.merge, args.out or ROOT / "eval" / "reviews" / "citation_resolutions.jsonl")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
