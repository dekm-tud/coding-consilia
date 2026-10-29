"""
Exhaustive LLM classification of the corpus.

Two tasks, two closed schemes:
  --task body      Layer B: every unit of output/units.jsonl (one text section or window) is judged
                   against the intimate-body codebook (src/codebooks/intimate_body.yaml): codes with a
                   role and a Latin evidence quote. Evidence must be a substring of the unit (after
                   normalisation), otherwise one retry, then the annotation is flagged invalid_evidence.
  --task subjects  Layer A: every consilium gets 1-3 central (usually 1-2) and <= 6 secondary concepts of the subject
                   scheme (src/codebooks/subjects.yaml), chosen from the Latin argumentum and the
                   translation (or the Latin text when no translation exists).

Backends (one code path each, JSON-schema constrained output):
  --backend openai   any OpenAI-compatible chat API: a vLLM server on the GPU box, Blablador
                     (https://api.blablador.fz-juelich.de/v1), OpenRouter, … Key from --key-env.
                     --thinking passes chat_template_kwargs.enable_thinking (Qwen on vLLM).
  --backend gemini   Google Gemini REST (GEMINI_API_KEY), responseJsonSchema + thinkingLevel.

Resumable: records already in --out are skipped. Every record carries model, backend, pass, prompt
hash, scheme version and token usage. Pass 2 (--pass 2) shuffles the code order and uses other
few-shot examples (both from the gold DEV split only), for the agreement tier.

Examples:
  python src/annotate/classify.py --task body --backend gemini --model gemini-3.8-flash --split dev
  python src/annotate/classify.py --task body --backend openai --base-url http://gpu:8000/v1 \\
         --model Qwen/Qwen3.6-27B --thinking --workers 16
  python src/annotate/classify.py --task subjects --backend gemini --model gemini-3.8-flash --split gold
Outputs: output/annotations/<task>_<model-slug>_p<pass>.jsonl
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
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
UNITS = ROOT / "output" / "units.jsonl"
CONSILIA = ROOT / "output" / "consilia.json"
CODEBOOK = ROOT / "src" / "codebooks" / "intimate_body.yaml"
SCHEME = ROOT / "src" / "codebooks" / "subjects.yaml"
GOLD_DIR = ROOT / "eval" / "gold"
OUT_DIR = ROOT / "output" / "annotations"


# ── text normalisation for the evidence check ──────────────────────────────
def norm(s: str) -> str:
    s = s.lower().replace("ſ", "s").replace("æ", "e").replace("ę", "e").replace("œ", "e").replace("ae", "e")
    s = s.replace("j", "i").replace("v", "u")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def evidence_ok(quote: str, unit_text: str) -> bool:
    q = norm(quote or "")
    return bool(q) and len(q) >= 8 and q in norm(unit_text)


# ── prompts ────────────────────────────────────────────────────────────────
def _items(xs) -> str:
    """codebook entries are strings or small dicts ({pattern: meaning}); render both."""
    out = []
    for v in xs or []:
        out.append(v if isinstance(v, str) else "; ".join(f"{k}: {w}" for k, w in v.items()))
    return " | ".join(out)


def render_codebook(cb: dict, shuffle_seed: int | None) -> str:
    codes = list(cb["codes"])
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(codes)
    groups = {g["id"]: g["label_en"] for g in cb["groups"]}
    out = [f"CODEBOOK v{cb['version']} — the intimate and reproductive body. ROLES: " + ", ".join(cb["roles"]) + ".",
           "topic = the bodily matter is (part of) the question decided; argument = a reasoning step depends on it; "
           "case_fact = narrated fact, neither decided nor argued; analogy = comparison/hypothetical; "
           "citation_only = appears only as the title of an allegation for an unrelated rule; formulaic = stock clause.", ""]
    for c in codes:
        out.append(f"- {c['id']} [{groups.get(c['group'], c['group'])}] {c['label_en']} ({c.get('label_la', '')}): {c.get('definition', '')}")
        if c.get("include"):
            out.append("    include: " + _items(c["include"]))
        if c.get("not"):
            out.append("    NOT: " + _items(c["not"]))
        if c.get("latin_cues"):
            out.append("    cues: " + _items(c["latin_cues"]).replace(" | ", ", "))
    out.append("")
    out.append("HOMONYMS (never code these): " + "; ".join((f"{h.get('pattern')} = {h.get('meaning')}" if isinstance(h, dict) else str(h)) for h in cb.get("homonyms", [])))
    return "\n".join(out)


def render_scheme(sc: dict) -> str:
    facets = {f["id"]: f["en"] for f in sc["facets"]}
    out = [f"SUBJECT SCHEME v{sc['version']} ({len(sc['concepts'])} concepts in {len(facets)} facets). "
           "Assign a concept only if the consilium DISCUSSES it (question, rule or argument), not for a mere word or allegation. "
           "Prefer the most specific concept. Genre descriptors are not concepts. Do not assign BODY concepts.", ""]
    cur = None
    for c in sc["concepts"]:
        if c["facet"] == "BODY":
            continue
        if c["facet"] != cur:
            cur = c["facet"]
            out.append(f"## {cur}: {facets.get(cur, cur)}")
        out.append(f"- {c['id']}: {c['en']} ({c.get('la', '')}). {c.get('scope', '')}")
    return "\n".join(out)


BODY_SYSTEM = """You annotate passages of late-medieval Latin legal opinions (consilia of Baldus de Ubaldis and Panormitanus) for a historical study of the intimate and reproductive body. Decide FROM THE LATIN; the English translation is only a reading aid and may be wrong. Apply the codebook below exactly. A unit may carry several codes, each with exactly one role and a short Latin evidence quote COPIED VERBATIM from the unit's Latin text (not from the context or argumentum). If nothing applies, relevant=false and annotations=[]. Figurative and homonymous uses are not annotated. Output JSON only, matching the schema.

{codebook}"""

BODY_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {"type": "boolean"},
        "annotations": {"type": "array", "items": {"type": "object", "properties": {
            "code": {"type": "string", "enum": []}, "role": {"type": "string", "enum": []},
            "evidence_la": {"type": "string"}, "explanation_en": {"type": "string"}},
            "required": ["code", "role", "evidence_la", "explanation_en"], "additionalProperties": False}},
        "translation_issue": {"type": ["string", "null"]},
    },
    "required": ["relevant", "annotations", "translation_issue"], "additionalProperties": False,
}

SUBJ_SYSTEM = """You index late-medieval Latin legal opinions (consilia) with a closed subject scheme. Read the Latin argumentum (the numbered propositions) and the text or its translation, then choose the CENTRAL concepts and up to 6 SECONDARY concepts. Use only ids from the scheme.
CENTRAL = the legal question(s) the consilium is asked and answers; the concept a reader would file it under. Usually ONE, often two, three only when the consilium really decides three separate questions. A concept the jurist discusses as a step, premise, objection, analogy or a side issue is SECONDARY, even if it takes much space. When in doubt between central and secondary, choose secondary. For every chosen concept give a short Latin evidence quote from the argumentum or text. Output JSON only.

{scheme}"""

SUBJ_SCHEMA = {
    "type": "object",
    "properties": {
        "central": {"type": "array", "items": {"type": "string", "enum": []}, "minItems": 1, "maxItems": 3},
        "secondary": {"type": "array", "items": {"type": "string", "enum": []}, "maxItems": 6},
        "evidence": {"type": "array", "items": {"type": "object", "properties": {
            "concept": {"type": "string"}, "evidence_la": {"type": "string"}}, "required": ["concept", "evidence_la"], "additionalProperties": False}},
        "note_en": {"type": "string"},
    },
    "required": ["central", "secondary", "evidence", "note_en"], "additionalProperties": False,
}


def body_user(u: dict) -> str:
    tl = u.get("tr_lang") or "en"      # reading aid language: en, or de where only German exists
    parts = [f"UNIT {u['unit']} — {u.get('title', '')}",
             f"ARGUMENTUM (la): {u.get('argumentum_la', '')[:1500]}",
             f"ARGUMENTUM ({tl}): {u.get('argumentum_en', '')[:1500]}",
             f"CONTEXT BEFORE (la): …{u.get('prev_la', '')}",
             f"UNIT TEXT (la):\n{u['la']}",
             f"UNIT TEXT ({tl}, machine translation):\n{u.get('en', '') or '(none)'}",
             f"CONTEXT AFTER (la): {u.get('next_la', '')}…",
             "Annotate the UNIT TEXT only."]
    return "\n\n".join(parts)


def subj_user(c: dict, max_chars: int = 14000) -> str:
    trs = c.get("translations") or {}
    tl = "en" if (trs.get("en") or {}).get("body") else "de"       # German where only that exists
    tr = trs.get(tl) or {}
    body_en = "\n".join(tr.get("body") or [])
    body = body_en if len(body_en) > 200 else "\n".join(c.get("body") or [])
    lang = f"{tl}, machine translation" if body is body_en else "la"
    arg = c.get("summary") or ""
    arg_en = tr.get("argument") or ""
    casus = " ".join(c.get("casus") or [])
    return (f"CONSILIUM {c['id']} — {c.get('title', '')}\n\nCASUS (la): {casus[:2000]}\n\nARGUMENTUM (la): {arg[:3000]}\n\n"
            f"ARGUMENTUM ({tl}): {arg_en[:3000]}\n\nTEXT ({lang}):\n{body[:max_chars]}")


def few_shots(task: str, pass_no: int, k: int = 6) -> tuple[list[dict], list[str]]:
    """(messages, ids) — examples from the DEV split only; pass 2 draws a different set."""
    rng = random.Random(1000 + pass_no)
    if task == "body":
        split = json.loads((GOLD_DIR / "intimate_body_split.json").read_text())
        units = {json.loads(l)["unit"]: json.loads(l) for l in UNITS.open(encoding="utf-8")}
        gold = [json.loads(l) for l in (GOLD_DIR / "intimate_body_gold.jsonl").open(encoding="utf-8")]
        dev = [g for g in gold if split.get(g["unit"]) == "dev" and g["unit"] in units]
        pos = [g for g in dev if g["relevant"] and all(a["role"] != "citation_only" for a in g["annotations"])]
        cit = [g for g in dev if any(a["role"] == "citation_only" for a in g["annotations"])]
        neg_h = [g for g in dev if not g["relevant"] and g.get("stratum") == "homonym"]
        neg = [g for g in dev if not g["relevant"] and g.get("stratum") != "homonym"]
        pick = rng.sample(pos, min(3, len(pos))) + rng.sample(cit, min(1, len(cit))) + rng.sample(neg_h, min(2, len(neg_h))) + rng.sample(neg, min(1, len(neg)))
        rng.shuffle(pick)
        msgs, ids = [], []
        for g in pick[:k + 1]:
            u = units[g["unit"]]
            ans = {"relevant": g["relevant"], "annotations": [{"code": a["code"], "role": a["role"], "evidence_la": a["evidence_la"], "explanation_en": g.get("note", "")} for a in g["annotations"]], "translation_issue": None}
            msgs += [{"role": "user", "content": body_user(u)}, {"role": "assistant", "content": json.dumps(ans, ensure_ascii=False)}]
            ids.append(g["unit"])
        return msgs, ids
    gold = [json.loads(l) for l in (GOLD_DIR / "subjects_gold.jsonl").open(encoding="utf-8")]
    cons = json.loads(CONSILIA.read_text(encoding="utf-8"))["consilia"]
    pick = rng.sample([g for g in gold if g["doc"] in cons], 3)
    msgs, ids = [], []
    for g in pick:
        ans = {"central": g["central"], "secondary": g["secondary"], "evidence": [], "note_en": g.get("note", "")}
        msgs += [{"role": "user", "content": subj_user(cons[g["doc"]], 6000)}, {"role": "assistant", "content": json.dumps(ans, ensure_ascii=False)}]
        ids.append(g["doc"])
    return msgs, ids


# ── backends ───────────────────────────────────────────────────────────────
class Backend:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.usage = {"prompt": 0, "output": 0, "calls": 0}

    def _post(self, url: str, payload: dict, headers: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        for attempt in range(6):
            try:
                req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
                with urllib.request.urlopen(req, timeout=self.args.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "ignore")[:400]
                if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                    time.sleep(min(60, 2 ** attempt * 3))
                    continue
                raise RuntimeError(f"HTTP {e.code}: {body}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < 5:
                    time.sleep(min(60, 2 ** attempt * 3))
                    continue
                raise RuntimeError(str(e)) from None
        raise RuntimeError("unreachable")

    def complete(self, system: str, shots: list[dict], user: str, schema: dict) -> tuple[dict, dict]:
        raise NotImplementedError


class OpenAIBackend(Backend):
    def complete(self, system, shots, user, schema):
        a = self.args
        msgs = [{"role": "system", "content": system}] + shots + [{"role": "user", "content": user}]
        payload = {"model": a.model, "messages": msgs, "temperature": a.temperature, "max_tokens": a.max_tokens,
                   "response_format": {"type": "json_schema", "json_schema": {"name": "annotation", "schema": schema, "strict": True}}}
        if a.thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": bool(a.thinking)}
        headers = {"Authorization": f"Bearer {os.environ.get(a.key_env, '')}"} if os.environ.get(a.key_env) else {}
        try:
            r = self._post(a.base_url.rstrip("/") + "/chat/completions", payload, headers)
        except RuntimeError as e:
            if "HTTP 400" in str(e) and "response_format" in str(e) or "json_schema" in str(e):
                payload["response_format"] = {"type": "json_object"}
                r = self._post(a.base_url.rstrip("/") + "/chat/completions", payload, headers)
            else:
                raise
        choice = r["choices"][0]
        msg = choice["message"]
        text = msg.get("content") or ""
        usage = r.get("usage") or {}
        if choice.get("finish_reason") == "length":
            raise RuntimeError(f"truncated at max_tokens={a.max_tokens} (thinking counts against it; raise --max-tokens)")
        return parse_json(text), {"prompt": usage.get("prompt_tokens", 0), "output": usage.get("completion_tokens", 0)}


class GeminiBackend(Backend):
    def complete(self, system, shots, user, schema):
        a = self.args
        contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in shots]
        contents.append({"role": "user", "parts": [{"text": user}]})
        gen = {"responseMimeType": "application/json", "responseJsonSchema": gemini_schema(schema),
               "temperature": a.temperature, "maxOutputTokens": a.max_tokens}
        if a.thinking_level:
            gen["thinkingConfig"] = {"thinkingLevel": a.thinking_level.upper()}
        payload = {"systemInstruction": {"parts": [{"text": system}]}, "contents": contents, "generationConfig": gen}
        key = os.environ.get(a.key_env or "GEMINI_API_KEY", "")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{a.model}:generateContent?key={key}"
        r = self._post(url, payload, {})
        cand = (r.get("candidates") or [{}])[0]
        text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []) if not p.get("thought"))
        um = r.get("usageMetadata") or {}
        return parse_json(text), {"prompt": um.get("promptTokenCount", 0), "output": um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0)}


def gemini_schema(schema: dict) -> dict:
    """Gemini's JSON schema dialect: no additionalProperties / strict, nullable via 'nullable'."""
    def walk(s):
        if isinstance(s, dict):
            # Gemini rejects minItems/maxItems combined with a long enum (checked 2026-10-17); the prompt states the limits
            s = {k: walk(v) for k, v in s.items() if k not in ("additionalProperties", "minItems", "maxItems")}
            if isinstance(s.get("type"), list):
                types = [t for t in s["type"] if t != "null"]
                s["type"] = types[0] if types else "string"
                s["nullable"] = True
            return s
        if isinstance(s, list):
            return [walk(x) for x in s]
        return s
    return walk(schema)


def parse_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("no JSON object in the answer")
    return json.loads(m.group(0))


# ── driver ─────────────────────────────────────────────────────────────────
def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["body", "subjects"], required=True)
    ap.add_argument("--backend", choices=["openai", "gemini"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--key-env", default=None, help="env var with the API key (default: BLABLADOR_API_KEY/OPENAI_API_KEY for openai, GEMINI_API_KEY for gemini)")
    ap.add_argument("--split", default="all", help="all | dev | test | rare | gold (units/docs of the gold sets) | comma-separated ids")
    ap.add_argument("--pass", dest="pass_no", type=int, default=1, choices=[1, 2])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=None, help="answer budget; default 1500, or 8000 with --thinking (thinking counts)")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--thinking", dest="thinking", action="store_true", default=None, help="openai/vLLM: enable_thinking=true")
    ap.add_argument("--no-thinking", dest="thinking", action="store_false")
    ap.add_argument("--thinking-level", default="low", help="gemini: low|medium|high|'' (off)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.max_tokens is None:
        args.max_tokens = 8000 if args.thinking else 1500
    if args.key_env is None:
        args.key_env = "GEMINI_API_KEY" if args.backend == "gemini" else ("BLABLADOR_API_KEY" if os.environ.get("BLABLADOR_API_KEY") else "OPENAI_API_KEY")

    # scheme + prompt
    if args.task == "body":
        cb = yaml.safe_load(CODEBOOK.read_text(encoding="utf-8"))
        system = BODY_SYSTEM.format(codebook=render_codebook(cb, None if args.pass_no == 1 else 20260923))
        schema = json.loads(json.dumps(BODY_SCHEMA))
        schema["properties"]["annotations"]["items"]["properties"]["code"]["enum"] = [c["id"] for c in cb["codes"]]
        schema["properties"]["annotations"]["items"]["properties"]["role"]["enum"] = cb["roles"]
        version = cb["version"]
    else:
        sc = yaml.safe_load(SCHEME.read_text(encoding="utf-8"))
        system = SUBJ_SYSTEM.format(scheme=render_scheme(sc))
        schema = json.loads(json.dumps(SUBJ_SCHEMA))
        ids = [c["id"] for c in sc["concepts"] if c["facet"] != "BODY"]
        schema["properties"]["central"]["items"]["enum"] = ids
        schema["properties"]["secondary"]["items"]["enum"] = ids
        version = sc["version"]
    shots, shot_ids = few_shots(args.task, args.pass_no)
    prompt_hash = hashlib.sha256((system + json.dumps(shots)).encode()).hexdigest()[:12]

    # items
    if args.task == "body":
        items = [json.loads(l) for l in UNITS.open(encoding="utf-8")]
        key = "unit"
        if args.split in ("dev", "test", "rare", "gold"):
            split = json.loads((GOLD_DIR / "intimate_body_split.json").read_text())
            want = set(split) if args.split == "gold" else {u for u, s in split.items() if s == args.split}
            items = [u for u in items if u["unit"] in want]
    else:
        cons = json.loads(CONSILIA.read_text(encoding="utf-8"))["consilia"]
        items = list(cons.values())
        key = "id"
        if args.split in ("gold", "dev", "test"):
            want = {json.loads(l)["doc"] for l in (GOLD_DIR / "subjects_gold.jsonl").open(encoding="utf-8")}
            items = [c for c in items if c["id"] in want]
    if args.split not in ("all", "dev", "test", "rare", "gold"):
        want = set(args.split.split(","))
        items = [x for x in items if x[key] in want]
    items = [x for x in items if x[key] not in set(shot_ids)]       # never classify the few-shot examples
    out = args.out or OUT_DIR / f"{args.task}_{slug(args.model)}_p{args.pass_no}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.open(encoding="utf-8"):
            try:
                rec = json.loads(l)
                if rec.get("ok", True) and rec.get("prompt_hash") == prompt_hash:   # failed items and old prompts are redone
                    done.add(rec[key])
            except Exception:
                pass
    todo = [x for x in items if x[key] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{args.task}: {len(items)} items, {len(done)} done, {len(todo)} to do -> {out}  "
          f"(model {args.model}, pass {args.pass_no}, prompt {prompt_hash}, few-shots {len(shot_ids)})")
    if not todo:
        return

    be = (OpenAIBackend if args.backend == "openai" else GeminiBackend)(args)
    lock = threading.Lock()
    fh = out.open("a", encoding="utf-8")
    t0 = time.time()
    n_ok = n_err = 0

    def work(x: dict) -> dict:
        rec = {key: x[key], "task": args.task, "model": args.model, "backend": args.backend, "pass": args.pass_no,
               "prompt_hash": prompt_hash, "scheme_version": version, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        user = body_user(x) if args.task == "body" else subj_user(x)
        try:
            res, usage = be.complete(system, shots, user, schema)
            if args.task == "body":
                bad = [a for a in res.get("annotations", []) if not evidence_ok(a.get("evidence_la", ""), x["la"])]
                if bad:
                    retry = user + "\n\nYour previous answer quoted evidence that is NOT a verbatim substring of the UNIT TEXT (la): " + \
                        json.dumps([a.get("evidence_la") for a in bad], ensure_ascii=False) + ". Copy the evidence exactly from the unit text."
                    res2, usage2 = be.complete(system, shots, retry, schema)
                    usage = {k: usage[k] + usage2[k] for k in usage}
                    res = res2
                for a in res.get("annotations", []):
                    a["evidence_valid"] = evidence_ok(a.get("evidence_la", ""), x["la"])
                res["relevant"] = bool(res.get("annotations"))
            rec.update(result=res, usage=usage, ok=True)
        except Exception as e:  # noqa: BLE001
            rec.update(ok=False, error=str(e)[:500])
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, x) for x in todo]
        for i, f in enumerate(as_completed(futs), 1):
            rec = f.result()
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                if rec["ok"]:
                    n_ok += 1
                    be.usage["prompt"] += rec["usage"]["prompt"]
                    be.usage["output"] += rec["usage"]["output"]
                else:
                    n_err += 1
                    if n_err >= 20 and n_ok == 0:
                        print(f"stopping: the first {n_err} items all failed, last error: {rec.get('error')}\n"
                              f"(is the model server at the base URL running? failed items are retried on the next run)")
                        os._exit(2)
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)}  ok {n_ok} err {n_err}  {el / i:.1f} s/item  tokens in {be.usage['prompt']} out {be.usage['output']}")
    fh.close()
    print(f"done: {n_ok} ok, {n_err} errors (re-run to retry the errors)")


if __name__ == "__main__":
    main()
