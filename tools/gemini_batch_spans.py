#!/usr/bin/env python3
"""
Layer C span annotation through the Gemini Batch API (half price, up to 24 h turnaround), same prompt,
schema and output shape as src/annotate/spans.py.

  python tools/gemini_batch_spans.py --submit --units output/units_all_20261027.jsonl --model gemini-4-argon
        -> uploads the request file, creates the job, writes output/qa/gemini_batch_<model>.json (job state file)
  python tools/gemini_batch_spans.py --status                  # job state
  python tools/gemini_batch_spans.py --fetch                   # when SUCCEEDED: results -> output/annotations/spans_<model>_p1.jsonl
  python tools/gemini_batch_spans.py --wait                    # poll every 5 min, then fetch

Units with an error or with spans that could not be located verbatim are listed at the end; rerun them
synchronously with `spans.py --units … --split <ids> --out <same file>` (failed records are redone).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "annotate"))
from classify import gemini_schema, slug  # noqa: E402
from spans import SCHEMA, SCHEME, SYSTEM, locate, render_labels, user_prompt  # noqa: E402

API = "https://generativelanguage.googleapis.com"
STATE_DIR = ROOT / "output" / "qa"


def key() -> str:
    k = os.environ.get("GEMINI_API_KEY", "")
    if not k:
        sys.exit("GEMINI_API_KEY is not set")
    return k


def prompt_parts():
    sc = yaml.safe_load(SCHEME.read_text(encoding="utf-8"))
    labels = [l["id"] for l in sc["labels"]]
    system = SYSTEM.format(labels=render_labels(sc))
    schema = json.loads(json.dumps(SCHEMA)); schema["properties"]["spans"]["items"]["properties"]["label"]["enum"] = labels
    return sc, labels, system, schema, hashlib.sha256(system.encode()).hexdigest()[:12]


def submit(args) -> None:
    sc, labels, system, schema, phash = prompt_parts()
    units = [json.loads(l) for l in args.units.open(encoding="utf-8")]
    if args.limit:
        units = units[:args.limit]
    gen = {"responseMimeType": "application/json", "responseJsonSchema": gemini_schema(schema),
           "temperature": args.temperature, "maxOutputTokens": args.max_tokens}
    if args.thinking_level:
        gen["thinkingConfig"] = {"thinkingLevel": args.thinking_level.upper()}
    lines = []
    for u in units:
        req = {"systemInstruction": {"parts": [{"text": system}]},
               "contents": [{"role": "user", "parts": [{"text": user_prompt(u)}]}],
               "generationConfig": gen}
        lines.append(json.dumps({"key": u["unit"], "request": req}, ensure_ascii=False))
    data = ("\n".join(lines) + "\n").encode("utf-8")
    print(f"{len(units)} requests, {len(data) / 1e6:.1f} MB, model {args.model}, prompt {phash}")
    hdr = {"x-goog-api-key": key()}
    r = requests.post(f"{API}/upload/v1beta/files", headers={**hdr, "X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                      "X-Goog-Upload-Header-Content-Length": str(len(data)), "X-Goog-Upload-Header-Content-Type": "application/jsonl",
                      "Content-Type": "application/json"}, json={"file": {"display_name": f"spans_{slug(args.model)}_{time.strftime('%Y%m%d-%H%M')}"}}, timeout=120)
    r.raise_for_status()
    up = r.headers["x-goog-upload-url"]
    r = requests.post(up, headers={"X-Goog-Upload-Offset": "0", "X-Goog-Upload-Command": "upload, finalize", "Content-Type": "application/jsonl"}, data=data, timeout=600)
    r.raise_for_status()
    fname = r.json()["file"]["name"]
    print("uploaded", fname)
    r = requests.post(f"{API}/v1beta/models/{args.model}:batchGenerateContent", headers={**hdr, "Content-Type": "application/json"},
                      json={"batch": {"display_name": f"consilia spans {args.model}", "input_config": {"file_name": fname}}}, timeout=120)
    r.raise_for_status()
    job = r.json()
    st = {"job": job["name"], "model": args.model, "units": str(args.units), "n": len(units), "prompt_hash": phash, "scheme_version": sc["version"],
          "input_file": fname, "submitted": time.strftime("%Y-%m-%dT%H:%M:%S"), "labels": labels}
    out = STATE_DIR / f"gemini_batch_{slug(args.model)}.json"
    out.write_text(json.dumps(st, indent=1), encoding="utf-8")
    print("job", job["name"], "->", out)


def load_state(args) -> dict:
    f = args.state or STATE_DIR / f"gemini_batch_{slug(args.model)}.json"
    return json.loads(Path(f).read_text(encoding="utf-8"))


def status(args) -> dict:
    st = load_state(args)
    r = requests.get(f"{API}/v1beta/{st['job']}", headers={"x-goog-api-key": key()}, timeout=60)
    r.raise_for_status()
    j = r.json()
    state = (j.get("metadata") or {}).get("state") or j.get("state")
    stats = (j.get("metadata") or {}).get("batchStats") or j.get("batchStats") or {}
    print(st["job"], state, json.dumps(stats))
    return {"job": j, "state": state, "st": st}


def fetch(args) -> None:
    s = status(args)
    if not str(s["state"]).endswith("SUCCEEDED"):        # JOB_STATE_* in the docs, BATCH_STATE_* in practice
        sys.exit(f"job is {s['state']}")
    j, st = s["job"], s["st"]
    resp = j.get("response") or (j.get("metadata") or {}).get("output") or {}
    dest = j.get("dest") or {}
    fname = resp.get("responsesFile") or dest.get("fileName")
    hdr = {"x-goog-api-key": key()}
    if fname:
        r = requests.get(f"{API}/download/v1beta/{fname}:download?alt=media", headers=hdr, timeout=1800)
        r.raise_for_status()
        results = [json.loads(l) for l in r.text.splitlines() if l.strip()]
    else:
        results = resp.get("inlinedResponses") or dest.get("inlinedResponses") or []
    units = {json.loads(l)["unit"]: json.loads(l) for l in Path(st["units"]).open(encoding="utf-8")}
    out = args.out or ROOT / "output" / "annotations" / f"spans_{slug(st['model'])}_p1.jsonl"
    n_ok = n_err = n_unm = 0; bad = []
    tot = {"prompt": 0, "output": 0}
    with out.open("w", encoding="utf-8") as fh:
        for item in results:
            k = item.get("key") or (item.get("metadata") or {}).get("key")
            u = units.get(k)
            if not u:
                continue
            rec = {"unit": k, "doc": u["doc"], "task": "spans", "model": st["model"], "backend": "gemini-batch", "pass": 1,
                   "prompt_hash": st["prompt_hash"], "scheme_version": st["scheme_version"], "ts": st["submitted"]}
            resp_i = item.get("response")
            if not resp_i or item.get("error"):
                rec.update(ok=False, error=json.dumps(item.get("error") or item)[:500]); n_err += 1; bad.append(k)
            else:
                try:
                    cand = (resp_i.get("candidates") or [{}])[0]
                    text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []) if not p.get("thought"))
                    um = resp_i.get("usageMetadata") or {}
                    usage = {"prompt": um.get("promptTokenCount", 0), "output": um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0)}
                    tot["prompt"] += usage["prompt"]; tot["output"] += usage["output"]
                    res = json.loads(text)
                    spans = [x for x in res.get("spans", []) if x.get("label") in st["labels"]]
                    found, unmatched = locate(u["la"], spans)
                    rec.update(spans=found, unmatched=[{"text": x.get("text"), "label": x.get("label")} for x in unmatched], usage=usage, ok=True)
                    n_ok += 1
                    if unmatched:
                        n_unm += 1
                except Exception as e:  # noqa: BLE001
                    rec.update(ok=False, error=str(e)[:500]); n_err += 1; bad.append(k)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    missing = sorted(set(units) - {(i.get("key") or (i.get("metadata") or {}).get("key")) for i in results})
    print(f"{out}: {n_ok} ok, {n_err} errors, {n_unm} with unmatched spans, {len(missing)} units without a result; tokens {tot}")
    if bad or missing:
        (STATE_DIR / "gemini_batch_retry_units.txt").write_text("\n".join(bad + missing) + "\n", encoding="utf-8")
        print("retry list -> output/qa/gemini_batch_retry_units.txt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--units", type=Path, default=ROOT / "output" / "units.jsonl")
    ap.add_argument("--model", default="gemini-4-argon")
    ap.add_argument("--state", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--thinking-level", default="low")
    args = ap.parse_args()
    if args.submit:
        submit(args)
    elif args.wait:
        while True:
            s = status(args)
            if str(s["state"]).split("_")[-1] in ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED"):
                break
            time.sleep(300)
        if str(s["state"]).endswith("SUCCEEDED"):
            fetch(args)
    elif args.fetch:
        fetch(args)
    else:
        status(args)


if __name__ == "__main__":
    main()
