#!/usr/bin/env python3
"""
Retranslate the body of consilia whose Latin paragraphs no longer match their stored translation
(paragraph counts differ after text corrections), with Gemini, directly into output/consilia.json and
the per-print JSON (output/<author>/<print>.json), so that the next build carries them on.

    python tools/retranslate_stale.py                 # all stale consilia, en and de
    python tools/retranslate_stale.py --dry-run       # list only
    python tools/retranslate_stale.py --limit 5
Prompt and style follow the original GPU translation (src/gpu/qwen_process.py): faithful, legal
terminology kept, one translation per paragraph. A consilium is written only when the model returns
exactly one translation per Latin paragraph; otherwise it stays stale and is listed at the end.
Needs GEMINI_API_KEY. Afterwards: python src/build_site_data.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANGS = {"en": "English", "de": "German"}
SYSTEM = ("You are a precise translator of Latin legal texts, 15th/16th-century ius commune consilia (jurist legal "
          "opinions), into {lang}. Translate faithfully, preserving legal terminology precision; keep citations of the "
          "Corpus iuris and of jurists in their abbreviated Latin form; keep the † marks and the leading marginal numbers "
          "where the paragraph has them. You receive the paragraphs of one consilium as a JSON list and answer with a JSON "
          "object {{\"paragraphs\": [...]}} containing exactly one translation per input paragraph, in the same order, "
          "no commentary.")
SCHEMA = {"type": "object", "properties": {"paragraphs": {"type": "array", "items": {"type": "string"}}}, "required": ["paragraphs"]}


def stale(c: dict, lang: str) -> bool:
    t = (c.get("translations") or {}).get(lang) or {}
    body = c.get("body") or []
    return bool(body) and len(t.get("body") or []) != len(body)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--langs", default="en,de")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    path = ROOT / "output" / "consilia.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    consilia = data["consilia"]
    jobs = [(cid, lang) for cid, c in consilia.items() for lang in args.langs.split(",") if stale(c, lang)]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} stale consilium translations ({len({j[0] for j in jobs})} consilia)")
    if args.dry_run or not jobs:
        return
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=types.HttpOptions(api_version="v1beta"))

    def work(cid: str, lang: str):
        body = consilia[cid]["body"]
        cfg = types.GenerateContentConfig(system_instruction=SYSTEM.format(lang=LANGS[lang]), response_mime_type="application/json",
                                          response_json_schema=SCHEMA, max_output_tokens=32768,
                                          thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW))
        for attempt in range(3):
            try:
                r = client.models.generate_content(model=args.model, contents=json.dumps(body, ensure_ascii=False), config=cfg)
                out = json.loads(r.text or "{}").get("paragraphs") or []
                if len(out) == len(body) and all(isinstance(s, str) and s.strip() for s in out):
                    return cid, lang, out, None
                err = f"{len(out)} paragraphs for {len(body)}"
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {str(e)[:120]}"
                time.sleep(5 * (attempt + 1))
        return cid, lang, None, err

    done, failed = 0, []
    touched_prints = set()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, cid, lang) for cid, lang in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            cid, lang, out, err = f.result()
            if out is None:
                failed.append((cid, lang, err)); print(f"[{i}/{len(jobs)}] {cid} {lang}: FAILED {err}", flush=True); continue
            c = consilia[cid]
            c.setdefault("translations", {}).setdefault(lang, {})["body"] = out
            c["translations"][lang]["retranslated"] = time.strftime("%Y-%m-%d") + f" {args.model}"
            touched_prints.add((c["author_viaf"], c["volume"])); done += 1
            print(f"[{i}/{len(jobs)}] {cid} {lang}: {len(out)} paragraphs", flush=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    for author, print_id in touched_prints:
        pp = ROOT / "output" / author / f"{print_id}.json"
        if pp.exists():
            pdata = json.loads(pp.read_text(encoding="utf-8"))
            for cid, c in pdata["consilia"].items():
                if cid in consilia and c.get("body") == consilia[cid].get("body"):
                    c["translations"] = consilia[cid].get("translations", c.get("translations"))
            pp.write_text(json.dumps(pdata, ensure_ascii=False, indent=1), encoding="utf-8")
        ap_ = ROOT / "output" / author / "consilia.json"
        if ap_.exists():
            adata = json.loads(ap_.read_text(encoding="utf-8"))
            for cid, c in adata["consilia"].items():
                if cid in consilia and c.get("body") == consilia[cid].get("body"):
                    c["translations"] = consilia[cid].get("translations", c.get("translations"))
            ap_.write_text(json.dumps(adata, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written: {done}, failed: {len(failed)}")
    for f_ in failed:
        print("  ", *f_)
    if done:
        print("next: python src/build_site_data.py")


if __name__ == "__main__":
    main()
