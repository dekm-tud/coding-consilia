"""
GPU-server script: translate + extract keywords for every consilium using
Qwen3.6-27B (https://huggingface.co/Qwen/Qwen3.6-27B) served locally via
vLLM's offline batch-inference API (no server process needed).

Meant to be COPIED to the GPU server and run there directly:

    pip install vllm

Input: a JSONL file produced by src/prepare_llm_input.py (one record per
consilium: id, title, argument_block, body_chunks).

Output: a JSONL file, one record per consilium:
    {"id": "...",
     "argument_en": "...", "argument_de": "...",
     "body_chunks_en": [...], "body_chunks_de": [...],
     "keywords_argument_raw": ["...", ...],   # ~20 English keywords
     "keywords_body_raw": ["...", ...]}       # ~20 English keywords

Batched for throughput: every --batch-size consilia are processed as two
sequential vLLM generate() calls. First, all translation requests for the
batch are sent in one flat list, so vLLM's continuous-batching scheduler
interleaves them on the GPU instead of running one conversation at a time.
Then keyword extraction is run the same way, but built from the *English
translations* just produced (not the raw Latin) — Qwen reads its own clean
English far more reliably than 16th-century legal Latin, so keywords come
out more accurate and more consistently phrased. Results are written and
flushed after each batch (resumable — ids already present in --output are
skipped on restart).

Usage:
    python qwen_process.py --input llm_input.jsonl --output llm_output.jsonl
    python qwen_process.py --model Qwen/Qwen3.6-27B --tensor-parallel-size 2
    python qwen_process.py --limit 5 --batch-size 5   # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from vllm_common import DEFAULT_MODEL, chat_batch, load_llm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_N_KEYWORDS = 20

_TRANSLATE_SYSTEM = (
    "You are a precise translator of Latin legal texts — 15th/16th-century ius "
    "commune consilia (jurist legal opinions) — into {lang}. Translate faithfully, "
    "preserving legal terminology precision and, if the source is a numbered list, "
    "the numbering. Respond with ONLY the translation, no commentary, no preamble."
)

_KEYWORD_SYSTEM = (
    "You are a legal-historical indexer analysing the English translation of a "
    "medieval Latin consilium (a jurist's legal opinion). Extract exactly {n} "
    "keywords in English that "
    "best index the given passage. Cover BOTH: (a) legal/juristic concepts "
    "(e.g. dowry, feudal succession, testamentary capacity, usufruct) and "
    "(b) the general life-topics / subject matter involved (e.g. marriage, "
    "divorce, inheritance, fertility, death, debt, guardianship). Prefer "
    "normalized, canonical English terms (e.g. 'divorce', not 'the dissolution "
    "of the marriage bond') so keywords are reusable and comparable across many "
    "documents. Respond with ONLY a JSON array of exactly {n} short strings — "
    "no commentary, no markdown fences."
)

_LANG_NAMES = {"en": "English", "de": "German"}
_RE_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


# ---------------------------------------------------------------------------
# Task construction
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """One chat request, tagged with where its result belongs in the output."""
    consilium_id: str
    kind: str        # "argument_en" | "argument_de" | "body_en" | "body_de" | "kw_argument" | "kw_body"
    index: int        # body-chunk index (0 for non-body-chunk kinds)
    messages: list[dict]
    max_tokens: int


def _translate_task(cid: str, kind: str, lang: str, text: str, index: int = 0) -> Optional[Task]:
    if not text.strip():
        return None
    system = _TRANSLATE_SYSTEM.format(lang=_LANG_NAMES[lang])
    max_tokens = max(64, int(len(text.split()) * 2.5))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
    return Task(cid, kind, index, messages, max_tokens)


def _keyword_task(cid: str, kind: str, text: str) -> Optional[Task]:
    if not text.strip():
        return None
    system = _KEYWORD_SYSTEM.format(n=_N_KEYWORDS)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
    return Task(cid, kind, 0, messages, 400)


def build_translation_tasks(rec: dict) -> list[Task]:
    cid = rec["id"]
    argument_block = rec.get("argument_block", "") or ""
    body_chunks = rec.get("body_chunks") or []

    tasks: list[Task] = []
    for kind, lang in (("argument_en", "en"), ("argument_de", "de")):
        t = _translate_task(cid, kind, lang, argument_block)
        if t:
            tasks.append(t)
    for i, chunk in enumerate(body_chunks):
        for kind, lang in (("body_en", "en"), ("body_de", "de")):
            t = _translate_task(cid, kind, lang, chunk, index=i)
            if t:
                tasks.append(t)
    return tasks


def build_keyword_tasks(rec: dict, translations: dict[tuple[str, str, int], str]) -> list[Task]:
    """Build keyword-extraction tasks from the English translations just produced
    (translations[(consilium_id, "argument_en"|"body_en", chunk_index)]), not the
    raw Latin — see module docstring for why."""
    cid = rec["id"]
    n_chunks = len(rec.get("body_chunks") or [])
    argument_en = translations.get((cid, "argument_en", 0), "")
    body_en_full = "\n\n".join(
        translations.get((cid, "body_en", i), "") for i in range(n_chunks)
    )

    tasks: list[Task] = []
    for kind, text in (("kw_argument", argument_en), ("kw_body", body_en_full)):
        t = _keyword_task(cid, kind, text)
        if t:
            tasks.append(t)
    return tasks


# ---------------------------------------------------------------------------
# Keyword JSON parsing
# ---------------------------------------------------------------------------


def _parse_keyword_json(raw: str) -> Optional[list[str]]:
    m = _RE_JSON_ARRAY.search(raw)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    return [str(x).strip() for x in data if str(x).strip()]


# ---------------------------------------------------------------------------
# Batch execution
# ---------------------------------------------------------------------------


def run_tasks(llm, tasks: list[Task]) -> dict[tuple[str, str, int], str]:
    """Execute a flat list of Tasks in one vLLM batch; return raw text per (id, kind, index)."""
    texts = chat_batch(llm, [t.messages for t in tasks], [t.max_tokens for t in tasks])
    return {(t.consilium_id, t.kind, t.index): text for t, text in zip(tasks, texts)}


def resolve_keywords(llm, kw_tasks: list[Task], results: dict[tuple[str, str, int], str]) -> dict[tuple[str, str], list[str]]:
    """Parse keyword-extraction results; retry once (batched) on unparseable output."""
    parsed: dict[tuple[str, str], list[str]] = {}
    retry_tasks: list[Task] = []
    for t in kw_tasks:
        raw = results[(t.consilium_id, t.kind, t.index)]
        kws = _parse_keyword_json(raw)
        if kws:
            parsed[(t.consilium_id, t.kind)] = kws
        else:
            retry_tasks.append(t)

    if retry_tasks:
        log.info("Retrying %d unparseable keyword generation(s) …", len(retry_tasks))
        retry_results = run_tasks(llm, retry_tasks)
        for t in retry_tasks:
            kws = _parse_keyword_json(retry_results[(t.consilium_id, t.kind, t.index)])
            if not kws:
                log.warning("Keyword extraction for %s/%s failed after retry; using empty list.",
                             t.consilium_id, t.kind)
            parsed[(t.consilium_id, t.kind)] = kws or []

    return parsed


def assemble_batch(records: list[dict], results: dict[tuple[str, str, int], str],
                    keywords: dict[tuple[str, str], list[str]]) -> list[dict]:
    out = []
    for rec in records:
        cid = rec["id"]
        n_chunks = len(rec.get("body_chunks") or [])
        body_en = [results.get((cid, "body_en", i), "") for i in range(n_chunks)]
        body_de = [results.get((cid, "body_de", i), "") for i in range(n_chunks)]
        out.append({
            "id": cid,
            "argument_en": results.get((cid, "argument_en", 0), ""),
            "argument_de": results.get((cid, "argument_de", 0), ""),
            "body_chunks_en": body_en,
            "body_chunks_de": body_de,
            "keywords_argument_raw": keywords.get((cid, "kw_argument"), []),
            "keywords_body_raw": keywords.get((cid, "kw_body"), []),
        })
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _chunks(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def run(model_id: str, input_path: Path, output_path: Path, tensor_parallel_size: int,
        dtype: str, gpu_memory_utilization: float, max_model_len: Optional[int],
        max_num_seqs: Optional[int], batch_size: int, limit: Optional[int]) -> None:
    records = _read_jsonl(input_path)
    if not records:
        log.error("No records found in %s", input_path)
        sys.exit(1)

    done_ids = {r["id"] for r in _read_jsonl(output_path)}
    todo = [r for r in records if r["id"] not in done_ids]
    log.info("%d total, %d already done, %d remaining", len(records), len(done_ids), len(todo))
    if limit:
        todo = todo[:limit]
        log.info("Limiting this run to %d record(s) (--limit).", limit)
    if not todo:
        log.info("Nothing to do.")
        return

    llm = load_llm(model_id, tensor_parallel_size, dtype, gpu_memory_utilization, max_model_len, max_num_seqs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    with output_path.open("a", encoding="utf-8") as out_fh:
        for batch in _chunks(todo, batch_size):
            log.info("Batch: %d consilia (%d–%d of %d)", len(batch), n_done + 1, n_done + len(batch), len(todo))
            translation_tasks = [t for rec in batch for t in build_translation_tasks(rec)]
            log.info("  %d translation request(s) in this batch", len(translation_tasks))
            try:
                translations = run_tasks(llm, translation_tasks)
                kw_tasks = [t for rec in batch for t in build_keyword_tasks(rec, translations)]
                log.info("  %d keyword-extraction request(s) in this batch", len(kw_tasks))
                kw_results = run_tasks(llm, kw_tasks)
                keywords = resolve_keywords(llm, kw_tasks, kw_results)
                batch_records = assemble_batch(batch, translations, keywords)
            except Exception:
                log.exception("Batch failed — skipping (rerun later to retry these ids).")
                continue
            for rec in batch_records:
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_fh.flush()
            n_done += len(batch)

    log.info("Done. Output: %s", output_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="GPU-side Qwen3.6 translation + keyword extraction (vLLM)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local path")
    ap.add_argument("--input", type=Path, default=Path("llm_input.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("llm_output.jsonl"))
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=None, help="Cap context length (default: model's own max)")
    ap.add_argument("--max-num-seqs", type=int, default=256,
                     help="Max concurrent sequences vLLM schedules. Hybrid Mamba/attention models "
                          "(like Qwen3.6) need one Mamba cache block per concurrent sequence — if vLLM "
                          "errors with 'exceeds available Mamba cache blocks (N)', pass a value below N, "
                          "or raise --gpu-memory-utilization for more blocks. Pass 0 to use vLLM's default.")
    ap.add_argument("--batch-size", type=int, default=25, help="Consilia per vLLM batch call")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N remaining records (smoke test)")
    args = ap.parse_args()
    run(args.model, args.input, args.output, args.tensor_parallel_size, args.dtype,
        args.gpu_memory_utilization, args.max_model_len, args.max_num_seqs or None,
        args.batch_size, args.limit)


if __name__ == "__main__":
    main()
