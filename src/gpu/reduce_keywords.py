"""
GPU-server script: stage 9b of the keyword pipeline — reduce each consilium's
already-vocabulary-compliant candidate keywords (src/prepare_keyword_reduction_input.py's
output, typically 13-39 terms per consilium) down to a curated ~10, via two
passes over the SAME small candidate list rather than retrieval against the
full 3,463-term controlled vocabulary.

Why not retrieve from the full vocabulary: tested empirically first. Embedding
the whole consilium text and comparing it to all 3,463 vocabulary term
embeddings systematically misses topics that are real but only briefly
mentioned in a long document — a whole-document embedding averages over
everything the text discusses. On consilia already correctly tagged "rape" by
the original (full-text-reading) extraction, embedding-similarity retrieval
ranked "rape" anywhere from #3 to #1525 out of 3,463 depending on the
document, and neither idf-reweighting nor chunk-level pooling fixed this
reliably (idf-reweighting actually made it worse — with ~1,100 singleton-
frequency terms in the vocabulary, their large idf multiplier drowns out the
similarity signal with noise). So instead of retrieving candidates, this
script starts from each consilium's OWN existing keywords.argument/
keywords.body (already produced by reading the actual text, already includes
the minor/rare topics) filtered to vocabulary-surviving terms, and just picks
the best ~10 of that already-small, already-relevant set:

  Pass A ("central"): the most central, defining themes of the consilium.
  Pass B ("distinctive"): notable/distinctive topics present, even if minor —
      each candidate is shown with its corpus-wide doc_freq so the model can
      recognize rarity, and is explicitly told not to skip a real rare topic
      just because it's a small part of the text.

keywords_final = union of both passes, deduplicated (order-preserving).
Consilia with <=10 candidates already are passed through unchanged with no
LLM call at all.

Both passes are constrained selection, not free generation: every returned
label is checked against that consilium's own candidate list (exact or
case/whitespace-normalized match) and anything else is dropped, with one
batched retry for records where a pass returned nothing valid.

Usage:
    python reduce_keywords.py --input keyword_reduction_input.jsonl --output keyword_reduction_output.jsonl
    python reduce_keywords.py --limit 5 --batch-size 5   # smoke test
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

_PASS_THROUGH_MAX = 10  # consilia with this many candidates or fewer skip the LLM entirely
_MAX_PER_PASS = 6
_TEXT_CHAR_LIMIT = 6000  # keep prompts bounded for very long consilia

_CENTRAL_SYSTEM = (
    "You are a legal-historical indexer. Below is the English translation of a medieval "
    "Latin consilium (a jurist's legal opinion), followed by a candidate list of topical "
    "labels already identified as relevant to it, drawn from a controlled vocabulary. "
    "Select the labels that represent the MOST CENTRAL, DEFINING subject matter of this "
    "consilium — the handful of themes that most completely describe what the case is "
    "actually about. Respond with ONLY a JSON array of at most {max_n} labels, copied "
    "EXACTLY as given in the candidate list (do not paraphrase, do not invent new labels, "
    "do not include anything not in the candidate list). No commentary, no markdown fences."
)

_DISTINCTIVE_SYSTEM = (
    "You are a legal-historical indexer building a discovery index for a large corpus of "
    "medieval consilia. Below is the English translation of a consilium, followed by a "
    "candidate list of topical labels already identified as relevant to it, each annotated "
    "with how many of the {n_total} consilia in the whole corpus also carry that label "
    "(lower = rarer, more distinctive). Select the labels that are NOTABLE OR DISTINCTIVE "
    "for THIS consilium — genuinely rare or unusual topics that are actually present in the "
    "text, even if only a minor or secondary part of it. Do not select a label just because "
    "it is common; only select a common label here if it is unusually central to this case. "
    "Do not skip a real rare topic just because it's a small part of the text. Respond with "
    "ONLY a JSON array of at most {max_n} labels, copied EXACTLY as given in the candidate "
    "list. No commentary, no markdown fences."
)

_RE_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


@dataclass
class Task:
    """One chat request, tagged with where its result belongs and what labels it may pick from."""
    consilium_id: str
    kind: str  # "central" | "distinctive"
    messages: list[dict]
    max_tokens: int
    candidate_labels: list[str]


def _normalize(s: str) -> str:
    return " ".join(s.strip().lower().split())


def _user_content(text_en: str, candidates: list[dict], with_freq: bool, n_total: int) -> str:
    if with_freq:
        cand_lines = "\n".join(f"- {c['label']} ({c['doc_freq']}/{n_total} consilia)" for c in candidates)
    else:
        cand_lines = "\n".join(f"- {c['label']}" for c in candidates)
    text = text_en[:_TEXT_CHAR_LIMIT]
    return f"CONSILIUM TEXT:\n{text}\n\nCANDIDATE LABELS:\n{cand_lines}"


def build_tasks(rec: dict, n_total: int) -> list[Task]:
    cid = rec["id"]
    candidates = rec["candidates"]
    labels = [c["label"] for c in candidates]
    max_n = min(_MAX_PER_PASS, len(candidates))

    central_messages = [
        {"role": "system", "content": _CENTRAL_SYSTEM.format(max_n=max_n)},
        {"role": "user", "content": _user_content(rec["text_en"], candidates, False, n_total)},
    ]
    distinctive_messages = [
        {"role": "system", "content": _DISTINCTIVE_SYSTEM.format(max_n=max_n, n_total=n_total)},
        {"role": "user", "content": _user_content(rec["text_en"], candidates, True, n_total)},
    ]
    return [
        Task(cid, "central", central_messages, 300, labels),
        Task(cid, "distinctive", distinctive_messages, 300, labels),
    ]


def _parse_labels(raw: str, valid_labels: list[str]) -> Optional[list[str]]:
    m = _RE_JSON_ARRAY.search(raw)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    by_norm = {_normalize(l): l for l in valid_labels}
    out: list[str] = []
    for item in data:
        canonical = by_norm.get(_normalize(str(item)))
        if canonical and canonical not in out:
            out.append(canonical)
    return out


def run_tasks(llm, tasks: list[Task]) -> dict[tuple[str, str], str]:
    texts = chat_batch(llm, [t.messages for t in tasks], [t.max_tokens for t in tasks])
    return {(t.consilium_id, t.kind): text for t, text in zip(tasks, texts)}


def resolve_tasks(llm, tasks: list[Task], results: dict[tuple[str, str], str]) -> dict[tuple[str, str], list[str]]:
    """Parse selection results; retry once (batched) on unparseable/empty output."""
    parsed: dict[tuple[str, str], list[str]] = {}
    retry_tasks: list[Task] = []
    for t in tasks:
        raw = results[(t.consilium_id, t.kind)]
        labels = _parse_labels(raw, t.candidate_labels)
        if labels:
            parsed[(t.consilium_id, t.kind)] = labels
        else:
            retry_tasks.append(t)

    if retry_tasks:
        log.info("Retrying %d unparseable/empty selection(s) …", len(retry_tasks))
        retry_results = run_tasks(llm, retry_tasks)
        for t in retry_tasks:
            labels = _parse_labels(retry_results[(t.consilium_id, t.kind)], t.candidate_labels)
            if not labels:
                log.warning("Selection for %s/%s failed after retry; using empty list.",
                            t.consilium_id, t.kind)
            parsed[(t.consilium_id, t.kind)] = labels or []

    return parsed


def assemble_batch(records: list[dict], selections: dict[tuple[str, str], list[str]]) -> list[dict]:
    out = []
    for rec in records:
        cid = rec["id"]
        if len(rec["candidates"]) <= _PASS_THROUGH_MAX:
            final = [c["label"] for c in rec["candidates"]]
        else:
            central = selections.get((cid, "central"), [])
            distinctive = selections.get((cid, "distinctive"), [])
            final = list(dict.fromkeys(central + distinctive))  # union, order-preserving dedup
        out.append({"id": cid, "keywords_final": final})
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
    n_total = len(records)

    done_ids = {r["id"] for r in _read_jsonl(output_path)}
    todo = [r for r in records if r["id"] not in done_ids]
    log.info("%d total, %d already done, %d remaining", len(records), len(done_ids), len(todo))
    if limit:
        todo = todo[:limit]
        log.info("Limiting this run to %d record(s) (--limit).", limit)
    if not todo:
        log.info("Nothing to do.")
        return

    n_llm_needed = sum(1 for r in todo if len(r["candidates"]) > _PASS_THROUGH_MAX)
    log.info("%d record(s) need LLM passes; %d pass through unchanged (<=%d candidates already)",
              n_llm_needed, len(todo) - n_llm_needed, _PASS_THROUGH_MAX)

    llm = load_llm(model_id, tensor_parallel_size, dtype, gpu_memory_utilization, max_model_len, max_num_seqs) \
        if n_llm_needed else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    with output_path.open("a", encoding="utf-8") as out_fh:
        for batch in _chunks(todo, batch_size):
            log.info("Batch: %d consilia (%d–%d of %d)", len(batch), n_done + 1, n_done + len(batch), len(todo))
            needs_llm = [rec for rec in batch if len(rec["candidates"]) > _PASS_THROUGH_MAX]
            try:
                selections: dict[tuple[str, str], list[str]] = {}
                if needs_llm:
                    tasks = [t for rec in needs_llm for t in build_tasks(rec, n_total)]
                    log.info("  %d selection request(s) in this batch", len(tasks))
                    results = run_tasks(llm, tasks)
                    selections = resolve_tasks(llm, tasks, results)
                batch_records = assemble_batch(batch, selections)
            except Exception:
                log.exception("Batch failed — skipping (rerun later to retry these ids).")
                continue
            for rec in batch_records:
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_fh.flush()
            n_done += len(batch)

    log.info("Done. Output: %s", output_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="GPU-side keyword reduction (vLLM) — stage 9b")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local path")
    ap.add_argument("--input", type=Path, default=Path("keyword_reduction_input.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("keyword_reduction_output.jsonl"))
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
