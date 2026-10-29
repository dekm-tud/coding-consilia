"""
Stage 7b of keyword refinement: one LLM call per keyword cluster (produced by
src/refine_keywords.py, stage 7a) to assign a canonical English label, and to
flag + split clusters that wrongly conflate distinct legal/general concepts.

Runs on the GPU server via vLLM's offline batch-inference API (see
vllm_common.py — keep it in the same directory when copying to the server):

    pip install vllm

Usage:
    python name_clusters.py --input keyword_clusters.json --output keyword_clusters_named.json
    python name_clusters.py --model Qwen/Qwen3.6-27B --limit 10   # smoke test

Output: one record per INPUT cluster id (a cluster that gets split produces
two output records sharing the same source_cluster_id):
    [{"source_cluster_id": 0, "label": "divorce",
      "members": ["divorce", "dissolution of marriage"], "frequency": 37,
      "split": false},
     …]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from vllm_common import DEFAULT_MODEL, chat_batch, load_llm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_SYSTEM = (
    "You are building a controlled keyword vocabulary for a corpus of medieval "
    "Latin legal opinions (consilia), covering both legal/juristic concepts "
    "(e.g. dowry, feudal succession, usufruct) and general life-topics (e.g. "
    "marriage, divorce, fertility, death). You are given a cluster of keyword "
    "phrases that an embedding model grouped as near-duplicates, with each "
    "phrase's frequency in the corpus.\n\n"
    "Respond with ONLY a JSON object, no commentary, no markdown fences, in "
    "exactly this shape:\n"
    '{"label": "<canonical English term for this cluster>", "split": false}\n'
    "or, if the phrases actually name two or more clearly DISTINCT concepts "
    "that should not share one label:\n"
    '{"split": true, "groups": [{"label": "<term A>", "members": [...]}, '
    '{"label": "<term B>", "members": [...]}]}\n'
    "Only split when concepts are genuinely distinct (not just differently "
    "phrased) — prefer NOT splitting when in doubt, since over-splitting "
    "defeats the purpose of building a compact vocabulary."
)

_RE_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)
_MAX_TOKENS = 500


def _format_cluster(cluster: dict) -> str:
    counts = cluster.get("member_counts") or {}
    lines = [f"- {m} (frequency: {counts.get(m, 1)})" for m in cluster["members"]]
    return "Keyword phrases:\n" + "\n".join(lines)


def _parse_response(raw: str) -> Optional[dict]:
    m = _RE_JSON_OBJ.search(raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _fallback_record(cluster: dict) -> dict:
    return {
        "source_cluster_id": cluster["cluster_id"],
        "label": cluster["members"][0],
        "members": cluster["members"],
        "frequency": cluster["frequency"],
        "split": False,
    }


def to_records(cluster: dict, parsed: dict, counts: dict[str, int]) -> list[dict]:
    if parsed.get("split") and parsed.get("groups"):
        return [
            {
                "source_cluster_id": cluster["cluster_id"],
                "label": g.get("label", g.get("members", ["?"])[0]),
                "members": g.get("members", []),
                "frequency": sum(counts.get(m, 0) for m in g.get("members", [])),
                "split": True,
            }
            for g in parsed["groups"]
        ]
    return [{
        "source_cluster_id": cluster["cluster_id"],
        "label": parsed.get("label", cluster["members"][0]),
        "members": cluster["members"],
        "frequency": cluster["frequency"],
        "split": False,
    }]


def run(model_id: str, input_path: Path, output_path: Path, tensor_parallel_size: int,
        dtype: str, gpu_memory_utilization: float, max_model_len: Optional[int],
        max_num_seqs: Optional[int], batch_size: int, limit: Optional[int]) -> None:
    clusters = json.loads(input_path.read_text(encoding="utf-8"))
    if not clusters:
        log.error("No clusters found in %s", input_path)
        sys.exit(1)

    results_so_far: list[dict] = []
    done_ids: set[int] = set()
    if output_path.exists():
        results_so_far = json.loads(output_path.read_text(encoding="utf-8"))
        done_ids = {rec["source_cluster_id"] for rec in results_so_far}

    todo = [c for c in clusters if c["cluster_id"] not in done_ids]
    log.info("%d total clusters, %d already named, %d remaining", len(clusters), len(done_ids), len(todo))
    if limit:
        todo = todo[:limit]
        log.info("Limiting this run to %d cluster(s) (--limit).", limit)
    if not todo:
        log.info("Nothing to do.")
        return

    llm = load_llm(model_id, tensor_parallel_size, dtype, gpu_memory_utilization, max_model_len, max_num_seqs)

    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        log.info("Batch: %d clusters (%d–%d of %d)", len(batch), i + 1, i + len(batch), len(todo))
        conversations = [[{"role": "system", "content": _SYSTEM},
                           {"role": "user", "content": _format_cluster(c)}] for c in batch]
        try:
            texts = chat_batch(llm, conversations, [_MAX_TOKENS] * len(batch))
        except Exception:
            log.exception("Batch failed — skipping (rerun later to retry these clusters).")
            continue

        for cluster, raw in zip(batch, texts):
            parsed = _parse_response(raw)
            if parsed is None:
                log.warning("Cluster %d: unparseable response, falling back to most frequent member.",
                             cluster["cluster_id"])
                results_so_far.append(_fallback_record(cluster))
                continue
            counts = cluster.get("member_counts") or {}
            results_so_far.extend(to_records(cluster, parsed, counts))

        output_path.write_text(json.dumps(results_so_far, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("Done. Output: %s (%d named record(s))", output_path, len(results_so_far))


def main() -> None:
    ap = argparse.ArgumentParser(description="GPU-side per-cluster canonical labeling (stage 7b, vLLM)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local path")
    ap.add_argument("--input", type=Path, default=Path("keyword_clusters.json"))
    ap.add_argument("--output", type=Path, default=Path("keyword_clusters_named.json"))
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--max-num-seqs", type=int, default=256,
                     help="Max concurrent sequences vLLM schedules — see qwen_process.py --help for why "
                          "this matters on hybrid Mamba/attention models. Pass 0 to use vLLM's default.")
    ap.add_argument("--batch-size", type=int, default=50, help="Clusters per vLLM batch call")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N remaining clusters (smoke test)")
    args = ap.parse_args()
    run(args.model, args.input, args.output, args.tensor_parallel_size, args.dtype,
        args.gpu_memory_utilization, args.max_model_len, args.max_num_seqs or None,
        args.batch_size, args.limit)


if __name__ == "__main__":
    main()
