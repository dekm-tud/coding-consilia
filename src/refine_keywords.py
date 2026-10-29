"""
Stage 7a of keyword refinement: cluster the raw per-consilium keywords
(produced by src/gpu/qwen_process.py, merged in via src/merge_llm_output.py)
into groups of near-duplicate phrasings, using the same BGE-M3 model already
used for semantic search (see src/embed_consilia.py).

This only deduplicates — it does not assign canonical labels. That happens in
src/gpu/name_clusters.py (one LLM call per cluster, run on the GPU server),
which reads this script's output.

Usage:
    python src/refine_keywords.py
    python src/refine_keywords.py --distance-threshold 0.2

Output: output/keyword_clusters.json
    [{"cluster_id": 0, "members": ["divorce", "dissolution of marriage"], "frequency": 37}, …]
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output"
MODEL_ID = "BAAI/bge-m3"


def collect_raw_keywords(consilia: dict) -> Counter:
    """Return a frequency count of every raw keyword string across the corpus.

    Case/whitespace-normalised for counting (so "Divorce" and "divorce" merge),
    but the most frequent original casing is kept as the representative form
    downstream (see main()).
    """
    counts: Counter = Counter()
    for c in consilia.values():
        raw = c.get("keywords_raw") or {}
        for kw in (raw.get("argument") or []) + (raw.get("body") or []):
            kw = kw.strip()
            if kw:
                counts[kw] += 1
    return counts


def _normalize(kw: str) -> str:
    return " ".join(kw.lower().split())


def dedupe_casing(counts: Counter) -> Counter:
    """Merge keywords that differ only in case/whitespace, keeping the most
    frequent original casing as the representative string."""
    groups: dict[str, Counter] = {}
    for kw, n in counts.items():
        groups.setdefault(_normalize(kw), Counter())[kw] += n
    merged: Counter = Counter()
    for norm_variants in groups.values():
        representative, _ = norm_variants.most_common(1)[0]
        merged[representative] = sum(norm_variants.values())
    return merged


def cluster_keywords(unique_keywords: list[str], distance_threshold: float, device: str | None) -> list[list[str]]:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering

    log.info("Loading %s …", MODEL_ID)
    model = SentenceTransformer(MODEL_ID, device=device)

    log.info("Embedding %d unique keyword phrases …", len(unique_keywords))
    embeddings = model.encode(unique_keywords, normalize_embeddings=True, show_progress_bar=True)

    log.info("Clustering (cosine distance_threshold=%.3f) …", distance_threshold)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    ).fit(embeddings)

    clusters: dict[int, list[str]] = {}
    for kw, label in zip(unique_keywords, clustering.labels_):
        clusters.setdefault(int(label), []).append(kw)
    return list(clusters.values())


def run(distance_threshold: float, device: str | None, out_path: Path) -> None:
    merged_path = OUTPUT_DIR / "consilia.json"
    consilia = json.loads(merged_path.read_text(encoding="utf-8"))["consilia"]

    raw_counts = collect_raw_keywords(consilia)
    if not raw_counts:
        log.error("No 'keywords_raw' found in %s — run src/gpu/qwen_process.py "
                   "and src/merge_llm_output.py first.", merged_path)
        return
    counts = dedupe_casing(raw_counts)
    log.info("%d raw keyword instances → %d unique phrases (case/whitespace-deduped)",
              sum(raw_counts.values()), len(counts))

    unique_keywords = list(counts.keys())
    member_groups = cluster_keywords(unique_keywords, distance_threshold, device)

    clusters = []
    for cid, members in enumerate(member_groups):
        members_sorted = sorted(members, key=lambda kw: -counts[kw])
        clusters.append({
            "cluster_id": cid,
            "members": members_sorted,
            "member_counts": {m: counts[m] for m in members_sorted},
            "frequency": sum(counts[m] for m in members),
        })
    clusters.sort(key=lambda c: -c["frequency"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(clusters, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote %d cluster(s) to %s", len(clusters), out_path)
    log.info("Largest cluster: %d members, frequency %d",
              len(clusters[0]["members"]), clusters[0]["frequency"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Cluster raw keywords with BGE-M3 (stage 7a)")
    ap.add_argument("--distance-threshold", type=float, default=0.25,
                     help="Cosine-distance merge threshold (lower = stricter, more clusters)")
    ap.add_argument("--device", default=None, help="Force device: 'cuda', 'cpu', etc.")
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "keyword_clusters.json")
    args = ap.parse_args()
    run(args.distance_threshold, args.device, args.out)


if __name__ == "__main__":
    main()
