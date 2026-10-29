#!/usr/bin/env python3
"""
Publish hf-export/ (the search page's vectors) to the Hugging Face dataset and tag the release.

Checks first that every model directory has chunks.f16.bin and chunks.json and that each chunks.json carries
the data revision of site-data/manifest.json; uploads only those two files per model and the README (no stale
int8 files). Creates the dataset (public, CC BY 4.0) if it does not exist. Run only on the editor's word:
uploading publishes.

    python tools/upload_embeddings_hf.py [--tag v0.2] [--repo dekmATtud/consilia-embeddings] [--dry-run] [--bump-models]

Then the HF URL goes first in site-data/models.json (done here with --bump-models).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "hf-export"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", default="v0.2")
    ap.add_argument("--repo", default="dekmATtud/consilia-embeddings")   # until v0.1 (23.10.2026): mschonhardt/consilia-embeddings
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--bump-models", action="store_true")
    a = ap.parse_args()
    rev = json.loads((ROOT / "site-data" / "manifest.json").read_text())["revision"]
    tags = [d.name for d in sorted(EXPORT.iterdir()) if d.is_dir()]
    for t in tags:
        cj = json.loads((EXPORT / t / "chunks.json").read_text())
        if cj.get("data_revision") != rev or not (EXPORT / t / "chunks.f16.bin").exists():
            raise SystemExit(f"{t}: data_revision {cj.get('data_revision')} != {rev} or no chunks.f16.bin; re-embed first")
    subprocess.run(["bash", "tools/upload_embeddings_hf.sh", a.tag], cwd=ROOT, check=True, capture_output=True)   # writes the README
    print(f"{len(tags)} models at revision {rev}: {', '.join(tags)}")
    if a.dry_run:
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.repo, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(repo_id=a.repo, repo_type="dataset", folder_path=str(EXPORT),
                      allow_patterns=["README.md", "*/chunks.f16.bin", "*/chunks.json"],
                      commit_message=f"{a.tag} (data revision {rev})")
    refs = api.list_repo_refs(a.repo, repo_type="dataset")
    if a.tag in [x.name for x in refs.tags]:
        api.delete_tag(a.repo, tag=a.tag, repo_type="dataset")
    api.create_tag(a.repo, tag=a.tag, repo_type="dataset", tag_message=f"data revision {rev}")
    print(f"uploaded to https://huggingface.co/datasets/{a.repo} and tagged {a.tag}")
    if a.bump_models:
        p = ROOT / "site-data" / "models.json"
        m = json.loads(p.read_text())
        for x in m["models"]:
            hf = f"https://huggingface.co/datasets/{a.repo}/resolve/{a.tag}/{x['tag']}/"
            x["embeddings"] = [hf] + [u for u in x.get("embeddings", []) if "huggingface.co" not in u]   # older HF releases out
        p.write_text(json.dumps(m, ensure_ascii=False, indent=1) + "\n")
        print("site-data/models.json: HF URLs first")


if __name__ == "__main__":
    main()
