"""
Consistency audit of the generated data: output/*.json, site-data/, hf-export/.

Writes output/qa/data_report.md (+ .json) and exits 1 when a HARD check fails, so the Pages
workflow can run it. Soft findings are listed for the editor.

HARD (break the site or the archive):
  - output/consilia.json == union of output/<author>/<print>.json (same ids, same text)
  - every consilium has a unique id, int n, str title, list argument/casus/body, list sources
  - site-data/index.json has the same ids as consilia.json; manifest.revision matches
  - every hf-export/<tag>/chunks.json data_revision equals site-data/manifest.json revision
  - image coverage: every source page has either an AVIF derivative (kind local) or the print is
    in iiif_manifests.json
SOFT (editorial):
  - numbering gaps/duplicates per print, empty body or argument, very short bodies (< 200 chars)
  - translations: body length mismatch, missing languages, "Bartolus" in a Baldus translation
  - keywords_final empty or > 15
  - sources without regions

Usage:
    python src/qa/check_data.py            # report + exit code
    python src/qa/check_data.py --quiet
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output"
SITE = ROOT / "site-data"
DATA = ROOT / "data"
QA = OUTPUT / "qa"


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    hard: list[str] = []
    soft: dict[str, list[str]] = defaultdict(list)

    consilia = load(OUTPUT / "consilia.json")["consilia"]
    # 1. per-print files are the source of consilia.json (written by src/build.py next to it)
    union: dict = {}
    for author_dir in sorted(OUTPUT.iterdir()):
        if not author_dir.is_dir() or not (author_dir / "consilia.json").exists():   # author folders only
            continue
        for f in sorted(author_dir.glob("*.json")):
            if f.name == "consilia.json" or f.name.endswith("_backup.json"):
                continue
            d = load(f)
            if isinstance(d, dict) and "consilia" in d:      # per-print edition files only
                union.update(d["consilia"])
    # the per-print files are local build copies (not in the repository): compared only where they exist
    if not union:
        pass
    elif set(union) != set(consilia):
        only_merged = sorted(set(consilia) - set(union))[:5]
        only_prints = sorted(set(union) - set(consilia))[:5]
        hard.append(f"consilia.json ids differ from per-print files (only merged: {only_merged}, "
                    f"only per-print: {only_prints})")
    else:
        diff = [cid for cid in consilia if consilia[cid].get("body") != union[cid].get("body")]
        if diff:
            hard.append(f"body text differs between consilia.json and per-print files for {len(diff)} ids, e.g. {diff[:3]}")

    # 2. schema per consilium
    by_vol: dict[str, list] = defaultdict(list)
    for cid, c in consilia.items():
        if c.get("id") != cid:
            hard.append(f"{cid}: id field mismatch")
        if not isinstance(c.get("n"), int):
            hard.append(f"{cid}: n is {type(c.get('n')).__name__}")
        if not isinstance(c.get("title"), str) or not c["title"].strip():
            hard.append(f"{cid}: title missing")
        for k in ("argument", "casus", "body", "sources"):
            if not isinstance(c.get(k), list):
                hard.append(f"{cid}: {k} is not a list")
        by_vol[c.get("volume", "?")].append(c)
        body_len = sum(len(s) for s in c.get("body") or [])
        if not c.get("body"):
            soft["empty_body"].append(cid)
        elif body_len < 200:
            soft["short_body"].append(f"{cid} ({body_len} chars)")
        if not c.get("argument") and not c.get("summary"):
            soft["empty_argument"].append(cid)
        tr = c.get("translations") or {}
        for lang in ("en", "de"):
            t = tr.get(lang)
            if not t:
                soft[f"no_translation_{lang}"].append(cid)
                continue
            tb = t.get("body") or []
            if tb and len(tb) != len(c.get("body") or []):
                soft[f"translation_misaligned_{lang}"].append(f"{cid} ({len(tb)} vs {len(c['body'])})")
            if lang == "en" and re.search(r"\bI, Bartolus\b|\bBartolus of Perugia\b", " ".join(tb)):
                soft["translation_bartolus"].append(cid)      # signature mistranslated; src/qa/fix_translation_names.py
        kw = c.get("keywords_final")
        if kw is not None and (not kw or len(kw) > 15):
            soft["keywords_count"].append(f"{cid} ({len(kw)})")
        for s in c.get("sources") or []:
            if not s.get("regions"):
                soft["source_without_regions"].append(f"{cid} {s.get('page')}")

    # 3. numbering per print (+ a CSV per print for the editor: every duplicate/gap with title and page)
    QA.mkdir(parents=True, exist_ok=True)
    for vol, lst in by_vol.items():
        ns_all = sorted(c["n"] for c in lst if isinstance(c.get("n"), int))
        cnt = Counter(ns_all)
        rows = ["id,n,roman,title,page,note"]
        prev = None
        for c in sorted(lst, key=lambda c: (c["n"], c["id"])):
            note = "duplicate" if cnt[c["n"]] > 1 else ("gap before" if prev is not None and c["n"] > prev + 1 else "")
            if note:
                page = (c.get("sources") or [{}])[0].get("page", "")
                rows.append(",".join([c["id"], str(c["n"]), c.get("roman", ""), '"' + c.get("title", "").replace('"', "'") + '"', page, note]))
            prev = c["n"]
        (QA / f"numbering_{vol}.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    for vol, lst in by_vol.items():
        ns = sorted(c["n"] for c in lst if isinstance(c.get("n"), int))
        dups = [n for n, k in Counter(ns).items() if k > 1]
        if dups:
            soft["numbering_duplicate"].append(f"{vol}: {dups[:10]}")
        gaps = [n for n in range(ns[0], ns[-1]) if n not in set(ns)] if ns else []
        if gaps:
            soft["numbering_gap"].append(f"{vol}: {gaps[:10]}{'…' if len(gaps) > 10 else ''}")

    # 4. site-data
    if (SITE / "index.json").exists():
        index = load(SITE / "index.json")
        if [e["id"] for e in index] != list(consilia):
            hard.append("site-data/index.json ids/order differ from output/consilia.json (run src/build_site_data.py)")
        manifest = load(SITE / "manifest.json")
        if manifest.get("n_consilia") != len(consilia):
            hard.append("site-data/manifest.json n_consilia is stale")
        rev = manifest.get("revision")
        for cj in (ROOT / "hf-export").glob("*/chunks.json"):
            dr = load(cj).get("data_revision")
            if dr != rev:
                hard.append(f"{cj.relative_to(ROOT)}: data_revision {dr} != site-data revision {rev} "
                            f"(re-run src/gpu/embed_web_models.py --tags {cj.parent.name})")
    else:
        hard.append("site-data/index.json missing")

    # 5. images
    iiif = load(SITE / "iiif_manifests.json") if (SITE / "iiif_manifests.json").exists() else {}
    srcs = load(SITE / "image_sources.json") if (SITE / "image_sources.json").exists() else {}
    missing_img: Counter = Counter()
    for cid, c in consilia.items():
        vol = c["volume"]
        if vol in iiif:
            continue
        entry = srcs.get(vol) or {}
        tpl = entry.get("path_template", "data/{author}/{print}/img/{stem}.avif")
        if tpl.startswith("http"):
            continue
        for s in c.get("sources") or []:
            stem = s["page"].replace(".xml", "")
            p = ROOT / tpl.replace("{author}", c["author_viaf"]).replace("{print}", vol).replace("{stem}", stem)
            if not p.exists():
                missing_img[vol] += 1
    for vol, n in missing_img.items():
        hard.append(f"{vol}: {n} source pages without image derivative and no IIIF entry")

    # report
    QA.mkdir(parents=True, exist_ok=True)
    lines = [f"# Data report ({len(consilia)} consilia, {len(by_vol)} prints)", ""]
    lines.append("## Hard failures" if hard else "## Hard failures: none")
    lines += [f"- {h}" for h in hard]
    lines += ["", "## Soft findings (for the editor)"]
    for k, v in sorted(soft.items()):
        lines.append(f"- **{k}**: {len(v)}" + (f" — e.g. {', '.join(v[:6])}" if v else ""))
    (QA / "data_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (QA / "data_report.json").write_text(json.dumps({"hard": hard, "soft": soft}, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    if not args.quiet:
        print("\n".join(lines))
    print(f"\n-> {QA / 'data_report.md'}")
    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    main()
