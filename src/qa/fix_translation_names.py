"""
Mechanical correction of a translation-layer error (QA 2026-10-17): the LLM rendered Baldus'
signature "Ego Bal." / "Bal. de Perusio" as "I, Bartolus" (EN) / "Ich, Bartolus" (DE) in
dozens of Baldo consilia. Bartolus is of Saxoferrato, so "Bartolus of Perugia" is a sure sign.

Rule: in a Baldo consilium, replace "I, Bartolus" / "Ich, Bartolus" / "Bartolus of Perugia" /
"Bartolus von Perugia" with the Baldus form ONLY in sections whose Latin does not mention
Bartolus at all (no "Bartol" / "Bart." / "Bar." stem in the Latin section), so genuine
citations of Bartolus are untouched. Writes output/consilia.json, the per-author and per-print
files, and logs every change to output/qa/translation_name_fixes.json.

Usage:
    python src/qa/fix_translation_names.py           # apply
    python src/qa/fix_translation_names.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output"
LATIN_BARTOLUS = re.compile(r"\bBart(?:ol|\.)|\bBar\.", re.I)
RULES = {
    "en": [(re.compile(r"\bI, Bartolus\b"), "I, Baldus"),
           (re.compile(r"\bBartolus of Perugia\b"), "Baldus of Perugia"),
           (re.compile(r"\bBartolus de Perusio\b"), "Baldus de Perusio")],
    "de": [(re.compile(r"\bIch, Bartolus\b"), "Ich, Baldus"),
           (re.compile(r"\bBartolus von Perugia\b"), "Baldus von Perugia"),
           (re.compile(r"\bBartolus de Perusio\b"), "Baldus de Perusio")],
}


def fix_consilium(c: dict, log: list) -> bool:
    changed = False
    if not c.get("author_viaf", "").startswith("Baldo"):
        return False
    tr = c.get("translations") or {}
    for lang, rules in RULES.items():
        t = tr.get(lang) or {}
        body = t.get("body") or []
        for i, sec in enumerate(body):
            la = c["body"][i] if i < len(c["body"]) else ""
            if LATIN_BARTOLUS.search(la):
                continue
            new = sec
            for rx, repl in rules:
                new = rx.sub(repl, new)
            if new != sec:
                log.append({"id": c["id"], "lang": lang, "section": i,
                            "before": sec[:0] or re.findall(r".{0,40}Bartolus.{0,40}", sec)[:2]})
                body[i] = new
                changed = True
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    files = [OUTPUT / "consilia.json"]
    for a in OUTPUT.iterdir():
        if a.is_dir() and a.name not in ("debug", "qa"):
            files += [f for f in a.glob("*.json") if not f.name.endswith("_backup.json")]
    log: list = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        n = sum(fix_consilium(c, log if f.name == "consilia.json" and f.parent == OUTPUT else []) for c in data["consilia"].values())
        if n and not args.dry_run:
            f.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{f.relative_to(ROOT)}: {n} consilia changed")
    (OUTPUT / "qa").mkdir(exist_ok=True)
    (OUTPUT / "qa" / "translation_name_fixes.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(log)} sections {'would be ' if args.dry_run else ''}changed; log in output/qa/translation_name_fixes.json")


if __name__ == "__main__":
    main()
