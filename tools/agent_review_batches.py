#!/usr/bin/env python3
"""
Batches for the agent review of disputed lines, and the write-back of the agents' verdicts.

    python tools/agent_review_batches.py make --min-cer 0.10 --per 20     # montages + manifests in output/qa/agent_review/
    python tools/agent_review_batches.py apply                             # verdicts_*.json -> PageXML + review_decisions.jsonl

make: takes the items the review tool shows (tools/review_server.collect), keeps those with difference
>= --min-cer, and writes for every batch of --per lines one montage image batch_NNN.jpg (numbered crops,
column width, the line framed) and batch_NNN.json (the two readings per number). An agent looks at the
image and writes batch_NNN.verdicts.json: {"1": {"choice": "gemini"|"pylaia"|"edit"|"unsure", "text": "…"
(for edit), "note": "…"}, …}.
apply: for every verdict file, choice gemini/pylaia swaps the readings if needed and marks the line
"reviewed <date>: agent <choice>"; edit writes the text; unsure leaves the line for the human tool.
"""

from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from review_server import collect, crop, apply_decision  # noqa: E402
sys.path.insert(0, str(ROOT / "src" / "qa"))
from compare_ocr import normalise, dist  # noqa: E402
import re
_PREFIX = re.compile(r"^(\d+)?(in\w*nomine(amen)?|casustalisest|factumtaleest|punctustalisest)")

OUT = ROOT / "output" / "qa" / "agent_review"


def make(args):
    items = [it for it in collect(None) if it["cer"] >= args.min_cer and it["kind"] == "readings differ"]
    if args.only_unknown:   # keep the lines the lexicon vote (tools/lexicon_pick.py) could not settle: unknown words on at least one side
        from lexicon_pick import vocabulary, decide
        voc = vocabulary()
        items = [it for it in items if any(decide(it, voc, 3)[1:])]
    OUT.mkdir(parents=True, exist_ok=True)
    for p in OUT.glob("batch_*"):
        if not p.name.endswith(".verdicts.json"):
            p.unlink()
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    n = 0
    for b in range(0, len(items), args.per):
        batch = items[b:b + args.per]
        tiles, manifest = [], {}
        for k, it in enumerate(batch, 1):
            im = Image.open(BytesIO(crop(it))).convert("RGB")
            # keep the framed line and a little context: middle 55 % of the three-line crop
            h = im.height; im = im.crop((0, int(h * 0.2), im.width, int(h * 0.8)))
            im.thumbnail((1400, 110))
            c = Image.new("RGB", (1460, im.height + 6), "white"); c.paste(im, (56, 3))
            ImageDraw.Draw(c).text((6, im.height // 2 - 12), f"{k:2d}", fill=(200, 0, 0), font=font)
            tiles.append(c)
            manifest[str(k)] = {"file": it["file"], "line_id": it["line_id"], "region": it["region"], "vol": it["vol"], "page": it["page"],
                                "gemini": it["gemini"], "pylaia": it["pylaia"], "valid": it["valid"], "cer": it["cer"]}
        H = sum(t.height + 4 for t in tiles); out = Image.new("RGB", (1460, H), "white"); y = 0
        for t in tiles:
            out.paste(t, (0, y)); y += t.height + 4
        n += 1
        out.save(OUT / f"batch_{n:03d}.jpg", quality=85)
        (OUT / f"batch_{n:03d}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(items)} lines -> {n} batches in {OUT}")


def apply(args):
    import collections
    counts = collections.Counter()
    for vf in sorted(OUT.glob("batch_*.verdicts.json")):
        mf = OUT / vf.name.replace(".verdicts.json", ".json")
        if not mf.exists():
            continue
        manifest = json.loads(mf.read_text(encoding="utf-8")); verdicts = json.loads(vf.read_text(encoding="utf-8"))
        for k, v in verdicts.items():
            it = manifest.get(str(k))
            if not it or it.get("applied") not in (None, "unsure"):
                continue
            choice = (v.get("choice") or "unsure").lower()
            item = {"file": it["file"], "region": it["region"], "line_id": it["line_id"], "kind": "readings differ", "valid": it["valid"], "gemini": it["gemini"], "pylaia": it["pylaia"]}
            # guards against verdicts that would duplicate the opener formula (Gemini carries "In Christi nomine."
            # etc. as a line of its own): a choice that only moves the formula between the readings is left open,
            # an edit that only prepends a chunk to one reading falls back to that reading
            g_l, p_l = normalise(it["gemini"], True), normalise(it["pylaia"], True)
            def strip_prefix(x):
                return _PREFIX.sub("", x)
            def only_prefix(longer, shorter):
                return len(shorter) >= 6 and longer != shorter and longer.endswith(shorter) and len(longer) - len(shorter) >= 6
            if choice in ("gemini", "pylaia") and strip_prefix(g_l) == strip_prefix(p_l) and g_l != p_l:
                choice = "unsure"; v["note"] = (v.get("note") or "") + " [not applied: readings differ only by the opener formula]"
            if choice == "edit" and (v.get("text") or "").strip():
                e_l = normalise(v["text"], True)
                if only_prefix(e_l, g_l):
                    choice = "gemini"; v["note"] = (v.get("note") or "") + " [edit only prepended a chunk to Gemini]"
                elif only_prefix(e_l, p_l):
                    choice = "pylaia"; v["note"] = (v.get("note") or "") + " [edit only prepended a chunk to PyLaia]"
            if choice in ("gemini", "pylaia"):
                result = apply_decision(item, choice, "", who="agent")
            elif choice == "edit" and (v.get("text") or "").strip():
                result = apply_decision(item, "save", v["text"].strip(), who="agent")
            else:
                result = "unsure"
            counts[result] += 1
            it["applied"] = result; it["applied_note"] = v.get("note") or ""
        mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print("applied:", dict(counts))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["make", "apply"])
    ap.add_argument("--min-cer", type=float, default=0.10)
    ap.add_argument("--per", type=int, default=20)
    ap.add_argument("--only-unknown", action="store_true")
    a = ap.parse_args()
    {"make": make, "apply": apply}[a.cmd](a)


if __name__ == "__main__":
    main()
