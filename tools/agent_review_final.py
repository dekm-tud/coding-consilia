#!/usr/bin/env python3
"""
Agent review of edition lines whose text still differs from Gemini's reading (round 2, after tools/final_text.py).

Since the final text was written (tools/apply_final_text.py) every TextLine carries TextEquiv index 0 = edition text,
1 = Gemini, 2 = PyLaia, 3 = kraken; tools/agent_review_batches.py assumed the older two-reading layout and would swap
the edition text into the Gemini slot, so this round has its own tool. Only index 0 (and the comments) is changed.

    python tools/agent_review_final.py make --min-cer 0.05 --per 20    # output/qa/agent_review2/batch_NNN.{jpg,json}
    python tools/agent_review_final.py make --word-pair quam quoniam    # targeted: edition "quam" where Gemini reads "quoniam"
    python tools/agent_review_final.py apply                           # batch_NNN.verdicts.json -> data/ PageXML

make: lines not yet decided by a human or reviewed, whose edition text and Gemini reading differ by >= --min-cer
(loose normalisation, as in the review tool). Per batch one montage (numbered crops: the line framed, a line of context
above and below) and a manifest with the distinct readings labelled A (edition text now), B, C, … (sources named).
An agent writes batch_NNN.verdicts.json: {"1": {"choice": "A"|"B"|…|"edit"|"unsure", "text": "…" (edit), "note": "…"}}.
apply: a letter other than A or an edit replaces the edition text; A confirms it; every decided line gets
"; reviewed <date>: agent2 <choice>" in its comments (so it is not shown again) and a record in
output/qa/review_decisions.jsonl. unsure leaves the line alone.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import sys
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "src" / "qa"))
from review_server import LOG, T, crop, utext  # noqa: E402
from compare_ocr import dist, normalise  # noqa: E402

OUT = ROOT / "output" / "qa" / "agent_review2"
SRC = {"0": "edition", "1": "gemini", "2": "pylaia", "3": "kraken"}


def collect(min_cer: float, word_pair: tuple[str, str] | None = None) -> list[dict]:
    items = []
    for f in sorted(glob.glob(str(ROOT / "data" / "Baldo_29618397" / "*" / "page" / "*.xml"))):
        vol = Path(f).parts[-3]
        page = ET.parse(f).getroot().find(T + "Page")
        for tr in page.findall(T + "TextRegion"):
            for tl in tr.findall(T + "TextLine"):
                cm = tl.get("comments") or ""
                if "reviewed " in cm or "human" in cm:
                    continue
                r = {SRC[i]: utext(tl, i) for i in SRC}
                if not r["gemini"].strip():
                    continue
                a, b = normalise(r["edition"], True), normalise(r["gemini"], True)
                cer = dist(a, b) / max(len(a), len(b), 1)
                if word_pair:                                   # targeted round: the edition has word 1 where Gemini has word 2
                    ew = [w.strip(".,:;").lower() for w in r["edition"].split()]
                    gw = [w.strip(".,:;").lower() for w in r["gemini"].split()]
                    if not (word_pair[0] in ew and word_pair[1] in gw and ew.count(word_pair[1]) < gw.count(word_pair[1])):
                        continue
                elif cer < min_cer:
                    continue
                items.append({"file": f, "vol": vol, "page": Path(f).name, "image": page.get("imageFilename"), "region": tr.get("id"),
                              "line_id": tl.get("id"), "cer": round(cer, 3), "readings": r})
    items.sort(key=lambda it: (-it["cer"], it["vol"], it["page"]))
    return items


def make(a) -> None:
    items = collect(a.min_cer, tuple(a.word_pair) if a.word_pair else None)
    OUT.mkdir(parents=True, exist_ok=True)
    n0 = len(list(OUT.glob("batch_*.json"))) - len(list(OUT.glob("batch_*.verdicts.json")))
    if any(not (OUT / p.name.replace(".json", ".verdicts.json")).exists() for p in OUT.glob("batch_[0-9][0-9][0-9].json")):
        sys.exit(f"{OUT} has batches without verdicts; finish them first")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    n = n0                                                  # new batches are numbered after the existing ones
    for b in range(0, len(items), a.per):
        tiles, manifest = [], {}
        for k, it in enumerate(items[b:b + a.per], 1):
            im = Image.open(BytesIO(crop(it))).convert("RGB")
            im.thumbnail((1400, 170))
            c = Image.new("RGB", (1460, im.height + 6), "white"); c.paste(im, (56, 3))
            ImageDraw.Draw(c).text((6, im.height // 2 - 12), f"{k:2d}", fill=(200, 0, 0), font=font)
            tiles.append(c)
            opts, labels = {}, {}
            for src in ("edition", "gemini", "pylaia", "kraken"):
                t = it["readings"][src].strip()
                if not t:
                    continue
                key = next((L for L, x in opts.items() if x == t), None)
                if key is None:
                    key = "ABCD"[len(opts)]; opts[key] = t; labels[key] = []
                labels[key].append(src)
            manifest[str(k)] = {"file": it["file"], "vol": it["vol"], "page": it["page"], "region": it["region"], "line_id": it["line_id"],
                                "cer": it["cer"], "options": opts, "sources": labels}
        H = sum(t.height + 4 for t in tiles); out = Image.new("RGB", (1460, H), "white"); y = 0
        for t in tiles:
            out.paste(t, (0, y)); y += t.height + 4
        n += 1
        out.save(OUT / f"batch_{n:03d}.jpg", quality=88)
        (OUT / f"batch_{n:03d}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(items)} lines -> {n} batches in {OUT}")


def apply(a) -> None:
    import collections
    today = datetime.date.today().isoformat(); counts = collections.Counter()
    for vf in sorted(OUT.glob("batch_*.verdicts.json")):
        mf = OUT / vf.name.replace(".verdicts.json", ".json")
        manifest = json.loads(mf.read_text(encoding="utf-8")); verdicts = json.loads(vf.read_text(encoding="utf-8"))
        for k, v in verdicts.items():
            it = manifest.get(str(k))
            if not it or it.get("applied"):
                continue
            choice = (v.get("choice") or "unsure").strip()
            if choice == "edit" and (v.get("text") or "").strip():
                new, how = v["text"].strip(), "edited"
            elif choice in it["options"]:
                new, how = it["options"][choice], "kept" if choice == "A" else "+".join(it["sources"][choice])
            else:
                counts["unsure"] += 1; continue
            tree = ET.parse(it["file"]); page = tree.getroot().find(T + "Page")
            region = next(tr for tr in page.findall(T + "TextRegion") if tr.get("id") == it["region"])
            tl = next(x for x in region.findall(T + "TextLine") if x.get("id") == it["line_id"])
            u0 = tl.find(f"{T}TextEquiv[@index='0']/{T}Unicode")
            if u0 is None or (u0.text or "").strip() != it["options"]["A"]:
                counts["changed meanwhile, skipped"] += 1; continue
            old = u0.text; u0.text = new
            tl.set("comments", ((tl.get("comments") or "") + f"; reviewed {today}: agent2 {how}").strip("; "))
            tree.write(it["file"], encoding="unicode", xml_declaration=True)
            with LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": datetime.datetime.now().isoformat(timespec="seconds"), "who": "agent2", "file": it["file"],
                                     "line_id": it["line_id"], "kind": "edition text vs readings", "action": how, "old": old, "text": new,
                                     "note": v.get("note") or ""}, ensure_ascii=False) + "\n")
            it["applied"] = how; counts["kept" if how == "kept" else "changed"] += 1
        mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print("applied:", dict(counts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["make", "apply"])
    ap.add_argument("--min-cer", type=float, default=0.05)
    ap.add_argument("--per", type=int, default=20)
    ap.add_argument("--word-pair", nargs=2, metavar=("EDITION", "GEMINI"), help="only lines where the edition has the first word and Gemini the second (e.g. quam quoniam)")
    a = ap.parse_args()
    make(a) if a.cmd == "make" else apply(a)


if __name__ == "__main__":
    main()
