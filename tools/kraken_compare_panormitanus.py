#!/usr/bin/env python3
"""
Compare the kraken reading of the Panormitanus print (Venice 1571) with its Gemini column transcription.

kraken (the Baldus segmentation and recognition models) wrote one PageXML per page to
output/kraken_panormitanus/pages/. The Gemini transcription has one file per column
(data/…/transkription_batch/<page>_c<n>.txt). The kraken lines of a page are assigned to the left or right
column by the centre of their baseline, running heads and lines holding only a marginal number are left out,
and the lines of each column are aligned with the Gemini lines (src/qa/compare_ocr.py, loose normalisation:
no spaces or punctuation, long s, ligatures, & = et, accents removed). Nothing in data/ is changed.

    python tools/kraken_compare_panormitanus.py [--out output/qa/kraken_panormitanus_compare.md]

Writes the report for the consilia pages, a JSON summary next to it and every line pair to <out>.jsonl.
"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import random
import re
import statistics
import unicodedata
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "qa"))
sys.path.insert(0, str(ROOT / "src"))
from compare_ocr import align, dist, normalise  # noqa: E402
from import_prebuilt import edition_form  # noqa: E402

PRINT = ROOT / "data" / "Panormitanus_32108367" / "Panormitanus_Cons_Print_Venice_1571_v1"
KRAKEN = ROOT / "output" / "kraken_panormitanus" / "pages"
NUMBER = re.compile(r"^\W*\d{1,3}\W*$")


def norm(s: str) -> str:
    """Loose normalisation after the spelling rules of the edition (edition_form: ligatures, accents, the
    character errors of the Gemini text), with the paragraph mark (¶, which kraken reads as †) unified."""
    s = unicodedata.normalize("NFC", s.replace("&#182;", "¶"))         # kraken writes ę decomposed
    return normalise(edition_form(s), True).replace("¶", "†")


def consilia_pages() -> set[str]:
    pre = json.loads((PRINT / "prebuilt.json").read_text(encoding="utf-8"))
    return {s["page"][:4] for r in pre["consilia"].values() for s in r.get("sources") or []}


def confusions(a: str, b: str) -> list[tuple[str, str]]:
    """Character substitutions, insertions and deletions turning the Gemini reading b into the kraken reading a."""
    sm = difflib.SequenceMatcher(None, b, a, autojunk=False)
    return [(b[i1:i2], a[j1:j2]) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]


def kraken_columns(path: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(line id, text) per column, top to bottom."""
    root = ET.parse(path).getroot()
    ns = root.tag.split("}")[0] + "}"
    lines = []
    for reg in root.iter(ns + "TextRegion"):
        if "header" in (reg.get("custom") or ""):
            continue
        for tl in reg.findall(ns + "TextLine"):
            u = tl.find(f"{ns}TextEquiv/{ns}Unicode")
            text = (u.text or "").strip() if u is not None else ""
            if not text or NUMBER.match(text):
                continue
            pts = [tuple(map(float, p.split(","))) for p in tl.find(ns + "Baseline").get("points").split()]
            xs = [p[0] for p in pts]
            lines.append(((min(xs) + max(xs)) / 2, statistics.mean(p[1] for p in pts), min(xs), max(xs), (tl.get("id"), text)))
    if not lines:
        return [], []
    gutter = (min(l[2] for l in lines) + max(l[3] for l in lines)) / 2
    left = sorted((l for l in lines if l[0] < gutter), key=lambda l: l[1])
    right = sorted((l for l in lines if l[0] >= gutter), key=lambda l: l[1])
    return [l[4] for l in left], [l[4] for l in right]


def gemini_column(page: str, col: int) -> list[str] | None:
    f = PRINT / "transkription_batch" / f"{page}_c{col}.txt"
    if not f.exists():
        return None
    return [l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def cer(a: str, b: str) -> float:
    return dist(a, b) / max(len(b), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "qa" / "kraken_panormitanus_compare.md")
    a = ap.parse_args()
    cols, pairs = [], []
    cons = consilia_pages()
    for xml in sorted(KRAKEN.glob("*.xml")):
        page = xml.stem
        if page not in cons:                 # repertorium and front matter are not part of the edition text
            continue
        kc = kraken_columns(xml)
        for c in (1, 2):
            g = gemini_column(page, c)
            if g is None:
                continue
            kid, k = [x[0] for x in kc[c - 1]], [x[1] for x in kc[c - 1]]
            gn, kn = [norm(x) for x in g], [norm(x) for x in k]
            gs, ks = "".join(gn), "".join(kn)
            cols.append({"page": page, "col": c, "part": "consilia" if page in cons else "other", "gemini_lines": len(g), "kraken_lines": len(k),
                         "gemini_chars": len(gs), "errors": dist(ks, gs), "cer": round(cer(ks, gs), 4)})
            for i, j in align(kn, gn):
                pairs.append({"page": page, "col": c, "part": "consilia" if page in cons else "other",
                              "line": kid[i] if i is not None else None, "kraken": k[i] if i is not None else None, "gemini": g[j] if j is not None else None,
                              "expanded": j is not None and "[" in g[j],
                              "cer": round(cer(kn[i], gn[j]), 4) if i is not None and j is not None else None})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    def stats(part: str) -> dict:
        cs = [c for c in cols if c["part"] == part or part == "all"]
        ps = [p for p in pairs if p["part"] == part or part == "all"]
        m = [p for p in ps if p["cer"] is not None]
        chars, errs = sum(c["gemini_chars"] for c in cs), sum(c["errors"] for c in cs)
        grp = collections.Counter("equal" if p["cer"] == 0 else "≤ 5 %" if p["cer"] <= .05 else "≤ 25 %" if p["cer"] <= .25
                                  else "> 25 %" for p in m)
        cers = sorted(c["cer"] for c in cs if c["gemini_chars"] > 200)
        return {"pages": len({c["page"] for c in cs}), "columns": len(cs), "gemini_chars": chars, "errors": errs,
                "cer": round(errs / max(chars, 1), 4), "column_cer_median": cers[len(cers) // 2] if cers else 0,
                "line_pairs": len(m), "line_groups": {k: grp[k] for k in ("equal", "≤ 5 %", "≤ 25 %", "> 25 %")},
                "cer_lines_without_expansions": round(sum(dist(norm(p["kraken"]), norm(p["gemini"])) for p in m if not p["expanded"])
                                                      / max(sum(len(norm(p["gemini"])) for p in m if not p["expanded"]), 1), 4),
                "lines_with_expansions": sum(p["expanded"] for p in m),
                "gemini_only_lines": sum(p["kraken"] is None for p in ps), "kraken_only_lines": sum(p["gemini"] is None for p in ps)}

    summary = {"consilia": stats("consilia")}
    conf = collections.Counter()
    for p in pairs:
        if p["part"] == "consilia" and p["cer"] is not None and 0 < p["cer"] <= .25 and not p["expanded"]:
            conf.update(confusions(norm(p["kraken"]), norm(p["gemini"])))
    summary["consilia"]["confusions"] = [[g, k, n] for (g, k), n in conf.most_common(40)]
    a.out.with_suffix(".json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")

    rnd = random.Random(1)
    out = ["# kraken and Gemini on Panormitanus, Venice 1571", "",
           "The comparison uses the loose normalisation (no spaces or punctuation, long s, ligatures, & = et, accents "
           "removed, ¶ = †). Only the pages with consilia (0003 to 0388) are compared. The brackets of the Gemini "
           "expansions are removed and their content kept (p[raese]nti → praesenti), as in the edition text. kraken "
           "follows the Baldus transcription (presenti, s. for supra), so the difference is also given for the line "
           "pairs whose Gemini text had no expansion.", "",
           "| | consilia pages |", "|---|---|"]
    S = summary["consilia"]
    rows = [("pages / columns", lambda x: f"{x['pages']} / {x['columns']}"),
            ("characters (Gemini)", lambda x: f"{x['gemini_chars']:,}"),
            ("character difference", lambda x: f"**{x['cer']:.2%}**"),
            ("difference in line pairs without Gemini expansions", lambda x: f"{x['cer_lines_without_expansions']:.2%}"),
            ("line pairs with Gemini expansions", lambda x: f"{x['lines_with_expansions']:,}"),
            ("median per column", lambda x: f"{x['column_cer_median']:.2%}"),
            ("aligned line pairs", lambda x: f"{x['line_pairs']:,}")]
    rows += [(f"line pairs {k}", lambda x, k=k: f"{x['line_groups'][k]:,} ({x['line_groups'][k] / max(x['line_pairs'], 1):.1%})")
             for k in ("equal", "≤ 5 %", "≤ 25 %", "> 25 %")]
    rows += [("lines only in Gemini", lambda x: f"{x['gemini_only_lines']:,}"),
             ("lines only in kraken", lambda x: f"{x['kraken_only_lines']:,}")]
    out += [f"| {name} | {f(S)} |" for name, f in rows]
    out += ["", "## Most frequent differences on the consilia pages", "",
            "Gemini → kraken, counted in line pairs with up to 25 % difference and without expansion brackets.", "", "| Gemini | kraken | n |", "|---|---|---|"]
    out += [f"| `{g or '∅'}` | `{k or '∅'}` | {n:,} |" for g, k, n in summary["consilia"]["confusions"][:30]]
    out += ["", "## Consilia columns with the largest difference", "", "| page | col | Gemini lines | kraken lines | CER |", "|---|---|---|---|---|"]
    for c in sorted((c for c in cols if c["part"] == "consilia" and c["gemini_chars"] > 200), key=lambda c: -c["cer"])[:20]:
        out.append(f"| {c['page']} | {c['col']} | {c['gemini_lines']} | {c['kraken_lines']} | {c['cer']:.1%} |")
    cp = [p for p in pairs if p["part"] == "consilia"]
    m = [p for p in cp if p["cer"] is not None]
    for title, sel in (("Samples of small differences (≤ 5 %)", [p for p in m if 0 < p["cer"] <= .05]),
                       ("Samples of larger differences (5 to 25 %)", [p for p in m if .05 < p["cer"] <= .25]),
                       ("Samples of line pairs over 25 %", [p for p in m if p["cer"] > .25]),
                       ("Samples of lines only in Gemini", [p for p in cp if p["kraken"] is None]),
                       ("Samples of lines only in kraken", [p for p in cp if p["gemini"] is None])):
        out += ["", f"## {title}", ""]
        for p in rnd.sample(sel, min(15, len(sel))):
            out.append(f"- {p['page']} c{p['col']}: G `{p['gemini']}`  \n  K `{p['kraken']}`")
    a.out.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(json.dumps({k: {x: v[x] for x in ("cer", "line_pairs", "line_groups", "gemini_only_lines", "kraken_only_lines")} for k, v in summary.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
