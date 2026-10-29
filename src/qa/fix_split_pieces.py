"""
Give the pieces that fix_layout.py creates their Gemini readings.

fix_layout.py splits the merged first line of a consilium into three TextLines in reading order:
  HL  the large-type opener ("In Christi nomine." or a large initial word such as "Super")
  L   the small-type rest of the same printed line (the truncated right part)
  H   the next printed line (the former merged line's remainder)
The texts of L and H are still the recognition model's, which fails exactly there. In the Gemini
transcription the same printed lines are two entries: G_k = HL + L, G_k+1 = H. This script finds G_k
for every triple (the line whose loose text is closest to HL + L, and which starts with the opener
when HL is one), splits it after the opener (or after the first word for a large initial word) and
writes the three readings, converted to the edition's conventions, with provenance in `comments`.
Guards: the match must be close (loose CER of HL+L against G_k below 0.45, or the opener prefix
matches and the remainders share more than half their characters); G_k+1 is taken for H only when its
length is within 0.7–1.4 of H's and it is not closer to another PageXML line; residual abbreviation
marks in the Gemini lines skip the triple; reviewed lines are never touched.

    python src/qa/fix_split_pieces.py --print Baldo_Cons_Print_Venice_1575_v1            # dry run
    python src/qa/fix_split_pieces.py --print Baldo_Cons_Print_Venice_1575_v1 --apply
Report: output/qa/split_pieces_<print>.md
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "qa"))
from compare_ocr import normalise, dist  # noqa: E402
from apply_ocr_fixes import convert_text, reading_index, ABBR_RE, MODEL  # noqa: E402

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
T = "{%s}" % NS
ET.register_namespace("", NS)
OUT = ROOT / "output" / "qa"
_OPENER_G = re.compile(r"^\s*(\{\d+\}\s*)?(\d+\s+)?(in\s+\S+\s+nomine\s*(amen)?\.?|in\s+nomine\s+domini\s*(amen)?\.?)", re.I)
_OPENER_X = re.compile(r"^\s*(\d+\s+)?in\s+\w+\s+nomine", re.I)


def text_of(tl):
    u = tl.find(f"{T}TextEquiv/{T}Unicode")
    return (u.text or "") if u is not None else ""


def set_text(tl, s):
    tl.find(f"{T}TextEquiv/{T}Unicode").text = s


def loose(s):
    return normalise(s, True)


def cer(a, b):
    return dist(a, b) / max(len(a), len(b), 1)


def process(print_dir: Path, ocr_dir: Path, apply: bool) -> list[dict]:
    today = datetime.date.today().isoformat()
    actions = []
    for xml in sorted((print_dir / "page").glob("*.xml")):
        tree = ET.parse(xml)
        page = tree.getroot().find(T + "Page")
        changed = False
        for tr in page.findall(T + "TextRegion"):
            m = re.search(r"type:(column_\d)", tr.get("custom", "") or "")
            if not m:
                continue
            txt = ocr_dir / f"{xml.stem}_c{m.group(1)[-1]}.txt"
            if not txt.exists():
                continue
            g = [l.rstrip() for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
            gl = [loose(x) for x in g]
            tls = sorted(tr.findall(T + "TextLine"), key=reading_index)
            ids = [t.get("id") for t in tls]
            for i, hl in enumerate(tls):
                lid = ids[i]
                if not lid.endswith("_left") or lid[:-5] not in ids:
                    continue
                hi = ids.index(lid[:-5])
                if hi != i + 2:
                    actions.append({"page": xml.name, "action": "skip", "reason": f"pieces not in the order HL, L, H ({i}, {hi})", "id": lid}); continue
                L, H = tls[i + 1], tls[i + 2]
                if any("reviewed " in (z.get("comments") or "") for z in (hl, L, H)):
                    continue
                hl_t, l_t, h_t = text_of(hl), text_of(L), text_of(H)
                combined = loose(hl_t + " " + l_t)
                opener = bool(_OPENER_X.match(hl_t))
                # candidate Gemini line
                best, best_c = None, 9.0
                for k, s in enumerate(gl):
                    if opener and not _OPENER_G.match(g[k]):
                        continue
                    c = cer(combined, s)
                    if c < best_c:
                        best, best_c = k, c
                if best is None:
                    actions.append({"page": xml.name, "action": "skip", "reason": "no Gemini candidate", "id": lid, "text": hl_t + " | " + l_t}); continue
                k = best
                gk = convert_text(g[k])
                if opener:
                    mo = _OPENER_G.match(gk)
                    if not mo:
                        actions.append({"page": xml.name, "action": "skip", "reason": "opener not found in the Gemini line", "id": lid, "text": gk}); continue
                    head, rest = gk[: mo.end()].strip(), gk[mo.end():].strip()
                    rest_ok = cer(loose(l_t), loose(rest)) < 0.6 if l_t.strip() else True
                    if best_c >= 0.45 and not rest_ok:
                        actions.append({"page": xml.name, "action": "skip", "reason": f"match too weak (CER {best_c:.2f})", "id": lid, "text": hl_t + " | " + l_t, "ocr": g[k]}); continue
                else:
                    if best_c >= 0.45:
                        actions.append({"page": xml.name, "action": "skip", "reason": f"match too weak (CER {best_c:.2f})", "id": lid, "text": hl_t + " | " + l_t, "ocr": g[k]}); continue
                    parts = gk.split(" ", 1)
                    head, rest = parts[0], (parts[1] if len(parts) > 1 else "")
                if ABBR_RE.search(g[k]) or (k + 1 < len(g) and ABBR_RE.search(g[k + 1])):
                    actions.append({"page": xml.name, "action": "skip", "reason": "unresolved abbreviation in the Gemini lines", "id": lid, "ocr": g[k]}); continue
                # next line for H
                h_new = None
                if k + 1 < len(g):
                    cand = convert_text(g[k + 1])
                    lh, lc = loose(h_t), loose(cand)
                    if 0.7 * len(lh) <= len(lc) <= 1.4 * len(lh) or not lh:
                        # make sure G_k+1 is not a better match for another PageXML line nearby
                        others = [cer(loose(text_of(z)), lc) for z in tls[i + 3:i + 5]]
                        if not others or cer(lh, lc) <= min(others) + 0.1:
                            h_new = cand
                if h_t.rstrip().endswith("¬") and h_new and not h_new.endswith("¬"):
                    h_new += "¬"
                if l_t.rstrip().endswith("¬") and rest and not rest.endswith("¬"):
                    rest += "¬"
                rec = {"page": xml.name, "action": "fixed", "id": lid, "cer": round(best_c, 2),
                       "hl": (hl_t, head), "l": (l_t, rest), "h": (h_t, h_new)}
                actions.append(rec)
                if apply:
                    for z, old, new in ((hl, hl_t, head), (L, l_t, rest), (H, h_t, h_new)):
                        if new is None or new == old:
                            continue
                        set_text(z, new)
                        z.set("comments", ((z.get("comments") or "") + f"; {MODEL} {today}: reading from the Gemini line by fix_split_pieces.py; was: {old[:120]}").strip("; "))
                    changed = True
        if changed:
            tree.write(xml, encoding="unicode", xml_declaration=True)
    return actions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--print", dest="print_id", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    print_dir = next(p for p in (ROOT / "data").glob(f"*/{args.print_id}"))
    acts = process(print_dir, ROOT / "output" / "ocr_gemini" / args.print_id, args.apply)
    fixed = [a for a in acts if a["action"] == "fixed"]
    skipped = [a for a in acts if a["action"] == "skip"]
    md = [f"# Split pieces from the Gemini line: {args.print_id}", "", f"{'Applied' if args.apply else 'Dry run'}: {len(fixed)} triples, {len(skipped)} skipped.", ""]
    for a in fixed:
        md.append(f"- {a['page']} `{a['id']}` (CER {a['cer']})  \n  opener: `{a['hl'][0]}` -> `{a['hl'][1]}`  \n  rest: `{a['l'][0]}` -> `{a['l'][1]}`  \n  next: `{a['h'][0]}` -> `{a['h'][1]}`")
    md += ["", "## Skipped", ""] + [f"- {a['page']} {a['id']}: {a['reason']} `{a.get('text', '')}` `{a.get('ocr', '')}`" for a in skipped]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"split_pieces_{args.print_id}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(md[2]); print(f"Report: {OUT / f'split_pieces_{args.print_id}.md'}")


if __name__ == "__main__":
    main()
