"""
Correct the PageXML from the Gemini comparison (src/qa/compare_ocr.py): the clear-cut cases only.

  insert   a line that exists in the Gemini transcription but not in the PageXML, when the geometry
           confirms it: the two neighbouring PageXML lines are at least 1.6 line pitches apart. A
           new TextLine is created between them (polygon and baseline interpolated from the
           neighbours, structure tag of the neighbours, text converted to the PageXML conventions).
  split    a PageXML line whose polygon is taller than 1.8 x the column's median line height and
           which the alignment matched to one Gemini line while a second Gemini line next to it is
           unmatched: the polygon is cut in two horizontally and both halves get the Gemini texts.
  replace  the reading of an opening line of a consilium (one of the first three text lines after a
           heading, or a line beginning with a casus formula such as "Casus talis est", "Factum tale
           est", "In Christi nomine", "Praemittendum", "Ad euidentiam") when it differs from the Gemini
           line by more than 30 % (loose CER): the recognition model fails most often next to the
           large initials and in the italic casus lines, Gemini reads them reliably. The old reading
           is kept in the comments attribute. Other differing readings stay in the review file.

Every new or changed line gets comments="gemini-3.8-flash <date>: inserted|split by apply_ocr_fixes.py",
the reading-order indices of the region are renumbered and the region's Unicode text is rebuilt.
Text conversion Gemini -> edition conventions (decided 2026-10-18): ſ -> s, & -> et, æ -> ae, œ -> oe,
expansion brackets removed ("co[n]sensu" -> "consensu"), {S} dropped, {n} -> "n " (marginal number as
the model writes it), a line-final hyphen -> ¬. ę is kept (not a ligature). Otherwise as read.

    python src/qa/apply_ocr_fixes.py --print Baldo_Cons_Print_Venice_1575_v1            # dry run, report only
    python src/qa/apply_ocr_fixes.py --print Baldo_Cons_Print_Venice_1575_v1 --apply    # write the PageXML
    python src/qa/apply_ocr_fixes.py --print … --pages 0123,0175 --apply
Report: output/qa/ocr_fixes_<print>.md. Rebuild afterwards (src/build.py --force --print …).
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import re
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "qa"))
from compare_ocr import normalise, dist  # noqa: E402

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
T = "{%s}" % NS
ET.register_namespace("", NS)
OUT = ROOT / "output" / "qa"
MODEL = "gemini-3.8-flash"


_LIG = str.maketrans({"ſ": "s", "æ": "ae", "Æ": "Ae", "œ": "oe", "Œ": "Oe"})
# punctuation spacing and the ét/etiam rule of the Gemini pipeline (src/gemini_ocr/pipeline.py)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("gemini_pipeline", ROOT / "src" / "gemini_ocr" / "pipeline.py")
_gp = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_gp)
normalize_line, ABBR_RE = _gp.normalize_line, _gp.ABBR_RE


def convert_text(t: str) -> str:
    """Gemini reading -> the conventions of the edition text (decided 2026-10-18): s for long s,
    ligatures resolved (& -> et, æ -> ae, œ -> oe), abbreviation expansions kept without brackets,
    {S}/{n} markers removed or turned into the marginal number prefix, line-final hyphen -> ¬."""
    t = html.unescape(t)                                           # Gemini sometimes writes HTML entities (&sect; &aelig; &#x00e8;)
    t = re.sub(r"^\s*\{S\}\s*", "", t)
    t = re.sub(r"^\s*\{(\d+)\}\s*", r"\1 ", t)
    t = t.translate(_LIG)
    t = re.sub(r"(?<=[A-Za-zſ])&(?=[A-Za-zſ])", "ae", t)          # "qu&dam": Gemini's & for æ inside a word
    t = re.sub(r"\[([^\]]*)\]", lambda m: "" if re.search(r"[A-ZÄÖÜß ]", m.group(1)) else m.group(1), t)   # drop annotations like "[Währung …]"
    t = re.sub(r"(?<!\S)&(?!\S)", "et", t)
    t = re.sub(r"&", "et", t)
    t = normalize_line(t)
    t = space_after_punctuation(t)
    t = re.sub(r"-\s*$", "¬", t)
    return t.strip()


def space_after_punctuation(t: str) -> str:
    """Gemini often glues punctuation to the next word ("ff.de ver.sig.l.nepos"); the print and the
    PyLaia text have a space there. Digits are left alone (l.1.C., 300.flor. stays as it is)."""
    return re.sub(r"([.,:;?!])(?=[A-Za-zſÀ-ɏ§¶†])", r"\1 ", t)


_CASUS = re.compile(r"^\s*(\{\d+\}\s*)?(\d+\s+)?(†\s*)?(Casus|Factum|Punctus|Thema|Quaestio|Qu[aæ]ritur|In Chri|In nomine|In Dei|Pr[aæ]mitt|Ad euidentiam|Ad euidentia)", re.I)
REPLACE_CER = 0.30
# the large-type opener is one printed line with the small type that follows it; the layout analysis
# often splits it off (or spans two lines with it), Gemini keeps it on the first line: not a reading error
_OPENER = re.compile(r"^(\d+\s+)?(in (christi|dei|domini) nomine( amen)?\.?\s*|in nomine domini( amen)?\.?\s*)", re.I)


def core_cer(xml_text: str, ocr_text: str) -> float:
    """loose CER after removing the opener formula and marginal numbers from both sides"""
    a = _OPENER.sub("", normalise(xml_text, False)).replace(" ", "")
    b = _OPENER.sub("", normalise(ocr_text, False)).replace(" ", "")
    a = re.sub(r"[^\w§¶†]", "", a); b = re.sub(r"[^\w§¶†]", "", b)
    return dist(a, b) / max(len(a), len(b), 1)


def pts(s: str) -> list[tuple[int, int]]:
    return [tuple(int(float(v)) for v in p.split(",")) for p in (s or "").split() if "," in p]


def pstr(points) -> str:
    return " ".join(f"{int(x)},{int(y)}" for x, y in points)


def line_info(tl) -> dict:
    c = tl.find(T + "Coords")
    lp = pts(c.get("points")) if c is not None else []
    bl = tl.find(T + "Baseline")
    bp = pts(bl.get("points")) if bl is not None else []
    y = statistics.median([p[1] for p in bp]) if bp else (statistics.median([p[1] for p in lp]) if lp else 0)
    u = tl.find(f"{T}TextEquiv/{T}Unicode")
    return {"el": tl, "id": tl.get("id"), "y": y, "poly": lp, "base": bp,
            "bbox": (min(x for x, _ in lp), min(py for _, py in lp), max(x for x, _ in lp), max(py for _, py in lp)) if lp else None,
            "text": (u.text or "") if u is not None else "", "custom": tl.get("custom", "")}


def structure_type(custom: str) -> str:
    m = re.search(r"structure\s*\{[^}]*type:([^;}]+)", custom or "")
    return m.group(1).strip() if m else ""


def reading_index(tl) -> float:
    m = re.search(r"readingOrder\s*\{[^}]*index:\s*(\d+)", tl.get("custom", "") or "")
    return float(m.group(1)) if m else 1e9


def set_reading_order(region) -> None:
    """Renumber the readingOrder indices and put the TextLine elements into that order.

    The order is the existing readingOrder index (fix_layout.py expresses its corrections through
    the indices, not through element positions), with new lines (no index yet, or inserted by
    make_line with a provisional index) placed where make_line put them. Renumbering by element
    order alone destroyed fix_layout's order in 8 regions on 2026-10-18."""
    lines = region.findall(T + "TextLine")
    others = [ch for ch in region if ch.tag != T + "TextLine"]
    keyed = []
    for pos, tl in enumerate(lines):
        idx = reading_index(tl)
        if tl.get("data-new") == "1":          # provisional: directly after the element before it
            prev = lines[pos - 1] if pos else None
            idx = (reading_index(prev) if prev is not None else -1) + 0.5
            del tl.attrib["data-new"]
        keyed.append((idx, pos, tl))
    keyed.sort(key=lambda k: (k[0], k[1]))
    for ch in list(region):
        region.remove(ch)
    for ch in others:
        region.append(ch)
    for i, (_, _, tl) in enumerate(keyed):
        custom = tl.get("custom", "")
        new = f"readingOrder {{index:{i};}}"
        tl.set("custom", re.sub(r"readingOrder\s*\{[^}]*\}", new, custom) if "readingOrder" in custom else (new + " " + custom).strip())
        region.append(tl)
    u = region.find(f"{T}TextEquiv/{T}Unicode")
    if u is not None and (u.text or "").strip():
        u.text = "\n".join((tl.find(f"{T}TextEquiv/{T}Unicode").text or "") for _, _, tl in keyed if tl.find(f"{T}TextEquiv/{T}Unicode") is not None)


def make_line(region, after_el, line_id: str, poly, base, text: str, stype: str, note: str):
    tl = ET.Element(T + "TextLine", {"id": line_id, "custom": f"readingOrder {{index:0;}} structure {{type:{stype};}}" if stype else "readingOrder {index:0;}",
                                     "comments": note, "data-new": "1"})
    ET.SubElement(tl, T + "Coords", {"points": pstr(poly)})
    if base:
        ET.SubElement(tl, T + "Baseline", {"points": pstr(base)})
    te = ET.SubElement(tl, T + "TextEquiv")
    ET.SubElement(te, T + "Unicode").text = text
    children = list(region)
    idx = children.index(after_el) + 1 if after_el is not None else len(children)
    region.insert(idx, tl)
    return tl


def shift(points, dy):
    return [(x, int(y + dy)) for x, y in points]


def process_page(xml: Path, col_results: list[dict], ocr_dir: Path, today: str) -> list[dict]:
    tree = ET.parse(xml)
    page = tree.getroot().find(T + "Page")
    actions = []
    for r in col_results:
        region = next((tr for tr in page.findall(T + "TextRegion") if tr.get("id") == r["region"]), None)
        if region is None:
            continue
        n = r["column"].split("_")[-1]
        txt = ocr_dir / f"{xml.stem}_c{n}.txt"
        if not txt.exists():
            continue
        ocr = [l.rstrip() for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
        lines = [line_info(tl) for tl in region.findall(T + "TextLine")]
        lines = [l for l in lines if l["bbox"]]
        if len(lines) < 5:
            continue
        by_y = sorted(lines, key=lambda l: l["y"])
        gaps = [b["y"] - a["y"] for a, b in zip(by_y, by_y[1:])]
        pitch = statistics.median(gaps)
        med_h = statistics.median([l["bbox"][3] - l["bbox"][1] for l in lines])
        used_ocr = set()

        # ---- inserts ---------------------------------------------------------------------
        for o in r["only_ocr"]:
            j = o["index"]
            if j >= len(ocr):
                continue
            prev_id = o.get("after_xml_id")
            prev = next((l for l in lines if l["id"] == prev_id or l["id"].split("+")[0] == prev_id), None)
            if prev is None:
                actions.append({"page": xml.name, "action": "skip_insert", "reason": "no preceding line", "ocr": ocr[j]})
                continue
            k = by_y.index(prev)
            nxt = by_y[k + 1] if k + 1 < len(by_y) else None
            gap = (nxt["y"] - prev["y"]) if nxt else None
            if nxt is None or gap < 1.6 * pitch:
                actions.append({"page": xml.name, "action": "skip_insert", "reason": f"no gap in the geometry (gap {round(gap) if gap else None}, pitch {round(pitch)})",
                                "ocr": ocr[j], "after": prev["id"]})
                continue
            # one or more missing lines fit into the gap; place this one at the first free pitch
            slots = round(gap / pitch) - 1
            dy = pitch
            new_y = prev["y"] + dy
            poly = shift(prev["poly"], dy) if prev["poly"] else [(prev["bbox"][0], int(new_y - 0.7 * med_h)), (prev["bbox"][2], int(new_y - 0.7 * med_h)),
                                                                  (prev["bbox"][2], int(new_y + 0.3 * med_h)), (prev["bbox"][0], int(new_y + 0.3 * med_h))]
            base = shift(prev["base"], dy) if prev["base"] else []
            stype = structure_type(prev["custom"]) or structure_type(nxt["custom"])
            text = convert_text(ocr[j])
            new_id = f"{prev['id']}_ins{j}"
            actions.append({"page": xml.name, "action": "insert", "region": r["region"], "after": prev["id"], "before": nxt["id"],
                            "gap_pitches": round(gap / pitch, 2), "slots": slots, "id": new_id, "type": stype, "text": text})
            make_line(region, prev["el"], new_id, poly, base, text, stype, f"{MODEL} {today}: inserted by apply_ocr_fixes.py (missing in the layout analysis)")
            used_ocr.add(j)

        # ---- splits ----------------------------------------------------------------------
        # a polygon that spans two printed lines (typically the large initial word with the two small-type
        # lines beside it): Gemini has two lines, the alignment matched one of them and left the other
        # unmatched nearby; the PageXML text contains the content of both
        unmatched = {o["index"] for o in r["only_ocr"]} - used_ocr
        for f in r["flagged"]:
            l = next((x for x in lines if x["id"] == f["xml_id"]), None)
            if l is None or f.get("ocr_index") is None or l["el"].get("comments"):
                continue
            if (l["bbox"][3] - l["bbox"][1]) <= 1.7 * med_h:
                continue
            j = f["ocr_index"]
            other = next((jj for d in (1, -1) for jj in [j + d] if jj in unmatched and 0 <= jj < len(ocr)), None)
            if other is None:
                actions.append({"page": xml.name, "action": "skip_split", "reason": "tall line but no unmatched Gemini line nearby", "id": l["id"], "text": l["text"][:60], "ocr": ocr[j][:60]})
                continue
            top_j, bot_j = sorted((j, other))
            both = normalise(ocr[top_j] + " " + ocr[bot_j], True)
            nx = normalise(l["text"], True)
            if dist(both, nx) / max(len(both), len(nx), 1) > 0.35 or len(nx) < 0.7 * len(both):
                actions.append({"page": xml.name, "action": "skip_split", "reason": "two Gemini lines do not match the tall line's content", "id": l["id"], "text": l["text"][:60], "ocr": (ocr[top_j] + " / " + ocr[bot_j])[:80]})
                continue
            if ABBR_RE.search(ocr[top_j]) or ABBR_RE.search(ocr[bot_j]):
                actions.append({"page": xml.name, "action": "skip_split", "reason": "unresolved abbreviation in the Gemini lines", "id": l["id"], "text": l["text"][:60], "ocr": (ocr[top_j] + " / " + ocr[bot_j])[:80]})
                continue
            x0, y0, x1, y1 = l["bbox"]
            ym = (y0 + y1) // 2
            stype = structure_type(l["custom"])
            note = f"{MODEL} {today}: split by apply_ocr_fixes.py (two printed lines in one polygon); was: {l['text'][:120]}"
            t_top, t_bot = convert_text(ocr[top_j]), convert_text(ocr[bot_j])
            c = l["el"].find(T + "Coords"); c.set("points", pstr([(x0, y0), (x1, y0), (x1, ym), (x0, ym)]))
            b = l["el"].find(T + "Baseline")
            if b is not None:
                b.set("points", pstr([(x0, ym - 6), (x1, ym - 6)]))
            l["el"].find(f"{T}TextEquiv/{T}Unicode").text = t_top
            l["el"].set("comments", note)
            make_line(region, l["el"], f"{l['id']}_b", [(x0, ym), (x1, ym), (x1, y1), (x0, y1)], [(x0, y1 - 6), (x1, y1 - 6)], t_bot, stype, note)
            unmatched.discard(other)
            actions.append({"page": xml.name, "action": "split", "region": r["region"], "id": l["id"], "old": l["text"][:80], "top": t_top, "bottom": t_bot})
        # ---- replacements: the opening lines of a consilium, where the recognition model fails most ----
        body_pos = {}          # line id -> position among the text lines after the last heading
        pos = None
        for l in by_y:
            st = structure_type(l["custom"])
            if st == "cons_title":
                pos = 0
            elif st == "cons_argument":
                continue
            elif pos is not None:
                pos += 1
                body_pos[l["id"]] = pos
        for f in r["flagged"]:
            if f["cer"] <= REPLACE_CER or f.get("ocr_index") is None:
                continue
            comps = f["xml_id"].split("+")          # compare_ocr may have merged two PageXML lines for the alignment
            targets = [next((x for x in lines if x["id"] == cid), None) for cid in comps]
            if any(x is None or x["el"].get("comments") for x in targets):
                continue
            j = f["ocr_index"]
            first = targets[0]
            opening = body_pos.get(first["id"], 99) <= 3 or _CASUS.match(ocr[j]) or _CASUS.match(first["text"])
            if not opening:
                continue
            if core_cer(f["xml"], ocr[j]) <= REPLACE_CER:
                continue            # only the opener formula sits on a different line; the reading agrees
            if any(_OPENER.match(normalise(x["text"], False)) for x in targets):
                # an opener line ("In Christi nomine.") is never replaced: on 2026-10-18 the pairing of two
                # merged pieces with two Gemini lines assigned the next line's text to 135 opener lines
                actions.append({"page": xml.name, "action": "skip_replace", "reason": "opener line, kept", "id": first["id"], "text": first["text"][:60], "ocr": ocr[j][:60]})
                continue
            split_texts = None
            if len(targets) == 2 and j + 1 in unmatched:
                pairs = [(targets[0], j), (targets[1], j + 1)]
            elif len(targets) == 1:
                pairs = [(targets[0], j)]
            elif len(targets) == 2 and False:
                # DISABLED 2026-10-18: distributing one Gemini line over two side-by-side PageXML pieces by
                # width ratio mis-assigned the opener formula to pieces of the next printed line in the dry
                # run (v1: 202 wrong candidates). These cases stay in the review file until the geometry
                # of the pieces is matched explicitly.
                left, right = sorted(targets, key=lambda z: z["bbox"][0])
                w0, w1 = left["bbox"][2] - left["bbox"][0], right["bbox"][2] - right["bbox"][0]
                g = convert_text(ocr[j])
                target_pos = int(len(g) * w0 / max(w0 + w1, 1))
                pos = g.rfind(" ", 0, target_pos + 1)
                if pos <= 0:
                    pos = g.find(" ", target_pos)
                if pos <= 0 or pos >= len(g) - 1:
                    actions.append({"page": xml.name, "action": "skip_replace", "reason": "cannot split the Gemini line for two side-by-side pieces", "id": f["xml_id"], "text": f["xml"][:60], "ocr": ocr[j][:60]})
                    continue
                split_texts = {left["id"]: g[:pos].strip(), right["id"]: g[pos:].strip()}
                pairs = [(left, j), (right, j)]
            else:
                actions.append({"page": xml.name, "action": "skip_replace", "reason": "merged PageXML lines without a second Gemini line", "id": f["xml_id"], "text": f["xml"][:60], "ocr": ocr[j][:60]})
                continue
            ok = True
            for l, jj in pairs:
                if split_texts:
                    nx, ng = normalise(f["xml"], True), normalise(ocr[jj], True)     # compare the whole printed line
                    continue_check = len(ng) < 0.85 * len(nx) or len(ng) > 1.3 * len(nx)
                    if continue_check:
                        ok = False
                        actions.append({"page": xml.name, "action": "skip_replace", "reason": f"length differs too much ({len(nx)} vs {len(ng)} characters)", "id": l["id"], "text": f["xml"][:60], "ocr": ocr[jj][:60]})
                    break
                nx, ng = normalise(l["text"], True), normalise(ocr[jj], True)
                if len(ng) < 0.85 * len(nx) or len(ng) > 1.3 * len(nx):
                    ok = False
                    actions.append({"page": xml.name, "action": "skip_replace", "reason": f"length differs too much ({len(nx)} vs {len(ng)} characters)", "id": l["id"], "text": l["text"][:60], "ocr": ocr[jj][:60]})
            if not ok:
                continue
            if any(ABBR_RE.search(ocr[jj]) for _, jj in pairs):
                actions.append({"page": xml.name, "action": "skip_replace", "reason": "unresolved abbreviation in the Gemini line (expansion pass pending)", "id": first["id"], "text": first["text"][:60], "ocr": ocr[j][:60]})
                continue
            for l, jj in pairs:
                new = split_texts[l["id"]] if split_texts else convert_text(ocr[jj])
                if l["text"].rstrip().endswith("¬") and not new.endswith("¬"):
                    new += "¬"                      # keep the hyphenation state the line joiner relies on
                if normalise(new, False) == normalise(l["text"], False) or new == l["text"]:
                    continue
                actions.append({"page": xml.name, "action": "replace", "region": r["region"], "id": l["id"], "cer": f["cer"], "old": l["text"], "new": new})
                l["el"].find(f"{T}TextEquiv/{T}Unicode").text = new
                l["el"].set("comments", f"{MODEL} {today}: reading replaced by apply_ocr_fixes.py (opening line, CER {f['cer']}); was: {l['text'][:120]}")
                unmatched.discard(jj)
        set_reading_order(region)
    return actions, tree


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--print", dest="print_id", required=True)
    ap.add_argument("--ocr", type=Path, help="Gemini texts (default output/ocr_gemini/<print>)")
    ap.add_argument("--pages", help="comma-separated page stems, e.g. 0123,0175")
    ap.add_argument("--apply", action="store_true", help="write the PageXML (default: report only)")
    args = ap.parse_args()
    ocr_dir = args.ocr or ROOT / "output" / "ocr_gemini" / args.print_id
    rep = json.loads((OUT / f"ocr_{args.print_id}.json").read_text(encoding="utf-8"))
    print_dir = next(p for p in (ROOT / "data").glob(f"*/{args.print_id}"))
    by_page: dict[str, list] = {}
    for c in rep["results"]:
        by_page.setdefault(c["page"], []).append(c)
    today = datetime.date.today().isoformat()
    sel = set(args.pages.split(",")) if args.pages else None
    all_actions = []
    for page, cols in sorted(by_page.items()):
        if sel and Path(page).stem not in sel:
            continue
        if not any(c["only_ocr"] or c["flagged"] for c in cols):
            continue
        xml = print_dir / "page" / page
        actions, tree = process_page(xml, cols, ocr_dir, today)
        all_actions += actions
        if args.apply and any(a["action"] in ("insert", "split", "replace") for a in actions):
            tree.write(xml, encoding="unicode", xml_declaration=True)
    n_ins = sum(a["action"] == "insert" for a in all_actions)
    n_split = sum(a["action"] == "split" for a in all_actions)
    n_rep = sum(a["action"] == "replace" for a in all_actions)
    skipped = [a for a in all_actions if a["action"].startswith("skip")]
    md = [f"# PageXML corrections from the Gemini comparison: {args.print_id}", "",
          f"{'Applied' if args.apply else 'Dry run'} {today}: {n_ins} lines inserted, {n_split} lines split, {n_rep} opening-line readings replaced, {len(skipped)} candidates skipped.", "", "## Changes", ""]
    for a in all_actions:
        if a["action"] == "insert":
            md.append(f"- {a['page']} insert after `{a['after']}` (gap {a['gap_pitches']} pitches): `{a['text']}`")
        elif a["action"] == "split":
            md.append(f"- {a['page']} split `{a['id']}`: `{a['top']}` / `{a['bottom']}`  (was: `{a['old']}`)")
        elif a["action"] == "replace":
            md.append(f"- {a['page']} replace `{a['id']}` (CER {a['cer']})  \n  was: `{a['old']}`  \n  now: `{a['new']}`")
    md += ["", "## Skipped (for manual review)", ""] + [f"- {a['page']} {a['action']}: {a.get('reason', '')}  \n  PageXML: `{a.get('text', '')}`  \n  Gemini: `{a.get('ocr', '')}`" for a in skipped]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"ocr_fixes_{args.print_id}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:3]))
    print(f"Report: {OUT / f'ocr_fixes_{args.print_id}.md'}")


if __name__ == "__main__":
    main()
