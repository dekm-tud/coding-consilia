"""
Rebuild the line structure of every column from the Gemini transcription, keeping the Transkribus
geometry and, where the two readings agree, the Transkribus text.

Why: the Transkribus layout breaks down at the openings of the consilia (large initial, "CASVS."
label, truncated right parts, two printed lines in one polygon) and the recognition model fails
there too; the Gemini column transcription (src/gemini_ocr, prompt_baldus.txt) gives one entry per
printed line in reading order with markers for summarium lines ({S}) and marginal numbers ({n}).

Per column (TextRegion of type column_*):
  1. Align the Gemini lines (G) with the PageXML lines (X) on loosely normalised text (banded
     edit-distance alignment as in compare_ocr.py).
  2. The output line sequence is G. Unmatched X lines whose text is contained in a neighbouring G
     line are attached to it as further geometry (the opener pieces); other unmatched X lines are
     kept as lines of their own in y order (a line Gemini missed). Unmatched G lines shorter than
     20 characters following a short line are fragments beside an initial and are joined to it;
     other unmatched G lines become new lines with interpolated geometry (a line Transkribus missed).
  3. Text: the X reading stays when it agrees with G (loose CER <= 0.10) and the line is a single
     piece; otherwise the Gemini reading (edition conventions: s, et, ae/oe, expansions without
     brackets, marginal number as prefix, hyphen as ¬). Both readings are stored as TextEquiv
     index 0 (valid) and 1 (the other source) with @conf 1.0 / 0.5; the parser reads index 0.
     Gemini lines with residual abbreviation marks keep the X reading.
  4. Structure: CONSILIVM heading -> cons_title, {S} -> cons_argument, else cons_body (the X tag is
     kept when it is one of these and no marker contradicts it).
  5. Geometry: union of the attached X polygons; baseline of the main piece; interpolated from the
     neighbours (full column width) for lines without X geometry (comments say so).
Every TextLine gets comments="relayout <date>: text=pylaia|gemini, geometry=pagexml|union|interpolated".
Columns without a Gemini file and regions that are not columns are left untouched.

    python src/qa/relayout_from_gemini.py --print Baldo_Cons_Print_Venice_1575_v1 --out scratch/relayout
    python src/qa/relayout_from_gemini.py --print … --apply        # writes data/<author>/<print>/page/
Report: output/qa/relayout_<print>.md (counts per decision, pages with unusual outcomes).
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import shutil
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "qa"))
sys.path.insert(0, str(ROOT / "src"))
from compare_ocr import normalise, dist, align  # noqa: E402
from apply_ocr_fixes import convert_text, ABBR_RE  # noqa: E402
from rapidfuzz.fuzz import partial_ratio  # noqa: E402

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
T = "{%s}" % NS
ET.register_namespace("", NS)
OUT = ROOT / "output" / "qa"
AGREE = 0.10
# a printed heading: upper-case CONSILIVM (with the usual misreadings) followed by a roman numeral or an
# ordinal word; case-sensitive on purpose, body lines often begin with "consilium"
_HEAD = re.compile(r"^\s*(\{\d+\}\s*)?(?:Super eodem\.\s*)?(CO(?:NS|S)S?ILI[VU]M|CONSILIV|CONSILLVM|CONSILIM)\.?\s+(?:[IVXLCDM]{1,12}\.?|PRIMVM|SECVNDVM|TERTIVM|QVARTVM|QVINTVM)\s*\.?\s*$")
_NUMONLY = re.compile(r"^\s*\d{1,2}\s*$")
_FOOT = re.compile(r"(Prima|Secunda|Tertia|Quarta|Quinta)\s+Pars\s+Consiliorum|^[A-Z]{1,2}\s?\d?\s*$")
_RO = re.compile(r"readingOrder\s*\{[^}]*index:\s*(\d+)")
_TYPE = re.compile(r"structure\s*\{[^}]*type:([^;}]+)")


def pts(s):
    return [tuple(int(float(v)) for v in p.split(",")) for p in (s or "").split() if "," in p]


def pstr(points):
    return " ".join(f"{int(x)},{int(y)}" for x, y in points)


def read_lines(region):
    out = []
    for tl in region.findall(T + "TextLine"):
        c = tl.find(T + "Coords"); lp = pts(c.get("points")) if c is not None else []
        if not lp:
            continue
        b = tl.find(T + "Baseline"); bp = pts(b.get("points")) if b is not None else []
        u = tl.find(f"{T}TextEquiv/{T}Unicode")
        m = _RO.search(tl.get("custom", "") or ""); mt = _TYPE.search(tl.get("custom", "") or "")
        out.append({"el": tl, "id": tl.get("id"), "text": ((u.text or "") if u is not None else "").strip(),
                    "poly": lp, "base": bp, "bbox": (min(x for x, _ in lp), min(y for _, y in lp), max(x for x, _ in lp), max(y for _, y in lp)),
                    "y": statistics.median([y for _, y in bp]) if bp else statistics.median([y for _, y in lp]),
                    "ro": int(m.group(1)) if m else 10 ** 6, "type": (mt.group(1).strip() if mt else "")})
    out.sort(key=lambda l: (l["ro"], l["y"]))
    return out


_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


_MOJIBAKE = str.maketrans({"ű": "ſ", "Ŵ": "ſ", "ų": "ſ", "Ű": "ſ", "ŷ": "ſ", "Ŷ": "ſ"})     # some Gemini column files carry these for the long s


_HEAD_INNER = re.compile(r"^(?P<pre>.{2,40}?[.:])\s+(?P<head>(?:CO(?:NS|S)S?ILI[VU]M|CONSILIV|CONSILLVM|CONSILIM)\.?\s+(?:[IVXLCDM]{1,12}\.?|PRIMVM|SECVNDVM|TERTIVM|QVARTVM|QVINTVM)\s*\.?)\s*$")
_NUMERAL = re.compile(r"(?:CO(?:NS|S)S?ILI[VU]M|CONSILIV|CONSILLVM|CONSILIM)\.?\s+([IVXLCDM]{1,12}|PRIMVM|SECVNDVM|TERTIVM|QVARTVM|QVINTVM)")
_ORD = {"PRIMVM": 1, "SECVNDVM": 2, "TERTIVM": 3, "QVARTVM": 4, "QVINTVM": 5}


def heading_number(text: str) -> int | None:
    m = _NUMERAL.search(text or "")
    if not m:
        return None
    tok = m.group(1).rstrip(".")
    if tok in _ORD:
        return _ORD[tok]
    try:
        from segment_consilia import roman_to_int
        return roman_to_int(tok) or None
    except Exception:
        return None


def split_inline_headings(lines: list[str]) -> list[str]:
    """"Ad idem. CONSILIVM CV." (a caption on the heading's line) -> two entries, so that the heading
    starts its own line as the segmenter expects."""
    out = []
    for s in lines:
        m = _HEAD_INNER.match(convert_text(s))
        if m and not _HEAD.match(convert_text(s)):
            out.append(m.group("pre")); out.append(m.group("head"))
        else:
            out.append(s)
    return out


def gemini_info(raw: str) -> dict:
    s = _CTRL.sub("", raw).translate(_MOJIBAKE).strip()
    is_sum = s.startswith("{S}")
    conv = convert_text(s)
    return {"raw": s, "conv": conv, "loose": normalise(s, True), "sum": is_sum, "head": bool(_HEAD.match(conv)), "abbr": bool(ABBR_RE.search(s))}


def pick_type(g: dict | None, x_type: str) -> str:
    if g and g["head"]:
        return "cons_title"
    if g and g["sum"]:
        return "cons_argument"
    if x_type in ("cons_title", "cons_argument", "cons_body"):
        return x_type            # Gemini's {S} marker adds argument lines, a missing marker never removes one
    return "cons_body"


def relayout_region(region, g_raw: list[str], stats: Counter, today: str, page_name: str, notes: list[str], state: dict) -> None:
    X = read_lines(region)
    G = [gemini_info(s) for s in split_inline_headings([s for s in g_raw if s.strip()])]
    if not G or not X:
        return
    rb = pts(region.find(T + "Coords").get("points"))
    rx0, rx1 = min(x for x, _ in rb), max(x for x, _ in rb)
    xl = [normalise(l["text"], True) for l in X]
    pairs = align(xl, [g["loose"] for g in G])
    # a pair whose readings differ by more than 60 % is not the same printed line (the alignment prefers a
    # bad match over two gaps); break it, the pieces are re-attached by text containment below
    broken = 0
    fixed_pairs = []
    for i, j in pairs:
        if i is not None and j is not None:
            c = dist(xl[i], G[j]["loose"]) / max(len(xl[i]), len(G[j]["loose"]), 1)
            if c > 0.6 and (len(xl[i]) >= 8 or len(G[j]["loose"]) >= 8):
                fixed_pairs.append((i, None)); fixed_pairs.append((None, j))
                if len(xl[i]) >= 20 and len(G[j]["loose"]) >= 20:
                    broken += 1
                continue
        fixed_pairs.append((i, j))
    pairs = fixed_pairs
    g_of_x = {i: j for i, j in pairs if i is not None and j is not None}
    # reliability guard: the Gemini column must cover the same text (page 0193 c2 of v1 has 141 Gemini
    # lines for 74 printed ones: the model wandered into the neighbouring column); then keep X as is
    long_x = [i for i in range(len(X)) if len(xl[i]) >= 20]
    matched_long = sum(1 for i in long_x if i in g_of_x)
    if len(G) > 1.25 * len(X) + 3 or len(G) < 0.75 * len(X) - 3 or (long_x and matched_long < 0.85 * len(long_x)) or broken >= 3:
        stats["column_gemini_unreliable"] += 1
        notes.append(f"{page_name} {region.get('id')}: Gemini unreliable ({len(X)} lines, {len(G)} Gemini lines, {matched_long}/{len(long_x)} long lines matched, {broken} broken pairs), kept as is")
        return
    x_of_g = {j: [i] for i, j in pairs if i is not None and j is not None}
    # step 2a: unmatched X pieces -> attach to a neighbouring G line that contains their text
    unmatched_x = [i for i, j in pairs if j is None]
    kept_x_only = []
    for i in unmatched_x:
        t = xl[i]
        if not t:
            continue
        # neighbours in G: the G lines of the nearest matched X lines above and below
        # neighbours by position on the page (the reading-order index of a stray piece is unreliable)
        y = X[i]["y"]
        ga = [g_of_x[k] for k in g_of_x if X[k]["y"] <= y + 5]
        gb = [g_of_x[k] for k in g_of_x if X[k]["y"] >= y - 5]
        above = max(ga) if ga else -1
        below = min(gb) if gb else len(G)
        lo, hi = min(above, below), max(above, below)
        cands = [j for j in range(max(0, lo - 2), min(len(G), hi + 3))]
        hit = None
        if len(t) >= 3:
            # 1. the piece lies inside one Gemini line
            for j in cands:
                if t in G[j]["loose"]:
                    hit = j; break
            # 2. it starts one (a stray left part) or is a close variant of its beginning
            if hit is None:
                for j in cands:
                    gl_ = G[j]["loose"]
                    if dist(t, gl_[: len(t) + 2]) <= max(1, len(t) // 4) or (len(t) >= 8 and partial_ratio(t, gl_) >= 85):
                        hit = j; break
            # 3. it spans the end of one line and the start of the next: attach where most of it lies
            if hit is None:
                for j in cands:
                    if j + 1 < len(G) and t in G[j]["loose"] + G[j + 1]["loose"]:
                        pos = (G[j]["loose"] + G[j + 1]["loose"]).find(t)
                        hit = j if pos + len(t) / 2 < len(G[j]["loose"]) else j + 1; break
        if hit is None and len(t) < 12 and not _HEAD.match(X[i]["text"]):
            stats["x_fragment_dropped"] += 1; continue   # a stray piece; the Gemini line sequence already covers the text
        if hit is None and _NUMONLY.match(X[i]["text"]):
            hit = cands[0] if cands else None          # a bare marginal number: give its geometry to the next line
        if hit is None and _FOOT.search(X[i]["text"]) and i >= len(X) - 3:
            stats["x_footer_dropped"] += 1; continue   # signature line or catchword that Gemini was told to omit
        if hit is not None:
            x_of_g.setdefault(hit, []).append(i); stats["x_piece_attached"] += 1
        else:
            kept_x_only.append(i); stats["x_only_kept"] += 1
    # step 2b: unmatched G lines: first pair them with a leftover X line of the same text (a pair the
    # alignment had broken), otherwise treat short ones as fragments
    seq = []          # output sequence: dicts with g index or x index
    skip_g = set()
    for j, g in enumerate(G):
        if j in x_of_g or j in skip_g or not g["loose"]:
            continue
        best, best_c = None, 1.0
        for i in kept_x_only:
            c = dist(xl[i], g["loose"]) / max(len(xl[i]), len(g["loose"]), 1)
            if c < best_c:
                best, best_c = i, c
        if best is not None and best_c < 0.4:
            x_of_g[j] = [best]; kept_x_only.remove(best); stats["x_only_kept"] -= 1; stats["pair_restored"] += 1
            continue
        # a Transkribus line that holds two printed lines (opener + next line) was paired with the previous
        # Gemini line; if it contains this Gemini line's text too, this line shares its geometry
        core = g["loose"][: max(8, len(g["loose"]) - 2)]
        if len(core) >= 8:
            for jj in (j - 1, j + 1):
                hit = next((i for i in x_of_g.get(jj, []) if core in xl[i]), None)
                if hit is not None:
                    x_of_g[j] = [hit]; stats["geometry_shared"] += 1; break
    for j, g in enumerate(G):
        if j in x_of_g or j in skip_g:
            continue
        first_matched = min(x_of_g) if x_of_g else 0
        if j < first_matched and first_matched >= 3 and not g["head"]:
            skip_g.add(j); stats["g_top_block_dropped"] += 1; continue     # title-page or running-head text in the crop
        if len(g["conv"]) < 20 and not g["head"] and not (j > 0 and G[j - 1]["head"]):
            # a short Gemini-only entry is a fragment beside an initial: drop it when its text is already
            # inside a PageXML line nearby, otherwise join it to the previous line
            near_x = [i for k in range(max(0, j - 3), min(len(G), j + 4)) for i in x_of_g.get(k, [])]
            if g["loose"] and any(g["loose"] in xl[i] for i in near_x):
                skip_g.add(j); stats["g_fragment_dropped"] += 1; continue
            if j > 0:
                G[j - 1]["conv"] = (G[j - 1]["conv"].rstrip("¬") + " " + g["conv"]).strip()
                G[j - 1]["loose"] = normalise(G[j - 1]["conv"], True)
                skip_g.add(j); stats["g_fragment_joined"] += 1
    for j in range(len(G)):
        if j in skip_g:
            continue
        seq.append(("g", j))
    # insert kept X-only lines by y between their neighbours
    for i in kept_x_only:
        y = X[i]["y"]
        pos = len(seq)
        for n, (kind, k) in enumerate(seq):
            ys = [X[m]["y"] for m in x_of_g.get(k, [])] if kind == "g" else [X[k]["y"]]
            if ys and min(ys) > y:
                pos = n; break
        seq.insert(pos, ("x", i))
    # step 3–5: build the lines
    pitch = statistics.median([b["y"] - a["y"] for a, b in zip(X, X[1:]) if b["y"] > a["y"]] or [48])
    built = []
    for kind, k in seq:
        if kind == "x":
            l = X[k]
            built.append({"id": l["id"], "text0": l["text"], "src0": "pylaia only", "text1": None, "poly": l["poly"], "base": l["base"],
                          "geom": "pagexml", "type": l["type"] if l["type"] in ("cons_title", "cons_argument", "cons_body") else "cons_body", "y": l["y"]})
            stats["line_from_pagexml_only"] += 1
            continue
        g = G[k]; xs = [X[i] for i in sorted(x_of_g.get(k, []), key=lambda i: X[i]["bbox"][0])]
        if xs:
            main = max(xs, key=lambda l: len(l["text"]))
            poly_pts = [p for l in xs for p in l["poly"]]
            bbox = (min(x for x, _ in poly_pts), min(y for _, y in poly_pts), max(x for x, _ in poly_pts), max(y for _, y in poly_pts))
            poly = main["poly"] if len(xs) == 1 else [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])]
            base = main["base"] if len(xs) == 1 else [(bbox[0], int(main["y"])), (bbox[2], int(main["y"]))]
            geom = "pagexml" if len(xs) == 1 else "union"
            x_text = " ".join(l["text"] for l in xs).strip()
            c = dist(normalise(x_text, True), g["loose"]) / max(len(g["loose"]), 1)
            x_head, g_head = heading_number(x_text), heading_number(g["conv"])
            if (x_head or g_head) and x_text and g["head"] | bool(_HEAD.match(x_text)):
                # a heading: the numeral decides; the reading that continues the sequence wins, else PageXML
                # PyLaia reads the numeral as printed (the print repeats numbers), Gemini tends to "correct" it
                # to the expected one; so PageXML wins unless its number is far off and Gemini's continues
                # the PageXML numeral is kept whenever it parses (Gemini garbles numerals: CUI, CCXXXXIIII, and
                # "corrects" the print's repetitions); a disagreement is recorded for the editor's review
                choose_x = x_head is not None
                text0, src0, text1 = (x_text, "pylaia", g["conv"]) if choose_x else (g["conv"], "gemini", x_text)
                state["last_heading"] = x_head if choose_x else g_head
                stats["heading_" + src0] += 1
                if x_head and g_head and x_head != g_head:
                    stats["heading_numeral_disagreement"] += 1
                    g = dict(g, note="heading numeral differs: Gemini reads " + g["conv"][:40])
            elif len(xs) == 1 and c <= AGREE:
                text0, src0, text1 = x_text, "pylaia", g["conv"]; stats["text_pylaia_agree"] += 1
            elif g["abbr"] and x_text:
                text0, src0, text1 = x_text, "pylaia", g["conv"]; stats["text_pylaia_abbr"] += 1
            else:
                text0, src0, text1 = g["conv"], "gemini", x_text; stats["text_gemini"] += 1
            if x_text.rstrip().endswith("¬") and src0 == "gemini" and not text0.endswith("¬"):
                text0 += "¬"
            x_type = "cons_title" if any(l["type"] == "cons_title" for l in xs) else main["type"]
            ltype = "cons_title" if heading_number(text0) and _HEAD.match(text0) else pick_type(g, x_type)
            built.append({"id": main["id"], "text0": text0, "src0": src0, "text1": text1, "poly": poly, "base": base, "geom": geom,
                          "type": ltype, "y": main["y"], "note": g.get("note")})
        else:
            if g["head"] and heading_number(g["conv"]):
                state["last_heading"] = heading_number(g["conv"])
            built.append({"id": None, "text0": g["conv"], "src0": "gemini", "text1": None, "poly": None, "base": None, "geom": "interpolated",
                          "type": pick_type(g, ""), "y": None})
            stats["line_from_gemini_only"] += 1
    # interpolate geometry
    for n, b in enumerate(built):
        if b["poly"] is not None:
            continue
        prev = next((built[m] for m in range(n - 1, -1, -1) if built[m]["y"] is not None), None)
        nxt = next((built[m] for m in range(n + 1, len(built)) if built[m]["y"] is not None), None)
        if prev and nxt:
            y = (prev["y"] + nxt["y"]) / 2 if nxt["y"] - prev["y"] < 1.8 * pitch else prev["y"] + pitch
        elif prev:
            y = prev["y"] + pitch
        elif nxt:
            y = nxt["y"] - pitch
        else:
            y = X[0]["y"]
        b["y"] = y
        b["poly"] = [(rx0, int(y - 0.75 * pitch)), (rx1, int(y - 0.75 * pitch)), (rx1, int(y + 0.25 * pitch)), (rx0, int(y + 0.25 * pitch))]
        b["base"] = [(rx0, int(y)), (rx1, int(y))]
    # write back
    for tl in region.findall(T + "TextLine"):
        region.remove(tl)
    used = set()
    for n, b in enumerate(built):
        lid = b["id"] if b["id"] and b["id"] not in used else f"g_{Path(page_name).stem}_{region.get('id')}_{n}"
        used.add(lid)
        tl = ET.SubElement(region, T + "TextLine", {"id": lid, "custom": f"readingOrder {{index:{n};}} structure {{type:{b['type']};}}",
                                                    "comments": f"relayout {today}: text={b['src0']}, geometry={b['geom']}" + (f"; {b['note']}" if b.get("note") else "")})
        ET.SubElement(tl, T + "Coords", {"points": pstr(b["poly"])})
        if b["base"]:
            ET.SubElement(tl, T + "Baseline", {"points": pstr(b["base"])})
        te = ET.SubElement(tl, T + "TextEquiv", {"index": "0", "conf": "1.0"})
        ET.SubElement(te, T + "Unicode").text = b["text0"]
        if b["text1"] is not None and b["text1"] != b["text0"]:
            te1 = ET.SubElement(tl, T + "TextEquiv", {"index": "1", "conf": "0.5", "comments": "gemini" if b["src0"].startswith("pylaia") else "pylaia"})
            ET.SubElement(te1, T + "Unicode").text = b["text1"]
    u = region.find(f"{T}TextEquiv/{T}Unicode")
    if u is not None and (u.text or "").strip():
        u.text = "\n".join(b["text0"] for b in built)
    if abs(len(built) - len(X)) > 3:
        notes.append(f"{page_name} {region.get('id')}: {len(X)} lines before, {len(built)} after")


def process_print(print_dir: Path, ocr_dir: Path, out_dir: Path | None, today: str) -> tuple[Counter, list[str]]:
    stats, notes, state = Counter(), [], {}
    target = print_dir if out_dir is None else out_dir / print_dir.name
    if out_dir is not None:
        (target / "page").mkdir(parents=True, exist_ok=True)
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
                stats["column_without_gemini"] += 1; continue
            relayout_region(tr, txt.read_text(encoding="utf-8").splitlines(), stats, today, xml.name, notes, state)
            changed = True; stats["columns"] += 1
        if out_dir is not None:
            tree.write(target / "page" / xml.name, encoding="unicode", xml_declaration=True)
        elif changed:
            tree.write(xml, encoding="unicode", xml_declaration=True)
    if out_dir is not None:
        for extra in ("mets.xml", "metadata.xml"):
            if (print_dir / extra).exists():
                shutil.copy(print_dir / extra, target / extra)
    return stats, notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--print", dest="print_id", required=True)
    ap.add_argument("--out", type=Path, help="write the rebuilt print here instead of data/ (default: scratch/relayout)")
    ap.add_argument("--apply", action="store_true", help="write into data/<author>/<print>/page/")
    args = ap.parse_args()
    print_dir = next(p for p in (ROOT / "data").glob(f"*/{args.print_id}"))
    out_dir = None if args.apply else (args.out or ROOT / "scratch" / "relayout")
    today = datetime.date.today().isoformat()
    stats, notes = process_print(print_dir, ROOT / "output" / "ocr_gemini" / args.print_id, out_dir, today)
    md = [f"# Relayout from Gemini: {args.print_id}", "", f"{'Applied to data/' if args.apply else 'Written to ' + str(out_dir)} on {today}.", "",
          "| decision | lines |", "|---|---|"] + [f"| {k} | {v} |" for k, v in sorted(stats.items())] + ["", "## Columns whose line count changed by more than 3", ""] + [f"- {n}" for n in notes]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"relayout_{args.print_id}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(dict(stats), indent=1)); print(f"{len(notes)} columns with a line-count change > 3; report {OUT / f'relayout_{args.print_id}.md'}")


if __name__ == "__main__":
    main()
