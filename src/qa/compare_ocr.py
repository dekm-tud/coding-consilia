"""
Compare a second transcription (Gemini column OCR, src/gemini_ocr) with the PageXML text of a print.

For every column the lines of the PageXML region are aligned with the lines of the Gemini
transcription (<ocr_dir>/<page>_c<n>.txt, one printed line per text line). The alignment is a
banded edit-distance alignment on normalised line texts, so it tolerates a few missing or extra
lines on either side. Two normalisations are compared:
  strict  lower case, ſ->s, v->u, j->i, æ/ę/œ->ae/oe, expansion brackets removed ("no[n]" -> "non"),
          combining marks removed, {S}/{n} markers and spacing differences removed
  loose   as strict, but compared as a character stream: punctuation, spaces and hyphens removed,
          & = et, so that spacing and punctuation habits of the two transcriptions do not count
Output per print:
  output/qa/ocr_<print>.json      per page and column: line counts, matched pairs, CER (strict,
                                  loose), the unmatched lines on both sides
  output/qa/ocr_<print>_review.md the columns with the largest differences and every line pair
                                  whose loose CER is above --review (default 0.25), for an editor
  output/qa/ocr_summary.md        one table over all prints
A line present in Gemini but unmatched in the PageXML is the strongest signal for a line missed by
the layout analysis (see check_lines.py); a pair with high loose CER is a reading to check.

    python src/qa/compare_ocr.py --print Baldo_Cons_Print_Venice_1575_v1 \
        --ocr output/ocr_gemini/Baldo_Cons_Print_Venice_1575_v1
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "qa"))
from check_lines import read_page  # noqa: E402

try:
    from rapidfuzz.distance import Levenshtein
    def dist(a: str, b: str) -> int:
        return Levenshtein.distance(a, b)
except ImportError:                       # pure-Python fallback, slow but correct
    def dist(a: str, b: str) -> int:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

OUT = ROOT / "output" / "qa"
_MARK = re.compile(r"^\s*(\{S\}\s*)?(\{\d+\}\s*)?")
_BR_KEEP = re.compile(r"\[([^\]]*)\]")
_PUNCT = re.compile(r"[^\w\s§¶†]")
_SP = re.compile(r"\s+")
_TRANS = str.maketrans({"ſ": "s", "v": "u", "j": "i", "æ": "ae", "ę": "ae", "œ": "oe", "Æ": "ae", "Œ": "oe", "¬": "-", "ß": "ss"})


def normalise(s: str, loose: bool) -> str:
    s = _MARK.sub(lambda m: (m.group(2) or "").strip("{} ") + (" " if m.group(2) else ""), s)   # {3} -> "3 ", {S} dropped
    s = s.lower().translate(_TRANS)
    s = _BR_KEEP.sub(r"\1", s)                      # no[n] -> non (the PageXML text expands too)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    if loose:                                       # character stream: no punctuation, no spaces, & = et
        s = s.replace("&", "et").replace("-", "")
        s = _PUNCT.sub("", s).replace(" ", "")
        return s
    return _SP.sub(" ", s).strip()


def align(a: list[str], b: list[str], band: int = 8) -> list[tuple[int | None, int | None]]:
    """Banded global alignment; cost of a pair = normalised edit distance, gap = 0.6."""
    n, m = len(a), len(b)
    INF = float("inf")
    gap = 0.6
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    B = [[None] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for i in range(0, n + 1):
        for j in range(max(0, i - band - abs(n - m)), min(m, i + band + abs(n - m)) + 1):
            if i == 0 and j == 0:
                continue
            best, arg = INF, None
            if i > 0 and D[i - 1][j] + gap < best:
                best, arg = D[i - 1][j] + gap, "a"
            if j > 0 and D[i][j - 1] + gap < best:
                best, arg = D[i][j - 1] + gap, "b"
            if i > 0 and j > 0 and D[i - 1][j - 1] < INF:
                la, lb = a[i - 1], b[j - 1]
                c = dist(la, lb) / max(len(la), len(lb), 1)
                if D[i - 1][j - 1] + c < best:
                    best, arg = D[i - 1][j - 1] + c, "ab"
            D[i][j], B[i][j] = best, arg
    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        step = B[i][j]
        if step == "ab":
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif step == "a":
            pairs.append((i - 1, None)); i -= 1
        elif step == "b":
            pairs.append((None, j - 1)); j -= 1
        else:   # outside band; fall back to gaps
            if i > 0:
                pairs.append((i - 1, None)); i -= 1
            else:
                pairs.append((None, j - 1)); j -= 1
    return pairs[::-1]


def merge_same_baseline(lines: list[dict]) -> list[dict]:
    """Join PageXML lines that sit on the same baseline (large-type openers such as "In Christi
    nomine." are often split from the small-type rest of the printed line) into one line, left to right."""
    if not lines:
        return lines
    gaps = sorted(b["y"] - a["y"] for a, b in zip(lines, lines[1:]))
    pitch = gaps[len(gaps) // 2] if gaps else 48
    out = [dict(lines[0])]
    for l in lines[1:]:
        prev = out[-1]
        # same baseline, or (large-type opener next to small type) polygons that overlap vertically by
        # more than half of the smaller one while not overlapping horizontally
        v_overlap = min(prev["bbox"][3], l["bbox"][3]) - max(prev["bbox"][1], l["bbox"][1])
        h_overlap = min(prev["bbox"][2], l["bbox"][2]) - max(prev["bbox"][0], l["bbox"][0])
        small_h = min(prev["bbox"][3] - prev["bbox"][1], l["bbox"][3] - l["bbox"][1]) or 1
        opener = v_overlap > 0.5 * small_h and h_overlap < 0.1 * small_h and "+" not in prev["id"]   # merge once, not a chain
        if abs(l["y"] - prev["y"]) < 0.4 * pitch or opener:
            left, right = sorted([prev, l], key=lambda z: z["bbox"][0])
            prev.update(text=f"{left['text']} {right['text']}", id=f"{prev['id']}+{l['id']}",
                        bbox=(min(prev["bbox"][0], l["bbox"][0]), min(prev["bbox"][1], l["bbox"][1]),
                              max(prev["bbox"][2], l["bbox"][2]), max(prev["bbox"][3], l["bbox"][3])))
        else:
            out.append(dict(l))
    return out


def compare_column(xml_lines: list[dict], ocr_lines: list[str], review: float) -> dict:
    xa = [normalise(l["text"], False) for l in xml_lines]
    xb = [normalise(t, False) for t in ocr_lines]
    la = [normalise(l["text"], True) for l in xml_lines]
    lb = [normalise(t, True) for t in ocr_lines]
    pairs = align(la, lb)
    strict_err = strict_len = loose_err = loose_len = 0
    matched, only_xml, only_ocr, flagged = 0, [], [], []
    for i, j in pairs:
        if i is not None and j is not None:
            matched += 1
            d_s, d_l = dist(xa[i], xb[j]), dist(la[i], lb[j])
            strict_err += d_s; strict_len += max(len(xa[i]), 1)
            loose_err += d_l; loose_len += max(len(la[i]), 1)
            cer = d_l / max(len(la[i]), len(lb[j]), 1)
            if cer > review:
                flagged.append({"xml_id": xml_lines[i]["id"], "cer": round(cer, 2), "xml": xml_lines[i]["text"], "ocr": ocr_lines[j], "ocr_index": j})
        elif i is not None:
            only_xml.append({"id": xml_lines[i]["id"], "text": xml_lines[i]["text"]})
        else:
            only_ocr.append({"index": j, "text": ocr_lines[j], "after_xml_id": next((xml_lines[p[0]]["id"] for p in reversed(pairs[:pairs.index((None, j))]) if p[0] is not None), None)})
    return {"xml_lines": len(xml_lines), "ocr_lines": len(ocr_lines), "matched": matched,
            "cer_strict": round(strict_err / strict_len, 4) if strict_len else None,
            "cer_loose": round(loose_err / loose_len, 4) if loose_len else None,
            "only_xml": only_xml, "only_ocr": only_ocr, "flagged": flagged}


def compare_print(print_dir: Path, ocr_dir: Path, review: float) -> dict:
    cols = []
    for xml in sorted((print_dir / "page").glob("*.xml")):
        page = read_page(xml)
        stem = xml.stem
        for c in page["columns"]:
            n = c["type"].split("_")[-1]
            txt = ocr_dir / f"{stem}_c{n}.txt"
            if not txt.exists():
                continue
            ocr_lines = [l.rstrip() for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
            xml_lines = merge_same_baseline([l for l in c["lines"] if l["text"].strip()])
            r = compare_column(xml_lines, ocr_lines, review)
            r.update(page=xml.name, column=c["type"], region=c["id"])
            cols.append(r)
    cer_s = [c["cer_strict"] for c in cols if c["cer_strict"] is not None]
    cer_l = [c["cer_loose"] for c in cols if c["cer_loose"] is not None]
    return {"print": print_dir.name, "columns": len(cols), "pages": len({c["page"] for c in cols}),
            "xml_lines": sum(c["xml_lines"] for c in cols), "ocr_lines": sum(c["ocr_lines"] for c in cols),
            "matched": sum(c["matched"] for c in cols),
            "only_xml": sum(len(c["only_xml"]) for c in cols), "only_ocr": sum(len(c["only_ocr"]) for c in cols),
            "flagged": sum(len(c["flagged"]) for c in cols),
            "cer_strict_median": round(statistics.median(cer_s), 4) if cer_s else None,
            "cer_loose_median": round(statistics.median(cer_l), 4) if cer_l else None,
            "results": cols}


def write_review(rep: dict, path: Path, top: int = 40) -> None:
    cols = sorted(rep["results"], key=lambda c: -(len(c["only_xml"]) + len(c["only_ocr"]) + (c["cer_loose"] or 0) * 10))
    md = [f"# OCR comparison: {rep['print']}", "",
          f"{rep['pages']} pages, {rep['columns']} columns. PageXML lines {rep['xml_lines']}, Gemini lines {rep['ocr_lines']}, matched {rep['matched']}.",
          f"Lines only in PageXML: {rep['only_xml']}; only in Gemini: {rep['only_ocr']} (candidates for lines the layout missed).",
          f"Median CER strict {rep['cer_strict_median']}, loose {rep['cer_loose_median']}. Line pairs flagged for review: {rep['flagged']}.", "",
          "## Columns with the largest differences", ""]
    for c in cols[:top]:
        md.append(f"### {c['page']} {c['column']}  (PageXML {c['xml_lines']} / Gemini {c['ocr_lines']} lines, CER loose {c['cer_loose']})")
        for l in c["only_ocr"]:
            md.append(f"- only Gemini (after {l['after_xml_id']}): `{l['text']}`")
        for l in c["only_xml"]:
            md.append(f"- only PageXML ({l['id']}): `{l['text']}`")
        for f in c["flagged"]:
            md.append(f"- CER {f['cer']} ({f['xml_id']})  \n  PageXML: `{f['xml']}`  \n  Gemini:  `{f['ocr']}`")
        md.append("")
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--print", dest="print_id", required=True)
    ap.add_argument("--ocr", type=Path, required=True, help="directory with <page>_c<n>.txt")
    ap.add_argument("--review", type=float, default=0.25, help="loose CER above which a line pair is listed")
    args = ap.parse_args()
    print_dir = next(p for p in (ROOT / "data").glob(f"*/{args.print_id}"))
    rep = compare_print(print_dir, args.ocr, args.review)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"ocr_{args.print_id}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    write_review(rep, OUT / f"ocr_{args.print_id}_review.md")
    summary = OUT / "ocr_summary.md"
    rows = {}
    if summary.exists():
        for ln in summary.read_text(encoding="utf-8").splitlines():
            if ln.startswith("| Baldo") or ln.startswith("| Panorm"):
                rows[ln.split("|")[1].strip()] = ln
    rows[rep["print"]] = (f"| {rep['print']} | {rep['pages']} | {rep['columns']} | {rep['xml_lines']} | {rep['ocr_lines']} | {rep['matched']} | "
                          f"{rep['only_xml']} | {rep['only_ocr']} | {rep['cer_strict_median']} | {rep['cer_loose_median']} | {rep['flagged']} |")
    summary.write_text("# OCR comparison (PageXML vs Gemini)\n\n| print | pages | columns | PageXML lines | Gemini lines | matched | only PageXML | only Gemini | CER strict | CER loose | flagged pairs |\n|---|---|---|---|---|---|---|---|---|---|---|\n"
                       + "\n".join(rows[k] for k in sorted(rows)) + "\n\nDetails: ocr_<print>.json and ocr_<print>_review.md; method in src/qa/compare_ocr.py.\n", encoding="utf-8")
    print(f"{rep['print']}: {rep['columns']} columns, matched {rep['matched']}/{rep['xml_lines']} PageXML lines, only Gemini {rep['only_ocr']}, "
          f"only PageXML {rep['only_xml']}, CER strict {rep['cer_strict_median']} loose {rep['cer_loose_median']}, flagged {rep['flagged']}")
    print(f"Review: {OUT / f'ocr_{args.print_id}_review.md'}")


if __name__ == "__main__":
    main()
