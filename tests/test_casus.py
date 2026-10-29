"""
Regression test for the misplaced-casus correction.

  pytest tests/test_casus.py

Two kinds of checks:
  * segmenter invariants, run from the PageXML with the heuristic dehyphenation
    (fast; the invariants do not depend on the LB model);
  * checks on the built output (output/consilia.json), skipped if it is missing.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import casus  # noqa: E402
import carry_derived  # noqa: E402
import pipeline  # noqa: E402
from segment_consilia import AnnotatedConsiliumSegmenter, iter_lines  # noqa: E402

AUTHOR = "Baldo_29618397"
PREFIX = "Baldo_Cons_Print_Venice_1575_"
VOLUMES = [f"{PREFIX}v{i}" for i in range(1, 6)]
V4 = f"{PREFIX}v4"
SNAPSHOT = ROOT / "archive" / "pre_casus_correction_2026-09-23" / "consilia.json"
OUTPUT = ROOT / "output" / "consilia.json"


def _chars(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def _consilium_chars(c: dict) -> int:
    n = sum(_chars(s) for s in c.get("body") or [])
    n += sum(_chars(s) for s in c.get("casus") or [])
    n += sum(_chars(it.get("text", "")) for it in c.get("argument") or [])
    return n


@pytest.fixture(scope="module")
def segmented():
    """volume -> (split segmenter+results, unsplit results), heuristic dehyphenation."""
    out = {}
    for vol in VOLUMES:
        pages = pipeline.load_pages(ROOT / "data" / AUTHOR / vol)
        seg = AnnotatedConsiliumSegmenter(vol, AUTHOR, use_lb_model=False, split_casus=True)
        new = seg.process(iter_lines(pages))
        old = AnnotatedConsiliumSegmenter(vol, AUTHOR, use_lb_model=False, split_casus=False) \
            .process(iter_lines(pages))
        out[vol] = (seg, new, old)
    return out


@pytest.fixture(scope="module")
def built():
    """The Baldo prints of the built output. Prebuilt prints (Panormitanus, import_prebuilt.py) are
    segmented elsewhere and have no EN translation, so the casus invariants do not apply to them."""
    if not OUTPUT.exists():
        pytest.skip("output/consilia.json not built")
    data = json.loads(OUTPUT.read_text(encoding="utf-8"))["consilia"]
    return {i: c for i, c in data.items() if c["volume"] in VOLUMES}


# --------------------------------------------------------------------------- segmenter

def test_ids_and_numbers_unchanged(segmented):
    for vol, (_, new, old) in segmented.items():
        assert [(c.id, c.n) for c in new] == [(c.id, c.n) for c in old], vol


def test_no_text_lost_or_duplicated_per_volume(segmented):
    """(d) casus + argument + body characters per volume are identical before/after."""
    for vol, (_, new, old) in segmented.items():
        before = sum(_consilium_chars(c.to_dict()) for c in old)
        after = sum(_consilium_chars(c.to_dict()) for c in new)
        assert after == before, f"{vol}: {before} -> {after}"


def test_text_moves_only_between_neighbours(segmented):
    """Per consilium: owner loses exactly what the receiver (next consilium) gains."""
    for vol, (seg, new, old) in segmented.items():
        old_by = {c.id: _consilium_chars(c.to_dict()) for c in old}
        new_by = {c.id: _consilium_chars(c.to_dict()) for c in new}
        delta = {i: new_by[i] - old_by[i] for i in old_by}
        expected = {i: 0 for i in old_by}
        ids = [c.id for c in new]
        for mv in seg.casus_moves:
            assert ids.index(mv["receiver_id"]) == ids.index(mv["owner_id"]) + 1
            moved = _chars(" ".join(next(c for c in new if c.id == mv["receiver_id"]).casus))
            expected[mv["owner_id"]] -= moved
            expected[mv["receiver_id"]] += moved
        assert delta == expected, vol


def test_v4_61_62_case(segmented):
    _, new, _ = segmented[V4]
    by = {c.id: c for c in new}
    c61, c62 = by[f"consilium-{V4.lower()}-61"], by[f"consilium-{V4.lower()}-62"]
    pat = re.compile(r"Nardus\s+maritauit\s+Mat")
    assert pat.search(" ".join(c62.casus)), "(b) v4 n.62 casus must contain Nardus' case"   # (b)
    assert not pat.search(" ".join(c61.body)), "(c) v4 n.61 body must no longer contain it"  # (c)
    assert c62.casus and c62.casus[0].lstrip().upper().startswith("CAS")
    # the signature stays with the consilium it closes
    assert re.search(r"Ego\s+Bal", " ".join(c61.body))
    # provenance moved with the text: n.62's sources now start on the page where the casus is
    assert c62.sources and c62.sources[0]["regions"]


def test_no_trailing_casus_left_in_a_body(segmented):
    """(a) after the correction no body ends in 'signature + casus marker' unless the
    decision for that consilium is an explicit 'keep' (casus_overrides.json) or the
    block is not the tail of the body (a later signature follows)."""
    overrides = casus.load_overrides()
    left = []
    for vol, (_, new, _) in segmented.items():
        for c in new:
            if overrides.get(c.id, {}).get("decision") == "keep":
                continue
            if _trailing_casus(" ".join(c.body)):
                left.append(c.id)
    assert not left, left


def test_every_candidate_is_decided(segmented):
    """No 'review' status may remain: undecided cases go to casus_overrides.json."""
    for vol, (seg, _, _) in segmented.items():
        undecided = [r["owner_id"] for r in seg.casus_report if r["status"] == "review"]
        assert not undecided, f"{vol}: {undecided}"


def test_direction_check_moves_only_towards_next(segmented):
    for vol, (seg, new, _) in segmented.items():
        ids = [c.id for c in new]
        for r in seg.casus_report:
            if r["status"] == "move":
                assert r["receiver_id"] == ids[ids.index(r["owner_id"]) + 1]


# --------------------------------------------------------------------------- unit level

def test_signature_and_marker_regexes():
    for txt in ["Ego Baldus. CASVS. In Christi", "ego Bal. de Peru. C A S VS. IN", "Ego Bald. OASV Punctus talis",
                "consulo ego¬ Baldus de Perusio etc. CASVS.", "Ego Baldus. Punctus. Super quo consilium petitur"]:
        sig = casus.RE_SIGNATURE.search(txt)
        assert sig, txt
        assert casus.RE_MARK.search(txt, sig.end()), txt


def test_translation_split_by_proportion():
    en = ("The heir is bound by the legacy, as I have shown. And so I say and advise. I, Baldus. "
          "CASE. The case is as follows: a certain Nardus married Matthea and gave her a dowry.")
    a, b = carry_derived.split_translated_section(en, p_expected=0.4)
    assert a.endswith("I, Baldus.") and b.startswith("CASE.")
    assert carry_derived.split_translated_section(en, p_expected=0.95) is None   # outside tolerance
    assert carry_derived.split_translated_section("No signature here at all. Nothing.", 0.5) is None


# --------------------------------------------------------------------------- built output

def _trailing_casus(flat: str) -> bool:
    found = []
    for m in casus.RE_SIGNATURE.finditer(flat):
        mk = casus.RE_MARK.search(flat[m.end(): m.end() + casus.MARK_WINDOW])
        if mk:
            found.append((m, mk))
    if not found:
        return False
    m, mk = found[-1]
    return all(len(flat) - m2.end() <= casus.TAIL_SIGNATURE_SLACK
               for m2 in casus.RE_SIGNATURE.finditer(flat, m.end() + mk.end()))


def test_output_v4_62_shows_nardus_case(built):
    c62 = built[f"consilium-{V4.lower()}-62"]
    assert re.search(r"Nardus\s+maritauit\s+Matthaeam", " ".join(c62["casus"]))
    assert not re.search(r"Nardus\s+maritauit", " ".join(built[f"consilium-{V4.lower()}-61"]["body"]))


def test_output_no_trailing_casus_in_bodies(built):
    overrides = casus.load_overrides()
    bad = [i for i, c in built.items()
           if overrides.get(i, {}).get("decision") != "keep" and _trailing_casus(" ".join(c["body"]))]
    assert not bad, bad


def test_output_translations_aligned(built):
    bad = []
    for i, c in built.items():
        tr = c.get("translations") or {}
        for lang in ("en", "de"):
            t = tr.get(lang) or {}
            if len(t.get("body") or []) != len(c["body"]):
                bad.append((i, lang, "body"))
            if c.get("casus") and len(t.get("casus") or []) != len(c["casus"]):
                bad.append((i, lang, "casus"))
    assert not bad, bad[:10]


def test_output_text_preserved_against_pre_correction_snapshot(built):
    """(d) on the real output: per volume, casus + argument + body characters equal the
    pre-correction output, so the LB-model build lost and duplicated nothing."""
    if not SNAPSHOT.exists():
        pytest.skip("pre-correction snapshot not present")
    old = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["consilia"]
    assert set(old) == set(built), "consilium ids changed"
    by_vol_old, by_vol_new = {}, {}
    for i, c in old.items():
        by_vol_old[c["volume"]] = by_vol_old.get(c["volume"], 0) + _consilium_chars(c)
    for i, c in built.items():
        by_vol_new[c["volume"]] = by_vol_new.get(c["volume"], 0) + _consilium_chars(c)
    # 2026-10-17: parse_pagexml now reads region types with a score field correctly, so lines tagged
    # cons_body inside *header* regions are skipped as intended; v5 lost exactly two such header
    # tokens ("56" in n.211, "oni" in n.473), 5 characters. Anything beyond that is a real loss.
    for vol in by_vol_old:
        assert 0 <= by_vol_old[vol] - by_vol_new.get(vol, 0) <= 5, (vol, by_vol_old[vol], by_vol_new.get(vol))


def test_output_derived_layers_carried(built):
    missing = [i for i, c in built.items() if "translations" not in c or "keywords_final" not in c]
    assert not missing, missing[:5]


_NAMES = re.compile(r"\bBa(?:rtolus|ldus)\b")   # "I, Bartolus" -> "I, Baldus" fix (src/qa/fix_translation_names.py) is deliberate


def _tr_chars(c: dict, lang: str) -> int:
    t = (c.get("translations") or {}).get(lang) or {}
    return sum(_chars(_NAMES.sub("", s)) for k in ("body", "casus") for s in t.get(k) or [])


def test_output_translations_preserved_against_snapshot(built):
    """No translated text is lost or duplicated by the split, per volume and language."""
    if not SNAPSHOT.exists():
        pytest.skip("pre-correction snapshot not present")
    old = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["consilia"]
    for lang in ("en", "de"):
        a, b = {}, {}
        for c in old.values():
            a[c["volume"]] = a.get(c["volume"], 0) + _tr_chars(c, lang)
        for c in built.values():
            b[c["volume"]] = b.get(c["volume"], 0) + _tr_chars(c, lang)
        assert a == b, lang


def test_output_translation_cut_lies_at_the_signature(built):
    """The casus translation must not start with the closing signature ('I, Baldus.'), and
    the translation of the owner's last section must not end with a casus label."""
    sig_start = re.compile(r"^\W*(?:I|Ich),?\s+(?:Bal|Bart|Bald)", re.I)
    label_end = re.compile(r"\b(?:CASE|FALL|CASUS|CASVS)\W*$")
    bad = []
    prev_by_vol: dict[str, dict] = {}
    for cid, c in built.items():
        prev = prev_by_vol.get(c["volume"])
        if c.get("casus") and prev is not None:
            for lang in ("en", "de"):
                first = ((c["translations"][lang].get("casus") or [""])[0])[:60]
                last = ((prev["translations"][lang]["body"] or [""])[-1])[-40:]
                if sig_start.match(first) or label_end.search(last):
                    bad.append((cid, lang, first[:30], last[-30:]))
        prev_by_vol[c["volume"]] = c
    assert not bad, bad[:5]
