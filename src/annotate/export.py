"""
Publish annotation layers for the site: output/annotations/*.jsonl ->
site-data/annotations/{intimate_body,subjects}.json, read by search.html (Subjects tab).

Layer B (intimate body): one or two passes. Status per annotation:
  verified_by_model  both passes agree on the code (role from pass 1)
  machine            one pass only, or the passes disagree (listed with pass=…)
  verified_by_editor set by hand in eval/reviews/body_reviews.jsonl ({unit, code, decision: accept|reject, role?})
Annotations whose evidence quote was not found in the unit are dropped unless an editor review accepts them.

Layer A (subjects): one pass; `central`/`secondary` per consilium.

    python src/annotate/export.py --body output/annotations/body_<model>_p1.jsonl [output/annotations/body_<model>_p2.jsonl]
    python src/annotate/export.py --subjects output/annotations/subjects_<model>_p1.jsonl
  python src/annotate/export.py --spans output/annotations/spans_<model>_p1.jsonl output/annotations/spans_flair.jsonl \\
        [--patch output/annotations/spans_<model>_patch.jsonl output/annotations/spans_flair_patch.jsonl --patch-units output/units_patch.jsonl]
    -> site-data/annotations/references_index.json (normalised citations and jurist names -> consilia) and
       site-data/annotations/references/<print>.json (all spans per consilium, with section offsets)
Then: python src/build_site_data.py  (lists the files in manifest.json so the site loads them).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from import_prebuilt import edition_form  # noqa: E402  (evidence quotes in the edition's spelling)
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site-data" / "annotations"
REVIEWS = ROOT / "eval" / "reviews" / "body_reviews.jsonl"


_LOOSE_DROP = {"\u0300", "\u0301", "\u0302", "\u0303", "\u0304", "\u0308"}


def _loose_char(c: str) -> str:
    import unicodedata
    d = "".join(x for x in unicodedata.normalize("NFD", c.lower()) if x not in _LOOSE_DROP)
    d = d.replace("v", "u").replace("j", "i").replace("ſ", "s").replace("l", "i").replace("1", "i")   # i/l/1: the usual OCR confusion
    return "".join(x for x in d if x.isalnum())


def loose(t: str) -> str:
    """letters and digits only, for re-anchoring spans after the text was cleaned"""
    t = t.replace("[", "").replace("]", "").replace("q;", "que")
    return "".join(_loose_char(c) for c in t)


def loose_index(t: str) -> tuple[str, list[int]]:
    """loose(t) and, for every character of it, its position in t"""
    key, pos = [], []
    for i, c in enumerate(t):
        for x in _loose_char(c):
            key.append(x); pos.append(i)
    return "".join(key), pos


def load(path: Path, key: str) -> dict:
    out = {}
    for l in path.open(encoding="utf-8"):
        r = json.loads(l)
        if r.get("ok"):
            out[r[key]] = r
    return out


# consilia renumbered by a rebuild (tools/recover_derived.py writes old id -> new id): the annotation runs
# and output/units.jsonl still carry the old ids, so every doc id is passed through this map
_ID_MAP_FILE = ROOT / "output" / "debug" / "id_map.json"
ID_MAP: dict[str, str] = json.loads(_ID_MAP_FILE.read_text(encoding="utf-8")) if _ID_MAP_FILE.exists() else {}


def doc_id(d: str) -> str:
    return ID_MAP.get(d, d)


def export_body(files: list[Path]) -> dict:
    cb = yaml.safe_load((ROOT / "src" / "codebooks" / "intimate_body.yaml").read_text(encoding="utf-8"))
    p1 = load(files[0], "unit")
    p2 = load(files[1], "unit") if len(files) > 1 else None
    reviews = {}
    if REVIEWS.exists():
        for l in REVIEWS.open(encoding="utf-8"):
            r = json.loads(l)
            reviews[(r["unit"], r["code"])] = r
    units_meta = {}
    for l in (ROOT / "output" / "units.jsonl").open(encoding="utf-8"):
        u = json.loads(l)
        units_meta[u["unit"]] = u
    cons_ids = set(json.loads((ROOT / "output" / "consilia.json").read_text(encoding="utf-8"))["consilia"])
    out = []
    for unit, rec in p1.items():
        um = units_meta.get(unit, {})
        if doc_id(um.get("doc") or unit.split("#")[0]) not in cons_ids:
            continue
        anns1 = rec["result"].get("annotations", [])
        codes2 = {a["code"]: a for a in (p2[unit]["result"].get("annotations", []) if p2 and unit in p2 else [])}
        seen = set()
        for a in anns1:
            rv = reviews.get((unit, a["code"]))
            if rv and rv.get("decision") == "reject":
                continue
            if not a.get("evidence_valid", True) and not (rv and rv.get("decision") == "accept"):
                continue
            status = "verified_by_editor" if rv else ("verified_by_model" if a["code"] in codes2 else ("machine" if p2 is None else "disputed"))
            out.append({"unit": unit, "doc": doc_id(um.get("doc") or unit.split("#")[0]), "section": um.get("section"), "window": um.get("window"),
                        "kind": um.get("kind"), "code": a["code"], "role": (rv or {}).get("role") or a["role"], "evidence_la": edition_form(a.get("evidence_la", "")),
                        "status": status, "pass_agreement": a["code"] in codes2 if p2 else None})
            seen.add(a["code"])
        if p2 and unit in p2:                      # pass 2 only: disputed
            for code, a in codes2.items():
                if code in seen or not a.get("evidence_valid", True):
                    continue
                out.append({"unit": unit, "doc": doc_id(um.get("doc") or unit.split("#")[0]), "section": um.get("section"), "window": um.get("window"),
                            "kind": um.get("kind"), "code": code, "role": a["role"], "evidence_la": edition_form(a.get("evidence_la", "")), "status": "disputed", "pass_agreement": False})
    models = sorted({rec["model"] for rec in p1.values()} | ({r["model"] for r in p2.values()} if p2 else set()))
    return {"codebook_version": cb["version"], "roles": cb["roles"], "counted_roles_default": cb["counted_roles_default"],
            "codes": [{"id": c["id"], "group": c["group"], "label_en": c["label_en"], "label_de": c.get("label_de", ""), "label_la": c.get("label_la", "")} for c in cb["codes"]],
            "groups": cb["groups"], "models": models, "units_classified": len(p1), "units": out,
            "note": "Gold set and codebook are model-annotated (Claude Opus 5.5) on the editor's behalf; statuses: machine, disputed, verified_by_model, verified_by_editor."}


def export_subjects(file: Path) -> dict:
    sc = yaml.safe_load((ROOT / "src" / "codebooks" / "subjects.yaml").read_text(encoding="utf-8"))
    preds = load(file, "id")
    cons = json.loads((ROOT / "output" / "consilia.json").read_text(encoding="utf-8"))["consilia"]
    docs = {doc_id(cid): {"central": r["result"].get("central", []), "secondary": [s for s in r["result"].get("secondary", []) if s not in r["result"].get("central", [])]}
            for cid, r in preds.items() if doc_id(cid) in cons}          # consilia that no longer exist after a rebuild are dropped
    return {"scheme_version": sc["version"], "facets": sc["facets"],
            "concepts": [{"id": c["id"], "facet": c["facet"], "en": c["en"], "de": c.get("de", ""), "la": c.get("la", ""), "scope": c.get("scope", "")} for c in sc["concepts"]],
            "models": sorted({r["model"] for r in preds.values()}), "docs": docs,
            "note": "Machine assignment (one pass); evaluation against the model-annotated gold in eval/results/subjects.md."}


def _ref_key(ref: str) -> str | None:
    """index key of a normalised citation: work and title only ("D. 2.14", "X 2.20", "C. 6.28", "Nov. 118",
    "LF 2.26"); the lex or capitulum number the model gives is unreliable (three quarters are ".1")."""
    import re
    r = re.sub(r"\s+", " ", ref.strip())
    m = re.match(r"^(D|Dig|C|Cod|X|VI|Clem|Inst|Nov|Auth|LF|Extrav)\.?\s*(\d+)(?:\.(\d+))?", r)
    if not m:
        return None
    work = {"Dig": "D", "Cod": "C"}.get(m.group(1), m.group(1))
    if work in ("Nov", "Auth"):
        return f"{work}. {m.group(2)}"
    return f"{work}{'.' if work in ('D', 'C', 'Inst', 'Clem', 'Nov', 'Auth') else ''} {m.group(2)}" + (f".{m.group(3)}" if m.group(3) else "")


# sigla of the jurists and works most often cited, mapped to one name (regex on the lower-cased,
# punctuation-free span text). Only mapped names enter the index: an unmapped siglum stays marked
# as JURIST in the text but gets no key, so that the name index holds names, not abbreviations.
JURIST_SIGLA = [
    (r"^(ego )?(bal|bald|baldus)( de peru.*)?$", "Baldus de Ubaldis"),
    (r"^(bar|bart|barth|bartho|barthol|barto|bartol|bartholus|bartolus)( de sax.*)?$", "Bartolus de Saxoferrato"),
    (r"^(inn|inno|innoc|innocen|innocent|innocentius)$", "Innocentius IV"),
    (r"^(cyn|cynus|cy)$", "Cynus de Pistorio"),
    (r"^(dyn|dynus|dinus|dy|di|din)$", "Dinus de Mugello"),
    (r"^(io|ioan|ioann|lo|iohan|ioannes|iohannes)( an| and| andr| andre| andreae)?$", "Johannes Andreae"),
    (r"^(io|ioan|ioann|iohan|ioannes|iohannes) (de )?(lig|lign|ligna|lignano).*$", "Johannes de Lignano"),
    (r"^(io|ioan|ioann|iohan|ioannes|iohannes) (cal|cald|calde|calder|calderinus).*$", "Johannes Calderinus"),
    (r"^(io|ioan|ioann|iohan|ioannes|iohannes) (de )?(imo|imol|imola).*$", "Johannes de Imola"),
    (r"^(io|ioan|ioann|iohan|ioannes|iohannes) (mo|mon|mona|monac|monachus).*$", "Johannes Monachus"),
    (r"^(io|ioan|ioann|iohan|ioannes|iohannes) (fab|faber).*$", "Johannes Faber"),
    (r"^(hos|host|hosti|hostien|hostiensis)$", "Hostiensis"),
    (r"^(spe|spec|specu|specul|speculator|speculatoris|speculatorem)$", "Speculator (Guilelmus Durantis)"),
    (r"^(arch|archi|archid|archidia|archidiaconus)$", "Archidiaconus (Guido de Baysio)"),
    (r"^(iac|iaco|iacob|lac|la|ia|iacobus) de are.*$", "Jacobus de Arena"),
    (r"^(iac|iaco|iacob|ia|iacobus) (but|bu|butr|butri|butriga|butrigarius).*$", "Jacobus Butrigarius"),
    (r"^(butr|butri|butriga|butrigarius)$", "Jacobus Butrigarius"),
    (r"^(iac|iaco|iacob|ia|iacobus) de bel.*$", "Jacobus de Belvisio"),
    (r"^(iac|iaco|iacob|ia|iacobus) (de )?ardi.*$", "Jacobus de Ardizone"),
    (r"^(iac|iaco|iacob|ia|iacobus) de isol.*$", "Jacobus de Isolanis"),
    (r"^(gul|guli|guil|gui|guliel|guiler|guilel|guilelmus|guillelmus)( de cu.*)?$", "Guilelmus de Cuneo"),
    (r"^(nic|nico|nicol|nicolaus) ab.*$", "Nicolaus de Tudeschis (Panormitanus)"),
    (r"^(abb|abba|abbas)( sic.*| mod.*| panor.*)?$", "Nicolaus de Tudeschis (Panormitanus)"),
    (r"^(nic|nico|nicol|nicolaus) de mat.*$", "Nicolaus de Matarellis"),
    (r"^azo$", "Azo"),
    (r"^(acc|accur|accurs|accursius)$", "Accursius"),
    (r"^(old|oldr|oldra|oldrad|oldradus)( de pon.*| de lau.*)?$", "Oldradus de Ponte"),
    (r"^(ant|anto|anton|antonius) de but.*$", "Antonius de Butrio"),
    (r"^(fede|feder|fed|federic|federicus|federicum|federici)( de sen.*)?$", "Federicus de Senis"),
    (r"^(odo|odof|odofr|odofredus)$", "Odofredus"),
    (r"^(pau|paul|paulus) de cast.*$", "Paulus de Castro"),
    (r"^(pau|paul|paulus) de (leaz|liaz|lyaz).*$", "Paulus de Liazariis"),
    (r"^(pet|petr|petrus) de (bell|bella).*$", "Petrus de Bellapertica"),
    (r"^(pet|petr|petrus) de anch.*$", "Petrus de Ancharano"),
    (r"^(ang|ange|angelus) de aret.*$", "Angelus de Aretio"),
    (r"^(ang|ange|angelus)( de per.*| de vbal.*| de ubal.*)?$", "Angelus de Ubaldis"),
    (r"^(ray|raym|rayn|raymundus|raynaldus)( de pen.*)?$", "Raymundus de Pennaforte"),
    (r"^(ric|rich|richardus|ricardus)( malumb.*)?$", "Ricardus Malumbra"),
    (r"^(raph|rapha|raphael|raphaelem)( fulg.*)?$", "Raphael Fulgosius"),
    (r"^(rof|rofr|rofred|rofredus|rofredum|roffredus)$", "Roffredus Beneventanus"),
    (r"^(gof|godof|goff|goffr|goffred|goffredus|goffreddus)( de tra.*)?$", "Goffredus de Trano"),
    (r"^(alb|albe|alber|alberi|albericus)( de ros.*)?$", "Albericus de Rosate"),
    (r"^(sal|sali|salic|salice|saliceto|saly)$", "Bartholomaeus de Saliceto"),
    (r"^(lud|ludo|ludov|ludovicus)( ro.*)?$", "Ludovicus Romanus"),
    (r"^(and|andr|andreas) de bar.*$", "Andreas de Barulo"),
    (r"^(arist|aristo|aristoteles|aristotelis)$", "Aristoteles"),
    (r"^(gl|glo|glos|gloss|gloff|glosa|glossa|glossam)( or| ord| ordi| ordin| ordina| ordinaria| mag| magn| magna)?$", "glossa ordinaria"),
    (r"^(glossat|glossato|glossatores|glossatori|glossatoribus)$", "glossatores"),
    (r"^(doc|doct|docto|doctor|doctores|doctorum|doctoribus)$", "doctores (communis opinio)"),
    (r"^(mod|moder|modern|moderni|modernos|modernorum)$", "moderni"),
]
# words the span often starts with that are not part of the name ("no. Bar." = notat Bartolus)
_JURIST_PREFIXES = ("no", "not", "nota", "notat", "do", "dominus", "d", "secundum", "sec", "per", "ut", "vt", "et", "idem", "ide", "in")


def jurist_key(text: str) -> str | None:
    import re
    k = re.sub(r"[^a-z]+", " ", text.lower().replace("ſ", "s")).strip()
    if len(k) < 2:
        return None
    words = k.split()
    while len(words) > 1 and words[0] in _JURIST_PREFIXES:
        words = words[1:]
    k = " ".join(words)
    for pat, name in JURIST_SIGLA:
        if re.match(pat, k):
            return name
    return None


# Citation spans that only the secondary run (the Flair tagger) found are not published: in the sample check of
# 2026-10-28 (150 consilia, output/qa/cit_sample/) 383 of 480 of them were prose fragments, against 1 of 4,194
# spans both runs found. Entities other than citations are kept as before.
KEEP_SECONDARY_ONLY_CIT = False


def load_resolutions() -> dict:
    """eval/reviews/citation_resolutions.jsonl: loci for citations the rules cannot resolve, found by
    language models and accepted only when two models agree ({doc, text, key, status, models});
    keyed by (doc, printed text)."""
    f = ROOT / "eval" / "reviews" / "citation_resolutions.jsonl"
    out = {}
    if f.exists():
        for l in f.open(encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r.get("key") and r.get("status") in ("verified_by_model", "verified_by_editor"):
                    out[(r["doc"], r["text"])] = r
    return out


def collect_spans(primary: Path, secondary: Path | None, units_path: Path) -> tuple[dict, dict, dict]:
    """Layer C: spans of the primary run (an LLM pass with normalised `ref`s) per print, marked
    verified_by_model when the secondary run (the Flair tagger) has an overlapping span of the same
    label; spans only in the secondary run are added as machine_tagger (citations excepted, see
    KEEP_SECONDARY_ONLY_CIT). Citations are resolved by rule
    (resolve_citations.py), a lex- or capitulum-only citation directly after a resolved one inherits its
    title (`rv: ctx`), and loci from eval/reviews/citation_resolutions.jsonl fill the rest (`rv: llm`).
    Returns (shards {vol: {doc: [spans]}}, stats, models)."""
    sys.path.insert(0, str(ROOT / "src" / "annotate"))
    from resolve_citations import resolve
    import re
    units_meta = {}
    for l in units_path.open(encoding="utf-8"):
        u = json.loads(l)
        units_meta[u["unit"]] = u
    cons = json.loads((ROOT / "output" / "consilia.json").read_text(encoding="utf-8"))["consilia"]
    resolutions = load_resolutions()
    p1 = load(primary, "unit")
    p2 = load(secondary, "unit") if secondary else {}
    shards: dict[str, dict] = {}
    n_ver = n_mach = n_sec = n_lost = n_moved = n_stale = n_ctx = n_llm = n_drop = 0
    GAP = re.compile(r"^[\s.,:;]*(?:et|vel|&|ac|item)?[\s.,:;]*$", re.I)

    def resolve_cit(sp: dict, prev_key, prev_end, sec_text: str, doc: str):
        """fills rk / rc / rv of a citation span; returns the key to carry on (or None)"""
        nonlocal n_ctx, n_llm
        adjacent = False
        if prev_end is not None and prev_end[0] == (sp["s"], sp["o"]) and sec_text:
            gap = sec_text[sp["o"] + prev_end[1]:sp["o"] + sp["a"]]
            adjacent = len(gap) <= 8 and bool(GAP.match(gap))
        rr = resolve(sp["t"], prev_key, adjacent=adjacent)
        if rr["status"] == "resolved" and rr.get("key"):
            sp["rk"] = rr["key"]
            if rr.get("via") == "adjacent":
                sp["rv"] = "ctx"; n_ctx += 1
            return rr["key"]
        if rr["status"] == "ambiguous":
            sp["rc"] = rr["candidates"][:4]
        r = resolutions.get((doc, sp["t"]))
        if r:
            sp["rk"] = r["key"]; sp["rv"] = "llm" if r["status"] == "verified_by_model" else "editor"; sp.pop("rc", None); n_llm += 1
            return r["key"]
        return None
    for unit, rec in p1.items():
        um = units_meta.get(unit)
        if not um:
            continue
        doc, vol = doc_id(um["doc"]), um["volume"]
        if doc not in cons:            # unit of a consilium that no longer exists after a rebuild (stale annotation run)
            n_stale += 1; continue
        def find_in(sec: str, text: str, exp: int):
            """occurrence of text in sec nearest to exp: exact, else loose; (start, text as it stands) or None"""
            best, bd = None, 10 ** 9
            i = sec.find(text)
            while i >= 0:
                if abs(i - exp) < bd:
                    best, bd = i, abs(i - exp)
                i = sec.find(text, i + 1)
            if best is not None:
                return best, text
            key, pos = loose_index(sec)
            k = loose(text)
            if len(k) < 3:
                return None
            j = key.find(k)
            while j >= 0:
                if abs(pos[j] - exp) < bd:
                    best, bd = j, abs(pos[j] - exp)
                j = key.find(k, j + 1)
            if best is None:
                return None
            a, b = pos[best], pos[best + len(k) - 1] + 1
            return a, sec[a:b]

        c = cons.get(doc) or {}
        sect_text = ((c.get("casus") if um["kind"] == "casus" else c.get("body")) or [None] * (um["section"] + 1))
        sect_text = sect_text[um["section"]] if um["section"] < len(sect_text) else None
        off = sect_text.find(um["la"]) if sect_text else -1      # window offset inside the section
        if off < 0 and sect_text:
            # the unit text predates later corrections: locate the window by its opening words, loosely
            head = um["la"][:80].strip()
            got = find_in(sect_text, head, 0) if len(head) >= 20 else None
            if got is None and len(head) >= 40:
                got = find_in(sect_text, head[:40], 0)
            off = got[0] if got else (0 if um.get("n_windows", 1) == 1 else -1)
        elif off < 0:
            off = -1
        sec_spans = [(s["start"], s["end"], s["label"]) for s in (p2.get(unit, {}).get("spans") or [])]
        out = []
        seen = []
        prev_key = None
        prev_end = None            # ((section, window offset), end of the last resolved citation) for adjacency

        all_secs = ((c.get("casus") if um["kind"] == "casus" else c.get("body")) or [])

        def anchor(start, text):
            """the span re-located in the current text (the annotated unit text may predate later
            corrections of the transcription): (section, offset, start, text as it now stands), first in
            its own section (exact, then loose: accents, spacing, punctuation, brackets, q;, u/v, i/j/l/1
            ignored), then in the two sections before and after it (the † sections can shift when the
            corrected text gains or loses a †); None when it no longer occurs"""
            if sect_text is None or off < 0:
                return um["section"], off, start, text
            got = find_in(sect_text, text, off + start)
            if got:
                return um["section"], off, got[0] - off, got[1]
            for d in (-1, 1, -2, 2):
                j = um["section"] + d
                if 0 <= j < len(all_secs) and all_secs[j]:
                    got = find_in(all_secs[j], text, 0 if d > 0 else len(all_secs[j]))
                    if got:
                        return j, 0, got[0], got[1]
            return None

        for s in rec.get("spans") or []:
            ok = any(l == s["label"] and a < s["end"] and s["start"] < b for a, b, l in sec_spans)
            status = "verified_by_model" if ok else ("machine" if not p2 else "disputed")
            got = anchor(s["start"], s["text"])
            if got is None:
                n_lost += 1; continue
            sec_i, off_i, st_, txt = got
            n_moved += st_ != s["start"] or sec_i != um["section"]
            s = dict(s, text=txt)
            n_ver += ok; n_mach += (not ok)
            sp = {"k": um["kind"], "s": sec_i, "o": off_i, "a": st_, "b": st_ + len(s["text"]), "t": s["text"], "l": s["label"], "st": status}
            # the model's normalised reference (s["ref"]) is not published (a check of 40 citations found its
            # work-and-title identification wrong in about two thirds); citations are resolved by rule instead
            if s["label"].startswith("CIT_"):
                k = resolve_cit(sp, prev_key, prev_end, all_secs[sec_i] if sec_i < len(all_secs) and all_secs[sec_i] else "", doc)
                if k:
                    prev_key = k; prev_end = ((sec_i, off_i), sp["b"])
            if s["label"] == "JURIST":
                sp["jk"] = jurist_key(s["text"])         # index key (jurist name)
            out.append(sp)
            seen.append((s["start"], s["end"], s["label"]))
        for a, b, l in sec_spans:
            if not any(l == l2 and a < b2 and a2 < b for a2, b2, l2 in seen):
                if l.startswith("CIT_") and not KEEP_SECONDARY_ONLY_CIT:
                    n_drop += 1; continue      # citations found only by the tagger: 80 % false in the 2026-10-28 sample
                t = um["la"][a:b]
                got = anchor(a, t) if t.strip() else None
                if got is None:
                    n_lost += 1; continue
                sec_i, off_i, a, t = got
                b = a + len(t)
                sp = {"k": um["kind"], "s": sec_i, "o": off_i, "a": a, "b": b, "t": t, "l": l, "st": "machine_tagger"}
                if l == "JURIST":
                    sp["jk"] = jurist_key(t)
                if l.startswith("CIT_"):
                    resolve_cit(sp, None, None, "", doc)
                out.append(sp); n_sec += 1
        if out:
            shards.setdefault(vol, {}).setdefault(doc, []).extend(out)
    stats = {"verified_by_model": n_ver, "disputed_or_machine": n_mach, "machine_tagger": n_sec, "re_anchored": n_moved, "dropped_text_not_found": n_lost,
             "skipped_stale_consilium_units": n_stale, "resolved_by_context": n_ctx, "resolved_by_models": n_llm,
             "dropped_tagger_only_citations": n_drop}
    models = {"primary": next(iter(p1.values()))["model"] if p1 else None, "secondary": next(iter(p2.values()))["model"] if p2 else None}
    return shards, stats, models


def build_index(shards: dict, stats: dict, models: dict) -> dict:
    sys.path.insert(0, str(ROOT / "src" / "annotate"))
    from resolve_citations import TABLE
    sc = yaml.safe_load((ROOT / "src" / "codebooks" / "spans.yaml").read_text(encoding="utf-8"))
    index: dict[str, dict] = {}
    jurists: dict[str, dict] = {}
    for vol, docs in shards.items():
        for doc, out in docs.items():
            for sp in out:
                if sp.get("rk"):
                    key = sp["rk"]
                    e = index.setdefault(key, {"label": sp["l"], "docs": {}, "forms": {}})
                    e["docs"][doc] = e["docs"].get(doc, 0) + 1
                    e["forms"][sp["t"]] = e["forms"].get(sp["t"], 0) + 1
                elif sp["l"] == "JURIST":
                    key = sp.get("jk")
                    if key:
                        e = jurists.setdefault(key, {"docs": {}, "forms": {}})
                        e["docs"][doc] = e["docs"].get(doc, 0) + 1
                        e["forms"][sp["t"]] = e["forms"].get(sp["t"], 0) + 1
    def title_of(key: str) -> str:
        import re as _re
        m = _re.match(r"^(D|C|Inst|Clem|X|VI|LF)\.? (\d+)\.(\d+)$", key)
        if m:
            return TABLE[m.group(1)]["books"].get(m.group(2), {}).get(m.group(3), "")
        m = _re.match(r"^Nov\. (\d+)$", key)
        if m:
            for coll in TABLE["Nov"]["collationes"].values():
                for e in coll.values():
                    if str(e["nov"]) == m.group(1):
                        return e["title"]
        if key.startswith("Decr."):
            return "Decretum Gratiani"
        return ""

    def compact(table):
        return {k: {**({"label": v["label"], "title": title_of(k)} if "label" in v else {}), "n": sum(v["docs"].values()), "docs": sorted(v["docs"]),
                    "forms": [f for f, _ in sorted(v["forms"].items(), key=lambda kv: -kv[1])[:4]]} for k, v in table.items() if len(v["docs"]) >= 1}
    idx = {"scheme_version": sc["version"], "labels": [{"id": l["id"], "en": l["en"], "de": l.get("de", "")} for l in sc["labels"]],
           "models": models, "statuses": stats,
           "refs": compact(index), "jurists": compact(jurists),
           "note": "Citations and entities found by a language model in the Latin text and checked against a Flair tagger trained on the same scheme; machine annotations throughout. Cited works are resolved to work and title by rule (src/annotate/resolve_citations.py, title tables in src/codebooks/corpus_titles.json); a lex- or capitulum-only citation directly after a resolved one inherits its title (rv: ctx); loci that two language models agreed on fill part of the rest (rv: llm); unresolved or ambiguous citations are listed as printed only."}
    return idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--body", nargs="+", type=Path)
    ap.add_argument("--subjects", type=Path)
    ap.add_argument("--spans", nargs="+", type=Path, help="primary (LLM) run, optionally the Flair run second")
    ap.add_argument("--patch", nargs="+", type=Path, help="a later span run (LLM, optionally Flair second) on re-annotated consilia: replaces their spans")
    ap.add_argument("--patch-units", type=Path, help="units file of the patch run (default output/units.jsonl)")
    args = ap.parse_args()
    SITE.mkdir(parents=True, exist_ok=True)
    if args.body:
        d = export_body(args.body)
        (SITE / "intimate_body.json").write_text(json.dumps(d, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        st = defaultdict(int)
        for u in d["units"]:
            st[u["status"]] += 1
        print(f"intimate_body.json: {len(d['units'])} annotations on {len({u['doc'] for u in d['units']})} consilia; {dict(st)}")
    if args.subjects:
        d = export_subjects(args.subjects)
        (SITE / "subjects.json").write_text(json.dumps(d, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"subjects.json: {len(d['docs'])} consilia")
    if args.spans:
        shards, stats, models = collect_spans(args.spans[0], args.spans[1] if len(args.spans) > 1 else None, ROOT / "output" / "units.jsonl")
        if args.patch:
            ps, pst, pm = collect_spans(args.patch[0], args.patch[1] if len(args.patch) > 1 else None, args.patch_units or ROOT / "output" / "units.jsonl")
            n_docs = 0
            for vol, docs in ps.items():
                for doc, spans in docs.items():
                    shards.setdefault(vol, {})[doc] = spans; n_docs += 1
            stats["patched_consilia"] = n_docs; stats["patch"] = pst; models["patch"] = pm
        idx = build_index(shards, stats, models)
        (SITE / "references_index.json").write_text(json.dumps(idx, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        d = SITE / "references"; d.mkdir(exist_ok=True)
        for vol, docs in shards.items():
            (d / f"{vol}.json").write_text(json.dumps(docs, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"references_index.json: {len(idx['refs'])} works and titles, {len(idx['jurists'])} jurist names; statuses {idx['statuses']}; shards: " +
              ", ".join(f"{v} {sum(len(x) for x in docs.values())} spans" for v, docs in shards.items()))
    print("now run: python src/build_site_data.py")


if __name__ == "__main__":
    main()
