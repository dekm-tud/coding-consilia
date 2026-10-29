#!/usr/bin/env python3
"""
Build the website bibliography (site-data/bibliography.json and bibliography.ris) from the Zotero collection
"Consilia", with researched additions and the controlled subject tags.

Sources, combined in this order:
  Zotero              the items of the collection that carry consilia_legal, consilia_med or consilia_general, plus
                      every item cited in the footnotes of the essay pages; read from Zotero's local API (Zotero 7,
                      "Allow other applications on this computer to communicate with Zotero") or from a saved
                      export (--from-file, the JSON of /api/users/0/collections/<key>/items)
  overrides           src/codebooks/bibliography_overrides.json: fields found by research for incomplete records
                      ({"KEY": {"editors": [...], "publisher": ..., "source": URL}}), duplicates to merge
                      ({"merge": {"DUPKEY": "KEEPKEY"}}) and items that are not in Zotero ({"new": {"new:id": {...}}})
  tags                src/codebooks/bibliography_item_tags.json ({"KEY": ["tag-id", ...]}), ids from
                      src/codebooks/bibliography_tags.yaml
  citations           src/codebooks/bibliography_citations.json: which footnote of which essay page cites which item

    python tools/build_bibliography.py                    # live from the local Zotero
    python tools/build_bibliography.py --from-file items.json
The automatic Zotero tags (auto:, field:) are not used; Zotero itself is not changed.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CB = ROOT / "src" / "codebooks"
SITE = ROOT / "site-data"
COLLECTION = "9R2B9N8U"                       # "Consilia" in the editor's Zotero library
SCOPE_TAGS = {"consilia_legal", "consilia_med", "consilia_general", "consilia", "consilia extravagantia"}
SKIP_TYPES = {"attachment", "note", "annotation"}
SNAP = ROOT / "output" / "bibliography_zotero.json"       # last Zotero export used (local, not published)


def zotero_items(api: str, include: list[str]) -> list[dict]:
    """the collection "Consilia", every item of the library that carries one of SCOPE_TAGS (wherever it is filed),
    and the items listed under "include" in the overrides"""
    def pages(path: str) -> list[dict]:
        out, start = [], 0
        while True:
            sep = "&" if "?" in path else "?"
            batch = json.load(urllib.request.urlopen(f"{api}/api/users/0/{path}{sep}limit=100&start={start}&format=json", timeout=60))
            if not batch:
                return out
            out += batch
            start += 100
    items = {i["key"]: i for i in pages(f"collections/{COLLECTION}/items")}
    for tag in sorted(SCOPE_TAGS):
        items.update({i["key"]: i for i in pages(f"items?tag={urllib.parse.quote(tag)}")})
    for k in include:
        if k not in items:
            items[k] = json.load(urllib.request.urlopen(f"{api}/api/users/0/items/{k}?format=json", timeout=60))
    return list(items.values())


def load_yaml_tags() -> dict:
    import yaml
    return yaml.safe_load((CB / "bibliography_tags.yaml").read_text(encoding="utf-8"))


def year_of(date: str) -> str:
    m = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", date or "")
    return m.group(1) if m else ""


PARTICLES = {"de", "da", "di", "del", "della", "dei", "degli", "van", "von", "der", "den", "du", "le", "la", "y", "zu"}


def cap(s: str) -> str:
    """a name part typed in lower case by mistake ("ingrid") gets its capital, particles stay as they are"""
    return " ".join(w[:1].upper() + w[1:] if w and w[0].islower() and w not in PARTICLES else w for w in (s or "").split(" "))


def person(c: dict) -> dict:
    if "name" in c:
        return {"name": c["name"]}
    return {"last": cap(c.get("lastName", "")), "first": cap(c.get("firstName", ""))}


def split_name(s: str) -> dict:
    last, _, first = s.partition(",")
    return {"last": last.strip(), "first": first.strip()} if first else {"name": s.strip()}


# journal names as the journals give them, replacing the subtitles, typos and imported citation residue of Zotero
JOURNALS = {
    "Zeitschrift für historische Forschung: Vierteljahresschrift zur Erforschung des Spätmittelalters und der frühen Neuzeit": "Zeitschrift für Historische Forschung",
    "Zeitschrift der Savigny-Stiftung für Rechtsgeschichte: Kanonistische Abteilung": "Zeitschrift der Savigny-Stiftung für Rechtsgeschichte. Kanonistische Abteilung",
    "Zeitschrift der Savigny-Stiftung für Rechtsgeschichte, Kanonistische Abteilung": "Zeitschrift der Savigny-Stiftung für Rechtsgeschichte. Kanonistische Abteilung",
    "Zeitschrift der Savigny-Stiftung für Rechtsgeschichte, Germanistische Abteilung": "Zeitschrift der Savigny-Stiftung für Rechtsgeschichte. Germanistische Abteilung",
    "Ius commune: Veröffentlichungen des Max-Planck-Instituts für europäische Rechtsgeschichte, Frankfurt am Main": "Ius commune",
    "Ius Commune": "Ius commune",
    "Historia et ius. Rivista di storia giuridica dell'età medievale e moderna": "Historia et ius",
    "Viator: Medieval and Renaissance Studies": "Viator",
    "Studien und Mitteilungen aus dem Benediktiner-und der Cistercensierordens": "Studien und Mitteilungen aus dem Benediktiner- und dem Cistercienser-Orden",
    "Clio@Themis, revue électronique d' histoire de droit": "Clio@Themis",
    "Studia historica: Historia moderna - Discursos teológicos y cuestiones económicas: siglos XVI-XVII": "Studia historica. Historia moderna",
    "Bibliothèque de l'Ëcole des Chartes": "Bibliothèque de l'École des chartes",
    ":Revue historique de droit français et étranger": "Revue historique de droit français et étranger",
    "genesis. Rivista della Società Iatliana dele Storiche": "Genesis. Rivista della Società italiana delle storiche",
    "Renassiance Quarterly": "Renaissance Quarterly",
    "Micrologus: Natura, scienze e società medievali / Nature, Sciences and Medieval Societies": "Micrologus",
    "Tijdschrift voor rechtsgeschiedenis": "Tijdschrift voor Rechtsgeschiedenis",
    "Tijdschrift voor Rechtsgeschiedenis vol. 52 (1984) p. 399-411": "Tijdschrift voor Rechtsgeschiedenis",
    "The American journal of legal history": "American Journal of Legal History",
    "Rivista di Storia del Diritto Italiano": "Rivista di storia del diritto italiano",
    "Thought. A review of culture and ideas": "Thought",
    "Rechtsgeschichte / Legal History: Zeitschrift des Max-Planck-Instituts für europäische Rechtsgeschichte": "Rechtsgeschichte – Legal History",
    "Medieval Encounters: Jewish, Christian and Muslim Culture in Confluence and Dialogue": "Medieval Encounters",
}


def clean_title(t: str) -> str:
    """remove what catalogue imports leave in Zotero titles: a responsibility statement ("/ CAVALLAR (Osvaldo)"),
    the closing period (the citation adds its own punctuation) and doubled spaces"""
    t = re.sub(r"\s+/\s+[A-ZÀ-Ý][A-ZÀ-Ý' -]+\s*\([^)]*\)\s*$", "", t.strip())
    t = re.sub(r"\s*\.{2,}$", "", t)
    t = re.sub(r"(?<![A-Z.])\.$", "", t)                  # not after an abbreviation such as "J." or "..."
    return re.sub(r"\s{2,}", " ", t).strip()


def clean_journal(c: str) -> str:
    c = JOURNALS.get(c.strip(), c.strip())
    if c.startswith(("Parergon:", "Bollettino della Società di studi valdesi")):
        c = c.split(":")[0] if c.startswith("Parergon") else c.split(" - ")[0]
    return c


def entry(d: dict, key: str) -> dict:
    creators = d.get("creators") or []
    e = {
        "id": key, "type": d.get("itemType"),
        "authors": [person(c) for c in creators if c.get("creatorType") in ("author", "contributor", "bookAuthor")],
        "editors": [person(c) for c in creators if c.get("creatorType") in ("editor", "seriesEditor")],
        "title": clean_title(d.get("title") or ""),
        "container": (d.get("bookTitle") or d.get("publicationTitle") or d.get("dictionaryTitle") or d.get("encyclopediaTitle")
                      or d.get("proceedingsTitle") or d.get("websiteTitle") or "").strip(),
        "year": year_of(d.get("date", "")),
        "kind": d.get("kind", "literature"),          # literature, or a source (manuscript, early print, catalogue, website)
    }
    if d.get("date") and not re.fullmatch(r"\s*\d{4}(-\d{1,2}(-\d{1,2})?)?\s*", d["date"]) and e["kind"] == "source":
        e["dateText"] = d["date"].strip()           # "22.II.1475", "14th century": kept as given for sources
    if e["type"] == "journalArticle" and e["container"]:
        e["container"] = clean_journal(e["container"])
    elif e["container"]:
        e["container"] = re.sub(r"\s+/\s+", ". ", e["container"])
    for k in ("volume", "issue", "pages", "series", "seriesNumber", "place", "publisher", "edition",
              "university", "thesisType", "DOI", "url", "ISBN", "numberOfVolumes", "archive", "archiveLocation",
              "institution", "reportType"):
        if d.get(k):
            e[k.lower() if k in ("DOI", "ISBN") else k] = str(d[k]).strip()
    if (d.get("university") or d.get("institution")) and not e.get("publisher"):
        e["publisher"] = d.get("university") or d.get("institution")
    if e["kind"] == "source" and re.search(r"\b(GW|ISTC|HC)\b", d.get("note") or ""):
        e["identifiers"] = d["note"].strip()        # GW, ISTC and similar numbers of early prints
    return {k: v for k, v in e.items() if v not in ("", [], None)}


def apply_override(e: dict, o: dict) -> list[str]:
    """fields found by research or corrected: authors/editors replace the lists (an empty list removes them, e.g.
    when Zotero has the editors of a volume as its authors), title_replace [old, new] mends a typo in the title,
    every other key sets that field"""
    changed = []
    for k, v in o.items():
        if k in ("source", "note") or (v in ("", None)) or (v == [] and k not in ("authors", "editors")):
            continue
        if k in ("authors", "editors"):
            e[k] = [split_name(x) for x in v]
        elif k == "title_replace":
            e["title"] = e["title"].replace(v[0], v[1])
        else:
            e[k] = str(v)
        changed.append(k)
    for k in ("authors", "editors"):
        if k in e and not e[k]:
            del e[k]
    if changed and o.get("source"):
        e.setdefault("research", []).append({"fields": changed, "source": o["source"]})
    return changed


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", s)


def dedupe(entries: list[dict], in_collection: set[str]) -> tuple[list[dict], dict]:
    """the same work entered twice: same DOI, or same first author with the same pages (or year) and the same
    beginning of the title. The record kept is the one filed in the collection, then the fuller one; it takes the
    fields it lacks and the tags of the other."""
    def first(e):
        p = (e.get("authors") or e.get("editors") or [{}])[0]
        return norm(p.get("last") or p.get("name") or "")
    groups: list[list[dict]] = []
    for e in entries:
        for g in groups:
            o = g[0]
            same_doi = e.get("doi") and o.get("doi") and e["doi"].lower() == o["doi"].lower()
            same = e.get("kind") != "source" and o.get("kind") != "source" and \
                first(e) == first(o) and norm(e["title"])[:25] == norm(o["title"])[:25] and \
                ((e.get("pages") and norm(e.get("pages")) == norm(o.get("pages"))) or e.get("year") == o.get("year"))
            if same_doi or same:
                g.append(e)
                break
        else:
            groups.append([e])
    out, merged = [], {}
    for g in groups:
        g.sort(key=lambda e: (e["id"] not in in_collection, -len(e)))
        keep = g[0]
        for o in g[1:]:
            for k, v in o.items():
                if k not in keep and k not in ("id", "research", "cited"):
                    keep[k] = v
            keep["tags"] = list(dict.fromkeys(keep.get("tags", []) + o.get("tags", [])))
            merged[o["id"]] = keep["id"]
        out.append(keep)
    return out, merged


def ris(e: dict) -> str:
    ty = {"book": "BOOK", "bookSection": "CHAP", "journalArticle": "JOUR", "thesis": "THES",
          "conferencePaper": "CPAPER", "dictionaryEntry": "DICT", "webpage": "ELEC"}.get(e["type"], "GEN")
    out = [f"TY  - {ty}"]
    name = lambda p: p.get("name") or f"{p['last']}, {p['first']}".strip(", ")
    out += [f"AU  - {name(p)}" for p in e.get("authors", [])]
    out += [f"ED  - {name(p)}" for p in e.get("editors", [])]
    out.append(f"TI  - {e['title']}")
    if e.get("container"):
        out.append(f"{'JO' if e['type'] == 'journalArticle' else 'T2'}  - {e['container']}")
    for tag, k in (("PY", "year"), ("VL", "volume"), ("IS", "issue"), ("CY", "place"), ("PB", "publisher"),
                   ("DO", "doi"), ("UR", "url"), ("SN", "isbn"), ("T3", "series")):
        if e.get(k):
            out.append(f"{tag}  - {e[k]}")
    if e.get("pages"):
        sp, _, ep = e["pages"].replace("–", "-").partition("-")
        out.append(f"SP  - {sp.strip()}")
        if ep:
            out.append(f"EP  - {ep.strip()}")
    out += [f"KW  - {t}" for t in e.get("tags", [])]
    out.append("ER  - ")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--api", default="http://localhost:23119")
    ap.add_argument("--from-file", type=Path)
    a = ap.parse_args()
    ov = json.loads((CB / "bibliography_overrides.json").read_text(encoding="utf-8")) if (CB / "bibliography_overrides.json").exists() else {}
    raw = json.loads(a.from_file.read_text(encoding="utf-8")) if a.from_file else zotero_items(a.api, ov.get("include", []))
    if not a.from_file:
        SNAP.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    by_key = {i["key"]: i["data"] for i in raw if i["data"]["itemType"] not in SKIP_TYPES}
    in_collection = {k for k, d in by_key.items() if COLLECTION in (d.get("collections") or [])}
    tags = json.loads((CB / "bibliography_item_tags.json").read_text(encoding="utf-8")) if (CB / "bibliography_item_tags.json").exists() else {}
    cites = json.loads((CB / "bibliography_citations.json").read_text(encoding="utf-8")) if (CB / "bibliography_citations.json").exists() else {"citations": []}
    scheme = load_yaml_tags()
    known = {t["id"] for t in scheme["tags"]}
    merge = ov.get("merge", {})

    cited = {}
    for c in cites.get("citations", []):
        k = merge.get(c.get("key"), c.get("key"))
        if k:
            cited.setdefault(k, set()).add((c["page"], c["note_id"]))
    keys = {k for k, d in by_key.items() if {t["tag"] for t in d.get("tags", [])} & SCOPE_TAGS}
    keys |= {k for k in cited if k in by_key} | {k for k in ov.get("include", []) if k in by_key}
    keys -= set(merge)

    entries, researched = [], 0
    for k in sorted(keys):
        e = entry(by_key[k], k)
        if apply_override(e, ov.get(k, {})):
            researched += 1
        e["tags"] = [t for t in tags.get(k, []) if t in known]
        entries.append(e)
    for k, d in ov.get("new", {}).items():
        e = entry(d, k.replace(":", "-"))           # anchor-friendly id: new-dean2007crime
        e["research"] = [{"fields": ["all"], "source": d.get("source", "")}]
        e["tags"] = [t for t in tags.get(k, []) if t in known]
        entries.append(e)
    entries, merged = dedupe(entries, in_collection)
    back = {}
    for dup, keep in merged.items():
        back.setdefault(keep, []).append(dup)
    for e in entries:
        refs = set()
        for k in [e["id"]] + back.get(e["id"], []):
            refs |= cited.get(k.replace("new-", "new:", 1) if k.startswith("new-") else k, set())
        if refs:
            e["cited"] = [{"page": p, "note": n} for p, n in sorted(refs)]
    sort_name = lambda e: ((e.get("authors") or e.get("editors") or [{"name": e["title"]}])[0].get("last")
                           or (e.get("authors") or e.get("editors") or [{"name": e["title"]}])[0].get("name", "")).lower()
    entries.sort(key=lambda e: (sort_name(e), e.get("year", ""), e["title"].lower()))
    used = {t for e in entries for t in e["tags"]}
    out = {"_meta": {"source": "Zotero collection Consilia of the editor, with researched additions",
                     "n": len(entries), "researched": researched, "new": len(ov.get("new", {}))},
           "facets": scheme["facets"], "merged_duplicates": merged, "tags": [t for t in scheme["tags"] if t["id"] in used], "entries": entries}
    (SITE / "bibliography.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (SITE / "bibliography.ris").write_text("\n\n".join(ris(e) for e in entries) + "\n", encoding="utf-8")
    print(f"site-data/bibliography.json: {len(entries)} entries ({researched} completed by research, "
          f"{len(ov.get('new', {}))} not in Zotero), {len(used)} tags, {len(cited)} cited in footnotes")


if __name__ == "__main__":
    main()
