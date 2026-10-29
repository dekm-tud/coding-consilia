"""
TEI P5 export, one file per print, for archiving (Zotero packages) and reuse.

Structure per consilium:
  <div type="consilium" xml:id=… n=…>
    <div type="casus">  <p> per section
    <head>              title as printed
    <div type="argumentum"> <list><item n=…>
    <div type="textus"> <p n=…> per section (the print's † sections; marginal numbers in @n)
    <note type="keywords" resp="#machine"> machine keywords
    <note type="subjects"> / <note type="intimate_body"> from site-data/annotations when present
  Inline, from site-data/annotations/references/<print>.json (Layer C, machine annotation):
    <ref type="citation" subtype="civ|can|feud|other" cRef="D. 2.14">ff. de pac. l. tale pactum</ref>
    <rs type="jurist" key="Bartolus de Saxoferrato">Bar.</rs>, <persName>, <placeName>, <orgName>,
    <date>, <rs type="statute">; cert="low" marks a span found by one of the two systems only.
  <pb n="0003" facs="…"/> before the first section that starts on a new page (facsimile URL:
  IIIF canvas or local derivative) — page breaks are at section granularity (the pipeline keeps
  page provenance per consilium, not per line).
Translations go into a second <text> per language only with --translations (large files).

    python src/export_tei.py                    # -> dist/tei/<print>.xml for every print
    python src/export_tei.py --print Baldo_Cons_Print_Venice_1575_v1 --translations
    python src/export_tei.py --doi 10.5281/zenodo.XXXXXXX     # before a Zenodo deposit
The header records editor (ORCID, GND, affiliation), responsibilities (recognition models from the
PageXML metadata, translation, annotation), licence, identifiers (site, repository, DOI), the print
and its digitised copies (site-data/editions.json), editorial declarations and the classification
schemes; a <facsimile> lists every page with its IIIF canvas or image URL, pb/@facs points to it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from xml.sax.saxutils import escape as _xml_escape
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_site_data import display_title  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "tei"
SITE = ROOT / "site-data"


_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def x(s) -> str:            # escape + drop control characters (not allowed in XML 1.0)
    return _xml_escape(_CTRL.sub("", str(s)))


def attr(s) -> str:
    return _xml_escape(_CTRL.sub("", str(s)), {'"': "&quot;"})


EDITOR = {"name": "Michael Schonhardt", "orcid": "https://orcid.org/0000-0002-2750-1900",
          "gnd": "https://d-nb.info/gnd/1048369242", "affiliation": "Technische Universität Darmstadt",
          "email": "michael.schonhardt@tu-darmstadt.de"}
SITE_URL = "https://dekm-tud.github.io/coding-consilia/"
REPO_URL = "https://github.com/dekm-tud/coding-consilia"
LICENCE_TEXT = "https://creativecommons.org/licenses/by/4.0/"


def header(print_id: str, author: dict, lst: list, revision: str, edition: dict, iiif: dict,
           models: list[str], doi: str, today: str) -> str:
    """teiHeader following the usual conventions for digital editions of early printed sources:
    full titleStmt with responsibilities, publicationStmt with licence and identifiers, sourceDesc
    as biblStruct of the print and the digitised copy, encodingDesc with editorialDecl and the
    classification schemes, profileDesc with languages, revisionDesc with the data revision."""
    n = len(lst)
    imprint = edition.get("imprint", "")                       # "Venice: Girolamo Polo, 1575" or "Venice, 1571 (…)"
    city = re.split(r"[:,(]", imprint)[0].strip()
    ym = re.search(r"\b(1[4-6]\d\d)\b", imprint)
    first_year = ym.group(1) if ym else ""
    printer = imprint.split(":")[1].split(",")[0].strip() if ":" in imprint else ""
    title_src = edition.get("title") or f"Consilia ({print_id})"
    vol = edition.get("volume", "")
    copies = [{"label": edition.get("copy", ""), "url": edition.get("catalogue", "")}] + list(edition.get("other_copies") or [])
    pages = sorted({pg["page"].replace(".xml", "") for c in lst for pg in (c.get("sources") or [])})
    tr_langs = sorted({l for c in lst for l, t in (c.get("translations") or {}).items() if t and t.get("body")})
    model_note = "; ".join(models) if models else "see sourceDesc"
    resp_rows = [
        ("Digital edition: concept, data model, segmentation, quality control", f'<persName ref="{EDITOR["orcid"]}">{EDITOR["name"]}</persName>'),
        ("Layout analysis, corrected by hand, and text recognition (PyLaia, Transkribus)" if models else "Column transcription (Gemini vision language model, column by column from the page images)",
         f"<orgName>READ-COOP, Transkribus platform</orgName><note>{x(model_note)}</note>" if models else "<name>Google Gemini 3.8 Flash, prompted transcription</name>"),
        ("Second reading: column transcription from the page images" if models else "", "<name>Google Gemini 3.8 Flash, prompted transcription</name>"),
        ("Third reading: line recognition with a kraken model fine-tuned on the lines where the first two readings agree" if models else "",
         "<name>kraken 7.1 (base model McCATMuS, doi:10.5281/zenodo.13788177)</name>"),
        ("Establishment of the text from the three readings, gold standard and review of disputed lines" if models else "", f'<persName ref="{EDITOR["orcid"]}">{EDITOR["name"]}</persName>'),
        ("Machine translation (reading aid, unreviewed)" if tr_langs else "", "<name>large language model, see encodingDesc</name>"),
        ("Automatic keywords and controlled subject layers (unreviewed, see encodingDesc)", "<name>Consilia annotation pipeline</name>"),
    ]
    resp_xml = "".join(f"<respStmt><resp>{x(r)}</resp>{who}</respStmt>" for r, who in resp_rows if r)
    copies_xml = "".join(
        f'<bibl type="digital_copy">{x(c.get("label", ""))}{f' <ref target="{attr(c["url"])}">{x(c["url"])}</ref>' if c.get("url") else ""}</bibl>'
        for c in copies if c.get("label") or c.get("url"))
    iiif_xml = f'<bibl type="iiif_manifest"><ref target="{attr(iiif["manifest"])}">{x(iiif["manifest"])}</ref></bibl>' if iiif.get("manifest") else ""
    return f"""<teiHeader>
  <fileDesc>
    <titleStmt>
      <title type="main">{x(author.get('name', ''))}: {x(title_src)}</title>
      <title type="sub">Digital edition of the printed text ({x(city)} {x(first_year)}; {x(imprint)}), structured by consilium, with facsimile links{', translations' if tr_langs else ''} and subject access</title>
      <title type="short">{x(print_id)}</title>
      <author><persName ref="http://viaf.org/viaf/{x(author.get('viaf', ''))}">{x(author.get('name', ''))}</persName></author>
      <editor><persName ref="{EDITOR['orcid']}"><forename>Michael</forename> <surname>Schonhardt</surname></persName><affiliation>{x(EDITOR['affiliation'])}</affiliation><email>{x(EDITOR['email'])}</email><idno type="GND">{x(EDITOR['gnd'])}</idno></editor>
      {resp_xml}
      <respStmt xml:id="editor"><resp>Editorial decisions</resp><persName ref="{EDITOR['orcid']}">{x(EDITOR['name'])}</persName></respStmt>
      <respStmt xml:id="machine"><resp>Machine-made layers (transcription consensus, translation, keywords, classification)</resp><name>Consilia pipeline, see encodingDesc</name></respStmt>
    </titleStmt>
    <editionStmt><edition>Consilia. A portal for medieval legal opinions, beta. Data revision <idno type="data_revision">{x(revision)}</idno><date when="{x(today)}">{x(today)}</date></edition></editionStmt>
    <extent>{n} consilia, {len(pages)} pages of the print</extent>
    <publicationStmt>
      <publisher><orgName>{x(EDITOR['affiliation'])}</orgName></publisher>
      <authority><persName ref="{EDITOR['orcid']}">{x(EDITOR['name'])}</persName></authority>
      <pubPlace>Darmstadt</pubPlace>
      <date when="{x(today)}">{x(today)}</date>
      <idno type="URI">{x(SITE_URL)}</idno>
      <idno type="repository">{x(REPO_URL)}</idno>
      {f'<idno type="DOI">{x(doi)}</idno>' if doi else '<!-- Zenodo DOI: pass it to export_tei.py (option doi) once the deposit exists -->'}
      <availability status="free"><licence target="{LICENCE_TEXT}">The text data of this file are released under the Creative Commons Attribution 4.0 International licence (CC BY 4.0). The underlying print is in the public domain; the page images belong to the holding libraries named in the sourceDesc and are not part of this file.</licence></availability>
    </publicationStmt>
    <sourceDesc>
      <biblStruct type="book">
        <monogr>
          <author><persName ref="http://viaf.org/viaf/{x(author.get('viaf', ''))}">{x(author.get('name', ''))}</persName></author>
          <title level="m">{x(title_src)}</title>
          <imprint><pubPlace>{x(city)}</pubPlace>{f'<publisher>{x(printer)}</publisher>' if printer and printer != city else ''}<date when="{x(first_year[:4])}">{x(first_year)}</date></imprint>
          {f'<biblScope unit="volume">{x(vol)}</biblScope>' if vol[:1].isdigit() else ''}
        </monogr>
        {f'<note type="extent_of_edition">{x(vol)}</note>' if vol and not vol[:1].isdigit() else ''}
        <note type="copies">{copies_xml}{iiif_xml}</note>
        <note type="editorial">The text was established from the digitised copy named first; where a second digitisation is listed it served as the source of the page images or as a control copy. Folio and page references follow the sequence of the scan (pb/@n), the consilium numbers follow the print.</note>
      </biblStruct>
    </sourceDesc>
  </fileDesc>
  <encodingDesc>
    <projectDesc><p>Consilia (<ref target="{x(SITE_URL)}">{x(SITE_URL)}</ref>) collects the legal opinions of the medieval ius commune as structured, searchable texts with facsimile links. The portal starts from the early printed editions; manuscripts are to follow. The data, the code and the TEI exports are maintained in <ref target="{x(REPO_URL)}">{x(REPO_URL)}</ref>.</p></projectDesc>
    <editorialDecl>
      <p>Status: machine transcription, not proof-read. Every consilium is a div with the number of the print (n), subdivided into the statement of facts (casus) where the print has one, the heading, the numbered summarium (argumentum) and the text (textus), whose paragraphs follow the print's section marks; the marginal numbers of the print that key the text to the summarium are given as p/@n.</p>
      <correction><p>Baldus: every line was read three times, by PyLaia (Transkribus), by Gemini from the page images and by a kraken model trained on the lines where the first two agree. Where two readings agree on the wording, that wording stands; where none agree, the earlier text stands; a reading that leaves an abbreviation mark unexpanded does not count. Spacing and punctuation of a changed line follow the Gemini reading when it shares the wording. Marginal numbers are taken where Gemini and kraken agree. Measured against a representative sample of 300 lines corrected by hand against the images, 97.5 % of the lines are right in their wording and the character error rate is 0.07 %; about 520 lines were decided by hand. The text is otherwise not proof-read. Apparent misprints of the source are kept.</p></correction>
      <normalization><p>Baldus: abbreviations are expanded silently; round s for long s; accents of the print are not reproduced; u/v and i/j follow the reading; one space after punctuation and on each side of the section mark †; line-end hyphenation is resolved by a line-break model and the lines are joined into the print's paragraphs; the marginal numbers stand outside the text, in p/@n. Panormitanus (Gemini text): abbreviations are expanded in square brackets, long s and ligatures follow the print, line ends are joined in the same way.</p></normalization>
      <segmentation><p>Segmentation into consilia and into casus, heading, argumentum and text rests on line-level structure tags of the layout analysis and on rules described in the repository (src/segment_consilia.py, src/casus.py).</p></segmentation>
      <interpretation><p>Keywords (note/@type='keywords') were derived automatically from the machine translations. Controlled subject terms (note/@type='subjects') and the body-related codes (note/@type='intimate_body') come from a language-model classification against fixed codebooks, each assignment anchored in a Latin quotation; they are unreviewed working layers and carry resp='#machine'. For the body-related codes, ana names the role of the passage (topic, argument, case_fact, citation_only, formulaic, analogy) and cert='high' means that two independent model passes agree, cert='low' that only one pass found the code. The casus (div/@type='casus') is printed before the heading; it follows the head here because the schema requires it, marked rend='printed-before-head'.</p>
      <p>Citations and names in the Latin text are marked inline: ref/@type='citation' (subtype civ = Corpus iuris civilis, can = Corpus iuris canonici, feud = Libri feudorum, other), rs/@type='jurist', persName, placeName, orgName, date and rs/@type='statute'. They were found by a language model and confirmed by a sequence tagger trained on the same scheme; a span found by one of the two only carries cert='low'. A citation resolved by rule to a work and title carries it in @cRef in the modern numbering (D. = Digesta, C. = Codex, Inst., Nov., X = Liber Extra, VI, Clem., LF, Decr.); an ambiguous one lists the candidate titles in @n; a jurist's siglum resolved to a name carries it in @key. The resolution rules and tables are src/annotate/resolve_citations.py and src/codebooks/corpus_titles.json in the repository.</p></interpretation>
    </editorialDecl>
    <classDecl>
      <taxonomy xml:id="subjects"><desc>Controlled subject scheme of the portal, src/codebooks/subjects.yaml in the repository; the keys used in note/@type='subjects' refer to it.</desc></taxonomy>
      <taxonomy xml:id="intimate_body"><desc>Codebook for body, sexuality and kinship, src/codebooks/intimate_body.yaml in the repository.</desc></taxonomy>
    </classDecl>
    <appInfo>
      <application ident="consilia-pipeline" version="1" n="{x(revision)}"><label>Consilia build pipeline, data revision {x(revision)}</label><ref target="{x(REPO_URL)}">source code</ref></application>
      {"".join(f'<application ident="transkribus-{attr(m.split(" ")[0].lower())}-model-{attr(m.rsplit(" ", 1)[-1])}" version="{attr(m.split(" ")[1].rstrip(","))}"><label>{x(m)}</label></application>' for m in models)}
      {'<application ident="kraken" version="7.1.1"><label>kraken, model fine-tuned from McCATMuS on consensus lines of the Baldus prints</label></application>' if models else ""}
    </appInfo>
  </encodingDesc>
  <profileDesc>
    <langUsage><language ident="la">Latin (text of the print)</language>{"".join(f'<language ident="{l}">{ {"en": "English", "de": "German"}[l] } (machine translation)</language>' for l in tr_langs)}</langUsage>
    <textClass><keywords scheme="http://id.loc.gov/authorities/subjects"><term>Law, Medieval</term><term>Legal opinions</term><term>Roman law</term><term>Canon law</term></keywords></textClass>
    <creation><date when="{x(first_year[:4])}">{x(first_year)}</date><placeName>{x(city)}</placeName></creation>
  </profileDesc>
  <revisionDesc status="draft">
    <change when="{x(today)}" who="#machine">Exported from data revision {x(revision)}.</change>
  </revisionDesc>
</teiHeader>"""


def page_id(pg: str) -> str:
    return "f" + re.sub(r"[^A-Za-z0-9_]", "_", pg)


def recognition_models(print_dir: Path) -> list[str]:
    """Text recognition models named in the PageXML metadata (Transkribus Creator strings), e.g.
    'PyLaia 2.44.0, model 571589'; empty for prebuilt prints."""
    import xml.etree.ElementTree as ET
    found = set()
    for xml in sorted((print_dir / "page").glob("*.xml"))[:400]:
        try:
            root = ET.parse(xml).getroot()
        except ET.ParseError:
            continue
        for el in root.iter():
            if el.tag.endswith("Creator") and el.text and "model_id" in el.text:
                m = re.search(r"name=([^:@]+)[^:]*:version=([^:]+):model_id=(\d+)", el.text)
                if m:
                    found.add(f"{m.group(1)} {m.group(2)}, model {m.group(3)}")
    return sorted(found)


def facs_url(c: dict, page: dict, iiif: dict, images: dict) -> str:
    v = c["volume"]
    if v in iiif:
        cm = iiif[v].get("canvas_map") or {}
        idx = cm.get(page["page"]) if page["page"] in cm else int(page["page"].replace("page_", "")) - 1 + iiif[v].get("canvas_offset", 0)
        return f"{iiif[v]['manifest']}#canvas-index-{idx}"
    tpl = (images.get(v) or {}).get("path_template", "data/{author}/{print}/img/{stem}.avif")
    return tpl.replace("{author}", c["author_viaf"]).replace("{print}", v).replace("{stem}", page["page"])


_SPAN_TAGS = {"CIT_CIV": ("ref", 'type="citation" subtype="civ"'), "CIT_CAN": ("ref", 'type="citation" subtype="can"'),
              "CIT_FEUD": ("ref", 'type="citation" subtype="feud"'), "CIT_OTHER": ("ref", 'type="citation" subtype="other"'),
              "JURIST": ("rs", 'type="jurist"'), "PERSON": ("persName", ""), "PLACE": ("placeName", ""), "ORG": ("orgName", ""),
              "DATE": ("date", ""), "STATUTE": ("rs", 'type="statute"')}


def marked(text: str, spans: list[dict]) -> str:
    """The section text with the Layer C spans as inline elements. A span is used only when its text still
    stands at its offsets (o+a..o+b) in the current text; overlapping spans keep the first."""
    sp = []
    for y in spans:
        a, b = y["o"] + y["a"], y["o"] + y["b"]
        if y["l"] in _SPAN_TAGS and 0 <= a < b <= len(text) and text[a:b] == y["t"]:
            sp.append((a, b, y))
    sp.sort(key=lambda t: (t[0], -t[1]))
    out, pos = [], 0
    for a, b, x_ in sp:
        if a < pos:
            continue
        tag, attrs = _SPAN_TAGS[x_["l"]]
        extra = ""
        if x_.get("rk"):
            extra += f' cRef="{attr(x_["rk"])}"'
        elif x_.get("rc"):
            extra += f' n="{attr(" | ".join(x_["rc"]))}"'          # candidates of an ambiguous citation
        if x_.get("jk"):
            extra += f' key="{attr(x_["jk"])}"'
        if x_.get("st") != "verified_by_model":
            extra += ' cert="low"'
        out.append(x(text[pos:a]))
        out.append(f"<{tag}{(' ' + attrs) if attrs else ''}{extra}>{x(text[a:b])}</{tag}>")
        pos = b
    out.append(x(text[pos:]))
    return "".join(out)


def consilium_xml(c: dict, iiif: dict, images: dict, body_ann: list, subj: dict | None, translations: bool, spans: list | None = None) -> str:
    o = [f'<div type="consilium" xml:id="{attr(c["id"])}" n="{attr(c["n"])}"' + (f' subtype="pars{c["pars"]}"' if c.get("pars") else "") + ">"]
    pages = c.get("sources") or []
    if pages:
        o.append(f'  <pb n="{attr(pages[0]["page"].replace(".xml", ""))}" facs="#{attr(page_id(pages[0]["page"].replace(".xml", "")))}"/>')
    disp = display_title(c)
    if disp and disp != (c.get("title") or "").strip():
        o.append(f"  <head><choice><orig>{x(c.get('title', ''))}</orig><reg>{x(disp)}</reg></choice></head>")
    else:
        o.append(f"  <head>{x(c.get('title', ''))}</head>")
    if c.get("casus"):   # in the print the casus stands before the heading; TEI wants the head first
        o.append('  <div type="casus" rend="printed-before-head">' + "".join(f"<p>{marked(s, [y for y in (spans or []) if y['k'] == 'casus' and y['s'] == i])}</p>" for i, s in enumerate(c["casus"])) + "</div>")
    if c.get("argument"):
        o.append('  <div type="argumentum"><list>' + "".join(f'<item n="{attr(a["n"])}">{x(a["text"])}</item>' for a in c["argument"]) + "</list></div>")
    marg = [m.get("marginal") for m in (c.get("body_meta") or [])]
    o.append('  <div type="textus">')
    for i, s in enumerate(c.get("body") or []):
        n = f' n="{marg[i]}"' if i < len(marg) and marg[i] else ""
        o.append(f"    <p{n}>{marked(s, [y for y in (spans or []) if y['k'] == 'body' and y['s'] == i])}</p>")
    o.append("  </div>")
    for extra_page in pages[1:]:
        o.append(f'  <pb n="{attr(extra_page["page"].replace(".xml", ""))}" facs="#{attr(page_id(extra_page["page"].replace(".xml", "")))}"/>')
    if c.get("keywords_final"):
        o.append('  <note type="keywords" resp="#machine">' + "".join(f"<term>{x(k)}</term>" for k in c["keywords_final"]) + "</note>")
    if subj:
        o.append('  <note type="subjects" resp="#machine">' + "".join(f'<term type="central" key="{attr(k)}"/>' for k in subj.get("central", [])) +
                 "".join(f'<term type="secondary" key="{attr(k)}"/>' for k in subj.get("secondary", [])) + "</note>")
    for a in body_ann:
        cert = "high" if a["status"] in ("verified_by_model", "verified_by_editor") else "low"
        resp = "#editor" if a["status"] == "verified_by_editor" else "#machine"
        o.append(f'  <note type="intimate_body" subtype="{attr(a["code"])}" ana="#{attr(a["role"])}" cert="{cert}" resp="{resp}"><quote xml:lang="la">{x(a.get("evidence_la", ""))}</quote></note>')
    if translations:
        for lang in ("en", "de"):
            t = (c.get("translations") or {}).get(lang) or {}
            if not t.get("body"):
                continue
            o.append(f'  <div type="translation" xml:lang="{lang}" resp="#machine">')
            if t.get("argument"):
                o.append(f"    <p rend=\"argumentum\">{x(t['argument'])}</p>")
            o += [f"    <p>{x(s)}</p>" for s in t["body"] if s]
            o.append("  </div>")
    o.append("</div>")
    return "\n".join(o)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--print", dest="print_filter")
    ap.add_argument("--translations", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--doi", default="", help="Zenodo DOI of the deposit, written to publicationStmt")
    args = ap.parse_args()
    import datetime
    today = datetime.date.today().isoformat()
    editions = json.loads((SITE / "editions.json").read_text(encoding="utf-8")) if (SITE / "editions.json").exists() else {}
    consilia = json.loads((ROOT / "output" / "consilia.json").read_text(encoding="utf-8"))["consilia"]
    authors = json.loads((ROOT / "output" / "authors.json").read_text(encoding="utf-8"))
    manifest = json.loads((SITE / "manifest.json").read_text(encoding="utf-8")) if (SITE / "manifest.json").exists() else {}
    iiif = json.loads((ROOT / "output" / "iiif_manifests.json").read_text(encoding="utf-8")); iiif.pop("_comment", None)
    images = json.loads((SITE / "image_sources.json").read_text(encoding="utf-8")) if (SITE / "image_sources.json").exists() else {}
    body_by_doc: dict[str, list] = {}
    subj_docs: dict = {}
    if (SITE / "annotations" / "intimate_body.json").exists():
        for u in json.loads((SITE / "annotations" / "intimate_body.json").read_text(encoding="utf-8"))["units"]:
            body_by_doc.setdefault(u["doc"], []).append(u)
    if (SITE / "annotations" / "subjects.json").exists():
        subj_docs = json.loads((SITE / "annotations" / "subjects.json").read_text(encoding="utf-8"))["docs"]
    ref_dir = SITE / "annotations" / "references"
    by_print: dict[str, list] = {}
    for c in consilia.values():
        by_print.setdefault(c["volume"], []).append(c)
    args.out.mkdir(parents=True, exist_ok=True)
    for print_id, lst in sorted(by_print.items()):
        if args.print_filter and print_id != args.print_filter:
            continue
        author = authors.get(lst[0]["author_viaf"], {})
        lst.sort(key=lambda c: (c.get("pars") or 0, c["n"], c["id"]))
        models = recognition_models(ROOT / "data" / lst[0]["author_viaf"] / print_id)
        pages = sorted({pg["page"].replace(".xml", "") for c in lst for pg in (c.get("sources") or [])})
        facs = ["<facsimile>"] + [f'  <surface xml:id="{attr(page_id(pg))}" n="{attr(pg)}"><graphic url="{attr(facs_url(lst[0], {"page": pg}, iiif, images))}"/></surface>' for pg in pages] + ["</facsimile>"]
        parts = ['<?xml version="1.0" encoding="UTF-8"?>', '<TEI xmlns="http://www.tei-c.org/ns/1.0">',
                 header(print_id, author, lst, manifest.get("revision", ""), editions.get(print_id) or {}, iiif.get(print_id) or {},
                        models, args.doi, today)] + facs + ['<text xml:lang="la"><body>']
        shard = json.loads((ref_dir / f"{print_id}.json").read_text(encoding="utf-8")) if (ref_dir / f"{print_id}.json").exists() else {}
        parts += [consilium_xml(c, iiif, images, body_by_doc.get(c["id"], []), subj_docs.get(c["id"]), args.translations, shard.get(c["id"])) for c in lst]
        parts += ["</body></text>", "</TEI>"]
        p = args.out / f"{print_id}.xml"
        p.write_text("\n".join(parts), encoding="utf-8")
        import xml.etree.ElementTree as ET
        ET.parse(p)   # well-formedness check
        print(f"{p.relative_to(ROOT)}: {len(lst)} consilia, {p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
