#!/usr/bin/env python3
"""
Gold standard for the transcription: a representative sample of lines, corrected by hand against the image.

Sample design (fixed seed, drawn once with --make-sample):
  two strata, 150 lines each
    agreed    all three readings (Gemini, PyLaia, kraken) equal after gold_eval.fold (letters level)
    disputed  at least one reading differs
  within each stratum a systematic sample in reading order (print, page, region, line) from a random
  start, so every volume, every part of a volume and every line type is represented in proportion.
  Each line carries its stratum weight (stratum size / 150); corpus-level figures are weighted
  averages, figures per stratum are unweighted.

    python tools/gold_server.py --make-sample       # writes output/gold/sample.json and the image crops
    python tools/gold_server.py --recrop            # renders the crops of the existing sample again
    python tools/gold_server.py --make-review       # review set from output/merge, served under /review/
    python tools/gold_server.py                     # http://localhost:8790/

The page shows the line in its context and as a tight crop, the distinct readings in random order and
without source names, and a text field. Clicking a reading (or 1/2/3) copies it into the field. The
field is the gold text: correct it if no reading is right, then save (Enter). Flags describe the line
(hard to read, initial, margin number, segmentation wrong); the note is free text.
Answers go to output/gold/gold.jsonl (one record per save, the last one per line counts). Nothing is
written to the PageXML.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import random
import sys
import threading
import unicodedata
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "qa"))
sys.path.insert(0, str(ROOT / "tools"))
from compare_ocr import normalise  # noqa: E402
from kraken_export import crop_line, points  # noqa: E402
from gold_eval import ACCENTS, fold, options  # noqa: E402

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"; T = "{%s}" % NS
OUT = ROOT / "output" / "gold"
DATA = ROOT / "data" / "Baldo_29618397"
MERGE = ROOT / "output" / "merge"
# readings and geometry: the working copy (output/merge) when it exists, else data/ and the first kraken run
READINGS = MERGE / "kraken_current.jsonl" if (MERGE / "kraken_current.jsonl").exists() else ROOT / "output" / "qa" / "kraken_all_e31.jsonl"
REVIEW = OUT / "review"
PER_STRATUM = 150
MARGIN_SIDE = 180      # px beyond line and column on both sides: the margin numbers stand in the gutter, outside both
SEED = 300
LOCK = threading.Lock()


def line_type(tl) -> str:
    cu = tl.get("custom") or ""
    i = cu.find("structure {type:")
    return cu[i + 16:].split(";")[0].split("}")[0] if i >= 0 else "none"


def make_sample() -> None:
    rows = [json.loads(l) for l in READINGS.open(encoding="utf-8")]
    if READINGS.parent == MERGE:
        for r in rows:
            r["file"] = str((MERGE / "Baldo_29618397" / r["print"] / "page" / r["page"]).relative_to(ROOT))   # relative: portable
    strata = collections.defaultdict(list)
    for r in rows:
        same = fold(r["gemini"], "letters") == fold(r["pylaia"], "letters") == fold(r["kraken"], "letters")
        strata["agreed" if same else "disputed"].append(r)
    rng = random.Random(SEED)
    sample = []
    for name in ("agreed", "disputed"):
        pop = strata[name]                                  # already in reading order
        step = len(pop) / PER_STRATUM
        start = rng.random() * step
        for j in range(PER_STRATUM):
            r = pop[int(start + j * step)]
            sample.append({**r, "stratum": name, "weight": round(len(pop) / PER_STRATUM, 2)})
    rng.shuffle(sample)                                     # strata mixed in the order of work
    for n, s in enumerate(sample, 1):
        s["id"] = f"g{n:03d}"
    render(sample)
    keep = ("id", "stratum", "weight", "file", "print", "page", "region", "line_id", "line_type", "valid",
            "gemini", "pylaia", "kraken", "conf", "prev", "next")
    (OUT / "sample.json").write_text(json.dumps([{k: s[k] for k in keep} for s in sample], ensure_ascii=False, indent=1), encoding="utf-8")
    c = collections.Counter((s["stratum"], s["print"][-2:], s["line_type"]) for s in sample)
    print(f"\n{len(sample)} lines; strata sizes: " + ", ".join(f"{k} {len(v):,}" for k, v in strata.items()))
    for k, v in sorted(c.items()):
        print(" ", k, v)


def make_review() -> None:
    """the lines flagged by tools/merge_layout.py and tools/audit_line_edges.py and the marginal numbers
    kraken and Gemini read differently, on the geometry of the working copy; not part of the gold sample.
    Rebuilding keeps the ids of the lines already in the set (answers stay valid), refreshes their
    readings and images, and appends new lines with the next ids."""
    cur = {}
    for l in (MERGE / "kraken_current.jsonl").open(encoding="utf-8"):
        r = json.loads(l); cur[(r["print"], r["page"], r["line_id"])] = r
    from gold_eval import margin_number
    items = [json.loads(l) for l in (MERGE / "review.jsonl").open(encoding="utf-8")]
    for k in reread_keys():
        r = cur[k]; g, kn = margin_number(r["gemini"]), margin_number(r["kraken"])
        if kn and kn != g:
            items.append(dict(print=k[0], page=k[1], line_id=k[2], reason="kraken reads another number than Gemini" if g else "kraken reads a number, Gemini none"))
    extra = MERGE / "review_extra.json"
    if extra.exists():
        items += json.loads(extra.read_text(encoding="utf-8"))
    order = {r: i for i, r in enumerate(dict.fromkeys(it["reason"] for it in items))}
    items.sort(key=lambda it: (order[it["reason"]], it["print"], it["page"]))
    old = {}
    if (REVIEW / "sample.json").exists():
        for it in json.loads((REVIEW / "sample.json").read_text(encoding="utf-8")):
            old[(it["print"], it["page"], it["line_id"])] = it
    seen, out = set(), []
    for it in list(old.values()) + items:                 # the existing set first, in its order
        k = (it["print"], it["page"], it["line_id"])
        if k in seen:
            continue
        seen.add(k); out.append(dict(it))
    nxt = max([int(o["id"][1:]) for o in old.values()] or [0]) + 1
    for it in out:
        k = (it["print"], it["page"], it["line_id"])
        if "id" not in it or k not in old:
            it["id"] = f"r{nxt:03d}"; nxt += 1
        r = cur.get(k, {})
        it.update(gemini=r.get("gemini", it.get("gemini", "")), pylaia=r.get("pylaia", it.get("pylaia", "")), kraken=r.get("kraken", it.get("kraken", "")),
                  file=str(MERGE / "Baldo_29618397" / it["print"] / "page" / it["page"]), stratum="review", weight=0, valid="", conf=r.get("conf", 0))
        it.pop("region", None)
    render(out, REVIEW)
    keep = ("id", "reason", "stratum", "weight", "file", "print", "page", "region", "line_id", "line_type", "valid",
            "gemini", "pylaia", "kraken", "conf", "prev", "next")
    (REVIEW / "sample.json").write_text(json.dumps([{k: it.get(k) for k in keep} for it in out], ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(out)} review lines ({len(out) - len(old)} new):", dict(collections.Counter(it["reason"] for it in out)))


def reread_keys() -> set:
    keys = set()
    for name in ("kraken_reread.jsonl", "kraken_reread_edges.jsonl"):
        f = MERGE / name
        if f.exists():
            for l in f.open(encoding="utf-8"):
                r = json.loads(l); keys.add((r["print"], r["page"], r["line_id"]))
    return keys


def render(sample: list[dict], out: Path = OUT) -> None:
    """line crop and context crop per sample line; also fills line_type, prev and next"""
    (out / "img").mkdir(parents=True, exist_ok=True)
    trees = {}
    for n, s in enumerate(sample, 1):
        if s["file"] not in trees:
            trees[s["file"]] = ET.parse(ROOT / s["file"]).getroot()      # ROOT / an absolute path stays that path
        page = trees[s["file"]].find(T + "Page")
        xml = ROOT / s["file"]
        img = next(p for p in (xml.parents[1] / page.get("imageFilename"), xml.parent / page.get("imageFilename"),
                               DATA / s["print"] / page.get("imageFilename")) if p.exists())
        region = next(tr for tr in page.iter(T + "TextRegion")
                      if (tr.get("id") == s["region"] if s.get("region") else any(tl.get("id") == s["line_id"] for tl in tr.findall(T + "TextLine"))))
        s["region"] = region.get("id")
        tls = region.findall(T + "TextLine")
        idx = next(i for i, tl in enumerate(tls) if tl.get("id") == s["line_id"])
        s["line_type"] = line_type(tls[idx])
        s["prev"] = rows_text(tls[idx - 1]) if idx else ""
        s["next"] = rows_text(tls[idx + 1]) if idx + 1 < len(tls) else ""
        im = Image.open(img); im.load()
        poly = points(tls[idx])
        crop_line(im, poly, side=MARGIN_SIDE).save(out / "img" / f"{s['id']}_line.png")
        # context: the region's width, one line above and below, the line framed
        rp = points(region); boxes = [points(t) for t in tls[max(0, idx - 1):idx + 2]]
        x0 = max(0, min(x for x, _ in rp) - MARGIN_SIDE); x1 = min(im.width, max(x for x, _ in rp) + MARGIN_SIDE)
        y0 = max(0, min(y for b in boxes for _, y in b) - 12); y1 = min(im.height, max(y for b in boxes for _, y in b) + 12)
        cr = im.convert("RGB").crop((x0, y0, x1, y1)); d = ImageDraw.Draw(cr)
        d.rectangle((min(x for x, _ in poly) - x0 - 4, min(y for _, y in poly) - y0 - 3,
                     max(x for x, _ in poly) - x0 + 4, max(y for _, y in poly) - y0 + 3), outline=(200, 30, 30), width=3)
        if cr.width > 1600:
            cr = cr.resize((1600, int(cr.height * 1600 / cr.width)))
        cr.save(out / "img" / f"{s['id']}_ctx.jpg", quality=88)
        print(f"\r{n}/{len(sample)}", end="", flush=True)


def rows_text(tl) -> str:
    te = tl.find(T + "TextEquiv")
    return (te.find(T + "Unicode").text or "").strip() if te is not None and te.find(T + "Unicode") is not None else ""


def load_answers(out: Path = OUT) -> dict:
    ans = {}
    p = out / "gold.jsonl"
    if p.exists():
        for l in p.open(encoding="utf-8"):
            if l.strip():
                a = json.loads(l); ans[a["id"]] = a
    return ans


PAGE = r"""<!doctype html><html lang="de"><head><meta charset="utf-8"><title>Goldstandard</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#f7f5f0;--fg:#222;--mut:#777;--card:#fff;--line:#ddd;--acc:#8a2b1e;--sel:#f3e3d0}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
header{display:flex;gap:16px;align-items:center;padding:10px 20px;border-bottom:1px solid var(--line);background:var(--card);position:sticky;top:0;z-index:2}
header b{font-size:16px} .bar{flex:1;height:6px;background:#eee;border-radius:3px;overflow:hidden} .bar i{display:block;height:100%;background:var(--acc)}
main{max-width:1250px;margin:0 auto;padding:16px 20px 60px}
img{max-width:100%;display:block;border:1px solid var(--line);background:#fff} .ctx{margin-bottom:10px} .tight{margin:8px 0 14px;image-rendering:auto}
.meta{color:var(--mut);font-size:13px;margin-bottom:8px} .nb{color:var(--mut);font:13px/1.4 ui-monospace,monospace;margin:2px 0}
.opt{display:flex;gap:10px;align-items:baseline;padding:8px 12px;margin:6px 0;background:var(--card);border:1px solid var(--line);border-radius:6px;cursor:pointer;font:16px/1.5 ui-monospace,Menlo,monospace}
.opt:hover{border-color:var(--acc)} .opt.on{background:var(--sel);border-color:var(--acc)} .opt kbd{color:var(--mut);font-size:12px} .opt mark{background:#ffe08a;padding:0 1px}
textarea{width:100%;box-sizing:border-box;font:17px/1.5 ui-monospace,Menlo,monospace;padding:10px;border:2px solid var(--line);border-radius:6px;resize:vertical}
textarea:focus{outline:none;border-color:var(--acc)}
.row{display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin:10px 0} label{user-select:none}
button{font:inherit;padding:7px 14px;border:1px solid var(--line);background:var(--card);border-radius:6px;cursor:pointer} button.pri{background:var(--acc);color:#fff;border-color:var(--acc)}
input[type=text]{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:6px;flex:1;min-width:200px}
.help{color:var(--mut);font-size:13px;margin-top:18px} .done{color:#2a7a2a;font-weight:600}
#grid{display:flex;flex-wrap:wrap;gap:3px;margin-top:14px} #grid a{width:14px;height:14px;border-radius:2px;background:#e3e0da;display:block} #grid a.d{background:#9bbf8c} #grid a.c{outline:2px solid var(--acc)}
details{margin-top:14px;color:var(--mut);font-size:13px}
</style></head><body>
<header><b id="title">Goldstandard</b><a id="switch" href="#"></a><span id="pos"></span><div class="bar"><i id="prog"></i></div><span id="cnt"></span>
<button onclick="go(-1)">&larr;</button><button onclick="go(1)">&rarr;</button><button onclick="nextOpen()">nächste offene</button></header>
<main>
<div class="meta" id="meta"></div>
<img class="ctx" id="ctx">
<div class="nb" id="prev"></div>
<img class="tight" id="tight">
<div class="nb" id="next"></div>
<div id="opts"></div>
<textarea id="gold" rows="2" spellcheck="false"></textarea>
<div class="row">
 <label><input type="checkbox" data-f="hard"> schwer lesbar</label>
 <label><input type="checkbox" data-f="initial"> Initiale / Auszeichnung</label>
 <label><input type="checkbox" data-f="margin_number"> Randziffer</label>
 <label><input type="checkbox" data-f="segmentation"> Zeile falsch geschnitten</label>
 <label><input type="checkbox" data-f="summary"> Summarium (als Haupttext getaggt)</label>
 <label><input type="checkbox" data-f="unsure"> unsicher</label>
</div>
<div class="row"><input type="text" id="note" placeholder="Notiz"><button class="pri" onclick="save()">Speichern (Enter)</button><span id="state"></span></div>
<details><summary>Regeln für den Goldtext</summary>
<p>Text so, wie er in der Edition stehen soll: Kürzungen aufgelöst, & als et, rundes s statt langem s, keine Akzente, kein Trennzeichen am Zeilenende, Satzzeichen wie gedruckt, Randziffern und † mit aufnehmen, wenn sie zur Zeile gehören. Beim Vergleich ohnehin gleichgesetzt: u/v, i/j, ae/oe/e (auch æ, œ, ę), Akzente, Trennzeichen. Nur die gerahmte Zeile, ohne Text aus den Nachbarzeilen.</p></details>
<div class="help">1/2/3 übernimmt eine Lesung &middot; Enter speichert und geht weiter &middot; Shift+Enter Zeilenumbruch &middot; Alt+&larr;/&rarr; blättern</div>
<div id="grid"></div>
</main>
<script>
let S=[],A={},i=0;const SET='__SET__';
const REASON={'number read, no † († missing or the line is a summary)':'Ziffer gelesen, aber kein † (fehlt das †, oder ist es ein Summarium?)',
 '† in a line piece that starts inside the column':'† in einem Zeilenstück mitten in der Spalte',
 '† and polygon in the gutter, no number read':'† und Polygon im Rand, aber keine Ziffer gelesen',
 'kraken reads another number than Gemini':'Lesungen haben verschiedene Randziffern',
 'kraken reads a number, Gemini none':'eine Lesung hat eine Randziffer, eine andere keine',
 'text runs beyond the polygon (not fixed automatically)':'Text ragt über das Polygon hinaus (nicht automatisch behoben)',
 'ink in the gutter, Gemini reads a number, kraken none':'Schwärze im Steg, eine Lesung hat eine Ziffer, die andere nicht'};
document.getElementById('title').textContent=SET?'Prüfzeilen':'Goldstandard';
document.getElementById('switch').textContent=SET?'zum Goldstandard':'zu den Prüfzeilen';document.getElementById('switch').href=SET?'/':'/review/';
const $=id=>document.getElementById(id), esc=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function hash(s){let h=0;for(const c of s)h=(h*31+c.charCodeAt(0))|0;return h}
const clean=t=>t.normalize('NFD').replace(/[\u0300\u0301\u0302\u0308]/g,'').normalize('NFC').replace(/¬/g,'').replace(/-\s*$/,'').trim();
const key=t=>t.replace(/[æœę]/g,'e').replace(/[ÆŒ]/g,'E').replace(/ae|oe/g,'e').replace(/A[Ee]|O[Ee]/g,'E').replace(/ſ/g,'s').replace(/v/g,'u').replace(/V/g,'U').replace(/j/g,'i').replace(/J/g,'I');
function opts(it){if(it.options){const a=it.options.map(clean);let h=hash(it.id);for(let j=a.length-1;j>0;j--){h=(h*1103515245+12345)|0;const r=Math.abs(h)%(j+1);[a[j],a[r]]=[a[r],a[j]]}return a}
  const seen=new Map();for(const t0 of [it.gemini,it.pylaia,it.kraken]){const t=clean(t0),k=key(t);if(t&&!seen.has(k))seen.set(k,t)}
  const a=[...seen.values()];let h=hash(it.id);for(let j=a.length-1;j>0;j--){h=(h*1103515245+12345)|0;const r=Math.abs(h)%(j+1);[a[j],a[r]]=[a[r],a[j]]}return a}
function marked(t,others){const ws=new Set(others.flatMap(o=>o.split(/\s+/)));return t.split(/(\s+)/).map(w=>/\s/.test(w)||!others.length||ws.has(w)?esc(w):'<mark>'+esc(w)+'</mark>').join('')}
function show(){const it=S[i],a=A[it.id];
  $('pos').textContent=`${i+1} / ${S.length}`;$('cnt').textContent=`${Object.keys(A).length} erledigt`;$('prog').style.width=(Object.keys(A).length/S.length*100)+'%';
  $('meta').textContent=`${it.id} · ${it.print.replace('Baldo_Cons_Print_Venice_1575_','')} · Seite ${it.page.replace('.xml','')} · ${it.line_type}`+(it.reason?` · ${REASON[it.reason]||it.reason}`:'');
  $('ctx').src='img/'+it.id+'_ctx.jpg';$('tight').src='img/'+it.id+'_line.png';
  $('prev').textContent=it.prev?'↑ '+clean(it.prev):'';$('next').textContent=it.next?'↓ '+clean(it.next):'';
  const o=opts(it);$('opts').innerHTML=o.map((t,k)=>`<div class="opt" data-t="${esc(t)}" onclick="pick(${k})"><kbd>${k+1}</kbd><span>${marked(t,o.filter((_,m)=>m!==k))}</span></div>`).join('');
  $('gold').value=a?a.gold:'';document.querySelectorAll('[data-f]').forEach(c=>c.checked=!!(a&&a.flags&&a.flags.includes(c.dataset.f)));
  $('note').value=a?(a.note||''):'';$('state').innerHTML=a?'<span class="done">gespeichert</span>':'';hl();
  $('grid').innerHTML=S.map((s,k)=>`<a href="#" class="${A[s.id]?'d':''} ${k===i?'c':''}" title="${s.id}" onclick="i=${k};show();return false"></a>`).join('');
  $('gold').focus()}
function hl(){const g=key(clean($('gold').value));document.querySelectorAll('.opt').forEach(e=>e.classList.toggle('on',key(e.dataset.t)===g))}
function pick(k){const e=document.querySelectorAll('.opt')[k];if(!e)return;$('gold').value=e.dataset.t;hl();$('gold').focus()}
async function save(){const it=S[i],g=$('gold').value.trim();if(!g){$('state').textContent='Goldtext fehlt';return}
  const rec={id:it.id,gold:g,flags:[...document.querySelectorAll('[data-f]:checked')].map(c=>c.dataset.f),note:$('note').value.trim()};
  const r=await fetch('save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(rec)});
  if(!r.ok){$('state').textContent='Fehler beim Speichern';return}A[it.id]=await r.json();nextOpen()}
function go(d){i=Math.max(0,Math.min(S.length-1,i+d));show()}
function nextOpen(){for(let k=1;k<=S.length;k++){const j=(i+k)%S.length;if(!A[S[j].id]){i=j;show();return}}show()}
$('gold').addEventListener('input',hl);
document.addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey&&(e.target.id==='gold'||e.target.id==='note')){e.preventDefault();save();return}
  if(e.altKey&&e.key==='ArrowLeft'){go(-1);return} if(e.altKey&&e.key==='ArrowRight'){go(1);return}
  if(['1','2','3'].includes(e.key)&&(e.target.id!=='gold'||e.altKey||$('gold').value.trim()==='')&&e.target.id!=='note'){e.preventDefault();pick(+e.key-1)}});
(async()=>{S=await (await fetch('sample.json')).json();A=await (await fetch('answers')).json();i=Math.max(0,S.findIndex(s=>!A[s.id]));show()})();
</script></body></html>"""


def inspect_page(n: int = 50, seed: int = 1023) -> str:
    """read-only view: n random disputed gold lines with the image, the three readings labelled by source,
    the gold text and what the majority rule chooses"""
    import html
    from gold_eval import body, has_abbreviation
    S = {x["id"]: x for x in json.loads((OUT / "sample.json").read_text(encoding="utf-8"))}
    A = load_answers(OUT)
    ids = sorted(i for i in A if S[i]["stratum"] == "disputed")
    random.Random(seed).shuffle(ids)
    rows = []
    for i in sorted(ids[:n], key=lambda x: int(x[1:])):
        x = S[i]; g = A[i]["gold"]
        f = {k: fold(body(x[k]), "letters") for k in ("gemini", "pylaia", "kraken") if not has_abbreviation(x[k])}
        win = next((a + "+" + b for a, b in (("pylaia", "kraken"), ("gemini", "kraken"), ("gemini", "pylaia"))
                    if a in f and b in f and f[a] == f[b]), None)
        choice = x[win.split("+")[0]] if win else x["gemini"]
        ok = lambda t: fold(body(t), "letters") == fold(body(g), "letters")
        lines = "".join(
            f'<tr class="{"ok" if ok(x[k]) else "bad"}"><th>{name}</th><td>{"✓" if ok(x[k]) else "✗"}</td><td>{html.escape(x[k])}</td></tr>'
            for k, name in (("gemini", "Gemini"), ("pylaia", "PyLaia"), ("kraken", "kraken")))
        rule = (f"Mehrheit {win.replace('pylaia', 'PyLaia').replace('gemini', 'Gemini').replace('+', ' + ')}" if win else "keine Mehrheit, Gemini")
        rows.append(f"""<section><div class="meta">{i} · {x['print'][-2:]} · {x['page'].replace('.xml', '')} · {x['line_type']}</div>
<img src="img/{i}_ctx.jpg"><img class="tight" src="img/{i}_line.png">
<table><tr class="gold"><th>Gold</th><td></td><td>{html.escape(g)}</td></tr>{lines}
<tr class="{"ok" if ok(choice) else "bad"} rule"><th>Regel</th><td>{"✓" if ok(choice) else "✗"}</td><td>{rule}</td></tr></table></section>""")
    right = sum(1 for r in rows if 'rule"><th>Regel</th><td>✓' in r)
    return f"""<!doctype html><html lang="de"><head><meta charset="utf-8"><title>Stichprobe</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{{--bg:#f7f5f0;--fg:#222;--mut:#777;--line:#ddd}} body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}}
main{{max-width:1250px;margin:0 auto;padding:16px 20px 60px}} section{{background:#fff;border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:14px 0}}
img{{max-width:100%;display:block;margin:6px 0}} .meta{{color:var(--mut);font-size:13px}} table{{border-collapse:collapse;width:100%;font:15px/1.5 ui-monospace,Menlo,monospace}}
th{{text-align:left;width:70px;color:var(--mut);font:600 13px system-ui}} td{{padding:2px 6px;vertical-align:top}} td:nth-child(2){{width:18px}}
tr.gold td{{font-weight:600}} tr.ok td:nth-child(2){{color:#2a7a2a}} tr.bad td:nth-child(2){{color:#a33}} tr.bad td:nth-child(3){{background:#fbeceb}} tr.rule td{{font-family:system-ui}}
</style></head><body><main><h2>{len(rows)} zufällige strittige Zeilen aus dem Goldstandard</h2>
<p class="meta">✓ = im Wortlaut gleich dem Goldtext (Randziffer separat, nicht mitgezählt). Regel: Mehrheit der drei Lesungen ohne Kürzungszeichen, sonst Gemini. Die Regel trifft hier {right} von {len(rows)}.</p>
{''.join(rows)}</main></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--make-sample", action="store_true")
    ap.add_argument("--recrop", action="store_true", help="render the image crops of the existing sample again")
    ap.add_argument("--make-review", action="store_true", help="build the review set from output/merge (separate from the sample)")
    ap.add_argument("--port", type=int, default=8790)
    a = ap.parse_args()
    if a.make_sample:
        if (OUT / "sample.json").exists():
            sys.exit("output/gold/sample.json exists; the sample is drawn once (delete it deliberately to redraw)")
        make_sample(); return
    if a.make_review:
        make_review(); return
    if a.recrop:
        render(json.loads((OUT / "sample.json").read_text(encoding="utf-8"))); print(); return
    SETS = {"": OUT, "review": REVIEW}
    # lines with a single option after the rules are answered automatically (flag "auto")
    for st, d in SETS.items():
        if not (d / "sample.json").exists():
            continue
        done = load_answers(d)
        with (d / "gold.jsonl").open("a", encoding="utf-8") as f:
            for it in json.loads((d / "sample.json").read_text(encoding="utf-8")):
                o = options(it)
                if it["id"] not in done and len(o) == 1 and st == "":
                    g = unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", o[0].replace("¬", "")) if c not in ACCENTS)).strip()
                    f.write(json.dumps(dict(id=it["id"], gold=g, gold_raw=o[0], flags=["auto"], note="one reading after the rules",
                                            time=datetime.datetime.now().isoformat(timespec="seconds"),
                                            match={k: fold(it[k]) == fold(g) for k in ("gemini", "pylaia", "kraken")}), ensure_ascii=False) + "\n")
    samples = {k: {s["id"]: s for s in json.loads((d / "sample.json").read_text(encoding="utf-8"))} for k, d in SETS.items() if (d / "sample.json").exists()}

    def split(path: str) -> tuple[str, str]:
        if path == "/review" or path.startswith("/review/"):
            return "review", path[len("/review"):] or "/"
        return "", path

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if urlparse(self.path).path == "/review":
                self.send_response(301); self.send_header("Location", "/review/"); self.end_headers(); return
            st, p = split(urlparse(self.path).path)
            if st not in samples:
                return self._send(404, b"set not built", "text/plain")
            d = SETS[st]
            if p in ("/", "/index.html"):
                return self._send(200, PAGE.replace("__SET__", st).encode(), "text/html; charset=utf-8")
            if p == "/inspect" and st == "":
                return self._send(200, inspect_page().encode(), "text/html; charset=utf-8")
            if p == "/sample.json":
                items = json.loads((d / "sample.json").read_text(encoding="utf-8"))
                for it in items:
                    it["options"] = options(it)
                return self._send(200, json.dumps(items, ensure_ascii=False).encode(), "application/json")
            if p == "/answers":
                return self._send(200, json.dumps(load_answers(d), ensure_ascii=False).encode(), "application/json")
            if p.startswith("/img/") and "/" not in p[5:]:
                f = d / "img" / p[5:]
                if f.exists():
                    return self._send(200, f.read_bytes(), "image/png" if f.suffix == ".png" else "image/jpeg")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            st, p = split(urlparse(self.path).path)
            if p != "/save" or st not in samples:
                return self._send(404, b"", "text/plain")
            d = SETS[st]
            rec = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            s = samples[st].get(rec.get("id"))
            if not s or not str(rec.get("gold", "")).strip():
                return self._send(400, b"bad record", "text/plain")
            raw = rec["gold"].strip()
            gold = unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", raw.replace("¬", "")) if c not in ACCENTS)).strip()
            g = fold(gold)
            rec.update(gold=gold, gold_raw=raw, time=datetime.datetime.now().isoformat(timespec="seconds"),
                       match={k: fold(s[k]) == g for k in ("gemini", "pylaia", "kraken")})
            with LOCK, (d / "gold.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._send(200, json.dumps(rec, ensure_ascii=False).encode(), "application/json")

    for st, d in SETS.items():
        if st in samples:
            print(f"{st or 'gold'}: {len(samples[st])} lines, {len(load_answers(d))} done  ->  http://localhost:{a.port}/{st + '/' if st else ''}")
    HTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
