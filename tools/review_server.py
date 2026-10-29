#!/usr/bin/env python3
"""
Local review tool for the lines where the two transcriptions disagree.

Item set: every TextLine of the Baldus volumes that carries two readings (TextEquiv index 0 = valid,
index 1 = the other source) whose texts still differ after the known equivalences are normalised
(et/&, æ/ae, ſ/s, u/v, i/j, expansion brackets, punctuation, spacing, hyphen; see compare_ocr.normalise),
plus lines present in one source only and headings whose numeral the sources read differently.
Sorted by the size of the difference (character error rate between the two normalised readings),
largest first; filters for volume and minimum difference.

    python tools/review_server.py            # http://localhost:8787/
Keys: 1 = keep/take the Gemini reading, 2 = keep/take the PageXML reading, e = edit the text field and
save with Ctrl+Enter, Space = skip. Each decision goes into the PageXML at once (readings swapped when
the other source is chosen; "reviewed <date>: gemini|pylaia|edited" appended to the line's comments)
and into output/qa/review_decisions.jsonl; reviewed lines are not shown again.
Afterwards: rebuild the prints that changed (src/build.py --force --print …), site-data.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import re
import sys
import threading
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "qa"))
from compare_ocr import normalise, dist  # noqa: E402

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
T = "{%s}" % NS
ET.register_namespace("", NS)
LOG = ROOT / "output" / "qa" / "review_decisions.jsonl"
LOCK = threading.Lock()


def pts(s):
    return [tuple(int(float(v)) for v in p.split(",")) for p in (s or "").split() if "," in p]


def utext(tl, idx):
    u = tl.find(f"{T}TextEquiv[@index='{idx}']/{T}Unicode")
    if u is None and idx == "0":
        u = tl.find(f"{T}TextEquiv/{T}Unicode")
    return ((u.text or "") if u is not None else "")


def collect(print_filter: str | None) -> list[dict]:
    items = []
    for f in sorted(glob.glob(str(ROOT / "data" / "Baldo_29618397" / "*" / "page" / "*.xml"))):
        vol = Path(f).parts[-3]
        if print_filter and vol != print_filter:
            continue
        root = ET.parse(f).getroot()
        page = root.find(T + "Page")
        for tr in page.findall(T + "TextRegion"):
            tls = tr.findall(T + "TextLine")
            for i, tl in enumerate(tls):
                cm = tl.get("comments") or ""
                if "reviewed " in cm:
                    continue
                t0, t1 = utext(tl, "0"), utext(tl, "1")
                src0 = "gemini" if "text=gemini" in cm else "pylaia"
                kind, cer = None, 0.0
                if "heading numeral differs" in cm:
                    kind = "heading numeral"; cer = 1.0
                elif t1.strip():
                    a, b = normalise(t0, True), normalise(t1, True)
                    if a != b:
                        kind = "readings differ"; cer = dist(a, b) / max(len(a), len(b), 1)
                elif "pylaia only" in cm:
                    kind, cer = "only in PageXML", 0.9
                elif "geometry=interpolated" in cm:
                    kind, cer = "only in Gemini", 0.9
                if not kind:
                    continue
                gem = t0 if src0 == "gemini" else t1
                pyl = t1 if src0 == "gemini" else t0
                items.append({"file": f, "vol": vol, "page": Path(f).name, "image": page.get("imageFilename"), "region": tr.get("id"),
                              "line_id": tl.get("id"), "kind": kind, "cer": round(cer, 3), "valid": src0, "gemini": gem, "pylaia": pyl,
                              "before": [utext(x, "0") for x in tls[max(0, i - 2):i]], "after": [utext(x, "0") for x in tls[i + 1:i + 3]]})
    items.sort(key=lambda it: (-it["cer"], it["vol"], it["page"]))
    return items


def crop(item: dict) -> bytes:
    root = ET.parse(item["file"]).getroot()
    page = root.find(T + "Page")
    img_path = next((p for p in [ROOT / "data" / "Baldo_29618397" / item["vol"] / item["image"], ROOT / "data" / "Baldo_29618397" / item["vol"] / "page" / item["image"]] if p.exists()), None)
    if img_path is None:
        return b""
    region = next(tr for tr in page.findall(T + "TextRegion") if tr.get("id") == item["region"])
    tls = region.findall(T + "TextLine")
    idx = next(i for i, tl in enumerate(tls) if tl.get("id") == item["line_id"])
    boxes = []
    for tl in tls[max(0, idx - 1):idx + 2]:
        p = pts(tl.find(T + "Coords").get("points")) if tl.find(T + "Coords") is not None else []
        if p:
            boxes.append((min(x for x, _ in p), min(y for _, y in p), max(x for x, _ in p), max(y for _, y in p)))
    mp = pts(tls[idx].find(T + "Coords").get("points"))
    rp = pts(region.find(T + "Coords").get("points"))
    x0 = min(x for x, _ in rp) - 10; x1 = max(x for x, _ in rp) + 10
    y0 = min(b[1] for b in boxes) - 10; y1 = max(b[3] for b in boxes) + 10
    im = Image.open(img_path).convert("RGB")
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(im.width, x1), min(im.height, y1)
    cr = im.crop((x0, y0, x1, y1)); d = ImageDraw.Draw(cr)
    # frame the whole printed line across the column (the line polygon may cover only a piece of it)
    d.rectangle((min(x for x, _ in rp) - x0 - 4, min(y for _, y in mp) - y0 - 4, max(x for x, _ in rp) - x0 + 4, max(y for _, y in mp) - y0 + 4), outline=(220, 0, 0), width=4)
    if cr.width > 1500:
        cr = cr.resize((1500, int(cr.height * 1500 / cr.width)))
    buf = BytesIO(); cr.save(buf, "JPEG", quality=88); return buf.getvalue()


def apply_decision(item: dict, action: str, text: str, who: str = "") -> str:
    today = datetime.date.today().isoformat()
    with LOCK:
        tree = ET.parse(item["file"]); root = tree.getroot(); page = root.find(T + "Page")
        region = next(tr for tr in page.findall(T + "TextRegion") if tr.get("id") == item["region"])
        tl = next(x for x in region.findall(T + "TextLine") if x.get("id") == item["line_id"])
        te0 = tl.find(f"{T}TextEquiv[@index='0']") or tl.find(T + "TextEquiv")
        te1 = tl.find(f"{T}TextEquiv[@index='1']")
        u0 = te0.find(T + "Unicode"); u1 = te1.find(T + "Unicode") if te1 is not None else None
        cm = tl.get("comments") or ""
        if action in ("gemini", "pylaia"):
            if action != item["valid"] and u1 is not None:
                u0.text, u1.text = u1.text, u0.text
                cm = cm.replace(f"text={item['valid']}", f"text={action}")
            result = action
        elif action == "save":
            u0.text = text; result = "edited"
        elif action == "delete":
            region.remove(tl); result = "deleted"
        else:
            result = "skipped"
        if result != "deleted":
            tl.set("comments", (cm + f"; reviewed {today}: {(who + ' ') if who else ''}{result}").strip("; "))
        tree.write(item["file"], encoding="unicode", xml_declaration=True)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": datetime.datetime.now().isoformat(timespec="seconds"), "who": who or "editor", "file": item["file"], "line_id": item["line_id"], "kind": item["kind"],
                                 "action": result, "text": text if action == "save" else None, "gemini": item["gemini"], "pylaia": item["pylaia"]}, ensure_ascii=False) + "\n")
    return result


PAGE = """<!doctype html><html lang="de"><head><meta charset="utf-8"><title>Zeilenprüfung</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#f4f1ea;color:#222}
header{display:flex;gap:1rem;align-items:center;padding:.5rem 1rem;background:#fff;border-bottom:1px solid #ddd;position:sticky;top:0;flex-wrap:wrap}
main{max-width:1500px;margin:0 auto;padding:1rem}
img{max-width:100%;border:1px solid #ccc;background:#fff}
.ctx{font-family:Georgia,serif;font-size:1.05rem;line-height:1.5;margin:.6rem 0;white-space:pre-wrap;color:#888}
.read{font-family:Georgia,serif;font-size:1.25rem;line-height:1.6;margin:.3rem 0;padding:.3rem .5rem;border-radius:4px}
.read.g{background:#e6f4ea}.read.p{background:#fdeedd}.read b{background:#ffd54a;font-weight:600}
.lab{font-family:system-ui;font-size:.8rem;color:#666;display:inline-block;width:9rem}
textarea{width:100%;font-family:Georgia,serif;font-size:1.1rem;padding:.4rem;box-sizing:border-box;margin-top:.4rem}
button{font-size:1rem;padding:.45rem .9rem;margin-right:.4rem;cursor:pointer;border:0;border-radius:4px}
.g1{background:#2a7;color:#fff}.p2{background:#c70;color:#fff}.sv{background:#27a;color:#fff}.del{background:#a33;color:#fff}
.meta{color:#666;font-size:.9rem} select{font-size:.95rem}
</style></head><body>
<header><strong>Zeilenprüfung</strong><span id="prog" class="meta"></span>
<label class="meta">Band <select id="vol"><option value="">alle</option></select></label>
<label class="meta">Abweichung ab <select id="min"><option value="0.25">25 %</option><option value="0.10" selected>10 %</option><option value="0.05">5 %</option><option value="0">alle</option></select></label>
<span class="meta">1 = Gemini, 2 = Transkribus, e = editieren (Strg+Enter speichert), Leertaste = überspringen</span></header>
<main>
<div id="meta" class="meta"></div>
<img id="img" alt="">
<div class="ctx" id="before"></div>
<div class="read g" id="rg"></div>
<div class="read p" id="rp"></div>
<div class="ctx" id="after"></div>
<textarea id="text" rows="2"></textarea>
<p><button class="g1" onclick="act('gemini')">1 Gemini</button><button class="p2" onclick="act('pylaia')">2 Transkribus</button><button class="sv" onclick="act('save')">Speichern (editiert)</button><button class="del" onclick="act('delete')">Zeile löschen</button><button onclick="next()">Überspringen</button></p>
<div id="msg" class="meta"></div>
</main>
<script>
let all=[], items=[], i=0;
function esc(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;');}
function tokens(s){return (s||'').split(/(\\s+)/);}
function diffMark(a,b){ // mark tokens of a that have no equal token in b (order-insensitive multiset)
  const bag={}; tokens(b).forEach(t=>{if(t.trim()){bag[t]=(bag[t]||0)+1;}});
  return tokens(a).map(t=>{ if(!t.trim()) return esc(t); if(bag[t]){bag[t]--; return esc(t);} return '<b>'+esc(t)+'</b>'; }).join('');
}
function filter(){ const v=document.getElementById('vol').value, m=parseFloat(document.getElementById('min').value); items=all.filter(it=>(!v||it.vol===v)&&it.cer>=m); i=0; show(); }
async function load(){ all=await (await fetch('/items')).json(); const vs=[...new Set(all.map(x=>x.vol))].sort(); const sel=document.getElementById('vol'); vs.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v.slice(-2);sel.appendChild(o);}); filter(); }
function show(){
  const it=items[i]; if(!it){document.getElementById('meta').textContent='Nichts mehr in dieser Auswahl.';document.getElementById('img').src='';document.getElementById('rg').innerHTML='';document.getElementById('rp').innerHTML='';return;}
  const n=all.indexOf(it);
  document.getElementById('prog').textContent=` ${i+1} / ${items.length} (gesamt offen ${all.length})`;
  document.getElementById('meta').textContent=`${it.vol.slice(-2)}  ${it.page}  ${it.kind}  Abweichung ${(it.cer*100).toFixed(0)} %  gültig: ${it.valid==='gemini'?'Gemini':'Transkribus'}`;
  document.getElementById('img').src=`/crop?i=${n}&t=${Date.now()}`;
  document.getElementById('before').textContent=it.before.join('\\n'); document.getElementById('after').textContent=it.after.join('\\n');
  document.getElementById('rg').innerHTML='<span class="lab">1 Gemini'+(it.valid==='gemini'?' (gültig)':'')+'</span>'+diffMark(it.gemini,it.pylaia);
  document.getElementById('rp').innerHTML='<span class="lab">2 Transkribus'+(it.valid==='pylaia'?' (gültig)':'')+'</span>'+diffMark(it.pylaia,it.gemini);
  document.getElementById('text').value=it.valid==='gemini'?it.gemini:it.pylaia; document.getElementById('msg').textContent='';
}
async function act(a){ const it=items[i]; if(!it) return; const n=all.indexOf(it); const text=document.getElementById('text').value;
  const r=await fetch('/decide',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({i:n,action:a,text})}); const j=await r.json();
  document.getElementById('msg').textContent=`${it.page} ${it.line_id}: ${j.result}`; all.splice(n,1); items.splice(i,1); show(); }
function next(){ i++; show(); }
document.getElementById('vol').onchange=filter; document.getElementById('min').onchange=filter;
document.addEventListener('keydown',e=>{ const ta=document.activeElement.tagName==='TEXTAREA';
  if(e.key==='Enter'&&e.ctrlKey){e.preventDefault();act('save');return;}
  if(ta) return;
  if(e.key==='1'){act('gemini');} else if(e.key==='2'){act('pylaia');} else if(e.key==='e'){document.getElementById('text').focus();} else if(e.key===' '){e.preventDefault();next();} });
load();
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--print", dest="print_filter")
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    items = collect(args.print_filter)
    from collections import Counter
    print(f"{len(items)} lines with differing readings  ->  http://localhost:{args.port}/   by difference:",
          dict(Counter("≥25%" if it["cer"] >= 0.25 else "≥10%" if it["cer"] >= 0.10 else "≥5%" if it["cer"] >= 0.05 else "<5%" for it in items)))

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif u.path == "/items":
                self._send(200, json.dumps(items, ensure_ascii=False).encode("utf-8"), "application/json")
            elif u.path == "/crop":
                i = int(parse_qs(u.query).get("i", ["0"])[0])
                self._send(200, crop(items[i]) if 0 <= i < len(items) else b"", "image/jpeg")
            else:
                self._send(404, b"", "text/plain")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            d = json.loads(self.rfile.read(n) or b"{}")
            it = items[int(d["i"])]
            result = apply_decision(it, d.get("action", "skip"), d.get("text", ""))
            if result != "skipped":
                items.pop(int(d["i"]))
            self._send(200, json.dumps({"result": result}).encode("utf-8"), "application/json")

    HTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
