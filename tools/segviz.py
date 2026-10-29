# Renders the PAGE XML segmentation of one dataset page as an image for the READMEs.
#   python tools/segviz.py Baldo_Cons_Print_Venice_1575_v1/0085.xml out.jpg   (path relative to the dataset folder)
#   python tools/segviz.py scan [--root DIR --list splits/test.txt]                  (line/region type counts of the listed pages)
#   python tools/segviz.py page/0100.xml out.jpg --root dist/hf-blb-st-georgen-48-pagexml --img <image dir>
import re, sys, collections
from pathlib import Path
from lxml import etree
from PIL import Image, ImageDraw, ImageFont
NS={"p":"http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
import argparse
ap=argparse.ArgumentParser(); ap.add_argument("xml"); ap.add_argument("out",nargs="?"); ap.add_argument("--root",default="dist/hf-consilia-baldus-venice1575-pagexml")
ap.add_argument("--img",help="directory of the images if not next to the XML"); ap.add_argument("--list",default="splits/test.txt")
A=ap.parse_args(); DS=Path(A.root)
sys.argv=[sys.argv[0],A.xml,A.out or ""]
def typ(el):
    m=re.search(r"(?:structure|type) \{[^}]*type:([^;}]+)", el.get("custom","")); return m.group(1) if m else "?"
def pts(s): return [tuple(map(int,p.split(","))) for p in s.split()]
COL={"cons_body":(31,119,180),"cons_argument":(255,127,14),"cons_title":(214,39,40),"header":(44,160,44),
     "page_number":(148,103,189),"catch-word":(140,86,75),
     "body":(31,119,180),"marginalia":(214,39,40),"page-number":(148,103,189),"quire-number":(255,127,14)}
REG={"column_1":(0,0,0),"column_2":(90,90,90),"header":(44,160,44),"page_number":(148,103,189),"footer":(140,86,75),
     "column_single":(0,0,0),"marginalia":(214,39,40),"page-number":(148,103,189),"quire-number":(255,127,14)}
if sys.argv[1]=="scan":
    for rel in (DS/A.list).read_text().split():
        t=etree.parse(str(DS/rel)); c=collections.Counter(typ(l) for l in t.findall(".//p:TextLine",NS))
        r=collections.Counter(typ(x) for x in t.findall(".//p:TextRegion",NS))
        print(rel, len(c), dict(c), dict(r)); 
    sys.exit()
rel, out = sys.argv[1], sys.argv[2]
t=etree.parse(str(DS/rel)); NS["p"]=t.getroot().nsmap.get(None) or NS["p"]; pg=t.find(".//p:Page",NS)  # PAGE 2013 or 2019 (kraken output)
imgname=Path(t.find(".//p:Page",NS).get("imageFilename")).name
im=Image.open((Path(A.img)/imgname) if A.img else (DS/rel).parent/imgname).convert("RGB")
ov=Image.new("RGBA",im.size,(0,0,0,0)); d=ImageDraw.Draw(ov)
for tl in t.findall(".//p:TextLine",NS):
    c=COL.get(typ(tl),(120,120,120)); co=tl.find("p:Coords",NS)
    if co is not None: d.polygon(pts(co.get("points")), fill=c+(70,), outline=c+(255,), width=3)
im=Image.alpha_composite(im.convert("RGBA"),ov); d=ImageDraw.Draw(im)
for tl in t.findall(".//p:TextLine",NS):
    bl=tl.find("p:Baseline",NS)
    if bl is not None: d.line(pts(bl.get("points")), fill=(0,0,0,255), width=5)
for r in t.findall(".//p:TextRegion",NS):
    c=REG.get(typ(r),(0,0,0)); d.polygon(pts(r.find("p:Coords",NS).get("points")), outline=c+(255,), width=9)
# legend strip below the page: line types, region types (wrapping rows)
font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 52)
bold=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52)
present_l={typ(l) for l in t.findall(".//p:TextLine",NS)}; present_r={typ(r) for r in t.findall(".//p:TextRegion",NS)}
rows=[("line type", [(k,c,"fill") for k,c in COL.items() if k in present_l]+[("baseline",(0,0,0),"line")]),
      ("region type", [(k,c,"outline") for k,c in REG.items() if k in present_r])]
W,H=im.size; LH=95; tmp=ImageDraw.Draw(Image.new("RGBA",(1,1)))
x0=60+max(tmp.textlength(t,font=bold) for t,_ in rows)+60
layout=[]; y=20
for title,items in rows:
    x=x0; layout.append(("title",title,60,y))
    for k,c,kind in items:
        w=80+tmp.textlength(k,font=font)+60
        if x+w>W-40: y+=LH; x=x0
        layout.append((kind,k,c,x,y)); x+=w
    y+=LH
strip=Image.new("RGBA",(W,y+20),(255,255,255,255)); d=ImageDraw.Draw(strip)
for it in layout:
    if it[0]=="title": d.text((it[2],it[3]+12), it[1], fill=(0,0,0,255), font=bold); continue
    kind,k,c,x,y=it
    if kind=="line": d.line([x,y+42,x+70,y+42], fill=(0,0,0,255), width=7)
    elif kind=="outline": d.rectangle([x,y+12,x+70,y+72], outline=c+(255,), width=8)
    else: d.rectangle([x,y+12,x+70,y+72], fill=c+(90,), outline=c+(255,), width=4)
    d.text((x+90,y+12), k, fill=(0,0,0,255), font=font)
canvas=Image.new("RGBA",(W,H+strip.height),(255,255,255,255)); canvas.paste(im,(0,0)); canvas.paste(strip,(0,H)); im=canvas
im=im.convert("RGB"); im.thumbnail((1400,2200)); im.save(out, quality=88, optimize=True)
print(out, im.size)
