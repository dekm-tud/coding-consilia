#!/usr/bin/env python3
"""
Panormitanus, Consilia (Transkribus-Dok. 18098597): Spalten-Crops aus PageXML
und zeilengenaue Transkription mit Gemini 3.8 Flash.

Schritte
  crop   Hauptspalten (TextRegions vom Typ column_N) mit Rand ausschneiden
  ocr    Crops einzeln an die Gemini-API senden (sofort, Standardpreis)
  batch-submit / batch-collect
         dasselbe über die Batch-API (halber Preis, Ergebnis binnen 24 h)
  merge  Alle Transkriptionen in Lesereihenfolge zu einer Datei zusammenführen

Beispiele (im Ordner mit den JPGs und dem Unterordner page/):
  python pipeline/pipeline.py crop  --data .
  python pipeline/pipeline.py ocr   --data . --pages 10-12
  python pipeline/pipeline.py ocr   --data .
  python pipeline/pipeline.py merge --data .

API-Key: Umgebungsvariable GEMINI_API_KEY (oder GOOGLE_API_KEY).
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageDraw

NS = {"p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
COLUMN_RE = re.compile(r"^column_(\d+)$")
HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #
def parse_pages(spec: str | None) -> set[int] | None:
    """'10-12,200' -> {10,11,12,200}"""
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def page_nr(stem: str) -> int:
    """'0123' / '0123_c1' / 'page_001' / 'page_001_c2' -> page number (file names of the Baldus volumes differ)."""
    m = re.search(r"(\d+)$", stem.split("_c")[0])
    if not m:
        raise ValueError(f"no page number in {stem!r}")
    return int(m.group(1))


def region_type(custom: str) -> str | None:
    m = re.search(r"type:([^;}]+);", custom or "")
    return m.group(1).strip() if m else None


def bbox(points: str) -> tuple[int, int, int, int]:
    xs, ys = [], []
    for pt in points.split():
        x, y = pt.split(",")
        xs.append(int(x))
        ys.append(int(y))
    return min(xs), min(ys), max(xs), max(ys)


def area(b) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


# --------------------------------------------------------------------------- #
# crop
# --------------------------------------------------------------------------- #
def read_layout(xml_path: Path) -> dict:
    """Spalten- und Kopfzeilenregionen einer Seite.

    Rückgabe: dict(img, W, H, header=(bbox)|None,
                   cols={n: dict(bbox, ids, n_regions)})
    Mehrere Regionen desselben Spaltentyps werden zusammengefasst: x-Ausdehnung der
    flächengrößten Region (Randnotizen anderer Teilregionen bleiben außen vor),
    y-Ausdehnung der Vereinigung aller Teilregionen.
    """
    root = ET.parse(xml_path).getroot()
    page = root.find("p:Page", NS)
    out = dict(img=page.get("imageFilename"), W=int(page.get("imageWidth")),
               H=int(page.get("imageHeight")), header=None, cols={})
    cands: dict[int, list] = {}
    headers = []
    for tr in page.findall("p:TextRegion", NS):
        coords = tr.find("p:Coords", NS)
        if coords is None:
            continue
        b = bbox(coords.get("points"))
        t = region_type(tr.get("custom", "")) or ""
        m = COLUMN_RE.match(t)
        if m:
            cands.setdefault(int(m.group(1)), []).append((b, tr.get("id")))
        elif t == "header":
            headers.append(b)
    if headers:
        out["header"] = max(headers, key=area)
    for n, lst in cands.items():
        big = max(lst, key=lambda c: area(c[0]))[0]
        y0 = min(c[0][1] for c in lst)
        y1 = max(c[0][3] for c in lst)
        out["cols"][n] = dict(bbox=(big[0], y0, big[2], y1),
                              ids=";".join(c[1] for c in lst), n_regions=len(lst))
    return out


def median(vals):
    vals = sorted(vals)
    if not vals:
        return None
    k = len(vals) // 2
    return vals[k] if len(vals) % 2 else (vals[k - 1] + vals[k]) / 2


def normalize_layouts(layouts: dict[int, dict], window: int, tol: int, ncols: int):
    """Vereinheitlicht die Spaltengeometrie.

    1. Beide Spalten einer Seite teilen Ober- und Unterkante des Satzspiegels:
       oben = Unterkante der Kopfzeile (falls vorhanden), sonst oberste Spaltenkante;
       unten = unterste Spaltenkante der Seite.
    2. Abgleich mit dem gleitenden Median der Nachbarseiten gleicher Parität
       (gerade/ungerade Aufnahmen sind unterschiedlich positioniert, die Lage driftet
       über den Band). Liegt die Oberkante mehr als `tol` px unter dem Median bzw. die
       Unterkante mehr als `tol` px über dem Median, wird bis zum Median erweitert.
       Es wird nur erweitert, nie beschnitten.
    3. Fehlt eine Spalte, wird sie aus dem Median-Abstand/-Breite der Nachbarseiten
       relativ zur vorhandenen Spalte ergänzt.
    """
    # Schritt 1: seiteninterne Kanten
    for pg, L in layouts.items():
        if not L["cols"]:
            continue
        cb = [c["bbox"] for c in L["cols"].values()]
        top = min(b[1] for b in cb)
        flags_top = "spalte"
        h = L["header"]
        if h is not None:
            # Kopfzeile nur verwenden, wenn sie horizontal über den Spalten liegt
            cx0, cx1 = min(b[0] for b in cb), max(b[2] for b in cb)
            if min(h[2], cx1) - max(h[0], cx0) > 0.3 * (cx1 - cx0):
                top, flags_top = h[3], "kopfzeile"
        L["top"], L["bottom"] = top, max(b[3] for b in cb)
        L["flags"] = [f"oben:{flags_top}"]

    def neighbours(pg):
        same = [p for p in layouts if p % 2 == pg % 2 and layouts[p]["cols"]
                and len(layouts[p]["cols"]) == ncols]
        same.sort(key=lambda p: abs(p - pg))
        return same[: 2 * window + 1]

    # Schritt 2 + 3 (Statistik aus den Rohwerten von Schritt 1)
    raw = {p: (L.get("top"), L.get("bottom")) for p, L in layouts.items()}
    for pg, L in layouts.items():
        if not L["cols"]:
            continue
        nb = neighbours(pg)
        if not nb:
            continue
        mtop = median([raw[p][0] for p in nb])
        mbot = median([raw[p][1] for p in nb])
        if L["top"] > mtop + tol:
            L["flags"].append(f"oben:median({L['top']}->{int(mtop)})")
            L["top"] = int(mtop)
        if L["bottom"] < mbot - tol:
            L["flags"].append(f"unten:median({L['bottom']}->{int(mbot)})")
            L["bottom"] = int(mbot)

        present = set(L["cols"])
        for n in range(1, ncols + 1):
            if n in present:
                continue
            ref = min(present, key=lambda k: abs(k - n))
            offs, widths = [], []
            for p in nb:
                c = layouts[p]["cols"]
                offs.append(c[n]["bbox"][0] - c[ref]["bbox"][0])
                widths.append(c[n]["bbox"][2] - c[n]["bbox"][0])
            x0 = L["cols"][ref]["bbox"][0] + int(median(offs))
            L["cols"][n] = dict(bbox=(x0, L["top"], x0 + int(median(widths)), L["bottom"]),
                                ids="(ergänzt)", n_regions=0)
            L["flags"].append(f"spalte_{n}:ergänzt")

    # Gemeinsame Kanten auf alle Spalten anwenden
    for L in layouts.values():
        if not L["cols"]:
            continue
        for c in L["cols"].values():
            x0, _, x1, _ = c["bbox"]
            c["norm"] = (max(0, x0), max(0, L["top"]), min(L["W"], x1), min(L["H"], L["bottom"]))


def cmd_crop(args):
    data = Path(args.data)
    page_dir = data / "page"
    out_dir = data / args.crops
    out_dir.mkdir(parents=True, exist_ok=True)
    dbg_dir = out_dir / "_debug"
    if args.debug:
        dbg_dir.mkdir(exist_ok=True)
    pages = parse_pages(args.pages)

    # Statistik immer über alle Seiten, auch wenn nur eine Auswahl ausgeschnitten wird
    layouts = {page_nr(p.stem): dict(read_layout(p), stem=p.stem) for p in sorted(page_dir.glob("*.xml"))}
    raw_cols = {p: {n: c["bbox"] for n, c in L["cols"].items()} for p, L in layouts.items()}
    if not args.no_normalize:
        normalize_layouts(layouts, args.window, args.tol, args.ncols)
    else:
        for L in layouts.values():
            L["flags"] = []
            for c in L["cols"].values():
                c["norm"] = c["bbox"]

    # Mediane Spaltenbreite je Spaltennummer über den ganzen Band
    widths: dict[int, list] = {}
    for L in layouts.values():
        for n, c in L["cols"].items():
            widths.setdefault(n, []).append(c["norm"][2] - c["norm"][0])
    med_w = {n: median(v) for n, v in widths.items()}

    rows, skipped = [], []
    for pg in sorted(layouts):
        if pages and pg not in pages:
            continue
        L = layouts[pg]
        stem = L.get("stem") or f"{pg:04d}"      # crop names follow the PageXML file names
        if not L["cols"]:
            skipped.append(stem)
            continue
        img_path = data / L["img"]
        if not img_path.exists():
            print(f"[WARN] Bild fehlt: {img_path}", file=sys.stderr)
            continue
        im = Image.open(img_path)
        im.load()
        W, H = im.size
        dbg = im.copy().convert("RGB") if args.debug else None
        draw = ImageDraw.Draw(dbg) if dbg else None

        for n in sorted(L["cols"]):
            c = L["cols"][n]
            x0, y0, x1, y1 = c["norm"]
            px = args.pad_x
            narrow = (x1 - x0) < args.narrow_ratio * med_w[n]
            if narrow:
                px += args.narrow_pad
            cx0, cy0 = max(0, x0 - px), max(0, y0 - args.pad_y)
            cx1, cy1 = min(W, x1 + px), min(H, y1 + args.pad_y)
            name = f"{stem}_c{n}.jpg"
            dst = out_dir / name
            if not dst.exists() or args.overwrite:
                im.crop((cx0, cy0, cx1, cy1)).convert("RGB").save(dst, quality=args.quality)
            rb = raw_cols[pg].get(n)
            rows.append(dict(page=stem, column=n, crop=name, region_ids=c["ids"],
                             n_regions=c["n_regions"],
                             region_bbox=",".join(map(str, rb)) if rb else "",
                             norm_bbox=",".join(map(str, c["norm"])),
                             crop_bbox=f"{cx0},{cy0},{cx1},{cy1}",
                             flags=" ".join(L["flags"] + ([f"schmal(+{args.narrow_pad}px)"]
                                                          if narrow else []))))
            if draw:
                if rb:
                    draw.rectangle(rb, outline=(220, 0, 0), width=6)
                draw.rectangle((cx0, cy0, cx1, cy1), outline=(0, 160, 0), width=8)
        if dbg:
            dbg.thumbnail((W // 4, H // 4))
            dbg.save(dbg_dir / f"{stem}.jpg", quality=80)
        print(f"{stem}: {len(L['cols'])} Spalte(n)  {' '.join(L['flags'])}")

    manifest = out_dir / ("_manifest.csv" if pages is None else "_manifest_auswahl.csv")
    fields = ["page", "column", "crop", "region_ids", "n_regions", "region_bbox",
              "norm_bbox", "crop_bbox", "flags"]
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} Crops in {out_dir}")
    if skipped:
        print(f"Ohne Spaltenregionen übersprungen ({len(skipped)}): {', '.join(skipped)}")
    corr = sorted({r["page"] for r in rows if "median" in r["flags"] or "ergänzt" in r["flags"]})
    if corr:
        print(f"An Nachbarseiten angeglichen ({len(corr)}): {', '.join(corr)}")
    narrow_crops = [r["crop"] for r in rows if "schmal" in r["flags"]]
    if narrow_crops:
        print(f"Schmale Spalten, seitlich +{args.narrow_pad} px ({len(narrow_crops)}): "
              f"{', '.join(narrow_crops)}")
    multi = sorted({r["page"] for r in rows if r["n_regions"] > 1})
    if multi:
        print(f"Mehrfachregionen zusammengefasst ({len(multi)}): {', '.join(multi)}")


# --------------------------------------------------------------------------- #
# ocr
# --------------------------------------------------------------------------- #
FENCE_RE = re.compile(r"^\s*```[^\n]*\n|\n?```\s*$")
RES_LEVELS = {
    "low": "MEDIA_RESOLUTION_LOW",
    "medium": "MEDIA_RESOLUTION_MEDIUM",
    "high": "MEDIA_RESOLUTION_HIGH",
    "ultra_high": "MEDIA_RESOLUTION_ULTRA_HIGH",
}
RETRY_CODES = {408, 429, 500, 502, 503, 504}
# Kürzungszeichen, die nach der Auflösung nicht mehr vorkommen dürfen
ABBR_RE = re.compile("[ãẽĩõũñỹāēīōūȳÃẼĨÕŨÑĀĒĪŌŪꝑꝓꝗꝙꝯꝰꝫꝝ\u0303\u0304\u0305\u0363-\u036f]")


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = FENCE_RE.sub("", text)
    lines = [ln.rstrip() for ln in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n"


def cmd_ocr(args):
    from google import genai
    from google.genai import errors, types

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY ist nicht gesetzt.")
    src = "GEMINI_API_KEY" if os.environ.get("GEMINI_API_KEY") else "GOOGLE_API_KEY"
    # Das SDK bevorzugt sonst stillschweigend GOOGLE_API_KEY, wenn beide gesetzt sind.
    os.environ.pop("GOOGLE_API_KEY", None)
    os.environ["GEMINI_API_KEY"] = key
    print(f"API-Key aus {src} (…{key[-4:]})")
    # Spaltenweise media_resolution erfordert v1beta.
    client = genai.Client(api_key=key, http_options=types.HttpOptions(api_version="v1beta"))

    prompt = Path(args.prompt).read_text(encoding="utf-8").strip()
    config = types.GenerateContentConfig(
        system_instruction=prompt,
        thinking_config=types.ThinkingConfig(thinking_level=getattr(types.ThinkingLevel, args.thinking.upper())),
        max_output_tokens=args.max_output_tokens,
        response_mime_type="application/json",
        response_json_schema={
            "type": "object",
            "properties": {"zeilen": {"type": "array", "items": {"type": "string"},
                                      "description": "eine gedruckte Zeile pro Eintrag"}},
            "required": ["zeilen"],
        },
    )
    main_prompt = prompt
    rules = main_prompt[main_prompt.index("ABKÜRZUNGEN"):main_prompt.index("ZEICHENTREUE")].strip()
    expand_prompt = Path(args.expand_prompt).read_text(encoding="utf-8").strip() \
        .replace("{ABKUERZUNGEN}", rules)
    expand_config = types.GenerateContentConfig(
        system_instruction=expand_prompt,
        thinking_config=types.ThinkingConfig(
            thinking_level=getattr(types.ThinkingLevel, args.expand_thinking.upper())),
        max_output_tokens=args.max_output_tokens,
        response_mime_type=config.response_mime_type,
        response_json_schema=config.response_json_schema,
    )
    res = types.PartMediaResolution(
        level=getattr(types.PartMediaResolutionLevel, RES_LEVELS[args.resolution]))

    data = Path(args.data)
    crop_dir = data / args.crops
    out_dir = data / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "_log.jsonl"
    log_lock = threading.Lock()

    pages = parse_pages(args.pages)
    todo = []
    for p in sorted(crop_dir.glob("*_c*.jpg")):
        page = page_nr(p.stem)
        if pages and page not in pages:
            continue
        if (out_dir / f"{p.stem}.txt").exists() and not args.overwrite:
            continue
        todo.append(p)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} Crops zu transkribieren (Modell {args.model}, "
          f"thinking_level={args.thinking}, media_resolution={args.resolution}).")
    if not todo:
        return

    def log(rec):
        with log_lock, log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def parse(text: str) -> list[str]:
        """JSON {"zeilen": [...]} -> Zeilenliste; Rückfall auf Klartext."""
        try:
            raw = json.loads(text)["zeilen"]
            if not isinstance(raw, list):
                raise ValueError
        except Exception:
            if text.lstrip().startswith("{"):
                return []     # unvollständiges JSON: verwerfen statt Bruchstücke zu speichern
            raw = clean(text).split("\n")
        lines = []
        for item in raw:
            for ln in str(item).replace("\r", "").split("\n"):
                ln = ln.strip()
                if ln:
                    lines.append(ln)
        return lines

    def check(lines: list[str]) -> dict:
        long = sum(1 for ln in lines if len(ln) > args.max_line_len)
        resid = sum(len(ABBR_RE.findall(ln)) for ln in lines)
        return dict(lines=len(lines), long_lines=long, residual_abbr=resid,
                    ok=bool(lines) and long == 0 and resid <= args.max_residual)

    class QuotaStop(Exception):
        pass

    stop = threading.Event()
    stop_msg = []

    def call(contents, cfg, label):
        """Eine API-Anfrage mit Wiederholung bei vorübergehenden Fehlern.

        Bei erschöpftem Kontingent mit langer Wartezeit (z. B. Tageslimit) wird der
        ganze Lauf angehalten, statt weitere Anfragen zu verbrauchen.
        """
        for attempt in range(1, args.retries + 1):
            if stop.is_set():
                raise QuotaStop()
            t0 = time.time()
            try:
                resp = client.models.generate_content(model=args.model, contents=contents,
                                                      config=cfg)
                return resp, attempt, round(time.time() - t0, 1)
            except errors.APIError as e:
                code = getattr(e, "code", None)
                msg = str(e)
                m = re.search(r"retryDelay['\"]?:\s*['\"](\d+)", msg)
                delay = int(m.group(1)) if m else None
                log(dict(crop=label, attempt=attempt, error=f"APIError {code}: {msg[:300]}",
                         ts=time.strftime("%Y-%m-%dT%H:%M:%S")))
                if code == 429 and (("PerDay" in msg) or (delay and delay > args.max_wait)):
                    if not stop.is_set():
                        stop_msg.append(msg.split("\\n")[0][:400])
                        stop.set()
                    raise QuotaStop()
                if code not in RETRY_CODES or attempt == args.retries:
                    raise
                time.sleep((delay + 1) if delay else min(60, 2 ** attempt) + random.random())
            except (QuotaStop, KeyboardInterrupt):
                raise
            except Exception as e:  # Netzwerk, Verbindungsabbruch
                log(dict(crop=label, attempt=attempt, error=f"{type(e).__name__}: {e}",
                         ts=time.strftime("%Y-%m-%dT%H:%M:%S")))
                if attempt == args.retries:
                    raise
                time.sleep(min(60, 2 ** attempt) + random.random())

    def record(label, step, resp, attempt, secs, lines):
        cand = resp.candidates[0] if resp.candidates else None
        um = resp.usage_metadata
        info = check(lines)
        finish = str(getattr(cand, "finish_reason", "")) if cand else ""
        info["truncated"] = "MAX_TOKENS" in finish
        if info["truncated"]:
            info["ok"] = False
        info.update(crop=label, step=step, attempt=attempt, seconds=secs,
                    finish_reason=finish,
                    prompt_tokens=getattr(um, "prompt_token_count", None),
                    output_tokens=getattr(um, "candidates_token_count", None),
                    thinking_tokens=getattr(um, "thoughts_token_count", None),
                    ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
        log(info)
        return info

    def work(path: Path):
        if stop.is_set():
            return f"{path.name}: übersprungen (Kontingent erschöpft)"
        part = types.Part.from_bytes(data=path.read_bytes(), mime_type="image/jpeg",
                                     media_resolution=res)
        try:
            # 1. Transkription; bei zusammengezogenen Zeilen vollständig wiederholen
            best = None
            for q in range(args.quality_retries + 1):
                resp, att, secs = call([part], config, path.name)
                lines = parse(resp.text or "")
                info = record(path.name, "transkription", resp, att, secs, lines)
                if info["truncated"]:
                    continue          # abgeschnittene Antwort ist unbrauchbar
                if best is None or (info["long_lines"], -info["lines"]) < \
                        (best[1]["long_lines"], -best[1]["lines"]):
                    best = (lines, info)
                if lines and info["long_lines"] == 0:
                    break
            if best is None:
                return (f"{path.name}: FEHLER – Antwort abgeschnitten (MAX_TOKENS); "
                        f"--max-output-tokens erhöhen")
            lines, info = best
            # 2. Gezielte Nachauflösung verbliebener Kürzungszeichen
            if args.expand and lines and info["residual_abbr"] > args.max_residual:
                inp = types.Part.from_text(text=json.dumps({"zeilen": lines}, ensure_ascii=False))
                resp, att, secs = call([part, inp], expand_config, path.name)
                new = parse(resp.text or "")
                info2 = record(path.name, "nachaufloesung", resp, att, secs, new)
                if not info2["truncated"] and len(new) == len(lines) \
                        and info2["residual_abbr"] < info["residual_abbr"]:
                    lines, info = new, info2
                else:
                    log(dict(crop=path.name, step="nachaufloesung", verworfen=True,
                             grund=f"{len(new)} statt {len(lines)} Zeilen" if len(new) != len(lines)
                             else "keine Verbesserung"))
        except QuotaStop:
            return f"{path.name}: abgebrochen (Kontingent erschöpft)"
        except Exception as e:
            return f"{path.name}: FEHLER – {type(e).__name__}: {str(e)[:200]}"

        if not lines:
            return f"{path.name}: FEHLER – keine Zeilen erkannt"
        with (out_dir / f"{path.stem}.txt").open("w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")
        msg = f"{path.name}: {info['lines']} Zeilen"
        if not info["ok"]:
            msg += (f"  [PRÜFEN: {info['long_lines']} überlange Zeile(n), "
                    f"{info['residual_abbr']} unaufgelöste Kürzungszeichen]")
        return msg

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, p) for p in todo]
        for i, f in enumerate(as_completed(futs), 1):
            print(f"[{i}/{len(todo)}] {f.result()}", flush=True)
    if stop.is_set():
        print("\nLauf angehalten: Kontingent erschöpft. Fertige Spalten sind gespeichert; "
              "ein erneuter Aufruf ohne --overwrite setzt fort.")
        if stop_msg:
            print(f"Meldung der API: {stop_msg[0]}")


# --------------------------------------------------------------------------- #
# batch
# --------------------------------------------------------------------------- #
SCHEMA = {
    "type": "object",
    "properties": {"zeilen": {"type": "array", "items": {"type": "string"},
                              "description": "eine gedruckte Zeile pro Eintrag"}},
    "required": ["zeilen"],
}


def parse_lines(text: str) -> list[str]:
    """JSON {"zeilen": [...]} -> Zeilenliste; unvollständiges JSON wird verworfen."""
    try:
        raw = json.loads(text)["zeilen"]
        if not isinstance(raw, list):
            raise ValueError
    except Exception:
        if text.lstrip().startswith("{"):
            return []
        raw = clean(text).split("\n")
    out = []
    for item in raw:
        for ln in str(item).replace("\r", "").split("\n"):
            ln = ln.strip()
            if ln:
                out.append(ln)
    return out


def check_lines(lines: list[str], args) -> dict:
    long = sum(1 for ln in lines if len(ln) > args.max_line_len)
    resid = sum(len(ABBR_RE.findall(ln)) for ln in lines)
    return dict(lines=len(lines), long_lines=long, residual_abbr=resid,
                ok=bool(lines) and long == 0 and resid <= args.max_residual)


def make_client():
    from google import genai
    from google.genai import types
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY ist nicht gesetzt.")
    src = "GEMINI_API_KEY" if os.environ.get("GEMINI_API_KEY") else "GOOGLE_API_KEY"
    os.environ.pop("GOOGLE_API_KEY", None)
    os.environ["GEMINI_API_KEY"] = key
    print(f"API-Key aus {src} (…{key[-4:]})")
    return genai.Client(api_key=key, http_options=types.HttpOptions(api_version="v1beta"))


def load_prompts(args) -> tuple[str, str]:
    main_prompt = Path(args.prompt).read_text(encoding="utf-8").strip()
    rules = main_prompt[main_prompt.index("ABKÜRZUNGEN"):main_prompt.index("ZEICHENTREUE")].strip()
    expand_prompt = Path(args.expand_prompt).read_text(encoding="utf-8").strip() \
        .replace("{ABKUERZUNGEN}", rules)
    return main_prompt, expand_prompt


def _g(d, *names, default=None):
    """Feldzugriff unabhängig von camelCase/snake_case."""
    for n in names:
        if isinstance(d, dict) and n in d:
            return d[n]
    return default


class BatchState:
    """Buchführung über eingereichte Batch-Jobs (<out>/_batch/_jobs.json)."""

    def __init__(self, out_dir: Path):
        self.dir = out_dir / "_batch"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "_jobs.json"
        self.pending_expand_path = self.dir / "_nachaufloesung_offen.json"
        self.jobs = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []
        self.pending_expand = set(json.loads(self.pending_expand_path.read_text(encoding="utf-8"))) \
            if self.pending_expand_path.exists() else set()

    def open_keys(self, kind: str) -> set[str]:
        return {k for j in self.jobs if j["kind"] == kind and not j.get("collected")
                for k in j["keys"]}

    def save(self):
        self.path.write_text(json.dumps(self.jobs, ensure_ascii=False, indent=1), encoding="utf-8")
        self.pending_expand_path.write_text(json.dumps(sorted(self.pending_expand)),
                                            encoding="utf-8")


def cmd_batch_submit(args):
    data = Path(args.data)
    crop_dir = data / args.crops
    out_dir = data / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    st = BatchState(out_dir)
    main_prompt, expand_prompt = load_prompts(args)
    kind = "nachaufloesung" if args.expand else "transkription"
    pages = parse_pages(args.pages)

    # Auswahl
    busy = st.open_keys(kind)
    if args.expand:
        keys = sorted(k for k in st.pending_expand if k not in busy
                      and (out_dir / f"{k}.txt").exists())
    else:
        keys = []
        for p in sorted(crop_dir.glob("*_c*.jpg")):
            if pages and page_nr(p.stem) not in pages:
                continue
            if p.stem in busy:
                continue
            if (out_dir / f"{p.stem}.txt").exists() and not args.overwrite:
                continue
            keys.append(p.stem)
    if args.limit:
        keys = keys[: args.limit]
    if busy:
        print(f"{len(busy)} Spalten sind bereits in offenen Jobs und werden übersprungen.")
    if not keys:
        print("Nichts einzureichen.")
        return

    gen = {
        "thinkingConfig": {"thinkingLevel": (args.expand_thinking if args.expand
                                             else args.thinking).upper()},
        "maxOutputTokens": args.max_output_tokens,
        "responseMimeType": "application/json",
        "responseJsonSchema": SCHEMA,
    }
    sysinst = {"parts": [{"text": expand_prompt if args.expand else main_prompt}]}

    def request(key: str) -> dict:
        img = base64.b64encode((crop_dir / f"{key}.jpg").read_bytes()).decode("ascii")
        parts = [{"inlineData": {"mimeType": "image/jpeg", "data": img},
                  "mediaResolution": {"level": RES_LEVELS[args.resolution]}}]
        if args.expand:
            lines = (out_dir / f"{key}.txt").read_text(encoding="utf-8").splitlines()
            parts.append({"text": json.dumps({"zeilen": lines}, ensure_ascii=False)})
        return {"key": key, "request": {"contents": [{"role": "user", "parts": parts}],
                                        "systemInstruction": sysinst,
                                        "generationConfig": gen}}

    # JSONL-Dateien in Paketen bis --chunk-mb schreiben
    stamp = time.strftime("%Y%m%d-%H%M%S")
    while any(p.name.startswith(f"{kind}_{stamp}") for p in st.dir.iterdir()) or \
            any(j["file"].startswith(f"{kind}_{stamp}") for j in st.jobs):
        time.sleep(1)
        stamp = time.strftime("%Y%m%d-%H%M%S")
    chunks, cur, cur_keys, size = [], None, [], 0
    limit = args.chunk_mb * 1024 * 1024
    for key in keys:
        line = json.dumps(request(key), ensure_ascii=False) + "\n"
        b = len(line.encode("utf-8"))
        if cur is None or size + b > limit:
            if cur is not None:
                cur.close()
            path = st.dir / f"{kind}_{stamp}_{len(chunks) + 1:02d}.jsonl"
            cur, cur_keys, size = path.open("w", encoding="utf-8"), [], 0
            chunks.append((path, cur_keys))
        cur.write(line)
        cur_keys.append(key)
        size += b
    cur.close()
    for path, ks in chunks:
        print(f"{path.name}: {len(ks)} Anfragen, {path.stat().st_size / 1e6:.0f} MB")
    if args.dry_run:
        print("Probelauf: nichts hochgeladen.")
        return

    from google.genai import types
    client = make_client()
    for path, ks in chunks:
        print(f"Lade {path.name} hoch …", flush=True)
        up = client.files.upload(file=str(path), config=types.UploadFileConfig(
            display_name=path.stem, mime_type="jsonl"))
        job = client.batches.create(model=args.model, src=up.name,
                                    config={"display_name": path.stem})
        st.jobs.append(dict(name=job.name, kind=kind, file=path.name, keys=ks,
                            model=args.model, created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                            collected=False))
        st.save()
        print(f"  Job {job.name} eingereicht ({len(ks)} Spalten).")
        if not args.keep_jsonl:
            path.unlink()
    print("\nStatus abfragen und Ergebnisse abholen mit:  pipeline.py batch-collect --data . "
          f"--out {args.out}")


def cmd_batch_collect(args):
    data = Path(args.data)
    out_dir = data / args.out
    st = BatchState(out_dir)
    if not st.jobs:
        sys.exit("Keine Batch-Jobs verzeichnet.")
    client = make_client()
    log_path = out_dir / "_log.jsonl"
    done_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
                   "JOB_STATE_EXPIRED"}

    def log(rec):
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    while True:
        open_jobs = [j for j in st.jobs if not j.get("collected")]
        if not open_jobs:
            break
        waiting = 0
        for j in open_jobs:
            try:
                job = client.batches.get(name=j["name"])
            except Exception as e:
                print(f"{j['file']}: Abfrage fehlgeschlagen ({type(e).__name__}: {e}) – "
                      f"nächster Versuch beim nächsten Durchlauf")
                waiting += 1
                continue
            state = job.state.name if hasattr(job.state, "name") else str(job.state)
            print(f"{j['file']} [{j['kind']}, {len(j['keys'])} Spalten]: {state}")
            if state not in done_states:
                waiting += 1
                continue
            j["state"] = state
            if state != "JOB_STATE_SUCCEEDED":
                j["collected"] = True       # Spalten werden beim nächsten Einreichen erneut erfasst
                st.save()
                continue
            try:
                raw = client.files.download(file=job.dest.file_name)
            except Exception as e:
                print(f"  Herunterladen fehlgeschlagen ({type(e).__name__}) – erneut beim nächsten Aufruf")
                waiting += 1
                continue
            res_path = st.dir / f"{Path(j['file']).stem}_ergebnis.jsonl"
            res_path.write_bytes(raw)
            stats = dict(geschrieben=0, fehler=0, nachaufloesung=0, verbessert=0)
            for ln in raw.decode("utf-8").splitlines():
                if not ln.strip():
                    continue
                rec = json.loads(ln)
                key = _g(rec, "key")
                resp = _g(rec, "response")
                if not resp:
                    stats["fehler"] += 1
                    log(dict(crop=f"{key}.jpg", step=j["kind"], batch=j["name"],
                             error=str(_g(rec, "error", "status"))[:300]))
                    continue
                cands = _g(resp, "candidates", default=[]) or [{}]
                cand = cands[0]
                parts = _g(_g(cand, "content", default={}), "parts", default=[]) or []
                text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                finish = str(_g(cand, "finishReason", "finish_reason", default=""))
                um = _g(resp, "usageMetadata", "usage_metadata", default={}) or {}
                lines = parse_lines(text)
                info = check_lines(lines, args)
                info.update(crop=f"{key}.jpg", step=j["kind"], batch=j["name"],
                            finish_reason=finish, truncated="MAX_TOKENS" in finish,
                            prompt_tokens=_g(um, "promptTokenCount", "prompt_token_count"),
                            output_tokens=_g(um, "candidatesTokenCount", "candidates_token_count"),
                            thinking_tokens=_g(um, "thoughtsTokenCount", "thoughts_token_count"),
                            ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
                log(info)
                txt = out_dir / f"{key}.txt"
                if j["kind"] == "transkription":
                    if info["truncated"] or not lines or info["long_lines"]:
                        stats["fehler"] += 1        # keine Datei -> wird erneut eingereicht
                        continue
                    with txt.open("w", encoding="utf-8", newline="\n") as fh:
                        fh.write("\n".join(lines) + "\n")
                    stats["geschrieben"] += 1
                    if info["residual_abbr"] > args.max_residual:
                        st.pending_expand.add(key)
                        stats["nachaufloesung"] += 1
                else:
                    old = txt.read_text(encoding="utf-8").splitlines() if txt.exists() else []
                    old_res = sum(len(ABBR_RE.findall(x)) for x in old)
                    if not info["truncated"] and len(lines) == len(old) \
                            and info["residual_abbr"] < old_res:
                        with txt.open("w", encoding="utf-8", newline="\n") as fh:
                            fh.write("\n".join(lines) + "\n")
                        stats["verbessert"] += 1
                    st.pending_expand.discard(key)
            j["collected"] = True
            j["stats"] = stats
            st.save()
            print(f"  abgeholt: {stats}")
        if not waiting or not args.wait:
            break
        print(f"{waiting} Job(s) laufen noch – nächste Abfrage in {args.poll} s …", flush=True)
        time.sleep(args.poll)

    missing = [p.stem for p in sorted((data / args.crops).glob("*_c*.jpg"))
               if not (out_dir / f"{p.stem}.txt").exists()]
    still_open = st.open_keys("transkription")
    retry = [k for k in missing if k not in still_open]
    print(f"\nTranskriptionen vorhanden: "
          f"{len(list(out_dir.glob('*_c*.txt')))}; offen in laufenden Jobs: {len(still_open)}; "
          f"neu einzureichen: {len(retry)}; Nachauflösung ausstehend: {len(st.pending_expand)}")
    if retry:
        print("  → pipeline.py batch-submit  (reicht fehlende Spalten erneut ein)")
    if st.pending_expand:
        print("  → pipeline.py batch-submit --expand  (Nachauflösung verbliebener Kürzungen)")


# --------------------------------------------------------------------------- #
# normalize
# --------------------------------------------------------------------------- #
_PREFIX = re.compile(r"^((?:\{S\} )?(?:\{\d+\} )?)")
_SP_BEFORE = re.compile(r"\s+([,;:.?!)])")
_SP_AFTER = re.compile(r"([,;:?!])(?=[A-Za-zÀ-ÿſæœęÆŒ&\[¶†(])")
_SP_MULTI = re.compile(r" {2,}")
# „ēt“ (etiam) wird vom Modell als „ét“ gelesen; im Druck steht ein Kürzungsstrich
_ET = re.compile(r"(?<![\wſ\[\]])([éÉ])t(?![\wſ\[])")
# akuter Akzent außerhalb von Enklitika (-que/-ue/-ne): meist verlesener Nasalstrich
_ACUTE = re.compile(r"[^\s.,;:]*[áéíóúÁÉÍÓÚ][^\s.,;:]*")
_ENCLITIC = re.compile(r"[áéíóú](que|ue|ne|q\[ue\])-?$")


def normalize_line(line: str) -> str:
    """Leerzeichen an Satzzeichen vereinheitlichen; Markierungen am Zeilenanfang bleiben.

    - kein Leerzeichen vor , ; : . ? ! )
    - genau ein Leerzeichen nach , ; : ? ! vor einem Wort (nicht nach Punkten,
      damit Allegationen wie „C.de cadu.tollen.l.1.“ unverändert bleiben)
    - mehrfache Leerzeichen zu einem
    - „ét“ (Kürzung für etiam) -> „et[iam]“
    """
    m = _PREFIX.match(line)
    pre, body = m.group(1), line[m.end():]
    body = _SP_BEFORE.sub(r"\1", body)
    body = _SP_AFTER.sub(r"\1 ", body)
    body = _SP_MULTI.sub(" ", body).strip()
    body = _ET.sub(lambda mm: ("e" if mm.group(1) == "é" else "E") + "t[iam]", body)
    return (pre + body) if body else pre.rstrip()


def cmd_normalize(args):
    data = Path(args.data)
    src = data / args.out
    dst = data / args.target
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*_c*.txt"))
    changed_lines = total = et = 0
    review = []
    for p in files:
        lines = p.read_text(encoding="utf-8").splitlines()
        out = []
        for i, ln in enumerate(lines, 1):
            new = normalize_line(ln)
            total += 1
            changed_lines += new != ln
            et += new.count("t[iam]") - ln.count("t[iam]")
            for tok in _ACUTE.findall(new):
                if not _ENCLITIC.search(tok):
                    review.append((p.stem.split("_c")[0], p.stem.split("_c")[1], i, tok, new))
            out.append(new)
        with (dst / p.name).open("w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(out) + "\n")
    rep = dst / "_akzente_pruefen.csv"
    with rep.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["seite", "spalte", "zeile", "wort", "zeilentext"])
        w.writerows(review)
    print(f"{len(files)} Spalten nach {dst} geschrieben; {changed_lines} von {total} Zeilen geändert, "
          f"{et}× „ét“ -> „et[iam]“.")
    print(f"{len(review)} Wörter mit akutem Akzent (vermutlich verlesener Nasalstrich) "
          f"zur Prüfung in {rep.name}.")


# --------------------------------------------------------------------------- #
# export-json
# --------------------------------------------------------------------------- #
ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
_HEAD = re.compile(r"^CON[SIL]{2,4}VM(PRIMVM|[IVXLC]+)\.?$")
_SIG = re.compile(r"(Nicolaus|Nicol\.|Nic\.|Nico\.)\s*Abb|\bAbb(as|a|\.)\.?\s*$", re.I)
_CASUS = re.compile(r"^(¶\s*)?(CASVS|Ca[sſf]us)\b")
_SUMHEAD = re.compile(r"^S\s*V\s*M\s*M\s*A\s*R\s*I\s*(?:[VY]\s*M|A)\.?$")
_PREFIXES = re.compile(r"^(?:\{(?:S|\d+)\}\s*)+")


def strip_prefixes(t: str) -> str:
    return _PREFIXES.sub("", t).strip()


def is_summ(t: str) -> bool:
    return "{S}" in t[:12] or bool(_SUMHEAD.match(strip_prefixes(t)))
_STOP = re.compile(r"^FINIS CONSILIORVM|^QVAESTIO(NES)?\b")
_MARG = re.compile(r"^\{(\d+)\}\s*")


def roman_to_int(s: str) -> int:
    if s == "PRIMVM":
        return 1
    total = 0
    for a, b in zip(s, s[1:] + " "):
        v = ROMAN[a]
        total += -v if b in ROMAN and ROMAN[b] > v else v
    return total


def int_to_roman(n: int) -> str:
    # Druckkonvention: IIII statt IV
    out = ""
    for v, r in ((100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (5, "V"), (1, "I")):
        while n >= v:
            out += r
            n -= v
    return out.replace("XC", "LXXXX").replace("XL", "XXXX")


def despace_caps(t: str) -> str:
    return re.sub(r"\b(?:[A-ZÆŒ] ){2,}[A-ZÆŒ]\b", lambda m: m.group(0).replace(" ", ""), t)


def norm_text(t: str) -> str:
    """Normalisierte Lesefassung: Auflösungen ohne Klammern, rundes s, ohne Markierungen."""
    t = strip_prefixes(t)
    t = re.sub(r"\[([^\]]*)\]", r"\1", t)
    t = t.replace("ſ", "s").replace("¶", " ")
    return re.sub(r"\s{2,}", " ", t).strip()


_WORD = re.compile(r"[A-Za-zÀ-ÿæœęÆŒ]+")
VOCAB: dict[str, int] = {}


def build_vocab(texts):
    VOCAB.clear()
    for t in texts:
        for w in _WORD.findall(t):
            w = w.lower()
            VOCAB[w] = VOCAB.get(w, 0) + 1


def _unhyphenated_break(left: str, right: str) -> bool:
    """Wort ohne Trennstrich über den Zeilenwechsel getrennt („sol“ / „uere“)?"""
    a = re.search(r"([A-Za-zÀ-ÿæœęÆŒ]+)$", left)
    b = re.match(r"([A-Za-zÀ-ÿæœęÆŒ]+)", right)
    if not a or not b or not VOCAB:
        return False
    wa, wb = a.group(1).lower(), b.group(1).lower()
    joined = VOCAB.get(wa + wb, 0)
    return joined >= 2 and joined > min(VOCAB.get(wa, 0), VOCAB.get(wb, 0))


def join_lines(texts: list[str]) -> str:
    """Zeilen verbinden; Trennungen am Zeilenende werden aufgelöst, auch ohne Trennstrich,
    wenn das zusammengesetzte Wort im Band häufiger belegt ist als seine Teile."""
    out = ""
    for t in texts:
        t = t.strip()
        if not t:
            continue
        if out.endswith(("-", "=", "⸗")):
            out = out[:-1] + t
        elif out and _unhyphenated_break(out, t):
            out = out + t
        else:
            out = (out + " " + t) if out else t
    return re.sub(r"\s{2,}", " ", out).strip()


def load_iiif(src: str):
    """IIIF-Manifest (URL oder lokale Datei) -> Liste je Canvas: id, Bilddienst, Maße, Blatt."""
    if re.match(r"https?://", src):
        import urllib.request
        with urllib.request.urlopen(src, timeout=60) as r:
            man = json.load(r)
    else:
        man = json.loads(Path(src).read_text(encoding="utf-8"))
    if "sequences" in man:      # Presentation 2
        raw = man["sequences"][0]["canvases"]
        get_id = lambda c: c["@id"]
        get_label = lambda c: c.get("label", "") if isinstance(c.get("label"), str) else ""
        get_service = lambda c: c["images"][0]["resource"]["service"]["@id"]
    else:                       # Presentation 3
        raw = man.get("items", [])
        get_id = lambda c: c["id"]
        get_label = lambda c: " ".join(v for vals in (c.get("label") or {}).values() for v in vals)
        def get_service(c):
            body = c["items"][0]["items"][0]["body"]
            svc = body.get("service") or [{}]
            svc = svc[0] if isinstance(svc, list) else svc
            return svc.get("id") or svc.get("@id")
    canv, fol = [], None
    for c in raw:
        label = get_label(c)
        m = re.search(r"fol\.\s*(\d+)\s*([rv])", label)
        if m:
            fol = (int(m.group(1)), m.group(2))
        elif fol and not re.search(r"unfoliated|blank|cover", label, re.I):
            n, side = fol
            fol = (n, "v") if side == "r" else (n + 1, "r")
        elif re.search(r"unfoliated|blank|cover", label, re.I):
            fol = None
        try:
            service = get_service(c)
        except Exception:
            service = None
        canv.append(dict(canvas=get_id(c), image=service, width=c.get("width"),
                         height=c.get("height"), folio=f"{fol[0]}{fol[1]}" if fol else None,
                         label=label))
    return canv, man.get("id") or man.get("@id") or src


def iiif_offset(canv, page_dir: Path):
    """Versatz Bild 0001 -> Canvas über exakt übereinstimmende Bildmaße bestimmen."""
    ours = {}
    for f in page_dir.glob("*.xml"):
        pg = ET.parse(f).getroot().find("p:Page", NS)
        ours[int(f.stem)] = (int(pg.get("imageWidth")), int(pg.get("imageHeight")))
    best = (-1, 0)
    for k in range(0, max(1, len(canv) - max(ours) + 1)):
        hits = sum(1 for p, wh in ours.items()
                   if k + p - 1 < len(canv) and (canv[k + p - 1]["width"], canv[k + p - 1]["height"]) == wh)
        best = max(best, (hits, k))
    return best[1], best[0], len(ours)


def cmd_export_json(args):
    data = Path(args.data)
    src = data / args.out
    vol = args.volume
    vol_slug = vol.lower()

    # Region-IDs und Spaltenkoordinaten aus dem Crop-Manifest
    regions = {}
    for r in csv.DictReader((data / args.crops / "_manifest.csv").open(encoding="utf-8")):
        x0, y0, x1, y1 = (int(v) for v in r["norm_bbox"].split(","))
        regions[(r["page"], r["column"])] = dict(
            region_id=r["region_ids"].split(";")[0], type=f"column_{r['column']}",
            coords=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

    canvases, man_id = None, None
    if not args.iiif and (data / "URN-3_FHCL.HOUGH_4884835").exists():
        args.iiif = str(data / "URN-3_FHCL.HOUGH_4884835")
    if args.iiif:
        canv, man_id = load_iiif(args.iiif)
        if args.iiif_offset is not None:
            off = args.iiif_offset
        else:
            off, hits, n = iiif_offset(canv, data / "page")
            print(f"IIIF: Versatz über Bildmaße bestimmt ({hits} von {n} Seiten stimmen exakt überein)")
            if hits < 0.8 * n:
                sys.exit("Die Bildmaße passen nicht zum Manifest – falsches Manifest oder "
                         "--iiif-offset von Hand angeben.")
        canvases = {f"{i + 1:04d}": canv[i + off] for i in range(len(canv) - off)}
        print(f"IIIF: {len(canv)} Canvases, Bild 0001 = Canvas {off + 1} ({canv[off]['label']})")

    # Alle Zeilen in Lesereihenfolge, bis zum Ende der Consilia
    L = []
    for p in sorted(src.glob("*_c*.txt"), key=lambda p: (p.stem[:4], p.stem[-1])):
        page, col = p.stem.split("_c")
        for i, ln in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            L.append(dict(page=page, col=col, no=i, ref=f"{page}_c{col}_l{i}",
                          text=normalize_line(ln)))
    if not L:
        sys.exit(f"Keine Transkriptionen in {src} gefunden – nichts exportiert.")
    stop = next((j for j, x in enumerate(L)
                 if _STOP.search(despace_caps(x["text"]).strip(" .¶"))), len(L))
    L = L[:stop]
    build_vocab(norm_text(x["text"]) for x in L)

    # Überschriften, Teil und korrigierte Nummer
    heads = []
    pars, prev = 1, 0
    for j, x in enumerate(L):
        t = despace_caps(re.sub(r"^\{S\}\s*|^\{\d+\}\s*", "", x["text"])).strip(" .¶")
        m = _HEAD.match(t.replace(" ", "") + ("" if t.endswith(".") else "."))
        if not m:
            continue
        printed = roman_to_int(m.group(1))
        if printed == 1 and prev > 1:
            pars, prev = pars + 1, 0
        n = printed if printed in (prev + 1, prev + 2) else prev + 1
        heads.append(dict(j=j, pars=pars, n=n, printed=m.group(1), title=t + "."))
        prev = n

    def block_before(h_idx, lo):
        """Summarium- und Casus-Zeilen vor einer Überschrift; lo = frühester Index."""
        j = heads[h_idx]["j"]
        # Summarium beginnt bei der letzten Überschrift „SVMMARIVM“ vor dem Consilium;
        # fehlt sie, zählen die unmittelbar vorangehenden {S}-Zeilen
        s = next((k for k in range(j - 1, max(lo, j - 80) - 1, -1)
                  if _SUMHEAD.match(strip_prefixes(L[k]["text"]))), None)
        if s is None:
            s = j
        while s - 1 >= lo and is_summ(L[s - 1]["text"]):
            s -= 1
        summ = L[s:j]
        win = range(max(lo, s - args.casus_window), s)
        cas_start = next((k for k in win if _CASUS.search(L[k]["text"])), None)
        if cas_start is None:
            sig = [k for k in win if _SIG.search(L[k]["text"])]
            if sig and sig[-1] + 1 < s:
                cas_start = sig[-1] + 1
        start = cas_start if cas_start is not None else s
        return start, (L[cas_start:s] if cas_start is not None else []), summ

    out, report = {}, []
    blocks = []
    for h in range(len(heads)):
        lo = heads[h - 1]["j"] + 1 if h else 0
        blocks.append(block_before(h, lo))
    for h, hd in enumerate(heads):
        start_next = blocks[h + 1][0] if h + 1 < len(heads) else len(L)
        body = [x for x in L[hd["j"] + 1:start_next]
                if not re.match(r"^FINIS\b", despace_caps(x["text"]))]
        _, casus, summ = blocks[h]

        # Abschnitte an † bzw. Randziffern
        segs, cur = [], None

        def new_seg(marg):
            nonlocal cur
            cur = dict(parts=[], refs=[], marg=marg)
            segs.append(cur)

        for x in body:
            t = x["text"]
            m = _MARG.match(t)
            marg = int(m.group(1)) if m else None
            t = _MARG.sub("", re.sub(r"^\{S\}\s*", "", t))
            pieces = t.split("†")
            if cur is None:
                new_seg(None)
            if marg is not None and len(pieces) == 1:
                new_seg(marg)
            for k, piece in enumerate(pieces):
                if k > 0:
                    new_seg(marg)
                cur["parts"].append(piece)
                cur["refs"].append(x["ref"])
        body_txt, body_meta = [], []
        for sg in segs:
            txt = join_lines([norm_text(p) for p in sg["parts"]])
            if not txt:
                continue
            if sg is not segs[0] or sg["marg"] is not None:
                txt = "† " + txt
            body_txt.append(txt)
            body_meta.append(dict(marginal=sg["marg"], first_line=sg["refs"][0],
                                  last_line=sg["refs"][-1]))

        summary = join_lines([norm_text(x["text"]) for x in summ
                              if not _SUMHEAD.match(strip_prefixes(x["text"]))])
        casus_txt = join_lines([norm_text(x["text"]) for x in casus])
        casus_txt = re.sub(r"^Ca[sf]us\s+c[o0]n[sf]ili[ij]{1,2}\s+[sf]equentis\.?\s*", "", casus_txt, flags=re.I) or None

        # Quellen: je Bild die Spaltenregionen mit erster/letzter Zeile
        lines_all = casus + summ + [L[hd["j"]]] + body
        sources, by_page = [], {}
        for x in lines_all:
            pg = by_page.setdefault(x["page"], {})
            reg = pg.setdefault(x["col"], [x["ref"], x["ref"]])
            reg[1] = x["ref"]
        for page, cols in by_page.items():
            entry = dict(page=f"{page}.xml", regions=[])
            if canvases and page in canvases:
                cv = canvases[page]
                entry.update(canvas=cv["canvas"], image=cv["image"], folio=cv["folio"])
            for col, (first, last) in cols.items():
                r = dict(regions.get((page, col), dict(region_id=None, type=f"column_{col}",
                                                       coords=None)))
                r.update(first_line=first, last_line=last)
                entry["regions"].append(r)
            sources.append(entry)

        cid = f"consilium-{vol_slug}-p{hd['pars']}-{hd['n']}"
        rec = dict(id=cid, n=hd["n"], pars=hd["pars"], roman=int_to_roman(hd["n"]) if hd["n"] > 1
                   else "PRIMVM", volume=vol, author_viaf=args.author, title=hd["title"],
                   casus=casus_txt, summary=summary, body=body_txt, body_meta=body_meta,
                   sources=sources)
        if roman_to_int(hd["printed"]) != hd["n"]:
            rec["roman_print"] = hd["printed"]
        out[cid] = rec
        report.append((cid, hd["printed"], len(body_txt), bool(casus_txt), bool(summary)))

    n_tr = attach_translations(out, data / args.tr_dir)
    if n_tr or any("body_de" in v for v in out.values()):
        print(f"Übersetzung übernommen: {n_tr} Consilia vollständig")
    if len(out) < 10:
        sys.exit(f"Nur {len(out)} Consilia erkannt – vorhandene Datei bleibt unverändert.")
    meta = dict(volume=vol, title=args.title, author_viaf=args.author, iiif_manifest=man_id,
                generated=time.strftime("%Y-%m-%dT%H:%M:%S"))
    target = data / args.target
    target.write_text(json.dumps(dict(meta=meta, consilia=out), ensure_ascii=False, indent=1),
                      encoding="utf-8")
    per_pars = {}
    for v in out.values():
        per_pars[v["pars"]] = per_pars.get(v["pars"], 0) + 1
    print(f"{len(out)} Consilia -> {target}  (je Teil: {per_pars})")
    fixed = [v["id"] for v in out.values() if "roman_print" in v]
    if fixed:
        print(f"Nummer nach Reihenfolge korrigiert ({len(fixed)}): {', '.join(fixed)}")
    print(f"mit Casus: {sum(r[3] for r in report)}, mit Summarium: {sum(r[4] for r in report)}, "
          f"Abschnitte gesamt: {sum(r[2] for r in report)}")


# --------------------------------------------------------------------------- #
# Übersetzung (Batch)
# --------------------------------------------------------------------------- #
TR_SCHEMA = {
    "type": "object",
    "properties": {
        "abschnitte": {"type": "array", "items": {"type": "string"}},
        "casus": {"type": ["string", "null"]},
        "summarium": {"type": ["string", "null"]},
    },
    "required": ["abschnitte", "casus", "summarium"],
}


def translation_chunks(cons: dict, max_chars: int):
    """Abschnitte eines Consilium in Pakete bis max_chars teilen (Casus/Summarium im ersten)."""
    chunks, cur, size = [], [], 0
    for i, seg in enumerate(cons["body"]):
        if cur and size + len(seg) > max_chars:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += len(seg)
    if cur or not chunks:
        chunks.append(cur)
    return chunks


def cmd_translate_submit(args):
    data = Path(args.data)
    src = json.loads((data / args.source).read_text(encoding="utf-8"))["consilia"]
    tdir = data / args.tr_dir
    tdir.mkdir(parents=True, exist_ok=True)
    st = BatchState(tdir)
    busy = st.open_keys("uebersetzung")
    prompt = Path(args.prompt).read_text(encoding="utf-8").strip()
    gen = {"thinkingConfig": {"thinkingLevel": args.thinking.upper()},
           "maxOutputTokens": args.max_output_tokens,
           "responseMimeType": "application/json", "responseJsonSchema": TR_SCHEMA}

    reqs = []
    for cid, cons in src.items():
        for k, idx in enumerate(translation_chunks(cons, args.chunk_chars)):
            key = f"{cid}__{k:02d}"
            if key in busy or (tdir / f"{key}.json").exists():
                continue
            payload = {"titel": cons["title"], "abschnitte": [cons["body"][i] for i in idx]}
            if k == 0:
                payload["casus"] = cons.get("casus")
                payload["summarium"] = cons.get("summary") or None
            else:
                payload["summarium_kontext"] = cons.get("summary") or None
            reqs.append((key, dict(cid=cid, chunk=k, idx=idx), payload))
    if args.only:
        wanted = set(args.only.split(","))
        reqs = [r for r in reqs if r[1]["cid"].rsplit("-", 2)[-2] + "-" + r[1]["cid"].rsplit("-", 1)[-1]
                in wanted or r[1]["cid"] in wanted]
    if args.limit:
        reqs = reqs[: args.limit]
    if not reqs:
        print("Nichts einzureichen.")
        return
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = st.dir / f"uebersetzung_{stamp}.jsonl"
    meta = {}
    with path.open("w", encoding="utf-8") as fh:
        for key, m, payload in reqs:
            meta[key] = m
            fh.write(json.dumps({"key": key, "request": {
                "contents": [{"role": "user", "parts": [
                    {"text": json.dumps(payload, ensure_ascii=False)}]}],
                "systemInstruction": {"parts": [{"text": prompt}]},
                "generationConfig": gen}}, ensure_ascii=False) + "\n")
    (st.dir / f"uebersetzung_{stamp}_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"{path.name}: {len(reqs)} Anfragen, {path.stat().st_size / 1e6:.1f} MB")
    if args.dry_run:
        return
    from google.genai import types
    client = make_client()
    up = client.files.upload(file=str(path), config=types.UploadFileConfig(
        display_name=path.stem, mime_type="jsonl"))
    job = client.batches.create(model=args.model, src=up.name, config={"display_name": path.stem})
    st.jobs.append(dict(name=job.name, kind="uebersetzung", file=path.name,
                        keys=[r[0] for r in reqs], meta=f"uebersetzung_{stamp}_meta.json",
                        model=args.model, created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                        collected=False))
    st.save()
    print(f"Job {job.name} eingereicht. Abholen mit: pipeline.py translate-collect --data .")


def cmd_translate_collect(args):
    data = Path(args.data)
    tdir = data / args.tr_dir
    st = BatchState(tdir)
    client = make_client()
    done_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
                   "JOB_STATE_EXPIRED"}
    while True:
        open_jobs = [j for j in st.jobs if not j.get("collected")]
        waiting = 0
        for j in open_jobs:
            try:
                job = client.batches.get(name=j["name"])
            except Exception as e:
                print(f"{j['file']}: Abfrage fehlgeschlagen ({type(e).__name__}: {e}) – "
                      f"nächster Versuch beim nächsten Durchlauf")
                waiting += 1
                continue
            state = job.state.name if hasattr(job.state, "name") else str(job.state)
            print(f"{j['file']} [{len(j['keys'])} Anfragen]: {state}")
            if state not in done_states:
                waiting += 1
                continue
            j["state"] = state
            if state != "JOB_STATE_SUCCEEDED":
                j["collected"] = True
            else:
                meta = json.loads((st.dir / j["meta"]).read_text(encoding="utf-8"))
                try:
                    raw = client.files.download(file=job.dest.file_name)
                except Exception as e:
                    print(f"  Herunterladen fehlgeschlagen ({type(e).__name__}) – erneut beim nächsten Aufruf")
                    waiting += 1
                    continue
                j["collected"] = True
                (st.dir / f"{Path(j['file']).stem}_ergebnis.jsonl").write_bytes(raw)
                ok = bad = 0
                for ln in raw.decode("utf-8").splitlines():
                    if not ln.strip():
                        continue
                    rec = json.loads(ln)
                    key, resp = _g(rec, "key"), _g(rec, "response")
                    cand = ((_g(resp, "candidates", default=[]) or [{}])[0]) if resp else {}
                    parts = _g(_g(cand, "content", default={}), "parts", default=[]) or []
                    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                    try:
                        out = json.loads(text)
                        assert len(out["abschnitte"]) == len(meta[key]["idx"])
                    except Exception:
                        bad += 1      # fehlt -> wird beim nächsten translate-submit erneut eingereicht
                        continue
                    out.update(meta[key])
                    (tdir / f"{key}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                      encoding="utf-8")
                    ok += 1
                j["stats"] = dict(gespeichert=ok, fehler=bad)
                print(f"  abgeholt: {j['stats']}")
            st.save()
        if not waiting or not args.wait:
            break
        time.sleep(args.poll)
    n = len(list(tdir.glob("consilium-*.json")))
    print(f"\nÜbersetzte Pakete: {n}. Fehlende reicht translate-submit erneut ein; "
          f"export-json übernimmt die Übersetzung in das JSON.")


def attach_translations(out: dict, tdir: Path):
    """Übersetzungspakete als casus_de / summary_de / body_de in die Datensätze übernehmen."""
    if not tdir.exists():
        return 0
    done = 0
    for f in sorted(tdir.glob("consilium-*.json")):
        t = json.loads(f.read_text(encoding="utf-8"))
        rec = out.get(t["cid"])
        if not rec:
            continue
        rec.setdefault("body_de", [None] * len(rec["body"]))
        if len(t["idx"]) and max(t["idx"]) >= len(rec["body"]):
            continue      # veraltet: Segmentierung hat sich geändert
        for i, tr in zip(t["idx"], t["abschnitte"]):
            rec["body_de"][i] = tr
        if t["chunk"] == 0:
            rec["casus_de"] = t.get("casus")
            rec["summary_de"] = t.get("summarium")
    for rec in out.values():
        if "body_de" in rec:
            rec["translation_complete"] = all(x is not None for x in rec["body_de"])
            done += rec["translation_complete"]
    return done


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
def cmd_merge(args):
    data = Path(args.data)
    out_dir = data / args.out
    files = sorted(out_dir.glob("*_c*.txt"),
                   key=lambda p: (page_nr(p.stem), int(p.stem.split("_c")[1])))
    target = out_dir / args.target
    with target.open("w", encoding="utf-8") as fh:
        for p in files:
            page, col = p.stem.split("_c")
            fh.write(f"##### Seite {page} · Spalte {col} #####\n")
            fh.write(p.read_text(encoding="utf-8"))
            fh.write("\n")
    print(f"{len(files)} Spalten -> {target}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--data", default=".", help="Ordner mit JPGs und page/ (Standard: .)")
        p.add_argument("--pages", help="Auswahl, z. B. 10-12,200")
        p.add_argument("--crops", default="crops", help="Unterordner für Crops")
        p.add_argument("--out", default="transkription", help="Unterordner für Texte")

    p = sub.add_parser("crop", help="Spalten ausschneiden")
    common(p)
    p.add_argument("--pad-x", type=int, default=50, help="Rand links/rechts in px (Standard 50)")
    p.add_argument("--pad-y", type=int, default=30, help="Rand oben/unten in px (Standard 30)")
    p.add_argument("--narrow-ratio", type=float, default=0.97,
                   help="Spalte gilt als schmal unter diesem Anteil der Medianbreite (Standard 0.97)")
    p.add_argument("--narrow-pad", type=int, default=5,
                   help="zusätzlicher Rand links/rechts für schmale Spalten in px (Standard 5)")
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--ncols", type=int, default=2, help="erwartete Spaltenzahl (Standard 2)")
    p.add_argument("--window", type=int, default=10,
                   help="Nachbarseiten gleicher Parität je Seite für den Median (Standard 10)")
    p.add_argument("--tol", type=int, default=120,
                   help="Abweichung vom Median in px, ab der erweitert wird (Standard 120)")
    p.add_argument("--no-normalize", action="store_true",
                   help="Regionen unverändert übernehmen (nur Rand)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--debug", action="store_true",
                   help="Kontrollbilder (rot: Region, grün: Crop) in crops/_debug/")
    p.set_defaults(func=cmd_crop)

    p = sub.add_parser("ocr", help="Transkription über Gemini")
    common(p)
    p.add_argument("--model", default="gemini-3.8-flash")
    p.add_argument("--prompt", default=str(HERE / "prompt_hauptspalte.txt"))
    p.add_argument("--resolution", choices=list(RES_LEVELS), default="ultra_high",
                   help="media_resolution des Bildes (Standard ultra_high ≈ 2240 Tokens)")
    p.add_argument("--thinking", choices=["low", "medium", "high"], default="medium",
                   help="thinking_level des Modells (Standard medium)")
    p.add_argument("--max-output-tokens", type=int, default=32768,
                   help="Budget für Denken + Ausgabe je Anfrage (Standard 32768)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--retries", type=int, default=6)
    p.add_argument("--limit", type=int, help="nur die ersten N offenen Crops")
    p.add_argument("--max-line-len", type=int, default=100,
                   help="längere Ausgabezeilen gelten als zusammengezogene Druckzeilen (Standard 100)")
    p.add_argument("--max-residual", type=int, default=3,
                   help="tolerierte unaufgelöste Kürzungszeichen je Spalte (Standard 3)")
    p.add_argument("--quality-retries", type=int, default=1,
                   help="Neuversuche bei zusammengezogenen Zeilen (Standard 1)")
    p.add_argument("--no-expand", dest="expand", action="store_false",
                   help="keine Nachauflösung verbliebener Kürzungszeichen")
    p.add_argument("--expand-prompt", default=str(HERE / "prompt_aufloesung.txt"))
    p.add_argument("--expand-thinking", choices=["low", "medium", "high"], default="medium",
                   help="thinking_level der Nachauflösung (Standard medium)")
    p.add_argument("--max-wait", type=int, default=300,
                   help="längere Wartezeiten bei 429 brechen den Lauf ab (Sekunden, Standard 300)")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_ocr)

    def model_opts(p):
        p.add_argument("--model", default="gemini-3.8-flash")
        p.add_argument("--prompt", default=str(HERE / "prompt_hauptspalte.txt"))
        p.add_argument("--expand-prompt", default=str(HERE / "prompt_aufloesung.txt"))
        p.add_argument("--resolution", choices=list(RES_LEVELS), default="ultra_high")
        p.add_argument("--thinking", choices=["low", "medium", "high"], default="medium")
        p.add_argument("--expand-thinking", choices=["low", "medium", "high"], default="medium")
        p.add_argument("--max-output-tokens", type=int, default=32768)
        p.add_argument("--max-line-len", type=int, default=100)
        p.add_argument("--max-residual", type=int, default=3)

    p = sub.add_parser("batch-submit", help="Spalten als Batch-Job einreichen (halber Preis)")
    common(p)
    model_opts(p)
    p.add_argument("--expand", action="store_true",
                   help="Nachauflösung für Spalten mit verbliebenen Kürzungszeichen einreichen")
    p.add_argument("--limit", type=int, help="nur die ersten N Spalten")
    p.add_argument("--overwrite", action="store_true", help="auch bereits transkribierte Spalten")
    p.add_argument("--chunk-mb", type=int, default=500,
                   help="maximale Größe je JSONL-Datei bzw. Job in MB (Standard 500)")
    p.add_argument("--dry-run", action="store_true", help="nur JSONL schreiben, nichts hochladen")
    p.add_argument("--keep-jsonl", action="store_true", help="JSONL nach dem Hochladen behalten")
    p.set_defaults(func=cmd_batch_submit)

    p = sub.add_parser("batch-collect", help="Status der Batch-Jobs abfragen, Ergebnisse abholen")
    common(p)
    model_opts(p)
    p.add_argument("--wait", action="store_true", help="warten, bis alle Jobs fertig sind")
    p.add_argument("--poll", type=int, default=300, help="Abfrageintervall in s (Standard 300)")
    p.set_defaults(func=cmd_batch_collect)

    p = sub.add_parser("normalize", help="Leerzeichen vereinheitlichen, „ét“ auflösen")
    common(p)
    p.add_argument("--target", default="transkription_final",
                   help="Zielordner für die normalisierten Texte (Standard transkription_final)")
    p.set_defaults(func=cmd_normalize)

    p = sub.add_parser("export-json", help="Consilia als JSON im Portal-Schema exportieren")
    common(p)
    p.set_defaults(out="transkription_batch")
    p.add_argument("--target", default="prebuilt.json")
    p.add_argument("--volume", default="Panormitanus_Cons_Print_Venice_1571_v1")
    p.add_argument("--author", default="Panormitanus_32108367")
    p.add_argument("--title", default="Abbatis Panormitani Consilia, Tractatus, Quaestiones & "
                                      "Practica, Venedig 1571")
    p.add_argument("--iiif", help="IIIF-Manifest: URL oder lokale Datei")
    p.add_argument("--iiif-offset", type=int,
                   help="0-basierter Canvas-Index von Bild 0001, falls das Manifest mehr Bilder hat")
    p.add_argument("--tr-dir", default="uebersetzung", help="Ordner mit Übersetzungspaketen")
    p.add_argument("--casus-window", type=int, default=60,
                   help="Zeilen vor dem Summarium, in denen ein Casus gesucht wird")
    p.set_defaults(func=cmd_export_json)

    p = sub.add_parser("translate-submit", help="Consilia zur Übersetzung einreichen (Batch)")
    p.add_argument("--data", default=".")
    p.add_argument("--source", default="prebuilt.json")
    p.add_argument("--tr-dir", default="uebersetzung")
    p.add_argument("--model", default="gemini-3.8-flash")
    p.add_argument("--prompt", default=str(HERE / "prompt_uebersetzung.txt"))
    p.add_argument("--thinking", choices=["low", "medium", "high"], default="medium")
    p.add_argument("--max-output-tokens", type=int, default=65536)
    p.add_argument("--chunk-chars", type=int, default=12000,
                   help="maximale Zeichen lateinischen Texts je Anfrage (Standard 12000)")
    p.add_argument("--only", help="nur bestimmte Consilia, z. B. p1-5,p1-6,p2-44")
    p.add_argument("--limit", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_translate_submit)

    p = sub.add_parser("translate-collect", help="Übersetzungen abholen")
    p.add_argument("--data", default=".")
    p.add_argument("--tr-dir", default="uebersetzung")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--poll", type=int, default=300)
    p.set_defaults(func=cmd_translate_collect)

    p = sub.add_parser("merge", help="Texte zusammenführen")
    common(p)
    p.add_argument("--target", default="_gesamt.txt")
    p.set_defaults(func=cmd_merge)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
