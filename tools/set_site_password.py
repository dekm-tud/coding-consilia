#!/usr/bin/env python3
"""
Set (or remove) the preview password of the website (assets/gate.js). Writes only a salted PBKDF2-SHA256 hash to
assets/gate-config.js; the password is never stored. The hash is public with the site, so choose a password of
12 or more characters. A first blocker for the preview, not access control: the data files stay reachable directly.

    python tools/set_site_password.py           # asks for the password twice (input hidden)
    python tools/set_site_password.py --off     # no gate
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import secrets
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "assets" / "gate-config.js"
ITERATIONS = 310_000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--off", action="store_true")
    a = ap.parse_args()
    if a.off:
        OUT.write_text("/* no preview password (tools/set_site_password.py) */\nwindow.CONSILIA_GATE = null;\n", encoding="utf-8")
        print(f"{OUT}: gate off"); return
    pw = getpass.getpass("Password: ")
    if pw != getpass.getpass("Again: "):
        raise SystemExit("the two entries differ; nothing written")
    if len(pw) < 10:
        raise SystemExit("at least 10 characters, please; nothing written")
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), salt.encode("utf-8"), ITERATIONS).hex()
    cfg = {"salt": salt, "iterations": ITERATIONS, "hash": h}
    OUT.write_text("/* preview password: salted PBKDF2-SHA256 hash, written by tools/set_site_password.py; the password is not stored */\n"
                   f"window.CONSILIA_GATE = {json.dumps(cfg)};\n", encoding="utf-8")
    print(f"{OUT}: gate on. Colleagues enter the password once per browser.")


if __name__ == "__main__":
    main()
