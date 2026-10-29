"""Draw the stratified gold-standard sample for Layer B.

Strata (fixed seed, reproducible):
  lexical   35  units with Latin cues of intimate-body topics
  homonym   20  units with known false-friend words (carnalis, sterilis, impotentia, ...)
  citation  20  units alleging Corpus iuris titles about these matters
  tagged    25  units from consilia carrying old protected tags (excluding birth-status-only)
  random    50  uniform random units
Run: python src/eval/sample_gold_units.py  -> eval/gold/intimate_body_sample.jsonl
"""
import json, random, re
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
SEED = 20260923


def norm(s):
    return s.lower().replace("æ", "e").replace("ę", "e").replace("ae", "e").replace("j", "i").replace("v", "u")


LEX = re.compile(r"\badulter|\bstupr|\bincest|\bconcubin|\bmeretri|\bpregnan|\bgrauid|\buentr[ei]|\bpartu[sm]?\b|\bpeperit|\bpuerper|\babort|\bfrigid|\buirginit|\bcastit|\bsodom|\bhermaphrod|\beunuch|\bnutri[cx]|copula carnal|carnaliter|\bspuri|\blegitimat|\bfocari|\bconcept")
HOM = re.compile(r"\bcarnal|\bsteril|\bimpoten|contra naturam|\bpregnantia|\buirgin|\bcastr[ao]|\bmalefic|\bcopulat|\bconsummat|\brapin|\bnatural")
CIT = re.compile(r"de adulter|ad l(eg)?\.? iul|de uentre|de inspic|de lib(eris)?\.? et post|de raptu|de incest|de frigid|qui filii sint|de agnosc|de concub|de natural(ibus)? lib")
PROT_BODY = {"rape", "adultery", "fornication", "incest", "sodomy", "bigamy", "concubinage", "prostitution",
             "sexual intercourse", "sexual misconduct", "seduction", "abduction", "consummation", "impotence",
             "virginity", "chastity", "vow of chastity", "pregnancy", "conception", "abortion", "birth",
             "stillbirth", "womb", "fetus", "sterility", "fertility", "wet-nurse", "hermaphroditism"}


def main():
    rnd = random.Random(SEED)
    units = [json.loads(l) for l in open(ROOT / "output/units.jsonl", encoding="utf-8")]
    cons = json.load(open(ROOT / "output/consilia.json", encoding="utf-8"))["consilia"]
    for u in units:
        u["_n"] = norm(u["la"])
    taken, out = set(), []

    def take(pool, k, stratum):
        pool = [u for u in pool if u["unit"] not in taken]
        for u in rnd.sample(pool, min(k, len(pool))):
            taken.add(u["unit"])
            out.append({"unit": u["unit"], "stratum": stratum})

    lex = [u for u in units if LEX.search(u["_n"])]
    take(lex, 35, "lexical")
    take([u for u in units if HOM.search(u["_n"]) and not LEX.search(u["_n"])], 20, "homonym")
    take([u for u in units if CIT.search(u["_n"])], 20, "citation")
    tagged_docs = {cid for cid, c in cons.items() if set(c.get("keywords_final") or []) & PROT_BODY}
    take([u for u in units if u["doc"] in tagged_docs], 25, "tagged")
    take(units, 50, "random")
    path = ROOT / "eval/gold/intimate_body_sample.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(len(out), "units ->", path)


if __name__ == "__main__" and "--rare" not in __import__("sys").argv:
    main()


# ---------------------------------------------------------------------------
# Supplement: targeted sample for rare codes (too rare for the main sample to
# contain them). Up to 3 units per code with strict Latin cues, excluding units
# already in the main sample. Run: python src/eval/sample_gold_units.py --rare
RARE_CUES = {
    "rape": r"per uim (cognou|oppress|stupr)|uiolenter (cognou|oppress|stupr)|uim intul\w* (mulier|uirgin)|oppressit uirgin",
    "stuprum": r"\bstupr",
    "sodomy": r"\bsodom|contra naturam (coi|comm|pecc\w* cum)|masculorum stupr",
    "abortion_miscarriage": r"\babort",
    "impotence": r"\bfrigid|coire non|impoten\w* coeundi|maleficiat",
    "sex_determination": r"\bhermaphrod|uterque sexus|androgyn",
    "virginity": r"\buirginit|\bdeflor|per obstetric",
    "prostitution": r"\bmeretri|lupanar|lenocin",
    "consummation": r"copula carnal|carnaliter cognou|matrimonium consummat",
    "nursing": r"\bnutri[cx]|\blact[ae]",
    "sexual_honour": r"\bpudicit|\bimpudic",
    "fertility_sterility": r"ad generandum|\bsteril\w* (mulier|uxor)|mulier\w* steril",
    "puberty_body": r"habitu corporis|inspectio\w* corporis|malitia supplet",
    "marital_cohabitation": r"debitum coniugal|separatio\w* thor|cohabitare",
    "chastity_conditions": r"caste uix|uitam uidual|castitat",
    "incest": r"\bincest",
}


def rare():
    import argparse  # noqa: F401
    rnd = random.Random(SEED + 1)
    units = [json.loads(l) for l in open(ROOT / "output/units.jsonl", encoding="utf-8")]
    taken = {json.loads(l)["unit"] for l in open(ROOT / "eval/gold/intimate_body_sample.jsonl")}
    out = []
    for code, pat in RARE_CUES.items():
        rx = re.compile(pat)
        pool = [u for u in units if u["unit"] not in taken and rx.search(norm(u["la"]))]
        for u in rnd.sample(pool, min(3, len(pool))):
            taken.add(u["unit"])
            out.append({"unit": u["unit"], "stratum": f"rare:{code}", "pool_size": len(pool)})
    path = ROOT / "eval/gold/intimate_body_rare_sample.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(len(out), "units ->", path)


if __name__ == "__main__" and "--rare" in __import__("sys").argv:
    rare()
