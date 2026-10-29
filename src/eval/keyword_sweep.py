"""Diagnostic lexical sweep: intimate-body stem hits (Latin + EN translation) vs keywords_final tags.
Spot-check tool from the 2026-09-23 review -- raw hits include homonyms/citations; not a precision/recall measure.
Run: python src/eval/keyword_sweep.py"""
import json,re,collections,random
cs=json.load(open('output/consilia.json'))['consilia']
prot=set(json.load(open('src/protected_keyword_categories.json'))['intimate_and_reproductive_body']['terms'])
def norm(s): return s.lower().replace('æ','e').replace('ę','e').replace('ae','e').replace('j','i').replace('v','u')
# category: (latin regex on normalised text, english regex, tags counting as hit)
CAT={
 'rape/stuprum':(r'\bstupr\w*|\brapt(us|u|um|a|am|or|oris|ore)\b|\brapu(it|erit)\b|uiolenter cognou|oppress\w* uirgin', r'\brap(e|ed|es|ing|ist)\b|\bravish|\bdefil', {'rape','abduction','seduction','sexual misconduct'}),
 'adultery':(r'\badulter\w*', r'\badulter\w*', {'adultery'}),
 'incest':(r'\bincest\w*', r'\bincest\w*', {'incest'}),
 'sodomy':(r'\bsodom\w*|contra naturam', r'\bsodom\w*|against nature', {'sodomy'}),
 'fornication/concubinage':(r'\bfornica\w*|\bconcubin\w*', r'\bfornicat\w*|\bconcubin\w*', {'fornication','concubinage'}),
 'prostitution':(r'\bmeretri\w*|\blenocin\w*|\bprostitu\w*|\blupanar\w*', r'\bprostitut\w*|\bharlot|\bbrothel|\bwhore', {'prostitution'}),
 'intercourse/consummation':(r'\bcoit\w*|\bconcubit\w*|copula\w* carnal\w*|carnaliter|\bcarnal\w*|commixtion\w*', r'\bintercourse\b|\bcarnal\w*|\bconsummat\w*', {'sexual intercourse','consummation','carnal marriage','conjugal life'}),
 'impotence':(r'\bimpoten\w*|\bfrigid\w*|\bmaleficiat\w*|coire non', r'\bimpoten\w*|\bfrigid\w*', {'impotence'}),
 'virginity/chastity':(r'\buirginit\w*|\bcastit\w*|\bdefloa\w*|\bdeflora\w*', r'\bvirginity\b|\bchastity\b|\bdeflower\w*', {'virginity','chastity','vow of chastity'}),
 'pregnancy/birth':(r'\bpregnan\w*|\bgrauid\w*|in utero|\bpuerper\w*|\bparturi\w*|\benix\w*|inspiciendo uentre|de uentre', r'\bpregnan\w*|\bchildbirth\b|\bgave birth\b|\bin the womb\b|\bwomb\b|\blabou?r pains', {'pregnancy','gestation','birth','womb','possession of the womb','substitution to the womb','conception','puerperium','fetus','gestational age','viability','caesarean section'}),
 'abortion/miscarriage':(r'\babort\w*|\babors\w*', r'\babort\w*|\bmiscarr\w*', {'abortion','stillbirth'}),
 'sterility/infertility':(r'\bsteril\w*', r'\bsteril\w*|\binfertil\w*|\bbarren\b', {'sterility','fertility'}),
 'wet-nurse/lactation':(r'\bnutric\w*|\bnutrix\b|\blactat\w*', r'\bwet.?nurse\w*|\bbreast.?feed\w*|\bsuckl\w*', {'wet-nurse'}),
 'sex/gender identity':(r'\bhermaphrod\w*|\beunuch\w*|\bcastrat\w*|\bspado\w*', r'\bhermaphrod\w*|\beunuch\w*|\bcastrat\w*', {'hermaphroditism','sex determination','sex'}),
 'menstruation':(r'\bmenstru\w*', r'\bmenstru\w*', set()),
 'illegitimate birth':(r'\bspuri\w*|\bbastard\w*|\bnothus\b|ex damnato coitu|\bincestuos\w*|\badulterin\w*', r'\bbastard\w*|\bspurious\b|\billegitimate\b', {'illegitimacy','spurious child','spurious grandson','bastard','bastardy','natural child'}),
}
rows=[]
for c in cs.values():
    la=norm(' '.join((c.get('casus') or [])+(c.get('body') or [])+[it['text'] for it in (c.get('argument') or [])]+[c.get('title') or '']))
    tr=c.get('translations',{}).get('en',{})
    a=tr.get('argument') or ''; a=' '.join(a) if isinstance(a,list) else a
    en=(' '.join((tr.get('casus') or [])+(tr.get('body') or []))+' '+a).lower()
    kf=set(c.get('keywords_final') or [])
    kall=set(sum(((c.get('keywords') or {}).get(k) or [] for k in ('argument','body')),[]))
    rows.append((c,la,en,kf,kall))
print(f"{'category':26} {'LAhit':>5} {'ENhit':>5} {'either':>6} {'tagFinal':>8} {'tagAny':>6} {'hit&tagF':>8} {'tagF_nohit':>10}")
ex={}
for cat,(lre,ere,tags) in CAT.items():
    L=re.compile(lre); E=re.compile(ere)
    la=sum(1 for c,l,e,kf,ka in rows if L.search(l)); en=sum(1 for c,l,e,kf,ka in rows if E.search(e))
    eith=[r for r in rows if L.search(r[1]) or E.search(r[2])]
    tf=[r for r in rows if r[3]&tags]; ta=[r for r in rows if (r[3]|r[4])&tags]
    both=[r for r in eith if r[3]&tags]; nohit=[r for r in tf if not(L.search(r[1]) or E.search(r[2]))]
    print(f"{cat:26} {la:5} {en:5} {len(eith):6} {len(tf):8} {len(ta):6} {len(both):8} {len(nohit):10}")
    ex[cat]=[r for r in eith if not (r[3]&tags)]
# protected share
pc=collections.Counter(k for r in rows for k in r[3] if k in prot)
print(pc.most_common(40))
print('docs with any protected final tag:',sum(1 for r in rows if r[3]&prot))

