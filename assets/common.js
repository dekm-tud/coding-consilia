/* Shared helpers for all pages and the search worker (plain script: <script src> and importScripts,
   no build step). Real data only: a failed request is an error, never a stand-in. */

/* ── Latin normalisation (j=i, v=u, ae=e) ──────────────────────────────── */
function normalise(term) {
  return term.toLowerCase()
    .replace(/æ/g, 'e').replace(/ę/g, 'e').replace(/ae/g, 'e')
    .replace(/j/g, 'i').replace(/v/g, 'u');
}
function termToRegexPart(t) {
  let pat = '';
  for (let i = 0; i < t.length; i++) {
    switch (t[i]) {
      case 'i': pat += '[ij]'; break;
      case 'u': pat += '[uv]'; break;
      case 'e': pat += '(?:ae|[eæę])'; break;
      default:  pat += t[i].replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
  }
  return pat;
}
/* ── lemmas (site-data/lemmas.json, tools/lemmatize_corpus.py) ───────────
   form -> lemma for the normalised word forms of the corpus. Indexing adds a lemma key ("~testamentum") to
   every word next to its form, a query word is searched as typed and by that key: "testamenti" finds
   testamentum, testamento, testamenta. A word in quotation marks is searched as typed, form only ("testamenti").
   Without the file (not built, or failed to load) everything works on word forms as before. */
let LEMMAS = null, LEMMA_FORMS = null, LEMMA_SET = null, lemmasLoading = null;
function setLemmas(d) { LEMMAS = (d && d.forms) || null; LEMMA_FORMS = null; LEMMA_SET = LEMMAS ? new Set(Object.values(LEMMAS)) : null; return !!LEMMAS; }
function loadLemmas(url = 'site-data/lemmas.json') {
  if (!lemmasLoading) lemmasLoading = fetch(url).then(r => r.ok ? r.json() : null).then(setLemmas).catch(() => false);
  return lemmasLoading;
}
function hasLemmas(site) { return !!((site && site.manifest && site.manifest.files) || {})['lemmas.json']; }   // listed by build_site_data.py when built
function lemmaOf(form) {               // the lemma of a normalised form; a form that is itself a lemma (heres) is its own
  if (!LEMMAS) return null;
  if (Object.hasOwn(LEMMAS, form)) return LEMMAS[form];      // own keys only: 'constructor' is a Latin word too
  return LEMMA_SET.has(form) ? form : null;
}
const LEMMA_KEY = '~';                 // marks lemma keys in the index, so that a form and a lemma of the same spelling stay apart
function indexTerm(term) {             // MiniSearch processTerm when indexing: the form and its lemma key
  const f = normalise(term); const l = lemmaOf(f);
  return l ? [f, LEMMA_KEY + l] : f;
}
function queryTerm(term) {             // MiniSearch processTerm when searching: the typed form (as a prefix) and, if known, its lemma key
  const f = normalise(term); const l = lemmaOf(f);
  return l ? [f, LEMMA_KEY + l] : f;    // the form keeps word beginnings working while typing ("hered" -> hereditas)
}
const isLemmaKey = t => t.startsWith(LEMMA_KEY);
const SEARCH_OPTIONS = {               // shared by the search worker and the reader; lemma keys match exactly
  boost: { title: 3, summary: 1.5, casus: 1.5, body: 1 }, processTerm: queryTerm,
  prefix: t => !isLemmaKey(t), fuzzy: t => !isLemmaKey(t) && t.length > 4 ? 0.15 : false,
};
/* ── query syntax of the full-text search ────────────────────────────────
   words            all of them in the same consilium (feudum vasallus)
   OR, |            either word (femina OR mulier)
   -word            consilia without that word (feudum -vasallus)
   "word"           that form only, without the other forms of its lemma ("heredem")
   a, b             several terms compared: each is searched on its own and the results are combined */
function splitGroups(raw) {            // terms to compare, separated by commas or semicolons outside quotation marks
  const out = []; let cur = '', quoted = false;
  for (const ch of raw) {
    if ('"“”„«»'.includes(ch)) quoted = !quoted;
    if (!quoted && (ch === ',' || ch === ';')) { if (cur.trim()) out.push(cur.trim()); cur = ''; } else cur += ch;
  }
  if (cur.trim()) out.push(cur.trim());
  return out;
}
function parseGroup(raw) {             // -> {clauses: [[{w, exact}]], neg: [{w, exact}]}: clauses are joined by AND, the words of a clause by OR
  const toks = []; const re = /(-?)["“”„«»]([^"“”„«»]*)["“”„«»]?|(\S+)/g; let m;
  while ((m = re.exec(raw)) !== null) {
    if (m[3] !== undefined) {
      if (m[3] === 'OR' || m[3] === '|') { toks.push({ or: true }); continue; }
      const neg = m[3].length > 1 && m[3][0] === '-', w = (neg ? m[3].slice(1) : m[3]).replace(/^[^\p{L}\d]+|[^\p{L}\d]+$/gu, '');
      if (w) toks.push({ w, exact: false, neg });
    } else m[2].split(/\s+/).filter(Boolean).forEach(w => toks.push({ w, exact: true, neg: m[1] === '-' }));
  }
  const clauses = [], neg = []; let joinNext = false;
  toks.forEach(t => {
    if (t.or) { joinNext = clauses.length > 0; return; }
    if (t.neg) { neg.push(t); joinNext = false; return; }
    if (joinNext) clauses[clauses.length - 1].push(t); else clauses.push([t]);
    joinNext = false;
  });
  return { clauses, neg };
}
function termQuery(t) {                // one word: form and lemma, or the quoted form only
  return t.exact ? { queries: [t.w], processTerm: normalise, prefix: false, fuzzy: false } : { queries: [t.w], combineWith: 'OR' };
}
function groupQuery(raw) {
  const { clauses, neg } = parseGroup(raw);
  if (!clauses.length) return null;
  const cl = clauses.map(c => c.length === 1 ? termQuery(c[0]) : { combineWith: 'OR', queries: c.map(termQuery) });
  const pos = cl.length === 1 ? cl[0] : { combineWith: 'AND', queries: cl };
  return neg.length ? { combineWith: 'AND_NOT', queries: [pos, ...neg.map(termQuery)] } : pos;
}
function buildQuery(raw) {             // MiniSearch query for the whole input: the compared terms joined by OR
  const qs = splitGroups(raw).map(groupQuery).filter(Boolean);
  if (!qs.length) return { queries: [], combineWith: 'OR' };
  return qs.length === 1 ? qs[0] : { combineWith: 'OR', queries: qs };
}
function queryWords(raw) {             // the words to mark in the text: every word that is searched for, not the excluded ones
  const out = [];
  splitGroups(raw).forEach(g => parseGroup(g).clauses.flat().forEach(t => out.push(t)));
  return out;
}
/* the text of a consilium as searched (title, argument, casus, text), its length in words and the matches of a
   highlight pattern in it: frequency curves on the search page and term_frequency in Ask */
function fullText(c) { return [c.title || '', c.summary || '', (c.casus || []).join(' '), (c.body || []).join(' ')].join(' '); }
function wordCount(c) { if (c._nw === undefined) c._nw = (fullText(c).match(/[^\s.,;:!?()]+/g) || []).length; return c._nw; }
function countMatches(c, re) { if (!re) return 0; re.lastIndex = 0; let n = 0; const t = fullText(c); while (re.exec(t) !== null) n++; return n; }
function formsOf(lemma) {             // all corpus forms of a lemma (for highlighting), the lemma itself included
  if (!LEMMAS) return [lemma];
  if (!LEMMA_FORMS) { LEMMA_FORMS = new Map(); for (const [f, l] of Object.entries(LEMMAS)) { if (!LEMMA_FORMS.has(l)) LEMMA_FORMS.set(l, []); LEMMA_FORMS.get(l).push(f); } }
  return [lemma, ...(LEMMA_FORMS.get(lemma) || [])];
}
const LETTER = '[A-Za-z\u00C0-\u024F\u0300-\u036F]';      // a letter, for whole-word matching without the u flag
const wholeWord = x => `(?<!${LETTER})${termToRegexPart(x)}(?!${LETTER})`;
function buildHighlightRe(raw) {
  const words = queryWords(raw);
  if (!words.length) return null;
  const parts = [];
  words.forEach(t => {
    const f = normalise(t.w);
    if (t.exact) { parts.push(wholeWord(f)); return; }                                        // quoted: that form only
    const l = lemmaOf(f);
    if (l) formsOf(l).slice(0, 80).forEach(x => parts.push(wholeWord(x)));                     // whole forms of the lemma
    parts.push(termToRegexPart(f));                                                            // and the typed text as a prefix
  });
  return new RegExp(`(${[...new Set(parts)].join('|')})`, 'gi');
}
function esc(s) {
  return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
function highlightText(text, re) {
  if (!re) return esc(text);
  re.lastIndex = 0;
  const parts = []; let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    parts.push(esc(text.slice(last, m.index)));
    parts.push(`<mark>${esc(m[0])}</mark>`);
    last = m.index + m[0].length;
  }
  parts.push(esc(text.slice(last)));
  return parts.join('');
}

/* ── names of prints ───────────────────────────────────────────────────── */
function parsePrint(v) {
  const m = String(v).match(/^([A-Za-z]+)_Cons_(?:Print_)?([A-Za-z]+)_(\d{4})_v(\d+)/i);
  return m ? { author: m[1], city: m[2], year: m[3], vol: m[4] } : null;
}
function volLabel(v) {
  const p = parsePrint(v);
  return p ? `${p.author}, Consilia (${p.city} ${p.year}), vol. ${p.vol}` : String(v).replace(/_/g, ' ');
}
function volShort(v) { const p = parsePrint(v); return p ? `${p.author.slice(0, 4)}. ${p.vol}` : String(v).slice(-4); }
function printLabel(v, authors) {
  const p = parsePrint(v);
  const viaf = authors && Object.keys(authors).find(k => (authors[k].prints || []).includes(v));
  const name = viaf ? (authors[viaf].name || viaf) : (p ? p.author : v);
  return p ? `${name}, ${p.city} ${p.year}, vol. ${p.vol}` : String(v).replace(/_/g, ' ');
}

/* ── data loading ──────────────────────────────────────────────────────── */
function fetchJSON(url) {
  return fetch(url).then(r => { if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`); return r.json(); });
}
function fetchJSONOpt(url) {
  return fetch(url).then(r => r.ok ? r.json() : ({})).catch(() => ({}));
}
/* What every page needs: the metadata index plus registries. */
async function loadSiteData() {
  const [index, authors, manifest, iiif, images, editions] = await Promise.all([
    fetchJSON('site-data/index.json'), fetchJSON('site-data/authors.json'), fetchJSON('site-data/manifest.json'),
    fetchJSONOpt('site-data/iiif_manifests.json'), fetchJSONOpt('site-data/image_sources.json'), fetchJSONOpt('site-data/editions.json'),
  ]);
  delete iiif._comment; delete images._comment;
  const byId = {}; index.forEach(e => { byId[e.id] = e; });
  // published annotation layers (src/annotate/export.py) are listed in the manifest when they exist
  const files = manifest.files || {};
  const [subjects, body] = await Promise.all([
    files['annotations/subjects.json'] ? fetchJSONOpt('site-data/annotations/subjects.json') : {},
    files['annotations/intimate_body.json'] ? fetchJSONOpt('site-data/annotations/intimate_body.json') : {}]);
  return { index, byId, authors, manifest, iiif, images, editions, annotations: { subjects, body } };
}
/* Annotation layers of one consilium: subjects (Layer A, per consilium) and the body/sexuality codes
   (Layer B, per text unit), with their labels; empty lists when a layer is not published. */
function annotationsFor(site, id) {
  const out = { subjects: [], body: [] };
  const S = site.annotations && site.annotations.subjects, B = site.annotations && site.annotations.body;
  if (S && S.concepts && S.docs && S.docs[id]) {
    const byId = {}; S.concepts.forEach(k => { byId[k.id] = k; });
    const d = S.docs[id];
    [...(d.central || []).map(k => [k, true]), ...(d.secondary || []).map(k => [k, false])].forEach(([k, central]) => {
      const c = byId[k]; if (c) out.subjects.push({ ...c, central });
    });
  }
  if (B && B.units && B.codes) {
    const codes = {}; B.codes.forEach(k => { codes[k.id] = k; });
    const counted = new Set(B.counted_roles_default || []);
    B.units.filter(u => u.doc === id).forEach(u => {
      const k = codes[u.code] || { id: u.code, label_en: u.code };
      out.body.push({ ...u, label_en: k.label_en, label_de: k.label_de || '', label_la: k.label_la || '', group: k.group, counted: counted.has(u.role) });
    });
  }
  return out;
}
/* `argument` in the text shard is the structured list; `summary` is rebuilt from it. */
function summaryOf(argument) {
  return (argument || []).map(it => `${it.n} ${it.text}`.trim()).join(' ').trim();
}
const _shardCache = {};
function getShard(kind, vol) {           // kind: 'text' | 'tr/en' | 'tr/de'
  const k = `${kind}/${vol}`;
  if (!_shardCache[k]) _shardCache[k] = fetchJSON(`site-data/${kind}/${vol}.json`).catch(e => { delete _shardCache[k]; throw e; });
  return _shardCache[k];
}

/* ── permanent addresses and citation ──────────────────────────────────── */
/* The public address of the portal comes from site-data/editions.json (`portal.url`), so that a local or
   offline copy still cites the published site. Without that entry the current location is used. */
function portalURL(site) {
  const p = site && site.editions && site.editions.portal;
  if (p && p.url) return p.url.replace(/\/?$/, '/');
  return `${location.origin}${location.pathname.replace(/[^/]*$/, '')}`;
}
function repoURL(site) {
  const p = site && site.editions && site.editions.portal;
  return (p && p.repository) || '';
}
function permalink(site, id) { return `${portalURL(site)}index.html?id=${encodeURIComponent(id)}`; }
function citeText(c, site) {
  const authors = site.authors || {}, manifest = site.manifest || {}, portal = (site.editions && site.editions.portal) || {};
  const p = parsePrint(c.volume) || {};
  const viaf = Object.keys(authors).find(k => (authors[k].prints || []).includes(c.volume));
  const name = viaf ? authors[viaf].name : (p.author || '');
  const d = new Date(), today = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
  const pars = c.pars ? `, pars ${c.pars}` : '';
  return `${name}, Consilia (${p.city || ''} ${p.year || ''}), vol. ${p.vol || ''}${pars}, cons. ${c.n}. ` +
    `In: ${portal.title || 'Consilia. A portal for medieval legal opinions'}, ed. ${portal.editor || 'Michael Schonhardt'} (beta, data revision ${manifest.revision || '?'}). ` +
    `${permalink(site, c.id)} (accessed ${today}).`;
}
async function copyText(s) {
  try { await navigator.clipboard.writeText(s); return true; } catch (e) { return false; }
}

/* Machine translations mark Latin words and titles with *asterisks* (markdown habit of the models);
   they are shown in italics. Everything else is escaped as usual. */
function trHTML(t) {
  return esc(t).replace(/(^|[\s(\[„"'])\*(\S(?:[^*\n]{0,78}?\S)?)\*(?=[\s.,;:)\]“"'?!]|$)/g, '$1<i>$2</i>');
}

/* ── theme toggle (system default; the button overrides and remembers) ─── */
function initTheme(btnId) {
  const KEY = 'consilia-theme';
  const root = document.documentElement;
  const apply = () => { const t = localStorage.getItem(KEY); if (t) root.dataset.theme = t; else root.removeAttribute('data-theme'); sync(); };
  const isDark = () => root.dataset.theme === 'dark' || (!root.dataset.theme && matchMedia('(prefers-color-scheme: dark)').matches);
  const SUN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5 5l1.5 1.5M17.5 17.5L19 19M19 5l-1.5 1.5M6.5 17.5L5 19"/></svg>';
  const MOON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M20 14.5A8 8 0 1 1 9.5 4a6.3 6.3 0 0 0 10.5 10.5z"/></svg>';
  const btn = () => document.getElementById(btnId);
  function sync() { const b = btn(); if (b) { b.innerHTML = isDark() ? SUN : MOON; b.title = isDark() ? 'Light mode' : 'Dark mode'; } }
  try { apply(); } catch (e) { /* storage blocked */ }
  document.addEventListener('DOMContentLoaded', () => {
    sync();
    const b = btn();
    if (b) b.addEventListener('click', () => { try { localStorage.setItem(KEY, isDark() ? 'light' : 'dark'); } catch (e) {} apply(); });
  });
}
if (typeof document !== 'undefined') initTheme('theme-toggle');

/* Layer C (citations and entities): the index (normalised citations and jurist names -> consilia) and the
   per-print shards with every span, loaded on demand; {} when the layer is not published. */
const _refCache = {};
function loadReferencesIndex(site) {
  if (!_refCache.index) _refCache.index = (site.manifest.files || {})['annotations/references_index.json'] ? fetchJSONOpt('site-data/annotations/references_index.json') : Promise.resolve({});
  return _refCache.index;
}
function loadReferencesShard(site, volume) {
  const k = 'shard:' + volume;
  if (!_refCache[k]) _refCache[k] = (site.manifest.files || {})[`annotations/references/${volume}.json`] ? fetchJSONOpt(`site-data/annotations/references/${volume}.json`) : Promise.resolve({});
  return _refCache[k];
}
/* a section text with its spans marked: <mark class="sp sp-LABEL" data-key=…>; overlapping spans are skipped */
function spanHTML(text, spans) {
  const sp = (spans || []).map(x => ({ a: x.o + x.a, b: x.o + x.b, l: x.l, key: x.rk || x.jk || '', layer: x.rk ? 'references' : 'jurists', st: x.st })).filter(x => x.a >= 0 && x.b <= text.length && x.b > x.a).sort((x, y) => x.a - y.a || y.b - x.b);
  let out = '', pos = 0;
  for (const x of sp) {
    if (x.a < pos) continue;
    out += esc(text.slice(pos, x.a));
    const title = (x.layer === 'references' && x.key ? x.key + ' · ' : '') + x.l.replace(/_/g, ' ').toLowerCase() + (x.st === 'verified_by_model' ? '' : ', unverified');
    out += `<mark class="sp sp-${esc(x.l)}" title="${esc(title)}"${x.key ? ` data-layer="${x.layer}" data-key="${esc(x.key)}"` : ''}>${esc(text.slice(x.a, x.b))}</mark>`;
    pos = x.b;
  }
  return out + esc(text.slice(pos));
}

