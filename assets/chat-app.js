/* Ask: a tool-using assistant over the static edition data, driven by a chat API the reader pays for:
   any OpenAI-compatible endpoint (OpenAI, Gemini's compatibility endpoint, a local vLLM) or
   the Anthropic Messages API (Claude). The browser executes the tools; the model only decides which
   to call. Key and endpoint live in localStorage. No server of this site is involved. */
const $ = id => document.getElementById(id);
const KEY = 'consilia-chat-cfg';
const PRESETS = {
  anthropic: { kind: 'anthropic', url: 'https://api.anthropic.com/v1', model: 'claude-sonnet-5-5',
    hint: 'Anthropic API key from <a href="https://console.anthropic.com/settings/keys" target="_blank" rel="noopener">console.anthropic.com</a>. Billed to your account; the browser calls the API directly.' },
  openai: { kind: 'openai', url: 'https://api.openai.com/v1', model: 'gpt-5-mini',
    hint: 'OpenAI API key from <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener">platform.openai.com</a>. Test connection lists the models your key can use.' },
  gemini: { kind: 'openai', url: 'https://generativelanguage.googleapis.com/v1beta/openai', model: 'gemini-3.8-flash',
    hint: 'Google AI Studio key from <a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener">aistudio.google.com</a>, used through Gemini\'s OpenAI-compatible endpoint.' },
  custom: { kind: 'openai', url: '', model: '', hint: 'Any OpenAI-compatible chat endpoint that accepts browser requests, for example a local vLLM server.' },
};
function preset() { return PRESETS[$('cfg-preset').value] || PRESETS.custom; }
function showHint() { $('cfg-hint').innerHTML = preset().hint; }
let site = null, consilia = {}, worker = null, seq = 0; const waiters = {};
let messages = [], busy = false, toolsSupported = true, textsReady = null, aborter = null;
/* semantic search: passage vectors of the chosen browser model (assets/vectors.js); the query model is
   downloaded only when the user clicks "Load model" */
let vecModels = [], vecStore = null, vecTag = null, encoder = null;
function semStatus(msg) { const el = $('sem-status'); if (el) el.textContent = msg; }
async function initSemantic(s) {
  try { const m = await fetchJSON('site-data/models.json'); vecModels = (m.models || m).filter(x => x.onnx_ready && !x.requires_webgpu); } catch (e) { vecModels = []; }
  vecStore = new VectorStore(s, vecModels);
  const sel = $('sem-model'); if (!sel) return;
  try { vecTag = localStorage.getItem('consilia-chat-vec') || null; } catch (e) {}
  if (!vecModels.find(x => x.tag === vecTag)) vecTag = (vecModels.find(x => x.default) || vecModels[0] || {}).tag || null;
  sel.innerHTML = vecModels.map(x => `<option value="${esc(x.tag)}"${x.tag === vecTag ? ' selected' : ''}>${esc(x.label)} (${x.download_mb && x.download_mb.wasm ? x.download_mb.wasm + ' MB' : 'browser'})</option>`).join('') || '<option value="">no browser model published</option>';
  sel.addEventListener('change', () => { vecTag = sel.value; try { localStorage.setItem('consilia-chat-vec', vecTag); } catch (e) {} if (encoder) { encoder.terminate(); encoder = null; } semStatus(''); $('sem-load').disabled = false; $('sem-load').textContent = 'Load model'; });
  $('sem-load').addEventListener('click', async () => {
    const cfg = vecStore.cfg(vecTag); if (!cfg) return;
    $('sem-load').disabled = true; $('sem-load').textContent = 'Loading…';
    try { await vecStore.load(vecTag, semStatus); encoder = new QueryEncoder(cfg, semStatus); await encoder.load(); $('sem-load').textContent = `${cfg.label} loaded`; }
    catch (e) { semStatus('Could not load: ' + e.message); $('sem-load').disabled = false; $('sem-load').textContent = 'Load model'; encoder = null; }
  });
}
async function semanticHits(rows) {
  const out = [];
  for (const r of rows) { const c = consilia[r.id]; if (!c) continue; const p = await vecStore.passage(vecTag, r.row); out.push({ ...ref(c), score: +r.score.toFixed(3), argumentum: (c.summary || '').slice(0, 200), passage: p.text.slice(0, 400), passage_lang: p.lang }); }
  return out;
}
const MAX_ROUNDS = 30;      // tool rounds per question before the model is asked to answer with what it has
const NUDGE_ROUND = 8;      // after this many rounds the model is reminded to answer

/* ── config ────────────────────────────────────────────────────────────── */
function cfg() { return { kind: preset().kind, url: $('cfg-url').value.trim().replace(/\/$/, ''), key: $('cfg-key').value.trim(), model: $('cfg-model').value.trim() }; }
function loadCfg() { try { const c = JSON.parse(localStorage.getItem(KEY) || '{}'); if (c.url) $('cfg-url').value = c.url; if (c.key) $('cfg-key').value = c.key; if (c.model) $('cfg-model').value = c.model; if (c.preset) $('cfg-preset').value = c.preset; } catch (e) {} showHint(); }
$('cfg-save').addEventListener('click', () => { try { localStorage.setItem(KEY, JSON.stringify({ ...cfg(), preset: $('cfg-preset').value })); $('cfg-status').textContent = 'Saved.'; } catch (e) { $('cfg-status').textContent = 'Storage blocked; settings last for this page only.'; } });
$('cfg-preset').addEventListener('change', e => { const p = PRESETS[e.target.value]; $('cfg-url').value = p.url; $('cfg-model').value = p.model; $('cfg-key').value = ''; $('model-options').innerHTML = ''; $('cfg-status').textContent = ''; showHint(); });
function authHeaders(c) {
  return c.kind === 'anthropic'
    ? { 'x-api-key': c.key, 'anthropic-version': '2023-06-01', 'anthropic-dangerous-direct-browser-access': 'true' }
    : { Authorization: 'Bearer ' + c.key };
}
$('cfg-test').addEventListener('click', async () => {
  const c = cfg(); $('cfg-status').textContent = 'Testing…';
  try {
    const r = await fetch(c.url + '/models', { headers: authHeaders(c) });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const j = await r.json(); const ids = (j.data || []).map(m => m.id);
    $('model-options').innerHTML = ids.map(i => `<option value="${esc(i)}">`).join('');
    $('cfg-status').textContent = `OK: ${ids.length} models available.`; $('cfg-status').className = 'status ready';
  } catch (e) { $('cfg-status').className = 'status error'; $('cfg-status').textContent = netError(e); }
});
function netError(e) {
  if (e && e.name === 'AbortError') return 'Stopped.';
  if (e instanceof TypeError) return 'The browser could not reach this endpoint. Causes: the provider does not allow browser requests (Anthropic, OpenAI and Gemini do), a content or tracking blocker stops the API host, a mistyped base URL, or a conversation that grew too large for the provider after many rounds (press New).';
  return 'Error: ' + e.message;
}

/* ── corpus tools ──────────────────────────────────────────────────────── */
loadSiteData().then(s => {
  site = s; s.index.forEach(e => { e.summary = e.argument_short; consilia[e.id] = e; });
  loadCfg(); initSemantic(s);
  worker = new Worker('assets/search-worker.js');
  if (hasLemmas(s)) { loadLemmas(); worker.postMessage({ type: 'lemmas', url: '../site-data/lemmas.json' }); }
  worker.onmessage = e => { const m = e.data, w = waiters[m.seq]; if (w) { delete waiters[m.seq]; m.type === 'error' ? w.reject(new Error(m.message)) : w.resolve(m.results); } };
  const vols = [...new Set(s.index.map(c => c.volume))].sort();
  textsReady = vols.reduce((p, v) => p.then(async () => {
    const text = await getShard('text', v); const docs = [];
    s.index.filter(c => c.volume === v).forEach(c => { const t = text[c.id] || {}; c.argument = t.argument || []; c.casus = t.casus || []; c.body = t.body || []; c.summary = summaryOf(c.argument) || c.summary; docs.push({ id: c.id, title: c.title || '', summary: c.summary, casus: c.casus.join(' '), body: c.body.join(' ') }); });
    worker.postMessage({ type: 'add', docs });
  }), Promise.resolve());
  addMsg('system', `Corpus loaded: ${s.manifest.n_consilia} consilia in ${vols.length} volumes. Set up an endpoint, then ask.`);
}).catch(e => addMsg('system', 'Could not load corpus: ' + e, true));

function fulltext(q) { return new Promise((resolve, reject) => { const s = ++seq; waiters[s] = { resolve, reject }; worker.postMessage({ type: 'search', q, seq: s }); }); }
function ref(c) { return { id: c.id, heading: c.title_display || c.title, print: volLabel(c.volume), print_short: volShort(c.volume), n: c.n, pars: c.pars || null, title: c.title, url: `index.html?id=${encodeURIComponent(c.id)}` }; }
const TOOLS = {
  corpus_stats: { description: 'Counts of consilia per print and author, data revision. Use for any question about size or coverage; never guess totals.', parameters: { type: 'object', properties: {} },
    run: async () => { const by = {}; site.index.forEach(c => { by[c.volume] = (by[c.volume] || 0) + 1; }); return { total: site.index.length, revision: site.manifest.revision, prints: Object.entries(by).map(([v, n]) => ({ print: v, label: volLabel(v), consilia: n })), authors: Object.fromEntries(Object.entries(site.authors).map(([k, a]) => [a.name, a.prints])) }; } },
  search_fulltext: { description: 'Full-text search over title, argumentum, casus and Latin text of all consilia, the "Full text" search of the website. total is the complete number of matching consilia. A word is found in all its forms (lemma list: heredem also finds heres, heredibus); i/j, u/v and ae/e count as equal; word beginnings match. Syntax: "feudum vasallus" = both words in the same consilium; "femina OR mulier" = either word; "feudum -vasallus" = without the second word; a word in double quotation marks = that form only; "feudum, vasallus" = terms compared, results combined (use term_frequency for the numbers). Search with Latin words; join synonyms with OR, forms of one word need no OR.', parameters: { type: 'object', properties: { query: { type: 'string' }, limit: { type: 'integer', default: 10 } }, required: ['query'] },
    run: async ({ query, limit = 10 }) => { await textsReady; const res = await fulltext(query); const re = buildHighlightRe(query); if (!re) return { total: 0, hits: [] }; return { total: res.length, search_page: `search.html?mode=keyword&q=${encodeURIComponent(query)}`, hits: res.slice(0, Math.min(limit, 25)).map(r => { const c = consilia[r.id]; const body = (c.body || []).join(' '); re.lastIndex = 0; const m = re.exec(body); const snip = m ? body.slice(Math.max(0, m.index - 150), m.index + 200) : body.slice(0, 200); return { ...ref(c), score: +r.score.toFixed(1), matched_fields: Object.values(r.match || {}).flat(), argumentum: (c.summary || '').slice(0, 300), snippet: snip }; }) }; } },
  term_frequency: { description: 'Compare how often words occur: for each term the number and share of consilia that contain it and its occurrences per 10,000 words, in total, per print and along the prints in sections of consecutive consilia (the frequency chart of the search page). Each term is a query in the syntax of search_fulltext ("feudum", "femina OR mulier"). Returns chart, a link to the search page that draws the curves; give it to the user.', parameters: { type: 'object', properties: { terms: { type: 'array', items: { type: 'string' }, description: 'one to six queries to compare' }, sections: { type: 'boolean', default: false, description: 'include the values per section of about 70 consilia' } }, required: ['terms'] },
    run: async ({ terms, sections = false }) => {
      await textsReady; terms = (terms || []).map(t => String(t).trim()).filter(Boolean).slice(0, 6); if (!terms.length) return { error: 'no terms given' };
      const pool = site.index, hits = [];
      for (const t of terms) { const re = buildHighlightRe(t), occ = {}; (await fulltext(t)).forEach(r => { const c = consilia[r.id]; if (c) occ[r.id] = countMatches(c, re); }); hits.push(occ); }
      const words = list => list.reduce((s, c) => s + wordCount(c), 0), sum = (list, occ) => list.reduce((s, c) => s + (occ[c.id] || 0), 0), has = (list, occ) => list.filter(c => c.id in occ).length;
      const pct = (a, b) => +(100 * a / Math.max(b, 1)).toFixed(2), p10k = (a, b) => +(1e4 * a / Math.max(b, 1)).toFixed(2);
      const W = words(pool), byVol = {}; pool.forEach(c => { (byVol[c.volume] = byVol[c.volume] || []).push(c); });
      const size = Math.max(10, Math.ceil(pool.length / 40)), out = { terms: [], prints: [], section_size: size };
      terms.forEach((t, i) => out.terms.push({ term: t, consilia: Object.keys(hits[i]).length, share_pct: pct(Object.keys(hits[i]).length, pool.length), occurrences: sum(pool, hits[i]), per_10k_words: p10k(sum(pool, hits[i]), W) }));
      if (sections) out.sections = [];
      Object.keys(byVol).sort().forEach(v => {
        const list = byVol[v].sort((a, b) => (a.pars || 0) - (b.pars || 0) || a.n - b.n), w = words(list);
        out.prints.push({ print: volLabel(v), consilia: list.length, terms: terms.map((t, i) => ({ term: t, consilia: has(list, hits[i]), share_pct: pct(has(list, hits[i]), list.length), per_10k_words: p10k(sum(list, hits[i]), w) })) });
        if (!sections) return;
        const nb = Math.max(1, Math.round(list.length / size)), lab = c => (c.pars ? c.pars + '.' : '') + c.n;
        for (let b = 0; b < nb; b++) { const part = list.slice(Math.floor(b * list.length / nb), Math.floor((b + 1) * list.length / nb)); if (!part.length) continue; const pw = words(part);
          out.sections.push({ print: volShort(v), from_n: lab(part[0]), to_n: lab(part[part.length - 1]), share_pct: hits.map(o => +(100 * has(part, o) / part.length).toFixed(1)), per_10k_words: hits.map(o => +(1e4 * sum(part, o) / Math.max(pw, 1)).toFixed(1)) }); }
      });
      out.chart = `search.html?mode=keyword&q=${encodeURIComponent(terms.join(', '))}`;
      out.note = 'share_pct: consilia containing the term in percent of the consilia of the print or section. per_10k_words: occurrences of the searched word forms per 10,000 words.';
      return out;
    } },
  list_keywords: { description: 'List the automatic keywords (English labels extracted by a language model) that contain a substring, with document counts. Then use consilia_by_keyword.', parameters: { type: 'object', properties: { contains: { type: 'string' } }, required: ['contains'] },
    run: async ({ contains }) => { const cnt = {}; site.index.forEach(c => (c.keywords_final || []).forEach(k => { cnt[k] = (cnt[k] || 0) + 1; })); const q = contains.toLowerCase(); return { keywords: Object.entries(cnt).filter(([k]) => k.toLowerCase().includes(q)).sort((a, b) => b[1] - a[1]).slice(0, 40).map(([k, n]) => ({ keyword: k, consilia: n })) }; } },
  consilia_by_keyword: { description: 'All consilia tagged with an exact automatic keyword.', parameters: { type: 'object', properties: { keyword: { type: 'string' }, limit: { type: 'integer', default: 30 } }, required: ['keyword'] },
    run: async ({ keyword, limit = 30 }) => { const hits = site.index.filter(c => (c.keywords_final || []).includes(keyword)); return { total: hits.length, hits: hits.slice(0, limit).map(c => ({ ...ref(c), argumentum: (c.summary || '').slice(0, 300) })) }; } },
  search_semantic: { description: 'Search by meaning with passage embeddings: the question (Latin, English or German) is embedded in the browser and compared with every body and casus passage; returns the consilia with the best passage each. Needs the semantic search model the user loads in the settings; if the result says it is not loaded, say so and use search_fulltext instead. Set lang to "en" to compare with the English translations, "la" with the Latin text, default both.', parameters: { type: 'object', properties: { query: { type: 'string' }, lang: { type: 'string', enum: ['both', 'la', 'en'], default: 'both' }, limit: { type: 'integer', default: 10 } }, required: ['query'] },
    run: async ({ query, lang = 'both', limit = 10 }) => { if (!vecStore || !vecTag) return { error: 'no semantic model published' }; if (!encoder || !encoder.ready) return { error: 'semantic search model not loaded: the user has to click "Load model" under "Semantic search model" above the chat; use search_fulltext meanwhile' }; await vecStore.load(vecTag, semStatus); const qv = await encoder.embed(query); const rows = vecStore.rank(qv, vecTag, { lang, limit: Math.min(limit, 25) }); return { model: vecStore.cfg(vecTag).label, hits: await semanticHits(rows) }; } },
  similar_consilia: { description: 'Consilia whose passages are closest to the passages of a given consilium (cosine similarity of stored embeddings; works without the query model, the vectors are loaded on first use).', parameters: { type: 'object', properties: { id: { type: 'string' }, lang: { type: 'string', enum: ['both', 'la', 'en'], default: 'both' }, limit: { type: 'integer', default: 10 } }, required: ['id'] },
    run: async ({ id, lang = 'both', limit = 10 }) => { if (!vecStore || !vecTag) return { error: 'no embedding model published' }; if (!consilia[id]) return { error: 'unknown id' }; await vecStore.load(vecTag, semStatus); const rows = vecStore.similar(id, vecTag, { lang, limit: Math.min(limit, 25) }); if (!rows.length) return { error: 'no passage vectors for this consilium' }; return { model: vecStore.cfg(vecTag).label, reference: ref(consilia[id]), hits: await semanticHits(rows) }; } },
  list_subjects: { description: 'The controlled subject scheme (Layer A): legal concepts grouped in facets, assigned per consilium as central or secondary by a language model. Returns concepts whose label (English, German or Latin) contains the substring, with consilium counts. Then use consilia_by_subject. Prefer this over keywords for legal topics (dowry, testament form, appeal, usury, statutes).', parameters: { type: 'object', properties: { contains: { type: 'string', description: 'substring, empty for the most frequent concepts' } } },
    run: async ({ contains = '' }) => { const S = site.annotations.subjects; if (!S || !S.concepts) return { error: 'subject layer not published' }; const cnt = {}, cen = {}; Object.entries(S.docs || {}).forEach(([cid, d]) => { (d.central || []).forEach(k => { cnt[k] = (cnt[k] || 0) + 1; cen[k] = (cen[k] || 0) + 1; }); (d.secondary || []).forEach(k => { cnt[k] = (cnt[k] || 0) + 1; }); }); const q = contains.toLowerCase(); const facets = Object.fromEntries((S.facets || []).map(f => [f.id, f.en])); return { concepts: S.concepts.filter(k => !q || [k.en, k.de, k.la, k.id].some(x => (x || '').toLowerCase().includes(q))).map(k => ({ id: k.id, label_en: k.en, label_la: k.la, facet: facets[k.facet] || k.facet, scope: k.scope, consilia: cnt[k.id] || 0, central: cen[k.id] || 0 })).sort((a, b) => b.consilia - a.consilia).slice(0, 60) }; } },
  consilia_by_subject: { description: 'All consilia assigned to a subject concept (Layer A), central assignments first. Use the concept id from list_subjects.', parameters: { type: 'object', properties: { concept: { type: 'string' }, central_only: { type: 'boolean', default: false }, limit: { type: 'integer', default: 30 } }, required: ['concept'] },
    run: async ({ concept, central_only = false, limit = 30 }) => { const S = site.annotations.subjects; if (!S || !S.docs) return { error: 'subject layer not published' }; const hits = []; Object.entries(S.docs).forEach(([cid, d]) => { const c = consilia[cid]; if (!c) return; if ((d.central || []).includes(concept)) hits.push({ ...ref(c), assignment: 'central', argumentum: (c.summary || '').slice(0, 300) }); else if (!central_only && (d.secondary || []).includes(concept)) hits.push({ ...ref(c), assignment: 'secondary', argumentum: (c.summary || '').slice(0, 300) }); }); hits.sort((a, b) => (a.assignment === 'central' ? 0 : 1) - (b.assignment === 'central' ? 0 : 1)); return { total: hits.length, central: hits.filter(h => h.assignment === 'central').length, hits: hits.slice(0, limit) }; } },
  list_body_codes: { description: 'The codebook of the layer "body and sexuality" (Layer B): codes such as adultery, impotence, pregnancy, virginity, with the number of consilia in which a language model found them as topic or argument (counted) or only as case fact, citation or formula. Then use consilia_by_body_code.', parameters: { type: 'object', properties: {} },
    run: async () => { const B = site.annotations.body; if (!B || !B.units) return { error: 'body layer not published' }; const counted = new Set(B.counted_roles_default || []); const by = {}; B.units.forEach(u => { const t = (by[u.code] = by[u.code] || { all: new Set(), counted: new Set() }); t.all.add(u.doc); if (counted.has(u.role)) t.counted.add(u.doc); }); return { roles: B.roles, counted_roles: B.counted_roles_default, codes: (B.codes || []).map(k => ({ id: k.id, label_en: k.label_en, label_la: k.label_la, group: k.group, consilia: (by[k.id] || { all: new Set() }).all.size, consilia_counted: (by[k.id] || { counted: new Set() }).counted.size })).filter(k => k.consilia).sort((a, b) => b.consilia_counted - a.consilia_counted) }; } },
  consilia_by_body_code: { description: 'All consilia carrying a Layer B code, with the Latin evidence quotes and the role of each occurrence (topic, argument, case_fact, citation_only, formulaic). By default only the counted roles topic and argument.', parameters: { type: 'object', properties: { code: { type: 'string' }, roles: { type: 'array', items: { type: 'string' }, description: 'roles to include; default the counted roles' }, limit: { type: 'integer', default: 30 } }, required: ['code'] },
    run: async ({ code, roles, limit = 30 }) => { const B = site.annotations.body; if (!B || !B.units) return { error: 'body layer not published' }; const want = new Set(roles && roles.length ? roles : (B.counted_roles_default || [])); const by = {}; B.units.filter(u => u.code === code && want.has(u.role)).forEach(u => { const c = consilia[u.doc]; if (!c) return; (by[u.doc] = by[u.doc] || { ...ref(c), argumentum: (c.summary || '').slice(0, 200), occurrences: [] }).occurrences.push({ role: u.role, evidence_la: u.evidence_la, status: u.status, section: u.section }); }); const hits = Object.values(by).sort((a, b) => b.occurrences.length - a.occurrences.length); return { total: hits.length, hits: hits.slice(0, limit) }; } },
  list_references: { description: 'Cited works (Layer C): the titles of the Corpus iuris civilis (D. = Digest, C. = Codex, Inst., Nov.), the Corpus iuris canonici (X = Liber Extra, VI, Clem., Decr.) and the Libri feudorum (LF) that the consilia cite, resolved by rule from the printed abbreviations (about half of the citations resolve; the rest stay as printed). Keys like "D. 2.14" (Digest, De pactis), "X 2.20" (De testibus). Filter by substring of the key, the title or the printed forms. Then use consilia_by_reference.', parameters: { type: 'object', properties: { contains: { type: 'string', description: 'e.g. "D. 2.14", "pactis", "de testibus"' } } },
    run: async ({ contains = '' }) => { const ix = await loadReferencesIndex(site); if (!ix || !ix.refs) return { error: 'citation layer not published' }; const q = contains.toLowerCase(); return { total: Object.keys(ix.refs).length, refs: Object.entries(ix.refs).filter(([k, v]) => !q || k.toLowerCase().includes(q) || (v.title || '').toLowerCase().includes(q) || (v.forms || []).some(f => f.toLowerCase().includes(q))).sort((a, b) => b[1].docs.length - a[1].docs.length).slice(0, 60).map(([k, v]) => ({ key: k, title: v.title, consilia: v.docs.length, passages: v.n, printed_forms: v.forms })) }; } },
  consilia_by_reference: { description: 'All consilia citing a work and title (key from list_references, e.g. "D. 35.2").', parameters: { type: 'object', properties: { key: { type: 'string' }, limit: { type: 'integer', default: 30 } }, required: ['key'] },
    run: async ({ key, limit = 30 }) => { const ix = await loadReferencesIndex(site); const v = ix && ix.refs && ix.refs[key]; if (!v) return { error: 'unknown key; use list_references' }; return { key, title: v.title, printed_forms: v.forms, total: v.docs.length, hits: v.docs.map(id => consilia[id]).filter(Boolean).slice(0, limit).map(c => ({ ...ref(c), argumentum: (c.summary || '').slice(0, 200) })) }; } },
  list_jurists: { description: 'Jurists and glosses cited by name (Layer C): Bartolus, Innocentius IV, Cynus, Johannes Andreae, the glossa ordinaria and others, with the number of consilia citing each. Then use consilia_by_jurist.', parameters: { type: 'object', properties: { contains: { type: 'string' } } },
    run: async ({ contains = '' }) => { const ix = await loadReferencesIndex(site); if (!ix || !ix.jurists) return { error: 'citation layer not published' }; const q = contains.toLowerCase(); return { jurists: Object.entries(ix.jurists).filter(([k, v]) => !q || k.toLowerCase().includes(q) || (v.forms || []).some(f => f.toLowerCase().includes(q))).sort((a, b) => b[1].docs.length - a[1].docs.length).slice(0, 60).map(([k, v]) => ({ name: k, consilia: v.docs.length, passages: v.n, printed_forms: v.forms })) }; } },
  consilia_by_jurist: { description: 'All consilia citing a jurist (name from list_jurists).', parameters: { type: 'object', properties: { name: { type: 'string' }, limit: { type: 'integer', default: 30 } }, required: ['name'] },
    run: async ({ name, limit = 30 }) => { const ix = await loadReferencesIndex(site); const v = ix && ix.jurists && ix.jurists[name]; if (!v) return { error: 'unknown name; use list_jurists' }; return { name, printed_forms: v.forms, total: v.docs.length, hits: v.docs.map(id => consilia[id]).filter(Boolean).slice(0, limit).map(c => ({ ...ref(c), argumentum: (c.summary || '').slice(0, 200) })) }; } },
  get_consilium: { description: 'Full record of one consilium by id: casus, argumentum, Latin text sections, the English translation if available, and its subject and body annotations.', parameters: { type: 'object', properties: { id: { type: 'string' }, with_translation: { type: 'boolean', default: true } }, required: ['id'] },
    run: async ({ id, with_translation = true }) => { const c = consilia[id]; if (!c) return { error: 'unknown id' }; await textsReady; const ann = annotationsFor(site, c.id); const out = { ...ref(c), casus: c.casus, argumentum: c.argument, text: c.body, keywords: c.keywords_final || [], subjects: ann.subjects.map(k => ({ id: k.id, label_en: k.en, central: k.central })), body_annotations: ann.body.map(u => ({ code: u.code, label_en: u.label_en, role: u.role, evidence_la: u.evidence_la })) }; try { const sh = await loadReferencesShard(site, c.volume); const sp = (sh && sh[c.id]) || []; const groups = { CIT_CIV: 'civil', CIT_CAN: 'canon', CIT_FEUD: 'feudal', CIT_OTHER: 'other' }; out.citations_as_printed = sp.filter(x => groups[x.l]).map(x => ({ corpus: groups[x.l], text: x.t, resolved: x.rk || null })); const cnt = {}; sp.forEach(x => { if (x.jk) cnt[x.jk] = (cnt[x.jk] || 0) + 1; }); out.jurists_cited = Object.entries(cnt).sort((a, b) => b[1] - a[1]).map(([k, n]) => ({ name: k, n })); } catch (e) {} if (with_translation) { try { const tr = await getShard('tr/en', c.volume); out.translation_en = tr[id] || null; } catch (e) { out.translation_en = null; } } return out; } },
};
function toolSpecs() { return Object.entries(TOOLS).map(([name, t]) => ({ type: 'function', function: { name, description: t.description, parameters: t.parameters } })); }
const SYSTEM = `You are the research assistant of "Consilia", a digital edition of late-medieval legal opinions (Latin) by Baldus de Ubaldis (Consilia, Venice 1575, 5 volumes) and Panormitanus (Consilia, Venice 1571). Answer from the edition's data only, using the tools; never invent consilia, numbers or quotations. Two annotation layers exist besides the automatic keywords: a controlled subject scheme per consilium (list_subjects, consilia_by_subject) and a codebook on body and sexuality with Latin evidence quotes (list_body_codes, consilia_by_body_code); both are machine assignments, say so when you rely on them. A third layer indexes the cited works by title (list_references, consilia_by_reference; resolved by rule from the printed abbreviations, about half of all citations) and the jurists cited by name (list_jurists, consilia_by_jurist). search_semantic finds passages by meaning when the user has loaded the browser embedding model; similar_consilia finds neighbours of a consilium from stored vectors. Combine a semantic search with full-text searches for Latin terms. Search in Latin: search_fulltext finds all forms of a word by itself (dos also finds dotis, dotem), several words must all occur, synonyms are joined with OR (femina OR mulier) and a leading minus excludes a word. Its total is the complete number of matching consilia, whereas search_semantic and similar_consilia return only the nearest consilia; give totals only from search_fulltext or the annotation layers. For questions about how often words occur, how they compare or where in the prints they are frequent, use term_frequency and give the user its chart link as a markdown link. The consilia you name are examples from your searches: say so when a search found more than you discuss, and give the number. Read the full text of the most relevant consilia with get_consilium before summarising their content. Cite every consilium you mention as a markdown link with its print and number, e.g. [Baldus, vol. 4, cons. 62](index.html?id=consilium-baldo_cons_print_venice_1575_v4-62). Say when you rely on a machine translation or an automatic keyword. Answer in the language of the question. Be concise and precise.`;

/* ── chat loop ─────────────────────────────────────────────────────────── */
function mdInline(h) {        // h is escaped text
  h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
  h = h.replace(/\[([^\]]+)\]\(((?:index|search)\.html\?[^)\s]+|https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  h = h.replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/__(.+?)__/g, '<b>$1</b>');
  return h.replace(/(^|[^*\w])\*(?!\s)(.+?)(?<!\s)\*(?!\w)/g, '$1<i>$2</i>');
}
/* the markdown a chat model writes: headings, rules, quotes, lists, tables, paragraphs; inline marks per block */
function md(s) {
  const lines = esc(s).split('\n'); let out = '', list = null, para = [];
  const flushPara = () => { if (para.length) { out += `<p>${mdInline(para.join(' '))}</p>`; para = []; } };
  const closeList = () => { if (list) { out += `</${list}>`; list = null; } };
  for (let i = 0; i < lines.length; i++) {
    const ln = lines[i];
    if (!ln.trim()) { flushPara(); closeList(); continue; }
    const hd = ln.match(/^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/);
    if (hd) { flushPara(); closeList(); const lv = hd[1].length <= 2 ? 3 : hd[1].length === 3 ? 4 : 5; out += `<h${lv}>${mdInline(hd[2])}</h${lv}>`; continue; }
    if (/^\s{0,3}([-*_])(\s*\1){2,}\s*$/.test(ln)) { flushPara(); closeList(); out += '<hr>'; continue; }
    if (/^\s*\|.*\|\s*$/.test(ln) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
      flushPara(); closeList();
      const cells = r => r.trim().replace(/^\||\|$/g, '').split('|').map(c => mdInline(c.trim()));
      out += '<table><thead><tr>' + cells(ln).map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
      i += 2;
      for (; i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i]); i++) out += '<tr>' + cells(lines[i]).map(c => `<td>${c}</td>`).join('') + '</tr>';
      i--; out += '</tbody></table>'; continue;
    }
    const q = ln.match(/^\s*&gt;\s?(.*)$/);
    if (q) { flushPara(); closeList(); out += `<blockquote>${mdInline(q[1])}</blockquote>`; continue; }
    const li = ln.match(/^\s*(?:[-*+]|\d+[.)])\s+(.*)$/);
    if (li) { flushPara(); const kind = /^\s*\d/.test(ln) ? 'ol' : 'ul'; if (list !== kind) { closeList(); out += `<${kind}>`; list = kind; } out += `<li>${mdInline(li[1])}</li>`; continue; }
    if (list && /^\s{2,}\S/.test(ln)) { out = out.replace(/<\/li>$/, ' ' + mdInline(ln.trim()) + '</li>'); continue; }   // continuation of a list item
    closeList(); para.push(ln.trim());
  }
  flushPara(); closeList(); return out.replace(/(<\/blockquote>)<blockquote>/g, '<br>');
}
function addMsg(who, text, isErr) {
  const d = document.createElement('div'); d.className = 'msg' + (who === 'tool' ? ' tool' : '');
  d.innerHTML = `<div class="who">${esc(who)}</div><div class="body${isErr ? ' error' : ''}">${who === 'assistant' ? md(text) : who === 'tool' ? text : `<p>${esc(text)}</p>`}</div>`;
  $('log').appendChild(d); d.scrollIntoView({ block: 'end' }); return d;
}
/* Anthropic Messages API: same conversation (kept in OpenAI shape) translated on the way out and back. */
function toAnthropic(msgs) {
  const out = []; let system = '';
  for (const m of msgs) {
    if (m.role === 'system') { system = m.content; continue; }
    if (m.role === 'tool') {
      const block = { type: 'tool_result', tool_use_id: m.tool_call_id, content: m.content };
      const last = out[out.length - 1];
      if (last && last.role === 'user' && Array.isArray(last.content) && last.content[0].type === 'tool_result') last.content.push(block);
      else out.push({ role: 'user', content: [block] });
      continue;
    }
    if (m.role === 'assistant' && m.tool_calls && m.tool_calls.length) {
      const content = [];
      if (m.content) content.push({ type: 'text', text: m.content });
      m.tool_calls.forEach(tc => { let input = {}; try { input = JSON.parse(tc.function.arguments || '{}'); } catch (e) {} content.push({ type: 'tool_use', id: tc.id, name: tc.function.name, input }); });
      out.push({ role: 'assistant', content }); continue;
    }
    out.push({ role: m.role, content: m.content || '' });
  }
  return { system, messages: out };
}
async function callAnthropic(c, msgs, useTools) {
  const { system, messages: am } = toAnthropic(msgs);
  const body = { model: c.model, max_tokens: 4096, temperature: 0.2, system, messages: am };
  if (useTools) { body.tools = Object.entries(TOOLS).map(([name, t]) => ({ name, description: t.description, input_schema: t.parameters })); if (useTools === 'none') body.tool_choice = { type: 'none' }; }
  const r = await fetch(c.url + '/messages', { method: 'POST', headers: { 'Content-Type': 'application/json', ...authHeaders(c) }, body: JSON.stringify(body), signal: aborter && aborter.signal });
  if (!r.ok) { const t = await r.text(); const err = new Error(`HTTP ${r.status}: ${t.slice(0, 300)}`); err.status = r.status; err.body = t; throw err; }
  const j = await r.json();
  const text = (j.content || []).filter(b => b.type === 'text').map(b => b.text).join('\n');
  const tool_calls = (j.content || []).filter(b => b.type === 'tool_use').map(b => ({ id: b.id, type: 'function', function: { name: b.name, arguments: JSON.stringify(b.input || {}) } }));
  return tool_calls.length ? { role: 'assistant', content: text, tool_calls } : { role: 'assistant', content: text };
}
/* older tool results are shortened on the way out (the log keeps them): after many rounds the full
   history would grow to megabytes, which some providers reject at the edge without CORS headers */
const KEEP_FULL_TOOL_RESULTS = 6, OLD_TOOL_RESULT_CHARS = 1500;
function compactHistory(msgs) {
  const toolIdx = msgs.map((m, i) => m.role === 'tool' ? i : -1).filter(i => i >= 0);
  const cut = new Set(toolIdx.slice(0, Math.max(0, toolIdx.length - KEEP_FULL_TOOL_RESULTS)));
  return msgs.map((m, i) => cut.has(i) && m.content.length > OLD_TOOL_RESULT_CHARS ? { ...m, content: m.content.slice(0, OLD_TOOL_RESULT_CHARS) + ' …[earlier result shortened; call again if you need the details]' } : m);
}
async function callLLM(msgs, useTools) {
  const c = cfg(); if (!c.url || !c.key || !c.model) throw new Error('Configure endpoint, key and model first.');
  msgs = compactHistory(msgs);
  if (c.kind === 'anthropic') return callAnthropic(c, msgs, useTools);
  const body = { model: c.model, messages: msgs, temperature: 0.2 };
  // final round: no tools at all (Gemini's OpenAI endpoint ignores tool_choice "none" and keeps calling tools)
  if (useTools && useTools !== 'none') { body.tools = toolSpecs(); body.tool_choice = 'auto'; }
  const r = await fetch(c.url + '/chat/completions', { method: 'POST', headers: { 'Content-Type': 'application/json', ...authHeaders(c) }, body: JSON.stringify(body), signal: aborter && aborter.signal });
  if (!r.ok) { const t = await r.text(); const err = new Error(`HTTP ${r.status}: ${t.slice(0, 300)}`); err.status = r.status; err.body = t; throw err; }
  return (await r.json()).choices[0].message;
}
/* fallback for models without function calling: ask for a JSON tool call in the text */
function fallbackSystem() { return SYSTEM + `\n\nTOOLS (no function-calling API available): to call a tool, answer ONLY with one JSON object {"tool": "<name>", "arguments": {...}} and nothing else; you will receive the result and can call again. Tools: ` + Object.entries(TOOLS).map(([n, t]) => `${n}${JSON.stringify(t.parameters.properties)}: ${t.description}`).join('\n'); }
function parseFallbackCall(text) { const m = text.match(/\{[\s\S]*\}/); if (!m) return null; try { const j = JSON.parse(m[0]); return j.tool && TOOLS[j.tool] ? j : null; } catch (e) { return null; } }

/* one line per tool result, readable without opening the details */
function resultSummary(name, args, result) {
  if (result.error) return 'error: ' + result.error;
  if (name === 'get_consilium') return `${result.print_short || result.print || ''} n ${result.n || ''}: ${(result.title || '').slice(0, 60)}`;
  if (name === 'list_keywords') return `${(result.keywords || []).length} keywords` + ((result.keywords || []).length ? ': ' + result.keywords.slice(0, 4).map(k => k.keyword).join(', ') + (result.keywords.length > 4 ? ', …' : '') : '');
  if (result.total != null) return `${result.total} consilia` + (result.hits && result.hits.length ? ': ' + result.hits.slice(0, 3).map(h => `${h.print_short || h.print} n ${h.n}`).join(', ') + (result.total > 3 ? ', …' : '') : '');
  return 'ok';
}
function argsText(args) { return Object.entries(args).filter(([k]) => k !== 'with_translation').map(([k, v]) => `${k}: ${typeof v === 'string' ? '"' + v + '"' : v}`).join(', '); }
function setProgress(el, text) { el.querySelector('.progress').textContent = text; }

$('form').addEventListener('submit', async e => {
  e.preventDefault(); if (busy) return;
  const q = $('q').value.trim(); if (!q) return;
  $('q').value = ''; busy = true; $('send').disabled = true; $('stop').classList.remove('hidden'); aborter = new AbortController();
  addMsg('you', q);
  if (!messages.length) messages.push({ role: 'system', content: toolsSupported ? SYSTEM : fallbackSystem() });
  messages.push({ role: 'user', content: q });
  const thinking = addMsg('assistant', '');
  thinking.querySelector('.body').innerHTML = '<p class="progress">Asking the model…</p>';
  const t0 = performance.now(); let nCalls = 0; const done = new Map();   // name+args -> result text, against loops
  const elapsed = () => `${((performance.now() - t0) / 1000).toFixed(0)} s`;
  try {
    for (let round = 0; ; round++) {
      let msg;
      const last = round >= MAX_ROUNDS;      // round limit: let the model answer with what it has
      if (last) messages.push({ role: 'user', content: 'Stop searching now and answer the question with the evidence gathered so far, citing the consilia you have read. Say what you could not verify.' });
      else if (round === NUDGE_ROUND || round === 2 * NUDGE_ROUND) messages.push({ role: 'user', content: `You have made ${nCalls} tool calls. Unless one more specific lookup is indispensable, answer the question now from what you have gathered, with citations.` });
      setProgress(thinking, `Round ${round + 1}, ${nCalls} tool call${nCalls === 1 ? '' : 's'}, ${elapsed()}: ${last ? 'asking for the final answer' : 'the model is deciding what to do next'}…`);
      try { msg = await callLLM(messages, last ? (toolsSupported ? 'none' : false) : toolsSupported); }
      catch (err) {
        if (toolsSupported && err.status === 400 && /tool/i.test(err.body || '')) { toolsSupported = false; messages[0] = { role: 'system', content: fallbackSystem() }; addMsg('system', 'This endpoint has no function calling; switching to the text protocol for tool calls.'); msg = await callLLM(messages, false); }
        else throw err;
      }
      const calls = last ? [] : (msg.tool_calls || []).map(tc => ({ id: tc.id, name: tc.function.name, args: tc.function.arguments }));
      if (!calls.length && !toolsSupported && !last) { const fb = parseFallbackCall(msg.content || ''); if (fb) calls.push({ id: null, name: fb.tool, args: JSON.stringify(fb.arguments || {}) }); }
      if (!calls.length) {
        let text = msg.content || '';
        if (!text.trim() && last && (msg.tool_calls || []).length) {   // the model still wanted tools: one more plain request without them
          messages.push({ role: 'user', content: 'No more tool calls are possible. Write the answer now as text.' });
          try { const m2 = await callLLM(messages, 'none'); text = m2.content || ''; } catch (e) { text = ''; }
        }
        messages.push({ role: 'assistant', content: text });
        thinking.querySelector('.body').innerHTML = md(text || 'The model ended without a text answer. Try again with a more specific question, another model, or press New.') + `<p class="progress">${nCalls} tool call${nCalls === 1 ? '' : 's'} in ${round + 1} round${round ? 's' : ''}, ${elapsed()}.</p>`;
        break;
      }
      messages.push(toolsSupported ? msg : { role: 'assistant', content: msg.content });
      // what the model said while deciding to call tools (its plan), shown muted
      if (msg.content && msg.content.trim() && !parseFallbackCall(msg.content)) { const note = addMsg('assistant', msg.content); note.classList.add('note'); thinking.remove(); $('log').appendChild(thinking); }
      for (const call of calls) {
        let args = {}; try { args = JSON.parse(call.args || '{}'); } catch (e) {}
        nCalls++;
        setProgress(thinking, `Round ${round + 1}, tool call ${nCalls}, ${elapsed()}: ${call.name}(${argsText(args)})…`);
        const t = TOOLS[call.name]; let result, txt;
        const sig = call.name + JSON.stringify(args);
        if (done.has(sig)) {   // identical call repeated: do not run it again, tell the model to answer
          txt = JSON.stringify({ note: 'This exact call was already made in this conversation and its result is above. Do not repeat calls; answer the question from the evidence gathered.', earlier_result_start: done.get(sig).slice(0, 1200) }); result = JSON.parse(txt);
        } else {
          try { result = t ? await t.run(args) : { error: 'unknown tool' }; } catch (err) { result = { error: String(err.message || err) }; }
          txt = JSON.stringify(result); done.set(sig, txt);
        }
        addMsg('tool', `<details><summary><b>${esc(call.name)}</b>(${esc(argsText(args))}) <span class="res">${esc(resultSummary(call.name, args, result))}</span></summary><pre style="white-space:pre-wrap;font-size:.68rem;max-height:14rem;overflow:auto">${esc(txt.slice(0, 4000))}</pre></details>`);
        thinking.remove(); $('log').appendChild(thinking);
        if (toolsSupported) messages.push({ role: 'tool', tool_call_id: call.id, content: txt.slice(0, 60000) });
        else messages.push({ role: 'user', content: `TOOL RESULT ${call.name}: ${txt.slice(0, 60000)}` });
      }
    }
  } catch (err) {
    thinking.querySelector('.body').innerHTML = `<p class="error">${esc(netError(err))}</p><p class="progress">${nCalls} tool call${nCalls === 1 ? '' : 's'}, ${elapsed()}. The conversation so far is kept; ask again or press New.</p>`;
    // keep the message list consistent: drop a trailing user turn that got no answer
    if (messages.length && messages[messages.length - 1].role === 'user') messages.pop();
  }
  busy = false; $('send').disabled = false; $('stop').classList.add('hidden'); aborter = null; thinking.scrollIntoView({ block: 'end' });
});
$('stop').addEventListener('click', () => { if (aborter) aborter.abort(); });
$('reset').addEventListener('click', () => { messages = []; $('log').innerHTML = ''; addMsg('system', 'New conversation.'); });
$('q').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('form').requestSubmit(); } });
$('q').addEventListener('input', e => { e.target.style.height = 'auto'; e.target.style.height = Math.min(160, e.target.scrollHeight) + 'px'; });
