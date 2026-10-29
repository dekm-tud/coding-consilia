/* Search: Full text (MiniSearch in a worker), Semantic (query model in the browser or via an
   OpenAI-compatible embeddings API), Similar (cosine), Subjects (annotation layers / machine keywords).
   Data: site-data/ (index + text shards + models.json + annotations/). Vectors: one row per body/casus
   section (Latin and English); a document scores by the max over its rows. No demo fallback: errors
   are shown, never invented results. URL state: ?mode=&q=&scope=&model=&lang=&similar=&keyword=&layer= */

const $ = id => document.getElementById(id);
function snippet(text, re, len = 190) {
  if (!text) return '';
  if (re) { re.lastIndex = 0; const m = re.exec(text); if (m) { const s = Math.max(0, m.index - 70); return (s > 0 ? '…' : '') + text.slice(s, s + len) + (s + len < text.length ? '…' : ''); } }
  return text.slice(0, len) + (text.length > len ? '…' : '');
}
const CLUSTER_PALETTE = ['#b23a2c', '#3f6b5e', '#9a7a2e', '#56688c', '#7c4a6b', '#6b853f', '#b06a35', '#4a7d86', '#86564a', '#5a5f7a', '#7d8a4a', '#9c4f57'];
const volColorMap = {};
function volColor(v) { if (!(v in volColorMap)) volColorMap[v] = CLUSTER_PALETTE[Object.keys(volColorMap).length % CLUSTER_PALETTE.length]; return volColorMap[v]; }
function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
const LAT_STOP = new Set('an de et in ad per cum non sed quia quod vel aut est si ut ex pro qui quae dum nam iam nec etiam tamen autem vero enim ergo ideo item idem igitur quando quomodo quoniam quam inter post ante apud sine sive atque neque quin qua quo ubi unde nisi possit potest possint posse debet debent esse fuit sunt fuerit sit sint erat erit habet habere dici dictum dicta dicitur fieri facta factum facti facere tenetur quid quis aliquid aliquis ille illa illud ipse ipsa ipsum hoc haec hic tale talis eadem propter contra super ultra secundum coram circa consilivm consilium consilii the of and to is are that from this with for not but when by be'.split(' '));

/* ── state ─────────────────────────────────────────────────────────────── */
let site = null, consilia = {}, allItems = [], models = [];
let mode = 'keyword', scope = 'all', lang = 'both', langTouched = false;
let searchWorker = null, shardsIndexed = 0, shardsTotal = 0, workerSeq = 0; const workerWaiters = {};
let lastResults = [], searchTimer = null, searchSeq = 0, simRefId = null;
let E = null, meta = null, idToRows = {}, selectedTag = null, embeddingsLoading = null; const embCache = {}, modelState = {};
let embedWorker = null, embedWorkerTag = null, embedSeq = 0, embedRuntime = ''; const embedWaiters = {};
let docEmbeddings = null, docIds = null, docIdToIdx = {}, pcaCoords = null, clusterAssignments = null, clusterLabels = null, pcaColorMode = 'cluster';
// subjects: layers = {machine, subjects?, body?}; each: {label, terms:{term:{label, ids:[...], ev?:{}}}}
let layers = {}, layerKey = 'machine', currentTerm = null, refIndex = null;

/* ── boot ──────────────────────────────────────────────────────────────── */
loadSiteData().then(async s => {
  site = s;
  const mods = await fetchJSON('site-data/models.json');
  const subj = site.annotations.subjects, body = site.annotations.body;   // optional layers, loaded with the site data when published
  models = mods.models || mods;
  site.index.forEach(e => { e.summary = e.argument_short; consilia[e.id] = e; });
  allItems = site.index;
  buildLayers(subj, body);
  const bootSearch = location.search;      // syncUrl() rewrites the URL during boot; the citation layers arrive later
  loadReferencesIndex(s).then(ix => {
    if (!ix || !ix.jurists) return;
    refIndex = ix; buildLayers(...annData);
    const sp0 = new URLSearchParams(bootSearch);
    if (sp0.get('layer') && layers[sp0.get('layer')] && layerKey !== sp0.get('layer')) { layerKey = sp0.get('layer'); if (sp0.get('keyword')) currentTerm = sp0.get('keyword'); }
    if (mode === 'index') { renderLayerSeg(); renderCloud(); if (currentTerm) { syncUrl(); runSearch(); } }
  });
  buildScope();
  const sp = new URLSearchParams(location.search);
  selectedTag = (models.find(m => m.tag === sp.get('model')) || models.find(m => m.default) || models[0]).tag;
  if (sp.get('lang')) { lang = sp.get('lang'); $('lang-select').value = lang; langTouched = true; }
  if (sp.get('scope')) { scope = sp.get('scope'); $('scope-select').value = scope; }
  if (sp.get('layer') && layers[sp.get('layer')]) layerKey = sp.get('layer');
  startFulltextIndex();
  initModelPicker();
  const simArg = sp.get('similar'), kw = sp.get('keyword'), q = sp.get('q');
  if (simArg) { setMode('similar', true); if (consilia[simArg]) setSimRef(simArg); }
  else if (kw) { setMode('index', true); selectTerm(kw); }
  else { setMode(sp.get('mode') || 'keyword', true); if (q) { $('search-input').value = q; runSearch(); } }
}).catch(e => { $('empty-state').textContent = 'Could not load corpus: ' + e; $('empty-state').classList.add('error'); });

function buildScope() {
  const sel = $('scope-select'); sel.innerHTML = '<option value="all">All volumes</option>';
  Object.entries(site.authors).forEach(([viaf, a]) => {
    const og = document.createElement('optgroup'); og.label = a.name || viaf;
    const all = document.createElement('option'); all.value = 'author:' + viaf; all.textContent = `All by ${a.name || viaf}`; og.appendChild(all);
    (a.prints || []).slice().sort().forEach(v => { const o = document.createElement('option'); o.value = 'vol:' + v; o.textContent = printLabel(v, site.authors).replace(`${a.name}, `, ''); og.appendChild(o); });
    sel.appendChild(og);
  });
  sel.addEventListener('change', () => { scope = sel.value; syncUrl(); runSearch(); });
  const sv = $('sim-vol'); sv.innerHTML = '<option value="">All prints</option>';
  [...new Set(allItems.map(c => c.volume))].sort().forEach(v => { const o = document.createElement('option'); o.value = v; o.textContent = printLabel(v, site.authors); sv.appendChild(o); });
  sv.addEventListener('change', () => { $('sim-input').focus(); renderSugg(); });
}
function inScope(c) {
  if (scope === 'all') return true;
  if (scope.startsWith('author:')) return c.author_viaf === scope.slice(7);
  if (scope.startsWith('vol:')) return c.volume === scope.slice(4);
  return true;
}
function scopedItems() { return scope === 'all' ? allItems : allItems.filter(inScope); }
function syncUrl() {
  const u = new URL(location.href); u.search = '';
  u.searchParams.set('mode', mode);
  const q = $('search-input').value.trim(); if (q && (mode === 'keyword' || mode === 'semantic')) u.searchParams.set('q', q);
  if (scope !== 'all') u.searchParams.set('scope', scope);
  if (mode === 'semantic' || mode === 'similar') { u.searchParams.set('model', selectedTag); if (lang !== 'both') u.searchParams.set('lang', lang); }
  if (mode === 'similar' && simRefId) u.searchParams.set('similar', simRefId);
  if (mode === 'index') { if (layerKey !== 'machine') u.searchParams.set('layer', layerKey); if (currentTerm) u.searchParams.set('keyword', currentTerm); }
  history.replaceState(null, '', u);
}

/* ── full text (worker) ────────────────────────────────────────────────── */
function startFulltextIndex() {
  searchWorker = new Worker('assets/search-worker.js');
  if (hasLemmas(site)) { loadLemmas(); searchWorker.postMessage({ type: 'lemmas', url: '../site-data/lemmas.json' }); }   // main thread: highlighting
  searchWorker.onmessage = e => { const m = e.data; const w = workerWaiters[m.seq]; if ((m.type === 'results' || m.type === 'error') && w) { delete workerWaiters[m.seq]; m.type === 'error' ? w.reject(new Error(m.message)) : w.resolve(m.results); } };
  searchWorker.onerror = e => setIndexStatus('Full-text index failed: ' + e.message);
  const vols = [...new Set(allItems.map(c => c.volume))].sort();
  shardsTotal = vols.length; shardsIndexed = 0; setIndexStatus();
  vols.reduce((p, v) => p.then(async () => {
    const text = await getShard('text', v);
    const docs = [];
    allItems.filter(c => c.volume === v).forEach(c => {
      const t = text[c.id] || {};
      c.argument = t.argument || []; c.casus = t.casus || []; c.body = t.body || []; c.summary = summaryOf(c.argument) || c.summary;
      docs.push({ id: c.id, title: c.title || '', summary: c.summary || '', casus: c.casus.join(' '), body: c.body.join(' ') });
    });
    searchWorker.postMessage({ type: 'add', docs }); shardsIndexed++; setIndexStatus();
    if (mode === 'keyword' && $('search-input').value.trim()) runSearch();
  }), Promise.resolve()).catch(err => setIndexStatus('Could not load the text of the prints: ' + err.message));
}
function setIndexStatus(msg) { $('index-progress').textContent = msg || (shardsIndexed < shardsTotal ? `Indexing full text… ${shardsIndexed}/${shardsTotal} prints` : ''); }
function workerSearch(q) { return new Promise((resolve, reject) => { const seq = ++workerSeq; workerWaiters[seq] = { resolve, reject }; searchWorker.postMessage({ type: 'search', q, seq }); }); }

/* ── vectors ───────────────────────────────────────────────────────────── */
function setModelStatus(msg, cls = '') { const el = $('model-status'); el.textContent = msg; el.className = 'status ' + cls; }
async function fetchBuffer(url, label) {
  const r = await fetch(url); if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const total = Number(r.headers.get('Content-Length')) || 0;
  if (!r.body || !total) return r.arrayBuffer();
  const reader = r.body.getReader(), buf = new Uint8Array(total); let got = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; buf.set(value, got); got += value.length; setModelStatus(`Loading ${label}… ${(got / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`, 'busy'); }
  return buf.buffer.slice(0, got);
}
async function cachedBuffer(url, revision, label) {
  const key = new Request(`${url}?rev=${encodeURIComponent(revision || '')}`);
  try { const c = await caches.open('consilia-embeddings'); const hit = await c.match(key); if (hit) return await hit.arrayBuffer(); } catch (e) {}
  const buf = await fetchBuffer(url, label);
  try { const c = await caches.open('consilia-embeddings'); await c.put(key, new Response(buf.slice(0))); } catch (e) {}
  return buf;
}
let F16 = null;
function f16ToF32(buf) {
  if (!F16) { F16 = new Float32Array(65536); for (let h = 0; h < 65536; h++) { const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff; F16[h] = e === 0 ? s * Math.pow(2, -14) * (m / 1024) : e === 31 ? (m ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + m / 1024); } }
  const u = new Uint16Array(buf), out = new Float32Array(u.length); for (let i = 0; i < u.length; i++) out[i] = F16[u[i]]; return out;
}
function modelCfg(tag) { return models.find(m => m.tag === (tag || selectedTag)); }
async function ensureEmbeddings() {
  if (E) return;
  const tag = selectedTag;
  if (!embeddingsLoading) embeddingsLoading = (async () => {
    const cfg = modelCfg(tag); setModelStatus(`Loading ${cfg.label} vectors…`, 'busy');
    try {
      let lastErr = null, chunks = null, buf = null;
      for (const base of cfg.embeddings) {
        try {
          chunks = await fetchJSON(base + 'chunks.json');
          if (chunks.data_revision !== site.manifest.revision) throw new Error(`vectors are for data revision ${chunks.data_revision}, the site is ${site.manifest.revision}`);
          buf = await cachedBuffer(base + 'chunks.f16.bin', `${chunks.revision}-${chunks.data_revision}`, 'embeddings'); break;
        } catch (e) { lastErr = e; chunks = null; buf = null; }
      }
      if (!buf) throw lastErr || new Error('no embedding source');
      const vec = f16ToF32(buf);
      if (vec.length !== chunks.n * chunks.dims) throw new Error('embedding file size does not match chunks.json');
      if (chunks.dims !== cfg.dims) throw new Error(`vectors have ${chunks.dims} dimensions, models.json says ${cfg.dims}`);
      if (chunks.doc.some(d => d < 0 || d >= site.index.length)) throw new Error('embedding rows refer to unknown consilia');
      const m = { model: chunks.model, dims: chunks.dims, n: chunks.n, ids: chunks.doc.map(d => site.index[d].id), chunk: chunks.chunk, lang: chunks.lang, kind: chunks.kind };
      const rows = {}; m.ids.forEach((id, i) => { (rows[id] = rows[id] || []).push(i); });
      embCache[tag] = { E: vec, meta: m, idToRows: rows };
      if (tag === selectedTag) activateVectors(tag);
      setModelStatus(`Passage vectors loaded (${m.n.toLocaleString("en")} passages, ${cfg.label}).`, 'ready');
    } catch (e) {
      setModelStatus(`${cfg.label} is not available right now (${e.message}).`);
      throw new Error(`${cfg.label} could not be loaded (${e.message}).`);
    } finally { embeddingsLoading = null; }
  })();
  return embeddingsLoading;
}
function activateVectors(tag) {
  const c = embCache[tag]; E = c ? c.E : null; meta = c ? c.meta : null; idToRows = c ? c.idToRows : {};
  docEmbeddings = null; docIds = null; docIdToIdx = {}; pcaCoords = null; clusterAssignments = null; clusterLabels = null;
  if (c) buildDocEmbeddings();
}
function buildDocEmbeddings() {
  const D = meta.dims; docIds = Object.keys(idToRows); docIdToIdx = {}; docEmbeddings = new Float32Array(docIds.length * D);
  docIds.forEach((id, di) => { const rows = idToRows[id], off = di * D; for (const ri of rows) { const roff = ri * D; for (let d = 0; d < D; d++) docEmbeddings[off + d] += E[roff + d]; } for (let d = 0; d < D; d++) docEmbeddings[off + d] /= rows.length; docIdToIdx[id] = di; });
}
function rowAllowed(ri) { return lang === 'both' || (lang === 'la' ? meta.lang[ri] === 0 : meta.lang[ri] === 1); }

/* ── query encoders: browser model (worker) or embeddings API ──────────── */
const ONNX_FILE = { q8: 'model_quantized.onnx', int8: 'model_int8.onnx', uint8: 'model_uint8.onnx', fp16: 'model_fp16.onnx', fp32: 'model.onnx', q4: 'model_q4.onnx', q4f16: 'model_q4f16.onnx', bnb4: 'model_bnb4.onnx' };
let webgpuOK = false;
async function detectWebGPU() { try { webgpuOK = !!(navigator.gpu && await navigator.gpu.requestAdapter()); } catch (e) { webgpuOK = false; } }
function runtimeFor(cfg) {
  if (!cfg.dtype) return null;
  if (webgpuOK && cfg.dtype.webgpu) return { device: 'webgpu', dtype: cfg.dtype.webgpu, mb: cfg.download_mb.webgpu };
  if (cfg.dtype.wasm) return { device: 'wasm', dtype: cfg.dtype.wasm, mb: cfg.download_mb.wasm };
  return null;
}
async function isCached(cfg) {
  const rt = runtimeFor(cfg); if (!rt || !cfg.onnx_ready) return false;
  try { const c = await caches.open('transformers-cache'); return !!(await c.match(`https://huggingface.co/${cfg.onnx_repo}/resolve/${cfg.onnx_revision || 'main'}/onnx/${ONNX_FILE[rt.dtype]}`)); } catch (e) { return false; }
}
function apiCfg(cfg) { return cfg.query_api || null; }       // {base_url, model, label} — key lives in localStorage only
function apiKey(cfg) { try { return localStorage.getItem('consilia-api-key:' + new URL(apiCfg(cfg).base_url).host) || ''; } catch (e) { return ''; } }
async function initModelPicker() {
  await detectWebGPU();
  const cur = modelCfg(); if (cur && cur.requires_webgpu && !webgpuOK) { selectedTag = (models.find(m => m.default) || models[0]).tag; }
  renderModelPicker();
}
async function renderModelPicker() {
  const list = $('model-list');
  const rows = await Promise.all(models.filter(m => !(m.requires_webgpu && !webgpuOK)).map(async cfg => {
    const rt = runtimeFor(cfg), cached = await isCached(cfg), st = modelState[cfg.tag], api = apiCfg(cfg);
    const state = st === 'ready' ? '<span class="badge ok">loaded</span>' : cached ? '<span class="badge ok">cached</span>' : api ? '<span class="badge">API</span>' : !cfg.onnx_ready ? '<span class="badge warn">similar only</span>' : '';
    const size = api ? `via ${esc(api.label || api.base_url)}` : rt ? `${rt.mb >= 1000 ? (rt.mb / 1000).toFixed(1) + ' GB' : rt.mb + ' MB'}${rt.device === 'webgpu' ? ', GPU' : ''}` : 'needs WebGPU';
    return `<button class="model" role="radio" aria-checked="${cfg.tag === selectedTag}" data-tag="${esc(cfg.tag)}"><span class="mn">${esc(cfg.label)}${state}</span><span class="mm"></span><span class="mm">${size}</span><span class="mt">${esc(cfg.description || '')}</span></button>`;
  }));
  list.innerHTML = rows.join('');
  const sel = $('model-select'); sel.innerHTML = '';
  models.filter(m => !(m.requires_webgpu && !webgpuOK)).forEach(cfg => {
    const rt = runtimeFor(cfg), api = apiCfg(cfg), o = document.createElement('option'); o.value = cfg.tag;
    o.textContent = cfg.label + (api ? ' (API)' : rt ? ` (${rt.mb >= 1000 ? (rt.mb / 1000).toFixed(1) + ' GB' : rt.mb + ' MB'})` : '');
    o.selected = cfg.tag === selectedTag; sel.appendChild(o);
  });
  updateModelActions();
}
$('model-list').addEventListener('click', e => { const card = e.target.closest('.model'); if (card) selectModel(card.dataset.tag); });
$('model-select').addEventListener('change', e => selectModel(e.target.value));
function selectModel(tag) {
  if (tag === selectedTag) return;
  selectedTag = tag; embeddingsLoading = null; activateVectors(embCache[tag] ? tag : null); syncUrl(); renderModelPicker();
  if (mode === 'similar' && simRefId) runSearch();
}
function updateModelActions() {
  const cfg = modelCfg(); if (!cfg) return;
  const btn = $('btn-load-model'), rt = runtimeFor(cfg), st = modelState[cfg.tag], api = apiCfg(cfg), busy = $('model-status').classList.contains('busy');
  if (mode === 'similar') { btn.classList.add('hidden'); if (!busy) setModelStatus(''); return; }
  btn.classList.remove('hidden');
  if (api) {
    const has = !!apiKey(cfg);
    btn.disabled = false; btn.textContent = has ? 'Change API key' : `Enter API key (${api.label || 'embeddings API'})`;
    if (!busy) setModelStatus(has ? `Queries are sent to ${api.label || api.base_url} with your key (kept in this browser).` : 'Queries are embedded by this provider with your key.');
    return;
  }
  btn.disabled = !cfg.onnx_ready || !rt || st === 'loading' || st === 'ready';
  btn.textContent = st === 'ready' ? 'Loaded' : st === 'loading' ? 'Loading…' : !cfg.onnx_ready ? 'Not available in the browser' : 'Load model';
  if (!busy) setModelStatus(!cfg.onnx_ready ? 'No browser build of this model; it works for Similar only.' : st === 'ready' ? `Runs in your browser (${embedRuntime}).` : 'Downloaded once, then kept by your browser.');
}
$('btn-load-model').addEventListener('click', async () => {
  const cfg = modelCfg(); if (!cfg) return;
  const api = apiCfg(cfg);
  if (api) { const k = prompt(`API key for ${api.label || api.base_url} (kept in this browser's localStorage only):`, apiKey(cfg)); if (k !== null) { try { localStorage.setItem('consilia-api-key:' + new URL(api.base_url).host, k.trim()); } catch (e) {} } updateModelActions(); return; }
  if (!cfg.onnx_ready || modelState[cfg.tag] === 'ready' || modelState[cfg.tag] === 'loading') return;
  modelState[cfg.tag] = 'loading'; updateModelActions();
  try { await loadQueryModel(cfg); modelState[cfg.tag] = 'ready'; setModelStatus(`${cfg.label} ready (${embedRuntime})`, 'ready'); }
  catch (e) { modelState[cfg.tag] = 'error'; setModelStatus('Model could not be loaded: ' + e.message); }
  renderModelPicker();
  if (modelState[cfg.tag] === 'ready' && mode === 'semantic' && $('search-input').value.trim()) runSearch();
});
function loadQueryModel(cfg) {
  return new Promise((resolve, reject) => {
    if (embedWorker) embedWorker.terminate();
    Object.keys(modelState).forEach(t => { if (t !== cfg.tag && modelState[t] === 'ready') modelState[t] = 'idle'; });
    embedWorker = new Worker('assets/embed-worker.js', { type: 'module' }); embedWorkerTag = cfg.tag;
    const prog = {};
    embedWorker.onmessage = e => {
      const m = e.data;
      if (m.type === 'progress') { prog[m.file] = [m.loaded, m.total]; const big = Object.values(prog).sort((a, b) => b[1] - a[1])[0]; setModelStatus(`Downloading ${cfg.label}… ${(big[0] / 1e6).toFixed(0)} / ${(big[1] / 1e6).toFixed(0)} MB`, 'busy'); }
      else if (m.type === 'ready') { embedRuntime = `${m.device.toUpperCase()} ${m.dtype}`; resolve(); }
      else if (m.type === 'embedding' || m.type === 'error') { const w = embedWaiters[m.id]; if (m.id === undefined || !w) { if (m.type === 'error') reject(new Error(m.message)); return; } delete embedWaiters[m.id]; m.type === 'error' ? w.reject(new Error(m.message)) : w.resolve(m.vec); }
    };
    embedWorker.onerror = e => reject(new Error(e.message || 'worker failed'));
    embedWorker.postMessage({ type: 'load', cfg: { onnx_repo: cfg.onnx_repo, onnx_revision: cfg.onnx_revision, dtype: cfg.dtype, pooling: cfg.pooling, normalize: cfg.normalize, query_prefix: cfg.query_prefix } });
  });
}
async function embedQuery(text) {
  const cfg = modelCfg(), api = apiCfg(cfg);
  if (api) {
    const key = apiKey(cfg); if (!key) throw new Error(`Enter an API key (button above).`);
    const r = await fetch(api.base_url.replace(/\/$/, '') + '/embeddings', { method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + key },
      body: JSON.stringify({ model: api.model, input: (cfg.query_prefix || '') + text }) });
    if (!r.ok) throw new Error(`embeddings API: HTTP ${r.status}`);
    const j = await r.json(); const v = Float32Array.from(j.data[0].embedding);
    if (cfg.normalize !== false) { let n = 0; for (const x of v) n += x * x; n = Math.sqrt(n) || 1; for (let i = 0; i < v.length; i++) v[i] /= n; }
    return v;
  }
  if (modelState[cfg.tag] !== 'ready' || embedWorkerTag !== cfg.tag) throw new Error(`Load the ${cfg.label} model first (button above).`);
  return new Promise((resolve, reject) => { const id = ++embedSeq; embedWaiters[id] = { resolve, reject }; embedWorker.postMessage({ type: 'embed', id, text }); });
}

/* ── modes ─────────────────────────────────────────────────────────────── */
const HINTS = {
  keyword: 'Searches the Latin text of all consilia. i/j, u/v and ae/e count as equal; word beginnings match.',
  semantic: 'Ranks passages by meaning rather than wording. Ask in Latin, English or German. Needs a model in your browser (one download).',
  similar: 'Lists the consilia whose passages are closest to a chosen one, by the stored passage vectors.',
  index: 'Browse the consilia by annotation layer.',
};
const LAYER_HINTS = {
  machine: 'Keywords assigned to each consilium by a language model (English).',
  subjects: 'Controlled vocabulary of legal subjects, assigned by a language model; each is central or secondary to a consilium.',
  references: 'Works cited in the text, identified by rule from the printed citations.',
  jurists: 'Jurists cited by name in the text.',
  body: 'Passages on body and sexuality, coded by a language model with a codebook; the evidence is shown.',
};
const EXAMPLES = {
  keyword: ['dos', 'statutum', 'usurae', 'testamentum'],
  semantic: ['Can a father disinherit a legitimate son?', 'Zinsverbot und Kaufvertrag', 'de iurisdictione episcopi'],
};
function showEmpty() {
  const el = $('empty-state');
  let text = '', eg = '';
  if (mode === 'keyword') { text = 'Type a Latin word. Other forms of the same word are found as well (lemma list from the <a href="https://huggingface.co/mschonhardt/latin-lemmatizer" target="_blank" rel="noopener">Latin lemmatizer</a>); a word in quotation marks only in the form given.'; eg = EXAMPLES.keyword.map(x => `<button data-q="${esc(x)}">${esc(x)}</button>`).join(''); }
  else if (mode === 'semantic') { text = 'Describe what you are looking for.'; eg = EXAMPLES.semantic.map(x => `<button data-q="${esc(x)}">${esc(x)}</button>`).join(''); }
  else if (mode === 'similar') { text = 'Enter a consilium above, or press “similar” on any result.'; }
  else { text = 'Choose a term above.'; }
  el.innerHTML = text + (eg ? `<div class="eg"><span class="lbl">Try</span>${eg}</div>` : '');
  showResults(false);
}
$('empty-state').addEventListener('click', e => { const b = e.target.closest('button[data-q]'); if (!b) return; $('search-input').value = b.dataset.q; syncUrl(); runSearch(); });
$('mode-tabs').addEventListener('click', e => { const b = e.target.closest('button[data-mode]'); if (b) setMode(b.dataset.mode); });
function setMode(m, silent) {
  mode = m;
  if (!langTouched) { lang = m === 'similar' ? 'la' : 'both'; $('lang-select').value = lang; }   // Similar defaults to Latin↔Latin (translator style would otherwise dominate)
  document.querySelectorAll('#mode-tabs button').forEach(b => b.classList.toggle('active', b.dataset.mode === m));
  $('mode-hint').textContent = m === 'index' ? (LAYER_HINTS[layerKey] || HINTS.index) : HINTS[m];
  $('sem-controls').classList.toggle('show', m === 'semantic' || m === 'similar');
  $('index-controls').classList.toggle('show', m === 'index');
  $('fields-block').style.display = m === 'keyword' ? '' : 'none';
  $('pca-block').style.display = (m === 'semantic' || m === 'similar') ? '' : 'none';
  $('q-box').classList.toggle('hidden', m === 'similar' || m === 'index');
  $('sim-box').classList.toggle('hidden', m !== 'similar' || simRefId !== null);
  $('sim-vol').classList.toggle('hidden', m !== 'similar' || simRefId !== null);
  $('sim-chip-row').classList.toggle('hidden', m !== 'similar' || simRefId === null);
  $('filter-box').classList.toggle('hidden', m !== 'index');
  updateModelActions();
  if (m === 'index') { $('sort-select').value = 'n'; renderLayerSeg(); renderCloud(); }
  if (!silent) { syncUrl(); runSearch(); } else if (!lastResults.length) showEmpty();
}
$('lang-select').addEventListener('change', e => { lang = e.target.value; langTouched = true; syncUrl(); runSearch(); });
$('topk-input').addEventListener('change', runSearch);
$('sort-select').addEventListener('change', () => { shown = PAGE; runSearch(); });
const mq = matchMedia('(max-width: 880px)'); const sideOpen = () => { $('side-more').open = !mq.matches; }; sideOpen(); mq.addEventListener('change', sideOpen);
$('search-input').addEventListener('input', () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { syncUrl(); runSearch(); }, 220); });
$('search-clear').addEventListener('click', () => { $('search-input').value = ''; syncUrl(); runSearch(); });
function showResults(on, msg) { $('results-layout').style.display = on ? 'grid' : 'none'; $('empty-state').style.display = on ? 'none' : 'block'; if (!on && msg) { $('empty-state').textContent = msg; $('empty-state').classList.add('error'); } else $('empty-state').classList.remove('error'); }
function topK() { return Math.max(1, parseInt($('topk-input').value || '20', 10)); }

async function runSearch() {
  const q = $('search-input').value.trim();
  $('search-clear').style.display = q ? 'flex' : 'none';
  const seq = ++searchSeq;
  if (mode === 'keyword') {
    if (!q) { showEmpty(); lastResults = []; return; }
    const re = buildHighlightRe(q);
    let res; try { res = await workerSearch(q); } catch (e) { showResults(false, e.message); return; }
    if (seq !== searchSeq) return;
    lastResults = res.filter(r => consilia[r.id] && inScope(consilia[r.id]));
    showResults(true); $('count-head').textContent = 'Matches';
    shown = PAGE; renderKeyword(re); renderStats(scopedItems().length); summary(`for “${q}”`);
    return;
  }
  if (mode === 'index') {
    const term = currentTerm && layers[layerKey] && layers[layerKey].terms[currentTerm];
    if (!term) { showEmpty(); lastResults = []; return; }
    lastResults = term.ids.filter(id => consilia[id] && inScope(consilia[id])).map(id => ({ id, score: 1 }));
    showResults(true); $('count-head').textContent = 'Consilia';
    shown = PAGE; renderIndexResults(term); renderStats(scopedItems().length); summary(`with “${term.label}”`);
    return;
  }
  if (mode === 'semantic' && !q) { showEmpty(); lastResults = []; return; }
  if (mode === 'similar' && simRefId === null) { showEmpty(); lastResults = []; return; }
  showResults(true); $('results-list').innerHTML = '<div class="empty">Loading…</div>'; $('result-summary').textContent = '';
  try {
    await ensureEmbeddings(); if (seq !== searchSeq) return;
    if (mode === 'semantic') await runSemantic(q, seq); else await runSimilar();
  } catch (e) { $('results-list').innerHTML = `<div class="empty error">${esc(e.message)}</div>`; setModelStatus(e.message); }
}
function summary(what) { const n = lastResults.length; $('result-summary').textContent = `${n} ${n === 1 ? 'consilium' : 'consilia'} ${what}`; }
const PAGE = 30; let shown = PAGE, rerender = null;
function paginate(list, sorted, draw) {
  rerender = () => paginate(list, sorted, draw);
  list.innerHTML = '';
  sorted.slice(0, shown).forEach((r, i) => list.appendChild(draw(r, i)));
  const rest = sorted.length - shown;
  if (rest > 0) { const b = document.createElement('button'); b.className = 'btn more'; b.textContent = `Show ${Math.min(PAGE, rest)} more (${rest} remaining)`; b.addEventListener('click', () => { shown += PAGE; rerender(); }); list.appendChild(b); }
}

async function runSemantic(q, seq) {
  const cfg = modelCfg(); setModelStatus(`Embedding the query with ${cfg.label}…`, 'busy');
  const qv = await embedQuery(q); if (seq !== searchSeq) return;
  if (qv.length !== meta.dims) throw new Error(`query vector has ${qv.length} dimensions, the corpus vectors ${meta.dims}`);
  const D = meta.dims, scored = [];
  for (const c of scopedItems()) {
    const rows = idToRows[c.id]; if (!rows) continue;
    let best = -Infinity, bestRow = -1;
    for (const ri of rows) { if (!rowAllowed(ri)) continue; const off = ri * D; let s = 0; for (let d = 0; d < D; d++) s += qv[d] * E[off + d]; if (s > best) { best = s; bestRow = ri; } }
    if (bestRow >= 0) scored.push({ id: c.id, score: best, row: bestRow });
  }
  scored.sort((a, b) => b.score - a.score); lastResults = scored.slice(0, topK());
  $('count-head').textContent = 'Nearest consilia';
  setModelStatus(`Runs in your browser (${embedRuntime}).`, 'ready');
  shown = PAGE; await renderSemantic(); renderStats(scopedItems().length); ensurePCA(); summary(`nearest to “${q}” (${cfg.label}, ${scored.length} compared)`);
}
function setSimRef(id) {
  simRefId = id; const c = consilia[id];
  $('sim-chip-row').classList.remove('hidden'); $('sim-box').classList.add('hidden'); $('sim-vol').classList.add('hidden'); hideSugg();
  $('chip-n').textContent = `${volShort(c.volume)}, n ${c.n}`; $('chip-title').textContent = c.title_display || c.title; syncUrl(); runSearch();
}
$('sim-clear').addEventListener('click', () => { simRefId = null; $('sim-chip-row').classList.add('hidden'); $('sim-box').classList.remove('hidden'); $('sim-vol').classList.remove('hidden'); $('sim-input').value = ''; $('sim-input').focus(); syncUrl(); runSearch(); });
/* reference consilium: a print from the dropdown plus a number, or "Bald. 2, 171", "pano 115", or a word of the title / argument */
function simCandidates(raw) {
  const q = raw.toLowerCase().replace(/[.,;:]/g, ' ').replace(/\s+/g, ' ').trim(); if (!q) return [];
  const nums = (q.match(/\d+/g) || []).map(Number), words = q.replace(/\d+/g, ' ').split(' ').filter(Boolean);
  let pool = scopedItems(); const sv = $('sim-vol').value; if (sv) pool = pool.filter(c => c.volume === sv);
  if (words.length) pool = pool.filter(c => { const vs = volShort(c.volume).toLowerCase().replace(/\./g, ''), t = ((c.title_display || c.title || '') + ' ' + (c.summary || '')).toLowerCase(); return words.every(w => vs.includes(w) || t.includes(w)); });
  if (nums.length) { const n = nums[nums.length - 1], vn = nums.length > 1 ? nums[0] : null; pool = pool.filter(c => c.n === n && (vn === null || c.volume.endsWith('v' + vn) || c.pars === vn)); }
  return pool.slice(0, 8);
}
function hideSugg() { $('sim-sugg').classList.add('hidden'); $('sim-sugg').innerHTML = ''; }
function renderSugg() {
  const cands = simCandidates($('sim-input').value), box = $('sim-sugg');
  if (!cands.length) { if ($('sim-input').value.trim()) { box.innerHTML = '<button disabled class="status">No consilium matches.</button>'; box.classList.remove('hidden'); } else hideSugg(); return; }
  box.innerHTML = cands.map((c, i) => `<button data-id="${esc(c.id)}"${i === 0 ? ' class="hot"' : ''}><span class="sn">${esc(volShort(c.volume))}, ${c.pars ? c.pars + '.' : ''}${c.n}</span><span class="st">${esc(snippet(c.summary || c.title || '', null, 90))}</span></button>`).join('');
  box.classList.remove('hidden');
}
$('sim-input').addEventListener('input', renderSugg);
$('sim-input').addEventListener('focus', renderSugg);
$('sim-input').addEventListener('keydown', e => { if (e.key === 'Enter') { const b = $('sim-sugg').querySelector('button[data-id]'); if (b) setSimRef(b.dataset.id); } if (e.key === 'Escape') hideSugg(); });
$('sim-sugg').addEventListener('mousedown', e => { const b = e.target.closest('button[data-id]'); if (b) { e.preventDefault(); setSimRef(b.dataset.id); } });
document.addEventListener('click', e => { if (!e.target.closest('#sim-box')) hideSugg(); });
async function runSimilar() {
  const ref = consilia[simRefId]; if (!ref) { showResults(false, 'Reference not found.'); return; }
  const refRows = (idToRows[simRefId] || []).filter(rowAllowed);
  if (!refRows.length) { $('results-list').innerHTML = '<div class="empty">No passage vectors for this consilium (in the chosen language).</div>'; return; }
  const D = meta.dims, scored = [];
  for (const c of scopedItems()) {
    if (c.id === simRefId) continue;
    const rows = idToRows[c.id]; if (!rows) continue;
    let best = -Infinity, bestRow = -1, bestRef = -1;
    for (const ri of refRows) { const roff = ri * D; for (const ci of rows) { if (!rowAllowed(ci)) continue; const coff = ci * D; let s = 0; for (let d = 0; d < D; d++) s += E[roff + d] * E[coff + d]; if (s > best) { best = s; bestRow = ci; bestRef = ri; } } }
    if (bestRow >= 0) scored.push({ id: c.id, score: best, row: bestRow, refRow: bestRef });
  }
  scored.sort((a, b) => b.score - a.score); lastResults = scored.slice(0, topK());
  $('count-head').textContent = 'Similar consilia';
  shown = PAGE; await renderSemantic(true); renderStats(scored.length); ensurePCA(); summary(`closest to ${volShort(ref.volume)}, n ${ref.n} (${modelCfg().label}, ${scored.length} compared)`);
}

/* ── rendering ─────────────────────────────────────────────────────────── */
function sortResults() { const m = $('sort-select').value; return [...lastResults].sort((a, b) => m === 'n' ? ((consilia[a.id].volume.localeCompare(consilia[b.id].volume)) || (consilia[a.id].n - consilia[b.id].n)) : (b.score - a.score)); }
function hitHead(c) {
  return `<div class="top"><span class="n">${c.pars ? c.pars + '.' : ''}${c.n}</span><span class="ttl">${esc(c.title_display || c.title)}</span><span class="vol" title="${esc(volLabel(c.volume))}">${esc(volShort(c.volume))}</span>` +
    `<button class="sim" data-sim-id="${esc(c.id)}" title="Find similar consilia">similar</button></div>`;
}
function renderKeyword(re) {
  const list = $('results-list'); list.innerHTML = '';
  const sorted = sortResults(); if (!sorted.length) { list.innerHTML = '<div class="empty">No matching consilia.</div>'; return; }
  const max = Math.max(...sorted.map(r => r.score), 1);
  paginate(list, sorted, r => {
    const c = consilia[r.id]; const matched = new Set(Object.values(r.match || {}).flat());
    let snip = '', field = '';
    if (matched.has('title')) { snip = highlightText(c.title, re); field = 'title'; }
    else if (matched.has('summary') && c.summary) { snip = highlightText(snippet(c.summary, re), re); field = 'argument'; }
    else if (matched.has('casus') && (c.casus || []).length) { snip = highlightText(snippet(c.casus.join(' '), re), re); field = 'casus'; }
    else { snip = highlightText(snippet((c.body || []).join(' '), re), re); field = 'text'; }
    const a = document.createElement('a'); a.className = 'hit'; a.href = `index.html?id=${encodeURIComponent(c.id)}&q=${encodeURIComponent($('search-input').value.trim())}`;
    a.innerHTML = hitHead(c) + `<div class="snip"><span class="ev">${field}</span>${snip}</div><div class="bar" title="score ${r.score.toFixed(1)}"><i style="width:${(r.score / max * 100).toFixed(0)}%"></i></div>`;
    return a;
  });
}
/* the passage behind a vector row: Latin from the text shard, English from the translation shard */
async function passageText(id, ri) {
  const c = consilia[id], kind = meta.kind[ri] === 1 ? 'casus' : 'body', idx = meta.chunk[ri];
  if (meta.lang[ri] === 1) { try { const tr = await getShard('tr/en', c.volume); return { text: ((tr[id] || {})[kind] || [])[idx] || '', lang: 'en' }; } catch (e) { return { text: '', lang: 'en' }; } }
  if (!c.body) { try { const t = await getShard('text', c.volume); const x = t[id] || {}; c.body = x.body || []; c.casus = x.casus || []; } catch (e) {} }
  return { text: ((c[kind]) || [])[idx] || '', lang: 'la' };
}
async function renderSemantic(isSimilar) {
  const list = $('results-list'); list.innerHTML = '';
  const sorted = sortResults(); if (!sorted.length) { list.innerHTML = '<div class="empty">No results in scope.</div>'; return; }
  const max = sorted[0].score || 1;
  const passages = await Promise.all(sorted.map(r => passageText(r.id, r.row)));
  paginate(list, sorted, (r, i) => {
    const c = consilia[r.id], p = passages[i];
    const a = document.createElement('a'); a.className = 'hit'; a.href = `index.html?id=${encodeURIComponent(c.id)}`;
    a.innerHTML = hitHead(c) + `<div class="snip"><span class="ev">${p.lang === 'en' ? 'passage (EN)' : 'passage (LA)'}</span>${p.lang === 'en' ? trHTML(snippet(p.text, null, 260)) : esc(snippet(p.text, null, 260))}</div><div class="bar" title="cosine similarity ${r.score.toFixed(3)}"><i style="width:${Math.max(0, r.score / max * 100).toFixed(0)}%"></i></div>`;
    return a;
  });
}
$('results-list').addEventListener('click', e => {
  const sim = e.target.closest('.sim'); if (sim) { e.preventDefault(); e.stopPropagation(); setMode('similar', true); setSimRef(sim.dataset.simId); window.scrollTo({ top: 0, behavior: 'smooth' }); return; }
  const kw = e.target.closest('.chip[data-term]'); if (kw && !kw.classList.contains('on')) { e.preventDefault(); e.stopPropagation(); setMode('index', true); selectTerm(kw.dataset.term); window.scrollTo({ top: 0, behavior: 'smooth' }); }
});

/* ── subjects: layers ──────────────────────────────────────────────────── */
let annData = null, kwLang = 'en';
try { kwLang = localStorage.getItem('consilia-kw-lang') || 'en'; } catch (e) {}
function buildLayers(subj, body) {
  annData = [subj, body];
  const machine = {};
  allItems.forEach(c => (c.keywords_final || []).forEach(k => { (machine[k] = machine[k] || { label: k, ids: [] }).ids.push(c.id); }));
  layers = { machine: { label: 'Keywords (automatic)', terms: machine } };   // label shown without the bracket in the segment
  if (subj && subj.concepts) {
    const terms = {}; const byId = {}; subj.concepts.forEach(k => { byId[k.id] = k; });
    Object.entries(subj.docs || {}).forEach(([cid, d]) => {
      [...(d.central || []), ...(d.secondary || [])].forEach(id => { const k = byId[id]; if (!k) return; (terms[id] = terms[id] || { label: `${(kwLang === 'de' && k.de) || k.en} (${k.la})`, facet: k.facet, ids: [], central: new Set() }).ids.push(cid); if ((d.central || []).includes(id)) terms[id].central.add(cid); });
    });
    layers.subjects = { label: 'Subjects', terms, facets: subj.facets || [] };
  }
  if (refIndex && refIndex.refs && Object.keys(refIndex.refs).length) {   // Layer C: cited works, resolved by rule to work and title
    const terms = {}; Object.entries(refIndex.refs).forEach(([k, v]) => { terms[k] = { label: `${k} ${v.title || ''}`.trim(), ids: v.docs, forms: v.forms, n_spans: v.n }; });
    layers.references = { label: 'Citations', terms };
  }
  if (refIndex && refIndex.jurists) {   // Layer C: jurists cited by name
    const jt = {}; Object.entries(refIndex.jurists || {}).forEach(([k, v]) => { jt[k] = { label: k, ids: v.docs, forms: v.forms, n_spans: v.n }; });
    layers.jurists = { label: 'Jurists', terms: jt };
  }
  if (body && body.units) {
    const terms = {}; const codes = {}; (body.codes || []).forEach(k => { codes[k.id] = k; });
    body.units.forEach(u => {
      if (u.role === 'citation_only' || u.role === 'formulaic') return;
      const k = codes[u.code] || { label_en: u.code }; const t = (terms[u.code] = terms[u.code] || { label: (kwLang === 'de' && k.label_de) || k.label_en, group: k.group, ids: [], ev: {} });
      if (!t.ids.includes(u.doc)) t.ids.push(u.doc);
      (t.ev[u.doc] = t.ev[u.doc] || []).push(u);
    });
    layers.body = { label: 'Body and sexuality', terms };
  }
}
function renderLayerSeg() {
  const seg = $('index-layer'); seg.innerHTML = '';
  Object.entries(layers).forEach(([k, l]) => { const b = document.createElement('button'); b.textContent = l.label.replace(/ \(.*$/, ''); b.title = l.label; b.classList.toggle('active', k === layerKey); b.addEventListener('click', () => { layerKey = k; currentTerm = null; $('mode-hint').textContent = LAYER_HINTS[k] || HINTS.index; renderLayerSeg(); renderCloud(); syncUrl(); runSearch(); }); seg.appendChild(b); });
  // label language of the controlled layers (the automatic keywords exist in English only)
  const ls = $('index-lang'); ls.innerHTML = ''; ls.classList.toggle('hidden', layerKey === 'machine');
  ['en', 'de'].forEach(l => { const b = document.createElement('button'); b.textContent = l.toUpperCase(); b.title = l === 'de' ? 'German labels' : 'English labels'; b.classList.toggle('active', l === kwLang); b.addEventListener('click', () => { if (l === kwLang) return; kwLang = l; try { localStorage.setItem('consilia-kw-lang', l); } catch (e) {} buildLayers(...annData); renderLayerSeg(); renderCloud(); runSearch(); }); ls.appendChild(b); });
}
$('index-filter').addEventListener('input', renderCloud);
function renderCloud() {
  const q = ($('index-filter').value || '').trim().toLowerCase(), cloud = $('index-cloud'), L = layers[layerKey];
  const terms = Object.entries(L.terms).map(([id, t]) => ({ id, label: t.label, n: t.ids.length }));
  const matched = (q ? terms.filter(t => t.label.toLowerCase().includes(q) || t.id.toLowerCase().includes(q)) : terms).sort((a, b) => b.n - a.n || a.label.localeCompare(b.label));
  const shown = matched.slice(0, q ? 300 : 80);
  cloud.innerHTML = shown.map(t => `<button class="chip${t.id === currentTerm ? ' on' : ''}" data-term="${esc(t.id)}">${esc(t.label)} <span class="num muted">${t.n}</span></button>`).join('') || '<span class="status">No matching terms.</span>';
  $('index-status').textContent = q ? `${shown.length} of ${matched.length} terms` : `${terms.length} terms, the ${shown.length} most frequent shown; filter to see others`;
}
$('index-cloud').addEventListener('click', e => { const b = e.target.closest('.chip[data-term]'); if (b) selectTerm(b.dataset.term); });
function selectTerm(term) {
  if (!layers[layerKey].terms[term]) { const k = Object.keys(layers).find(k => layers[k].terms[term]); if (k) layerKey = k; }
  currentTerm = term; renderLayerSeg(); renderCloud(); syncUrl(); runSearch();
}
function renderIndexResults(term) {
  const list = $('results-list'); list.innerHTML = '';
  const sorted = sortResults(); if (!sorted.length) { list.innerHTML = '<div class="empty">No consilia carry this term in scope.</div>'; return; }
  paginate(list, sorted, r => {
    const c = consilia[r.id];
    const a = document.createElement('a'); a.className = 'hit'; a.href = `index.html?id=${encodeURIComponent(c.id)}`;
    let extra = '';
    if (term.ev && term.ev[c.id]) extra = term.ev[c.id].slice(0, 3).map(u => `<div class="snip"><span class="ev">${esc(u.role)}${u.status ? ', ' + esc(u.status.replace(/_/g, ' ')) : ''}</span>${esc(u.evidence_la || '')}</div>`).join('');
    else if (layerKey === 'subjects') extra = `<div class="snip"><span class="ev">${term.central && term.central.has(c.id) ? 'central' : 'secondary'}</span>${esc(snippet(c.summary || '', null, 200))}</div>`;
    else if (layerKey === 'references' || layerKey === 'jurists') extra = `<div class="snip"><span class="ev">as printed</span>${esc((term.forms || []).slice(0, 3).join(' | '))}</div><div class="snip">${esc(snippet(c.summary || '', null, 160))}</div>`;
    else extra = `<div class="snip">${esc(snippet(c.summary || (c.body || []).join(' '), null, 190))}</div><div class="kw">${(c.keywords_final || []).map(k => `<span class="chip${k === currentTerm ? ' on' : ''}" data-term="${esc(k)}">${esc(k)}</span>`).join('')}</div>`;
    a.innerHTML = hitHead(c) + extra; return a;
  });
}

/* ── stats, distribution, map ──────────────────────────────────────────── */
function renderStats(total) {
  const count = lastResults.length; $('stat-count').textContent = count; $('stat-pct').textContent = `${total ? ((count / total) * 100).toFixed(1) : 0}% of ${total} consilia in scope`;
  if (mode === 'keyword') {
    const fc = { title: 0, summary: 0, casus: 0, body: 0 };
    lastResults.forEach(r => new Set(Object.values(r.match || {}).flat()).forEach(f => { if (f in fc) fc[f]++; }));
    const mx = Math.max(...Object.values(fc), 1);
    Object.keys(fc).forEach(f => { $('cnt-' + f).textContent = fc[f]; $('bar-' + f).style.width = `${(fc[f] / mx * 100).toFixed(0)}%`; });
  }
  renderDistribution();
}
function renderDistribution() {
  const el = $('dist-bars'), pool = scopedItems(); if (!pool.length) { el.innerHTML = ''; return; }
  const inResult = new Set(lastResults.map(r => r.id)), byVol = {};
  pool.forEach(c => { (byVol[c.volume] = byVol[c.volume] || []).push(c); });
  const rows = Object.keys(byVol).sort().map(v => {
    const list = byVol[v].sort((a, b) => (a.pars || 0) - (b.pars || 0) || a.n - b.n), nb = Math.min(24, Math.max(6, Math.round(list.length / 20))), counts = new Array(nb).fill(0);
    list.forEach((c, i) => { if (inResult.has(c.id)) counts[Math.min(nb - 1, Math.floor(i * nb / list.length))]++; });
    return { v, counts, total: counts.reduce((x, y) => x + y, 0) };
  });
  const mx = Math.max(...rows.flatMap(r => r.counts), 1);
  el.innerHTML = rows.map(r => `<div class="drow"><span title="${esc(volLabel(r.v))}">${esc(volShort(r.v))}</span><div class="mini">${r.counts.map(v => `<i style="height:${(v / mx * 100).toFixed(0)}%" title="${v}"></i>`).join('')}</div><span class="c num">${r.total}</span></div>`).join('');
}
function seededRng(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const MAP_SEED = 20260923;
$('pcb-cluster').addEventListener('click', () => setPcaColor('cluster')); $('pcb-volume').addEventListener('click', () => setPcaColor('volume'));
function setPcaColor(m) { pcaColorMode = m; $('pcb-cluster').classList.toggle('active', m === 'cluster'); $('pcb-volume').classList.toggle('active', m === 'volume'); if (m === 'cluster' && !clusterAssignments && pcaCoords) computeClusters(); else renderPCA(); }
function ensurePCA() { if (!docIds) return; if (!pcaCoords) computePCA(); else if (pcaColorMode === 'cluster' && !clusterAssignments) computeClusters(); else renderPCA(); }
async function computePCA() {
  const N = docIds.length, D = meta.dims, X = docEmbeddings, mean = new Float32Array(D);
  for (let i = 0; i < N; i++) { const off = i * D; for (let d = 0; d < D; d++) mean[d] += X[off + d]; } for (let d = 0; d < D; d++) mean[d] /= N;
  const rng = seededRng(MAP_SEED), u = [new Float32Array(D), new Float32Array(D)];
  for (let k = 0; k < 2; k++) { for (let d = 0; d < D; d++) u[k][d] = rng() - 0.5; let n = 0; for (let d = 0; d < D; d++) n += u[k][d] ** 2; n = Math.sqrt(n); for (let d = 0; d < D; d++) u[k][d] /= n; }
  for (let it = 0; it < 6; it++) for (let k = 0; k < 2; k++) {
    const a = new Float32Array(N); for (let i = 0; i < N; i++) { const off = i * D; let s = 0; for (let d = 0; d < D; d++) s += (X[off + d] - mean[d]) * u[k][d]; a[i] = s; }
    const v = new Float32Array(D); for (let i = 0; i < N; i++) { const off = i * D; for (let d = 0; d < D; d++) v[d] += a[i] * (X[off + d] - mean[d]); }
    for (let j = 0; j < k; j++) { let dp = 0; for (let d = 0; d < D; d++) dp += v[d] * u[j][d]; for (let d = 0; d < D; d++) v[d] -= dp * u[j][d]; }
    let n = 0; for (let d = 0; d < D; d++) n += v[d] ** 2; n = Math.sqrt(n); if (n > 0) for (let d = 0; d < D; d++) u[k][d] = v[d] / n;
  }
  const coords = new Float32Array(N * 2); let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < N; i++) { const off = i * D; let px = 0, py = 0; for (let d = 0; d < D; d++) { const x = X[off + d] - mean[d]; px += x * u[0][d]; py += x * u[1][d]; } coords[i * 2] = px; coords[i * 2 + 1] = py; minX = Math.min(minX, px); maxX = Math.max(maxX, px); minY = Math.min(minY, py); maxY = Math.max(maxY, py); }
  pcaCoords = { raw: coords, minX, maxX, minY, maxY, N };
  if (pcaColorMode === 'cluster') computeClusters(); else renderPCA();
}
function computeClusters() {
  const { raw: coords, N } = pcaCoords, k = Math.max(3, Math.min(CLUSTER_PALETTE.length, Math.round(Math.sqrt(N / 2)) || 6));
  const cx = new Float32Array(k), cy = new Float32Array(k), rng = seededRng(MAP_SEED + 1);
  const seed = Math.floor(rng() * N); cx[0] = coords[seed * 2]; cy[0] = coords[seed * 2 + 1];
  for (let ki = 1; ki < k; ki++) { const d2 = new Float32Array(N); let tot = 0; for (let i = 0; i < N; i++) { let md = Infinity; for (let j = 0; j < ki; j++) { const dx = coords[i * 2] - cx[j], dy = coords[i * 2 + 1] - cy[j]; md = Math.min(md, dx * dx + dy * dy); } d2[i] = md; tot += md; } let r = rng() * tot, acc = 0; for (let i = 0; i < N; i++) { acc += d2[i]; if (acc >= r) { cx[ki] = coords[i * 2]; cy[ki] = coords[i * 2 + 1]; break; } } }
  const asgn = new Int32Array(N);
  for (let it = 0; it < 25; it++) { let changed = 0; for (let i = 0; i < N; i++) { let bk = 0, bd = Infinity; for (let ki = 0; ki < k; ki++) { const dx = coords[i * 2] - cx[ki], dy = coords[i * 2 + 1] - cy[ki], d = dx * dx + dy * dy; if (d < bd) { bd = d; bk = ki; } } if (asgn[i] !== bk) { asgn[i] = bk; changed++; } } if (!changed) break; const sx = new Float32Array(k), sy = new Float32Array(k), cn = new Int32Array(k); for (let i = 0; i < N; i++) { sx[asgn[i]] += coords[i * 2]; sy[asgn[i]] += coords[i * 2 + 1]; cn[asgn[i]]++; } for (let ki = 0; ki < k; ki++) if (cn[ki]) { cx[ki] = sx[ki] / cn[ki]; cy[ki] = sy[ki] / cn[ki]; } }
  clusterAssignments = asgn; clusterLabels = clusterLabelsFor(asgn, k); renderPCA();
}
function clusterLabelsFor(asgn, k) {
  const MINF = Math.max(2, Math.round(docIds.length * 0.02)), cf = Array.from({ length: k }, () => ({})), corpus = {};
  asgn.forEach((ki, i) => { const c = consilia[docIds[i]]; if (!c) return; const text = ((c.title || '').replace(/CONSILI[VU]M?\s*[IVXLCDM\d.]+\s*/i, '') + ' ' + (c.summary || '')).toLowerCase().replace(/[^a-zäöüàáâãèéêëìíïòóôùúæœ\s]/g, ' '); const seen = new Set(); text.split(/\s+/).forEach(w => { if (w.length < 4 || LAT_STOP.has(w)) return; cf[ki][w] = (cf[ki][w] || 0) + 1; if (!seen.has(w)) { corpus[w] = (corpus[w] || 0) + 1; seen.add(w); } }); });
  const cdf = {}; cf.forEach(f => Object.keys(f).forEach(w => cdf[w] = (cdf[w] || 0) + 1));
  return cf.map(f => { const total = Object.values(f).reduce((s, n) => s + n, 0) || 1; const scored = Object.entries(f).filter(([w]) => (corpus[w] || 0) >= MINF).map(([w, cnt]) => [w, (cnt / total) * Math.log((k + 1) / (cdf[w] || 1))]).filter(([, s]) => s > 0.001); scored.sort((a, b) => b[1] - a[1]); return scored.slice(0, 3).map(([w]) => w).join(', ') || '—'; });
}
function renderPCA() {
  if (!pcaCoords) return;
  const canvas = $('pca-canvas'), W = canvas.width, H = canvas.height, ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, W, H);
  const { raw, minX, maxX, minY, maxY, N } = pcaCoords, pad = 20, sx = x => pad + (x - minX) / ((maxX - minX) || 1) * (W - 2 * pad), sy = y => H - pad - (y - minY) / ((maxY - minY) || 1) * (H - 2 * pad);
  const useCluster = pcaColorMode === 'cluster' && clusterAssignments;
  const color = i => useCluster ? CLUSTER_PALETTE[clusterAssignments[i] % CLUSTER_PALETTE.length] : volColor(consilia[docIds[i]]?.volume);
  const hi = new Set(lastResults.map(r => docIdToIdx[r.id]).filter(i => i != null)); if (simRefId != null && docIdToIdx[simRefId] != null) hi.add(docIdToIdx[simRefId]);
  for (let i = 0; i < N; i++) { ctx.beginPath(); ctx.arc(sx(raw[i * 2]), sy(raw[i * 2 + 1]), N > 400 ? 2 : 3, 0, 7); ctx.fillStyle = (hi.size && !hi.has(i)) ? color(i) + '2b' : color(i) + 'bb'; ctx.fill(); }
  hi.forEach(i => { const isRef = docIds[i] === simRefId; ctx.beginPath(); ctx.arc(sx(raw[i * 2]), sy(raw[i * 2 + 1]), isRef ? 7 : 5, 0, 7); ctx.fillStyle = isRef ? cssVar('--paper') : color(i); ctx.fill(); ctx.lineWidth = isRef ? 2.4 : 1.4; ctx.strokeStyle = isRef ? color(i) : cssVar('--paper'); ctx.stroke(); });
  const legend = $('pca-legend');
  if (useCluster) legend.innerHTML = [...new Set(clusterAssignments)].sort((a, b) => a - b).map(ki => `<span><i style="background:${CLUSTER_PALETTE[ki % CLUSTER_PALETTE.length]}"></i>${esc(clusterLabels[ki] || 'cluster ' + (ki + 1))}</span>`).join('');
  else legend.innerHTML = [...new Set(docIds.map(id => consilia[id]?.volume).filter(Boolean))].map(v => `<span><i style="background:${volColor(v)}"></i>${esc(volLabel(v))}</span>`).join('');
  const tip = $('tip'), nearest = e => { const rect = canvas.getBoundingClientRect(), mx = (e.clientX - rect.left) * (W / rect.width), my = (e.clientY - rect.top) * (H / rect.height); let best = -1, bd = 100; for (let i = 0; i < N; i++) { const d = (sx(raw[i * 2]) - mx) ** 2 + (sy(raw[i * 2 + 1]) - my) ** 2; if (d < bd) { bd = d; best = i; } } return best; };
  canvas.onmousemove = e => { const b = nearest(e); if (b >= 0) { const c = consilia[docIds[b]], hit = lastResults.find(r => r.id === docIds[b]); tip.innerHTML = `${esc(volShort(c.volume))} n ${c.n}, ${esc(c.title)}` + (hit ? `<br>score ${hit.score.toFixed(3)}` : ''); tip.style.display = 'block'; tip.style.left = (e.clientX + 12) + 'px'; tip.style.top = (e.clientY - 8) + 'px'; } else tip.style.display = 'none'; };
  canvas.onmouseleave = () => { tip.style.display = 'none'; };
  canvas.onclick = e => { const b = nearest(e); if (b >= 0) window.open(`index.html?id=${encodeURIComponent(docIds[b])}`, '_blank'); };
}
