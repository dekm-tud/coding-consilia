/* Passage vectors and query encoders without page-specific UI, shared by chat.html (plain script).
   search.html has its own copy of this logic with its picker; both read site-data/models.json and the
   hf-export/<tag>/ files (chunks.json + chunks.f16.bin), whose data_revision must equal the site's.

   const V = new VectorStore(site, models); await V.load(tag, onStatus)  -> {E, meta, idToRows}
   const Q = new QueryEncoder(cfg, onStatus); await Q.load(); const vec = await Q.embed(text)
   V.rank(vec, tag, {lang, limit, ids}) -> [{id, score, row}];  V.similar(id, tag, {...}) */
(function () {
  let F16 = null;
  function f16ToF32(buf) {
    if (!F16) { F16 = new Float32Array(65536); for (let h = 0; h < 65536; h++) { const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff; F16[h] = e === 0 ? s * Math.pow(2, -14) * (m / 1024) : e === 31 ? (m ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + m / 1024); } }
    const u = new Uint16Array(buf), out = new Float32Array(u.length); for (let i = 0; i < u.length; i++) out[i] = F16[u[i]]; return out;
  }
  async function fetchBuffer(url, onStatus) {
    const r = await fetch(url); if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
    const total = Number(r.headers.get('Content-Length')) || 0;
    if (!r.body || !total) return r.arrayBuffer();
    const reader = r.body.getReader(), buf = new Uint8Array(total); let got = 0;
    for (;;) { const { done, value } = await reader.read(); if (done) break; buf.set(value, got); got += value.length; if (onStatus) onStatus(`Loading vectors… ${(got / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`); }
    return buf.buffer.slice(0, got);
  }
  async function cachedBuffer(url, revision, onStatus) {
    const key = new Request(`${url}?rev=${encodeURIComponent(revision || '')}`);
    try { const c = await caches.open('consilia-embeddings'); const hit = await c.match(key); if (hit) return await hit.arrayBuffer(); } catch (e) {}
    const buf = await fetchBuffer(url, onStatus);
    try { const c = await caches.open('consilia-embeddings'); await c.put(key, new Response(buf.slice(0))); } catch (e) {}
    return buf;
  }

  class VectorStore {
    constructor(site, models) { this.site = site; this.models = models; this.cache = {}; this.loading = {}; }
    cfg(tag) { return this.models.find(m => m.tag === tag); }
    async load(tag, onStatus) {
      if (this.cache[tag]) return this.cache[tag];
      if (!this.loading[tag]) this.loading[tag] = (async () => {
        const cfg = this.cfg(tag); if (!cfg) throw new Error(`unknown model ${tag}`);
        let lastErr = null, chunks = null, buf = null;
        for (const base of cfg.embeddings) {
          try {
            chunks = await fetchJSON(base + 'chunks.json');
            if (chunks.data_revision !== this.site.manifest.revision) throw new Error(`vectors are for data revision ${chunks.data_revision}, the site is ${this.site.manifest.revision}`);
            buf = await cachedBuffer(base + 'chunks.f16.bin', `${chunks.revision}-${chunks.data_revision}`, onStatus); break;
          } catch (e) { lastErr = e; chunks = null; buf = null; }
        }
        if (!buf) throw lastErr || new Error('no embedding source');
        const E = f16ToF32(buf);
        if (E.length !== chunks.n * chunks.dims) throw new Error('embedding file size does not match chunks.json');
        if (chunks.dims !== cfg.dims) throw new Error(`vectors have ${chunks.dims} dimensions, models.json says ${cfg.dims}`);
        if (chunks.doc.some(d => d < 0 || d >= this.site.index.length)) throw new Error('embedding rows refer to unknown consilia');
        const meta = { model: chunks.model, dims: chunks.dims, n: chunks.n, ids: chunks.doc.map(d => this.site.index[d].id), chunk: chunks.chunk, lang: chunks.lang, kind: chunks.kind };
        const idToRows = {}; meta.ids.forEach((id, i) => { (idToRows[id] = idToRows[id] || []).push(i); });
        this.cache[tag] = { E, meta, idToRows }; delete this.loading[tag];
        if (onStatus) onStatus(`${cfg.label}: ${meta.n} passages ready`);
        return this.cache[tag];
      })().catch(e => { delete this.loading[tag]; throw e; });
      return this.loading[tag];
    }
    /* best passage per consilium for a query vector; lang: 'la' | 'en' | 'both' */
    rank(qv, tag, { lang = 'both', limit = 10, ids = null } = {}) {
      const { E, meta, idToRows } = this.cache[tag]; const D = meta.dims;
      if (qv.length !== D) throw new Error(`query vector has ${qv.length} dimensions, the corpus vectors ${D}`);
      const ok = ri => lang === 'both' || (lang === 'la' ? meta.lang[ri] === 0 : meta.lang[ri] === 1);
      const scored = [];
      for (const id of (ids || Object.keys(idToRows))) {
        const rows = idToRows[id]; if (!rows) continue;
        let best = -Infinity, bestRow = -1;
        for (const ri of rows) { if (!ok(ri)) continue; const off = ri * D; let s = 0; for (let d = 0; d < D; d++) s += qv[d] * E[off + d]; if (s > best) { best = s; bestRow = ri; } }
        if (bestRow >= 0) scored.push({ id, score: best, row: bestRow });
      }
      scored.sort((a, b) => b.score - a.score); return scored.slice(0, limit);
    }
    /* nearest consilia to one consilium by its own passage vectors (no query model needed) */
    similar(refId, tag, { lang = 'both', limit = 10 } = {}) {
      const { E, meta, idToRows } = this.cache[tag]; const D = meta.dims;
      const ok = ri => lang === 'both' || (lang === 'la' ? meta.lang[ri] === 0 : meta.lang[ri] === 1);
      const refRows = (idToRows[refId] || []).filter(ok); if (!refRows.length) return [];
      const scored = [];
      for (const [id, rows] of Object.entries(idToRows)) {
        if (id === refId) continue;
        let best = -Infinity, bestRow = -1, bestRef = -1;
        for (const ri of refRows) { const roff = ri * D; for (const ci of rows) { if (!ok(ci)) continue; const coff = ci * D; let s = 0; for (let d = 0; d < D; d++) s += E[roff + d] * E[coff + d]; if (s > best) { best = s; bestRow = ci; bestRef = ri; } } }
        if (bestRow >= 0) scored.push({ id, score: best, row: bestRow, refRow: bestRef });
      }
      scored.sort((a, b) => b.score - a.score); return scored.slice(0, limit);
    }
    /* the passage behind a vector row: {text, lang, kind, chunk} (loads the text/translation shard) */
    async passage(tag, row) {
      const { meta } = this.cache[tag]; const id = meta.ids[row], c = this.site.byId[id];
      const kind = meta.kind && meta.kind[row] === 1 ? 'casus' : 'body', chunk = meta.chunk[row], en = meta.lang[row] === 1;   // kind: 1 = casus
      let text = '';
      try {
        if (en) { const tr = await getShard('tr/en', c.volume); const t = tr[id] || {}; text = ((kind === 'casus' ? t.casus : t.body) || [])[chunk] || ''; }
        else { const sh = await getShard('text', c.volume); const t = sh[id] || {}; text = ((kind === 'casus' ? t.casus : t.body) || [])[chunk] || ''; }
      } catch (e) { text = ''; }
      return { id, text, lang: en ? 'en' : 'la', kind, chunk };
    }
  }

  /* query encoder in the browser (assets/embed-worker.js, transformers.js); a model is downloaded
     only when load() is called, and then kept by the browser's cache */
  class QueryEncoder {
    constructor(cfg, onStatus) { this.cfg = cfg; this.onStatus = onStatus; this.worker = null; this.ready = false; this.seq = 0; this.waiters = {}; this.runtime = ''; }
    load() {
      if (this.ready) return Promise.resolve();
      if (this.loadingP) return this.loadingP;
      const cfg = this.cfg;
      this.loadingP = new Promise((resolve, reject) => {
        this.worker = new Worker('assets/embed-worker.js', { type: 'module' });
        const prog = {};
        this.worker.onmessage = e => {
          const m = e.data;
          if (m.type === 'progress') { prog[m.file] = [m.loaded, m.total]; const big = Object.values(prog).sort((a, b) => b[1] - a[1])[0]; if (this.onStatus) this.onStatus(`Downloading ${cfg.label}… ${(big[0] / 1e6).toFixed(0)} / ${(big[1] / 1e6).toFixed(0)} MB`); }
          else if (m.type === 'ready') { this.ready = true; this.runtime = `${m.device.toUpperCase()} ${m.dtype}`; if (this.onStatus) this.onStatus(`${cfg.label} ready (${this.runtime})`); resolve(); }
          else if (m.type === 'embedding' || m.type === 'error') { const w = this.waiters[m.id]; if (m.id === undefined || !w) { if (m.type === 'error') reject(new Error(m.message)); return; } delete this.waiters[m.id]; m.type === 'error' ? w.reject(new Error(m.message)) : w.resolve(m.vec); }
        };
        this.worker.onerror = e => reject(new Error(e.message || 'worker failed'));
        this.worker.postMessage({ type: 'load', cfg: { onnx_repo: cfg.onnx_repo, onnx_revision: cfg.onnx_revision, dtype: cfg.dtype, pooling: cfg.pooling, normalize: cfg.normalize, query_prefix: cfg.query_prefix } });
      }).catch(e => { this.loadingP = null; throw e; });
      return this.loadingP;
    }
    embed(text) {
      if (!this.ready) return Promise.reject(new Error(`the ${this.cfg.label} model is not loaded`));
      return new Promise((resolve, reject) => { const id = ++this.seq; this.waiters[id] = { resolve, reject }; this.worker.postMessage({ type: 'embed', id, text }); });
    }
    terminate() { if (this.worker) this.worker.terminate(); this.worker = null; this.ready = false; this.loadingP = null; }
  }
  window.VectorStore = VectorStore; window.QueryEncoder = QueryEncoder;
})();
