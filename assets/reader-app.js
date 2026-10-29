/* Reader: index of one print, continuous Latin text, facsimile with region
   overlays. Data: site-data/ (index.json + one text shard). The machine translations are not shown on the site;
   the translation code below stays inert (trMode is always 'la').
   Deep links: ?id=<consilium id> | ?vol=<print>&n=<number> | ?q=<find in print> | &tr=en|de */
let site = null, volume = null, items = [], activeIndex = 0, srcPage = 0;
let observer = null, highlightRe = null, searchTimer = null, suppressObserver = false;
let miniSearch = null, trMode = 'la', trShard = null, fac = null, refShard = {};

const $ = id => document.getElementById(id);

/* ── boot ───────────────────────────────────────────────────────────────── */
loadSiteData().then(async s => {
  site = s;
  fac = new Facsimile($('osd'), { iiif: site.iiif, images: site.images });
  const sel = $('volume-select');
  const vols = [...new Set(site.index.map(c => c.volume))].sort();
  Object.entries(site.authors).forEach(([viaf, a]) => {
    const og = document.createElement('optgroup'); og.label = a.name || viaf;
    vols.filter(v => (a.prints || []).includes(v)).forEach(v => {
      const o = document.createElement('option'); o.value = v; o.textContent = printLabel(v, site.authors).replace(`${a.name}, `, '');
      og.appendChild(o);
    });
    if (og.children.length) sel.appendChild(og);
  });
  sel.addEventListener('change', () => { loadVolume(sel.value); pushState(); });

  const params = new URLSearchParams(location.search);
  const wantId = params.get('id'), wantVol = params.get('vol'), wantN = parseInt(params.get('n'));
  let startVol = vols[0], targetId = null, notice = '';
  if (wantId) {
    if (site.byId[wantId]) { startVol = site.byId[wantId].volume; targetId = wantId; }
    else notice = `Consilium “${wantId}” was not found.`;
  } else if (wantVol || wantN) {
    if (wantVol) { if (vols.includes(wantVol)) startVol = wantVol; else notice = `Print “${wantVol}” was not found.`; }
    if (wantN && !notice) {
      const c = site.index.find(c => c.volume === startVol && c.n === wantN);
      if (c) targetId = c.id; else notice = `Consilium ${wantN} was not found in ${printLabel(startVol, site.authors)}.`;
    }
  }
  sel.value = startVol;
  if (notice) { const n = document.createElement('div'); n.className = 'notice'; n.textContent = notice; $('rd-inner').parentElement.insertBefore(n, $('rd-inner')); }   // shown at once, above the text
  if (!(await loadVolume(startVol))) return;
  if (params.get('q')) { $('find-input').value = params.get('q'); runFind(params.get('q')); }
  if (targetId) { const i = items.findIndex(c => c.id === targetId); if (i >= 0) { activate(i, true); await settleScroll(i); } }
}).catch(err => { $('rd-inner').innerHTML = `<p class="error" style="padding:2rem">Could not load corpus: ${esc(String(err))}</p>`; });

function pushState() {
  const c = items[activeIndex]; if (!c) return;
  const u = new URL(location.href); u.search = '';
  u.searchParams.set('id', c.id); if (trMode !== 'la') u.searchParams.set('tr', trMode);
  history.replaceState(null, '', u);
}

/* ── one print ──────────────────────────────────────────────────────────── */
async function loadVolume(v) {
  volume = v; trShard = null;
  $('rd-inner').innerHTML = '<p class="empty">Loading text…</p>';
  let text;
  try { [text, refShard] = await Promise.all([getShard('text', v), loadReferencesShard(site, v)]); }
  catch (err) { $('rd-inner').innerHTML = `<p class="error" style="padding:2rem">Could not load the text of this print: ${esc(String(err))}</p>`; return false; }
  if (volume !== v) return false;
  items = site.index.filter(c => c.volume === v)
    .sort((a, b) => (a.pars || 0) - (b.pars || 0) || a.n - b.n || a.id.localeCompare(b.id))
    .map(e => { const t = text[e.id] || {}; return { ...e, argument: t.argument || [], casus: t.casus || [], body: t.body || [], marg: t.marg || [], summary: summaryOf(t.argument) }; });
  $('print-name').textContent = printLabel(v, site.authors);
  // legend of the marks, once per print, above the text (hidden with the toggle)
  const old = document.querySelector('.marks-legend'); if (old) old.remove();
  if (refShard && Object.keys(refShard).length) { const lg = document.createElement('div'); lg.className = 'marks-legend'; lg.title = 'Citations and names found by a language model in the Latin text and checked by a tagger (machine annotation). Click a resolved citation or a jurist to open the index.'; lg.innerHTML = '<span class="lg-civ">Corpus iuris civilis</span><span class="lg-can">Corpus iuris canonici</span><span class="lg-oth">Libri feudorum, other works</span><span class="lg-jur">jurists, glossa</span><span class="lg-per">persons</span><span class="lg-plc">places, institutions</span><span class="lg-dat">dates, statutes</span>'; $('rd-inner').parentElement.insertBefore(lg, $('rd-inner')); }
  await buildIndex(); buildList(); await buildText(); clearFind(); activate(0, false);
  return true;
}
async function buildIndex() {
  if (hasLemmas(site)) await loadLemmas();       // site-data/lemmas.json, see common.js; without it, word forms only
  miniSearch = new MiniSearch({ fields: ['title', 'summary', 'casus', 'body'], idField: 'idx', processTerm: indexTerm,
    searchOptions: SEARCH_OPTIONS });
  miniSearch.addAll(items.map((c, i) => ({ idx: i, title: c.title || '', summary: c.summary || '', casus: c.casus.join(' '), body: c.body.join(' ') })));
}
function buildList() {
  const list = $('rd-list'); list.innerHTML = '';
  items.forEach((c, i) => {
    const d = document.createElement('div'); d.className = 'ix'; d.dataset.idx = i;
    const first = (c.argument[0] && c.argument[0].text) || c.summary || '';
    d.innerHTML = `<span class="n">${c.pars ? c.pars + '.' : ''}${c.n ?? '?'}</span><span class="t">${esc(c.title_display || c.title || '')}<br><span class="sub">${esc(first.slice(0, 90))}</span></span>`;
    d.addEventListener('click', () => { activate(i, false); scrollTo(i); closeDrawers(); });
    list.appendChild(d);
  });
  setCount(null);
}
/* the print's own word for the numbered summary (site-data/editions.json argument_label): Argumentum, Summarium */
function argLabel(v) { const e = site.editions && site.editions[v]; return (e && e.argument_label) || 'Argumentum'; }
async function ensureTr() {
  if (trMode === 'la') return null;
  if (!trShard) { try { trShard = await getShard(`tr/${trMode}`, volume); } catch (e) { trShard = {}; } }
  return trShard;
}
async function buildText() {
  if (observer) { observer.disconnect(); observer = null; }
  const tr = await ensureTr();
  const inner = $('rd-inner'); inner.innerHTML = '';
  items.forEach((c, i) => {
    const a = document.createElement('article'); a.className = 'cons'; a.id = `c-${i}`; a.dataset.idx = i;
    a.innerHTML = consHTML(c, null, tr ? tr[c.id] : null);
    inner.appendChild(a);
  });
  observer = new IntersectionObserver(entries => {
    if (suppressObserver) return;
    let top = null, topY = Infinity;
    entries.forEach(e => { if (e.isIntersecting) { const y = e.boundingClientRect.top; if (y >= -40 && y < topY) { topY = y; top = e.target; } } });
    if (top) { const i = parseInt(top.dataset.idx); if (i !== activeIndex) activate(i, true); }
  }, { root: $('rd-text'), threshold: 0.12 });
  document.querySelectorAll('.cons').forEach(el => observer.observe(el));
}
/* section heading with a short note on how the layer was made (one wording for every layer) */
function secH(title, note) { return `<div class="sect-h">${title}${note ? ` <span class="muted" style="text-transform:none;letter-spacing:0">(${note})</span>` : ''}</div>`; }
const ROLE_LABELS = { topic: 'topic', argument: 'argument', case_fact: 'case fact', citation_only: 'citation only', formulaic: 'formula' };
function roleLabel(r) { return esc(ROLE_LABELS[r] || String(r || '').replace(/_/g, ' ')); }
function statusLabel(st) { return st === 'verified_by_model' ? ', found in both passes' : st === 'verified_by_editor' ? ', confirmed by the editor' : st && st !== 'machine' ? ', ' + esc(st.replace(/_/g, ' ')) : ''; }
function trP(t, lang) { return t ? `<p class="tr"><span class="trlang">${lang}</span>${trHTML(t)}</p>` : ''; }
/* translated argumentum: the machine translation is one string with the items as "1. …" lines;
   render it as the same numbered list as the Latin, with the print's numbers, not as one paragraph */
function trArgument(t, lang) {
  const items = [];
  String(t).split(/\n+/).forEach(line => {
    const m = line.trim().match(/^(\d{1,3})[.)]?\s+(.*)$/);
    if (m) items.push({ n: m[1], text: m[2] }); else if (line.trim() && items.length) items[items.length - 1].text += ' ' + line.trim(); else if (line.trim()) items.push({ n: '', text: line.trim() });
  });
  if (items.length < 2 && !(items[0] && items[0].n)) return `<div class="prose">${trP(t, lang)}</div>`;
  return `<ol class="arg tr"><span class="trlang">${lang}</span>` + items.map(it => `<li><span class="n">${esc(it.n)}</span><span>${trHTML(it.text)}</span></li>`).join('') + `</ol>`;
}
function consHTML(c, re, tr) {
  const h = re ? (t => highlightText(t, re)) : esc;
  const pages = (c.pages || []).map(p => p.page).join(', ');
  let html = `<div class="meta"><span>Consilium ${c.pars ? c.pars + '.' : ''}${c.n}</span><span class="cons-id">${esc(c.id)}</span><span class="pg">p. ${esc(pages)}</span></div>`;
  const spans = (refShard && refShard[c.id]) || [];
  const marked = (kind, i, s) => (!re && spans.length) ? spanHTML(s, spans.filter(x => x.k === kind && x.s === i)) : h(s);   // span marks only without a search highlight
  if (c.casus.length) {
    html += `<div class="casus-block"><div class="sect-h" style="margin-top:0">Casus</div><div class="prose">` +
      c.casus.map((s, i) => `<p class="sect">${marked('casus', i, s)}</p>` + (tr && tr.casus && tr.casus[i] ? trP(tr.casus[i], trMode) : '')).join('') + `</div></div>`;
  }
  html += `<h2>${h(c.title_display || c.title)}</h2>` + (c.title_display && c.title && c.title_display !== c.title ? `<div class="asprinted">${h(c.title)}</div>` : '');
  if (c.argument.length) {
    html += `<div class="sect-h">${esc(argLabel(c.volume))}</div><ol class="arg">` + c.argument.map(it => `<li><span class="n">${it.n}</span><span>${h(it.text)}</span></li>`).join('') + `</ol>`;
    if (tr && tr.argument) html += trArgument(tr.argument, trMode);
  }
  if (c.body.length) {
    html += `<div class="sect-h">Textus</div><div class="prose">`;
    if (tr && tr.stale) html += `<p class="tr"><span class="trlang">${trMode}</span>The translation of this text is withheld: the Latin paragraphs were corrected after it was made, and it will be renewed with the next translation run.</p>`;
    c.body.forEach((s, i) => {
      const m = c.marg[i];
      const txt = s.startsWith('† ') ? `<span class="dag">†</span>${(!re && spans.length) ? spanHTML(s.slice(2), spans.filter(x => x.k === 'body' && x.s === i).map(x => ({ ...x, o: x.o - 2 }))) : h(s.slice(2))}` : marked('body', i, s);
      html += `<p class="sect">${m ? `<span class="marg">${m}</span>` : ''}${txt}</p>` + (tr && tr.body && tr.body[i] ? trP(tr.body[i], trMode) : '');
    });
    html += `</div>`;
  }
  if (c.keywords_final && c.keywords_final.length) {
    html += secH('Keywords', 'assigned by a language model') + `<div class="kws">` +
      c.keywords_final.map(k => `<a class="chip" href="search.html?keyword=${encodeURIComponent(k)}">${esc(k)}</a>`).join('') + `</div>`;
  }
  if (spans.length) {   // citations as printed, grouped by corpus, and jurists linked to the index (machine annotation)
    const groups = { CIT_CIV: 'Corpus iuris civilis', CIT_CAN: 'Corpus iuris canonici', CIT_FEUD: 'Libri feudorum', CIT_OTHER: 'Other works' };
    const cits = spans.filter(x => groups[x.l]);
    const works = {}; cits.forEach(x => { if (!x.rk) return; const e = (works[x.rk] = works[x.rk] || { k: x.rk, n: 0, f: x.t }); e.n++; });
    const wl = Object.values(works).sort((p, q) => q.n - p.n);
    if (wl.length) html += secH('Cited works', `${wl.length} ${wl.length === 1 ? 'title' : 'titles'}, identified by rule from the printed citations`) + `<div class="kws">` +
      wl.slice(0, 60).map(e => `<a class="chip" href="search.html?mode=index&layer=references&keyword=${encodeURIComponent(e.k)}" title="${esc(e.f)}">${esc(e.k)}${e.n > 1 ? ` <span class="num muted">${e.n}</span>` : ''}</a>`).join('') + `</div>`;
    if (cits.length) {
      html += secH('Citations as printed', `${cits.length} ${cits.length === 1 ? 'passage' : 'passages'} found by a language model and a tagger, ${cits.filter(x => x.rk).length} resolved to a title`) + `<div class="cits">`;
      Object.entries(groups).forEach(([l, label]) => { const xs = cits.filter(x => x.l === l); if (xs.length) html += `<p class="cit-group"><span class="cit-label">${label}</span>${xs.map(x => `<span class="cit${x.st === 'verified_by_model' ? '' : ' unverified'}" title="${x.rk ? x.rk + ' · ' : (x.rc ? 'ambiguous: ' + x.rc.join(', ') + ' · ' : 'not resolved · ')}${x.st === 'verified_by_model' ? 'found by the language model and the tagger' : 'found by one of the two'}">${esc(x.t)}</span>`).join('')}</p>`; });
      html += `</div>`;
    }
    const cnt = {}; spans.forEach(x => { if (!x.jk) return; const e = (cnt[x.jk] = cnt[x.jk] || { k: x.jk, n: 0, f: x.t }); e.n++; });
    const jur = Object.values(cnt).sort((p, q) => q.n - p.n);
    if (jur.length) html += secH('Jurists cited', 'names found by a language model and a tagger, sigla resolved by rule') + `<div class="kws">` +
      jur.slice(0, 25).map(e => `<a class="chip" href="search.html?mode=index&layer=jurists&keyword=${encodeURIComponent(e.k)}" title="${esc(e.f)}">${esc(e.k)}${e.n > 1 ? ` <span class="num muted">${e.n}</span>` : ''}</a>`).join('') + `</div>`;
  }
  const ann = annotationsFor(site, c.id);
  if (ann.subjects.length) {
    html += secH('Subjects', 'controlled vocabulary, assigned by a language model') + `<div class="kws">` +
      ann.subjects.map(k => `<a class="chip${k.central ? ' on' : ''}" href="search.html?mode=index&layer=subjects&keyword=${encodeURIComponent(k.id)}" title="${esc(k.la || '')}${k.central ? ' (central)' : ' (secondary)'}">${esc(k.en)}</a>`).join('') + `</div>`;
  }
  if (ann.body.length) {
    const byCode = {}; ann.body.forEach(u => { (byCode[u.code] = byCode[u.code] || { u, ev: [] }).ev.push(u); });
    html += secH('Body and sexuality', 'codebook, assigned by a language model with the Latin evidence') + `<div class="prose ann">` +
      Object.values(byCode).map(({ u, ev }) => `<p class="sect"><a class="chip${ev.some(x => x.counted) ? ' on' : ''}" href="search.html?mode=index&layer=body&keyword=${encodeURIComponent(u.code)}">${esc(u.label_en)}</a> ` +
        ev.map(x => `<span class="muted">${roleLabel(x.role)}${statusLabel(x.status)}:</span> <q>${esc(x.evidence_la || '')}</q>`).join(' ')).join('') + `</div>`;
  }
  html += `<div class="actions"><a class="btn small" href="search.html?similar=${encodeURIComponent(c.id)}">Similar consilia</a>` +
    `<button class="btn small" data-act="cite" data-id="${esc(c.id)}">Cite</button>` +
    `<button class="btn small" data-act="link" data-id="${esc(c.id)}">Permalink</button>` +
    (repoURL(site) ? `<a class="btn small" href="${esc(repoURL(site))}/issues/new?title=${encodeURIComponent('Correction: ' + c.id)}" target="_blank" rel="noopener">Report an error</a>` : '') + `</div>`;
  return html;
}
$('rd-inner').addEventListener('click', async e => {
  const mk = e.target.closest('mark.sp[data-key]'); if (mk) { location.href = `search.html?mode=index&layer=${mk.dataset.layer}&keyword=${encodeURIComponent(mk.dataset.key)}`; return; }
  const b = e.target.closest('button[data-act]'); if (!b) return;
  const c = items.find(x => x.id === b.dataset.id); if (!c) return;
  const ok = await copyText(b.dataset.act === 'cite' ? citeText(c, site) : permalink(site, c.id));
  const old = b.textContent; b.textContent = ok ? 'Copied' : 'Copy failed'; setTimeout(() => { b.textContent = old; }, 1400);
});

/* ── find in this print ─────────────────────────────────────────────────── */
$('find-input').addEventListener('input', e => { clearTimeout(searchTimer); searchTimer = setTimeout(() => runFind(e.target.value), 200); });
$('find-clear').addEventListener('click', () => { $('find-input').value = ''; clearFind(); });
function setCount(shown) {
  $('rd-count').textContent = (shown != null && shown !== items.length) ? `${shown} of ${items.length} consilia` : `${items.length} consilia`;
}
function runFind(raw) {
  const q = raw.trim();
  $('corpus-link').href = q ? `search.html?q=${encodeURIComponent(q)}` : 'search.html';
  $('find-clear').style.display = q ? 'flex' : 'none';
  if (!q) { clearFind(); return; }
  highlightRe = buildHighlightRe(q);
  const hits = new Set(miniSearch.search(buildQuery(q)).map(r => r.id));
  let shown = 0, first = -1;
  items.forEach((c, i) => {
    const art = $(`c-${i}`), ix = document.querySelector(`.ix[data-idx="${i}"]`);
    if (hits.has(i)) { art.classList.remove('hidden'); art.innerHTML = consHTML(c, highlightRe, trShard ? trShard[c.id] : null); ix.classList.remove('hidden'); shown++; if (first < 0) first = i; }
    else { art.classList.add('hidden'); ix.classList.add('hidden'); }
  });
  setCount(shown);
  if (first >= 0) activate(first, true);
}
function clearFind() {
  highlightRe = null; $('corpus-link').href = 'search.html'; $('find-clear').style.display = 'none'; setCount(null);
  items.forEach((c, i) => { const art = $(`c-${i}`), ix = document.querySelector(`.ix[data-idx="${i}"]`); if (art) { art.classList.remove('hidden'); art.innerHTML = consHTML(c, null, trShard ? trShard[c.id] : null); } ix && ix.classList.remove('hidden'); });
}

/* ── activation, scrolling, facsimile ───────────────────────────────────── */
function activate(i, scrollIndex) {
  activeIndex = i; srcPage = 0;
  document.querySelectorAll('.ix').forEach(el => el.classList.toggle('active', parseInt(el.dataset.idx) === i));
  document.querySelectorAll('.cons.active').forEach(el => el.classList.remove('active'));
  $(`c-${i}`)?.classList.add('active');
  if (scrollIndex) document.querySelector(`.ix[data-idx="${i}"]`)?.scrollIntoView({ block: 'nearest' });
  updateImage(); pushState();
}
async function settleScroll(i) {
  suppressObserver = true;
  const go = () => $(`c-${i}`)?.scrollIntoView({ behavior: 'auto', block: 'start' });
  go(); try { await document.fonts.ready; } catch (e) {} go();
  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))); go();
  setTimeout(() => { suppressObserver = false; }, 400);
}
function scrollTo(i, behavior = 'smooth') { suppressObserver = true; $(`c-${i}`)?.scrollIntoView({ behavior, block: 'start' }); setTimeout(() => { suppressObserver = false; }, 650); }

let imageTimer = null;
function updateImage() {            // debounced: scrolling through many consilia must not start a request per step
  clearTimeout(imageTimer);
  imageTimer = setTimeout(updateImageNow, 250);
}
async function updateImageNow() {
  const c = items[activeIndex]; const pages = c?.pages ?? []; const total = pages.length;
  $('page-prev').disabled = srcPage <= 0; $('page-next').disabled = srcPage >= total - 1;
  $('page-counter').textContent = total > 1 ? `${srcPage + 1}/${total}` : '';
  if (!total) { $('page-label').textContent = '—'; $('osd-empty').textContent = 'no page source'; $('osd-empty').style.display = 'flex'; return; }
  const page = pages[srcPage];
  $('page-label').textContent = `Page ${page.page.replace(/^page_/, '')}`;
  $('page-tag').textContent = 'loading…';
  try {
    const info = await fac.show(c.volume, c.author_viaf, page);
    if (items[activeIndex] !== c) return;
    $('osd-empty').style.display = 'none';
    $('page-tag').textContent = info.kind === 'iiif' ? 'IIIF' : 'facsimile';
    let credit = esc(info.attribution || '');
    if (info.label) credit += `, ${esc(info.label)}`;
    if (info.link) credit += ` <a href="${esc(info.link)}" target="_blank" rel="noopener">${info.kind === 'iiif' ? 'Canvas at source ↗' : 'Full resolution at source ↗'}</a>`;
    $('fac-credit').innerHTML = credit;
  } catch (e) {
    $('osd-empty').textContent = 'page image unavailable'; $('osd-empty').style.display = 'flex'; $('page-tag').textContent = 'error';
  }
}
function pageTurn(d) { const total = items[activeIndex]?.pages?.length ?? 0; const n = srcPage + d; if (n >= 0 && n < total) { srcPage = n; updateImageNow(); } }
$('page-prev').addEventListener('click', () => pageTurn(-1));
$('page-next').addEventListener('click', () => pageTurn(1));

/* ── drawers (narrow screens) ───────────────────────────────────────────── */
$('ix-toggle').addEventListener('click', () => $('rd').classList.toggle('ix-open'));
$('fac-toggle').addEventListener('click', () => { $('rd').classList.toggle('fac-open'); setTimeout(() => fac.home(), 250); });
function closeDrawers() { $('rd').classList.remove('ix-open'); }

/* narrow screens: the print select lives in the index drawer */
function placeControls() {
  const narrow = matchMedia('(max-width: 760px)').matches;
  const home = document.querySelector('.topbar'), drawer = $('mobile-controls');
  ['volume-select'].forEach(id => {
    const el = $(id); const target = narrow ? drawer : home;
    if (el.parentElement !== target) { if (narrow) { drawer.appendChild(el); el.style.display = ''; } else { home.insertBefore(el, $('fac-toggle')); el.style.display = ''; } }
  });
}
placeControls(); addEventListener('resize', placeControls);
/* Layer C marks on/off (kept per browser) */
try { if (localStorage.getItem('consilia-marks') === 'off') { document.body.classList.add('no-marks'); $('mark-refs').checked = false; } } catch (e) {}
$('mark-refs').addEventListener('change', e => { document.body.classList.toggle('no-marks', !e.target.checked); try { localStorage.setItem('consilia-marks', e.target.checked ? 'on' : 'off'); } catch (err) {} });

