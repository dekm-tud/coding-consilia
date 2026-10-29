/* Bibliography page: renders site-data/bibliography.json (tools/build_bibliography.py) in the language of the
   essay switch (assets/essay.js sets <html lang>), with search, subject filters and one anchor per entry
   (bibliography.html#<Zotero key>), which the footnotes of the essay pages link to. */
(function () {
  const $ = id => document.getElementById(id);
  const T = {
    en: { search: 'Search author, title, journal', count: (n, m) => n === m ? `${n} titles` : `${n} of ${m} titles`,
          sortBy: 'Sort', sort: { author: 'Author', year: 'Newest first', yearAsc: 'Oldest first' }, clear: 'Clear all',
          filters: n => n ? `Filters (${n})` : 'Filters',
          ed: n => n > 1 ? 'eds.' : 'ed.', edBy: 'edited by', in: 'In', and: 'and', pp: '', vol: 'vol.',
          cited: 'Cited in', pages: { 'introduction.html': 'Introduction', 'baldus.html': 'Baldus', 'panormitanus.html': 'Panormitanus' },
          dl: 'Download as <a href="site-data/bibliography.ris" download>RIS</a> for Zotero, Citavi or EndNote, or as <a href="site-data/bibliography.json" download>JSON</a>. The list is based on the editor\'s Zotero collection, and missing data were completed from library catalogues.',
          thesis: 'Thesis', none: 'No titles match these filters.', literature: 'Research literature', sources: 'Sources cited on this site', undated: 'undated' },
    de: { search: 'Autor, Titel, Zeitschrift suchen', count: (n, m) => n === m ? `${n} Titel` : `${n} von ${m} Titeln`,
          sortBy: 'Sortierung', sort: { author: 'Autor', year: 'Neueste zuerst', yearAsc: 'Älteste zuerst' }, clear: 'Alle entfernen',
          filters: n => n ? `Filter (${n})` : 'Filter',
          ed: () => 'Hg.', edBy: 'hg. von', in: 'in:', and: 'und', pp: 'S. ', vol: 'Bd.',
          cited: 'Zitiert in', pages: { 'introduction.html': 'Einführung', 'baldus.html': 'Baldus', 'panormitanus.html': 'Panormitanus' },
          dl: 'Herunterladen als <a href="site-data/bibliography.ris" download>RIS</a> für Zotero, Citavi oder EndNote oder als <a href="site-data/bibliography.json" download>JSON</a>. Grundlage ist die Zotero-Sammlung des Herausgebers, fehlende Angaben wurden aus Bibliothekskatalogen ergänzt.',
          thesis: 'Dissertation', none: 'Keine Titel passen zu diesen Filtern.', literature: 'Forschungsliteratur', sources: 'Auf dieser Website zitierte Quellen', undated: 'ohne Jahr' },
  };
  let data = null, active = new Set(), q = '', sort = 'author', hashDone = false;
  const lang = () => (document.documentElement.lang === 'de' ? 'de' : 'en');
  const esc = s => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const dash = s => esc(s).replace(/\s*-+\s*/g, '–');

  function nameInv(p) { return p.name ? esc(p.name) : esc(p.first ? `${p.last}, ${p.first}` : p.last); }
  function nameFwd(p) { return p.name ? esc(p.name) : esc(p.first ? `${p.first} ${p.last}` : p.last); }
  function names(list, L, first) {
    const out = list.map((p, i) => (first && i === 0 ? nameInv(p) : nameFwd(p)));
    return out.length <= 1 ? out.join('') : out.slice(0, -1).join(', ') + ` ${T[L].and} ` + out[out.length - 1];
  }
  function cite(e, L) {
    const t = T[L], a = e.authors || [], ed = e.editors || [];
    const head0 = a.length ? names(a, L, true) : ed.length ? `${names(ed, L, true)} (${t.ed(ed.length)})` : '';
    const head = head0 ? `<span class="bib-au">${head0.replace(/\.$/, '')}</span>` : '';   // the period follows outside the name
    const pub = [e.place, e.publisher].filter(Boolean).map(esc).join(': ');
    const series = e.series ? esc(e.series) + (e.seriesNumber ? ' ' + esc(e.seriesNumber) : '') : '';
    const title = esc(e.title.replace(/["“”„]([^"“”„]*)["“”]/g, '‘$1’')), cont = esc(e.container || '');   // inner quotation marks become single ones
    const pages = e.pages ? t.pp + dash(e.pages) : '';
    const when = esc(e.dateText || e.year || '');
    const section = ['bookSection', 'dictionaryEntry', 'encyclopediaArticle', 'conferencePaper'].includes(e.type);
    let s;
    if (e.type === 'manuscript') {
      const where = [e.place, e.archive, e.archiveLocation].filter(Boolean).map(esc).join(', ');
      return `${where}${where ? ': ' : ''}${head ? head + ', ' : ''}<i>${title}</i>${when ? ` (${when})` : ''}.`;
    }
    if (e.type === 'webpage') return `${head ? head + (L === 'en' ? '. ' : ': ') : ''}${L === 'en' ? `“${title}.”` : title + ','} <i>${cont}</i>${e.year ? ', ' + e.year : ''}.`;
    if (L === 'en') {
      if (e.type === 'journalArticle' || e.type === 'magazineArticle') {
        s = `${head}. “${title}.” <i>${cont}</i>${e.volume ? ' ' + esc(e.volume) : ''}${e.issue ? ', no. ' + esc(e.issue) : ''}${e.year ? ` (${e.year})` : ''}${pages ? ': ' + pages : ''}.`;
      } else if (section) {
        const edPart = a.length && ed.length ? `, ${t.edBy} ${names(ed, L, false)}` : '';
        s = `${head}. “${title}.” ${t.in} <i>${cont}</i>${edPart}${pages ? ', ' + pages : ''}.${series ? ' ' + series + '.' : ''}${pub ? ' ' + pub + ',' : ''} ${e.year || ''}.`;
      } else {
        s = `${head}${head ? '. ' : ''}<i>${title}</i>.${e.type === 'thesis' ? ' ' + esc(e.thesisType || t.thesis) + '.' : ''}${e.reportType ? ' ' + esc(e.reportType) + '.' : ''}${series ? ' ' + series + '.' : ''}${pub ? ' ' + pub + ',' : ''} ${when}.`;
      }
    } else {
      if (e.type === 'journalArticle' || e.type === 'magazineArticle') {
        s = `${head}: ${title}, ${t.in} <i>${cont}</i>${e.volume ? ' ' + esc(e.volume) : ''}${e.issue ? '/' + esc(e.issue) : ''}${e.year ? ` (${e.year})` : ''}${pages ? ', ' + pages : ''}.`;
      } else if (section) {
        const edPart = a.length && ed.length ? `${names(ed, L, false)} (${t.ed(ed.length)}): ` : '';
        s = `${head}: ${title}, ${t.in} ${edPart}<i>${cont}</i>${series ? ` (${series})` : ''}, ${pub ? pub + ' ' : ''}${e.year || ''}${pages ? ', ' + pages : ''}.`;
      } else {
        s = `${head}${head ? ': ' : ''}<i>${title}</i>${e.type === 'thesis' ? ', ' + esc(e.thesisType || t.thesis) : ''}${e.reportType ? ', ' + esc(e.reportType) : ''}${series ? ` (${series})` : ''}, ${pub ? pub + ' ' : ''}${when}.`;
      }
    }
    return s.replace(/\s+([.,])/g, '$1').replace(/\.\./g, '.').replace(/,\s*\./g, '.');
  }
  const label = (id, L) => { const t = data.tags.find(x => x.id === id); return t ? t[L] : id; };
  const hay = e => [e.title, e.container, e.year, e.publisher, e.place, ...(e.authors || []).concat(e.editors || []).map(p => p.name || `${p.first} ${p.last}`)]
    .join(' ').toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');

  const sortKey = e => {
    const p = (e.authors || e.editors || [])[0];
    return ((p && (p.last || p.name)) || e.title).normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/^(de |da |di |van |von |del |della )/i, '').toUpperCase();
  };
  function renderFacets(L, counts) {
    const t = T[L];
    $('bib-facets').innerHTML = Object.entries(data.facets).map(([f, fl]) => {
      const tags = data.tags.filter(x => x.facet === f && (counts[x.id] || active.has(x.id)));
      if (!tags.length) return '';
      return `<details class="bib-facet" open><summary>${esc(fl[L])}</summary><ul>${tags.map(x =>
        `<li><button type="button" class="bib-opt${active.has(x.id) ? ' on' : ''}" data-tag="${x.id}" aria-pressed="${active.has(x.id)}"><span>${esc(x[L])}</span><span class="n">${counts[x.id] || 0}</span></button></li>`).join('')}</ul></details>`;
    }).join('');
    $('bib-active').innerHTML = active.size ? [...active].map(x => `<button type="button" class="bib-pill" data-tag="${x}" title="×">${esc(label(x, L))} <span aria-hidden="true">×</span></button>`).join('') +
      `<button type="button" class="bib-clear">${t.clear}</button>` : '';
    $('bib-filter-btn').textContent = t.filters(active.size);
  }
  function item(e, L) {
    const t = T[L];
    // only DOIs are linked; a web page cited as a source is linked to itself, publisher and shop pages are not
    const links = [e.doi ? `<a href="https://doi.org/${esc(e.doi)}" target="_blank" rel="noopener">doi:${esc(e.doi)}</a>` : '',
                   !e.doi && e.type === 'webpage' && e.url ? `<a href="${esc(e.url)}" target="_blank" rel="noopener">${esc(e.url.replace(/^https?:\/\/(www\.)?/, '').split('/')[0])}</a>` : ''];
    const cited = (e.cited || []).length ? `${t.cited} ${[...new Set(e.cited.map(c => c.page))].map(p =>
      `<a href="${p}#${e.cited.find(c => c.page === p).note}">${t.pages[p] || p}</a>`).join(', ')}` : '';
    const meta = [e.identifiers ? esc(e.identifiers) : '', ...links, cited].filter(Boolean).join('<span class="sep">·</span>');
    return `<li id="${esc(e.id)}"><div class="bib-cite">${cite(e, L)}</div>${meta ? `<div class="bib-meta">${meta}</div>` : ''}</li>`;
  }
  function render() {
    const L = lang(), t = T[L];
    $('bib-q').placeholder = t.search;
    $('bib-sort-l').textContent = t.sortBy;
    $('bib-sort').innerHTML = Object.entries(t.sort).map(([k, v]) => `<option value="${k}"${k === sort ? ' selected' : ''}>${v}</option>`).join('');
    const words = q.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '').split(/\s+/).filter(Boolean);
    let list = data.entries.filter(e => [...active].every(x => (e.tags || []).includes(x)) && words.every(w => e._hay.includes(w)));
    if (sort === 'author') list = [...list].sort((a, b) => a._key.localeCompare(b._key, 'de') || (a.year || '').localeCompare(b.year || ''));
    else list = [...list].sort((a, b) => (sort === 'year' ? -1 : 1) * ((a.year || '0').localeCompare(b.year || '0')) || a._key.localeCompare(b._key));
    $('bib-count').textContent = t.count(list.length, data.entries.length);
    const counts = {};
    list.forEach(e => (e.tags || []).forEach(x => { counts[x] = (counts[x] || 0) + 1; }));
    renderFacets(L, counts);
    // groups: initial letter (author order) or decade (year order); sources follow the literature
    const groupOf = e => sort === 'author' ? (e._key[0] || '#') : (e.year ? `${e.year.slice(0, 3)}0s` : t.undated);
    const block = (entries, title, withNav) => {
      if (!entries.length) return { html: '', groups: [] };
      const groups = [];
      entries.forEach(e => { const g = groupOf(e); if (!groups.length || groups[groups.length - 1].g !== g) groups.push({ g, items: [] }); groups[groups.length - 1].items.push(e); });
      const html = (title ? `<h2 class="bib-part">${title}</h2>` : '') + groups.map(({ g, items }) =>
        `<h3 class="bib-group"${withNav ? ` id="g-${esc(g)}"` : ''}>${esc(g)}</h3><ol class="bib-entries">${items.map(e => item(e, L)).join('')}</ol>`).join('');
      return { html, groups: groups.map(x => x.g) };
    };
    const lit = list.filter(e => e.kind !== 'source'), src = list.filter(e => e.kind === 'source');
    const A = block(lit, src.length ? t.literature : '', true), B = block(src, t.sources, false);
    $('bib-list').innerHTML = list.length ? A.html + B.html : `<p class="muted bib-none">${t.none}</p>`;
    $('bib-az').innerHTML = A.groups.length > 3 ? A.groups.map(g => `<a href="#g-${esc(g)}">${esc(g)}</a>`).join('') : '';
    $('bib-dl').innerHTML = t.dl;
    if (location.hash && !hashDone) {     // entries are rendered after load, so :target does not apply; mark the entry instead
      const el = document.getElementById(decodeURIComponent(location.hash.slice(1)));
      if (el && el.tagName === 'LI') { el.classList.add('hit'); el.scrollIntoView({ block: 'center' }); hashDone = true; }
    } else if (location.hash) {
      const el = document.getElementById(decodeURIComponent(location.hash.slice(1)));
      if (el && el.tagName === 'LI') el.classList.add('hit');
    }
  }
  function syncUrl() {
    const u = new URL(location.href);
    u.searchParams.delete('tag'); active.forEach(x => u.searchParams.append('tag', x));
    q ? u.searchParams.set('q', q) : u.searchParams.delete('q');
    history.replaceState(null, '', u);
  }
  document.addEventListener('click', ev => {
    const b = ev.target.closest('[data-tag]');
    if (b) { const x = b.dataset.tag; active.has(x) ? active.delete(x) : active.add(x); syncUrl(); render(); return; }
    if (ev.target.closest('.bib-clear')) { active.clear(); syncUrl(); render(); return; }
    if (ev.target.closest('#bib-filter-btn')) document.body.classList.toggle('bib-filters-open');
  });
  document.addEventListener('DOMContentLoaded', async () => {
    const p = new URLSearchParams(location.search);
    p.getAll('tag').forEach(x => active.add(x)); q = p.get('q') || '';
    $('bib-q').value = q;
    let timer = null;
    $('bib-q').addEventListener('input', ev => { clearTimeout(timer); timer = setTimeout(() => { q = ev.target.value.trim(); syncUrl(); render(); }, 150); });
    $('bib-sort').addEventListener('change', ev => { sort = ev.target.value; render(); });
    try { data = await fetchJSON('site-data/bibliography.json'); }
    catch (err) { $('bib-list').innerHTML = `<p class="error">${esc(String(err))}</p>`; return; }
    data.entries.forEach(e => { e._hay = hay(e); e._key = sortKey(e); });
    render();
    new MutationObserver(render).observe(document.documentElement, { attributes: true, attributeFilter: ['lang'] });
  });
})();
