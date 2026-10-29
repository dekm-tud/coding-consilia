// Language switch for the essay pages (introduction, authors): one page, two text blocks.
// The choice is shared with the keyword labels on the search page (localStorage 'consilia-kw-lang').
(function () {
  let lang = 'en';
  try { lang = localStorage.getItem('consilia-kw-lang') || 'en'; } catch (e) {}
  const blocks = () => document.querySelectorAll('.essay-block');
  function apply() {
    blocks().forEach(b => b.classList.toggle('hidden', b.dataset.lang !== lang));
    document.documentElement.lang = lang;
    document.querySelectorAll('#essay-lang button').forEach(b => b.classList.toggle('active', b.dataset.lang === lang));
    if (location.hash) { const t = document.querySelector(location.hash); if (t) t.scrollIntoView(); }
    if (document.querySelector('.essay-block')) buildToc();
  }
  /* side table of contents: the h2/h3 of the visible language block, placed next to the text column
     when the window is wide enough, hidden otherwise */
  function slug(t) { return t.toLowerCase().replace(/<[^>]+>/g, '').replace(/[^a-z0-9äöüß]+/g, '-').replace(/^-|-$/g, '').slice(0, 60); }
  function buildToc() {
    let nav = document.querySelector('nav.toc');
    const main = document.querySelector('main.page'); if (!main) return;
    if (!nav) { nav = document.createElement('nav'); nav.className = 'toc'; nav.setAttribute('aria-label', 'Contents'); document.body.appendChild(nav); }
    const block = [...document.querySelectorAll('.essay-block')].find(b => b.dataset.lang === lang && b.tagName === 'SECTION') || document.querySelector('main.page');
    const heads = [...block.querySelectorAll('h2, h3')].filter(h => h.textContent.trim());
    heads.forEach(h => { if (!h.id) { let id = slug(h.textContent), i = 2; while (document.getElementById(id)) id = slug(h.textContent) + '-' + i++; h.id = id; } });
    nav.innerHTML = heads.map(h => `<a class="${h.tagName.toLowerCase()}" href="#${h.id}">${h.textContent}</a>`).join('');
    nav.classList.toggle('hidden', heads.length < 3);
    placeToc(); watchToc(heads);
  }
  function placeToc() {
    const nav = document.querySelector('nav.toc'), main = document.querySelector('main.page'); if (!nav || !main) return;
    const r = main.getBoundingClientRect(), w = 15 * 16;
    const fits = window.innerWidth - r.right > w + 40;
    nav.style.display = fits && !nav.classList.contains('hidden') ? '' : 'none';
    if (fits) nav.style.left = Math.round(r.right + 24) + 'px';
  }
  let tocObs = null;
  function watchToc(heads) {
    if (tocObs) tocObs.disconnect();
    const links = [...document.querySelectorAll('nav.toc a')];
    tocObs = new IntersectionObserver(es => {
      es.forEach(e => { if (e.isIntersecting) links.forEach(l => l.classList.toggle('active', l.getAttribute('href') === '#' + e.target.id)); });
    }, { rootMargin: '-10% 0px -75% 0px' });
    heads.forEach(h => tocObs.observe(h));
  }
  window.addEventListener('resize', placeToc);
  document.addEventListener('DOMContentLoaded', () => {
    const seg = document.getElementById('essay-lang'); if (!seg) return;
    [['en', 'English'], ['de', 'Deutsch']].forEach(([l, title]) => {
      const b = document.createElement('button'); b.type = 'button'; b.dataset.lang = l; b.textContent = l.toUpperCase(); b.title = title;
      b.addEventListener('click', () => { if (l === lang) return; lang = l; try { localStorage.setItem('consilia-kw-lang', l); } catch (e) {} apply(); });
      seg.appendChild(b);
    });
    apply();
  });
})();
