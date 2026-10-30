// Smoke tests for the static site (real data only, no mocked corpus).
// Run: npx -y -p @playwright/test playwright test -c tests/playwright.config.js
const { test, expect } = require('@playwright/test');

const P = 'consilium-baldo_cons_print_venice_1575_';
const VOLS = [1, 2, 3, 4, 5].map(k => `Baldo_Cons_Print_Venice_1575_v${k}`);

function watchConsole(page) {
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  return errors;
}
async function activeCons(page) {
  const idx = await page.locator('.ix.active').getAttribute('data-idx');
  return page.locator(`#c-${idx}`);
}
async function nPrints(request) {
  const idx = await (await request.get('/site-data/index.json')).json();
  return new Set(idx.map(c => c.volume)).size;
}

// deep links: ?id= opens exactly that consilium, in its own print, for every Baldo volume
for (const k of [1, 2, 3, 4, 5]) {
  test(`reader deep link ?id= lands on the right consilium (v${k})`, async ({ page }) => {
    const errors = watchConsole(page);
    const id = `${P}v${k}-128`;
    await page.goto(`/index.html?id=${encodeURIComponent(id)}`);
    await expect(page.locator('#volume-select')).toHaveValue(VOLS[k - 1]);
    const c = await activeCons(page);
    await expect(c.locator('.cons-id')).toHaveText(id);
    await expect(c).toBeInViewport();
    expect(errors).toEqual([]);
  });
}

test('legacy ?vol=&n= link selects the volume, not just the number', async ({ page }) => {
  await page.goto(`/index.html?vol=${VOLS[1]}&n=128`);
  await expect(page.locator('#volume-select')).toHaveValue(VOLS[1]);
  await expect((await activeCons(page)).locator('.cons-id')).toHaveText(`${P}v2-128`);
});

test('an unresolvable link shows a notice instead of a wrong consilium', async ({ page }) => {
  await page.goto('/index.html?id=consilium-does-not-exist');
  await expect(page.locator('.notice')).toContainText('not found');
});

test('v4 n.62 opens with the case of Nardus (casus block above the title)', async ({ page }) => {
  await page.goto(`/index.html?id=${encodeURIComponent(P + 'v4-62')}`);
  const c = await activeCons(page);
  await expect(c.locator('.casus-block')).toContainText('Nardus');
  const order = await c.evaluate(el => {
    const cb = el.querySelector('.casus-block'), h = el.querySelector('h2');
    return cb.compareDocumentPosition(h) & Node.DOCUMENT_POSITION_FOLLOWING ? 'casus-first' : 'title-first';
  });
  expect(order).toBe('casus-first');
  await page.goto(`/index.html?id=${encodeURIComponent(P + 'v4-61')}`);
  await expect(await activeCons(page)).not.toContainText('Nardus maritauit');
});

test('the reader shows the Latin text only, and Panormitanus has a Summarium', async ({ page }) => {
  await page.goto(`/index.html?id=${encodeURIComponent(P + 'v1-17')}&tr=en`);
  const c = await activeCons(page);
  await expect(c.locator('.sect').first()).toBeVisible();
  await expect(c.locator('p.tr')).toHaveCount(0);
  await expect(page.locator('#tr-select')).toHaveCount(0);
  await expect(c.locator('.sect-h').first()).toHaveText('Argumentum');
  await page.goto('/index.html?vol=Panormitanus_Cons_Print_Venice_1571_v1&n=1');
  await expect((await activeCons(page)).locator('.sect-h', { hasText: 'Summarium' }).first()).toBeVisible();
});

test('reader loads the metadata index and ONE text shard, never the full corpus', async ({ page }) => {
  const urls = [];
  page.on('request', r => urls.push(new URL(r.url()).pathname));
  await page.goto(`/index.html?id=${encodeURIComponent(P + 'v2-10')}`);
  await expect((await activeCons(page)).locator('.cons-id')).toHaveText(P + 'v2-10');
  expect(urls).not.toContain('/output/consilia.json');
  expect(urls.filter(u => u.startsWith('/site-data/text/'))).toEqual([`/site-data/text/${VOLS[1]}.json`]);
  expect(urls.filter(u => u.startsWith('/site-data/tr/'))).toEqual([]);
});

test('facsimile: Mirador shows the Baldus page from the Internet Archive item with attribution and region annotations (needs network)', async ({ page }) => {
  const urls = [];
  page.on('request', r => urls.push(r.url()));
  await page.goto(`/index.html?id=${encodeURIComponent(P + 'v3-20')}`);
  await expect(page.locator('#osd .mirador-viewer canvas').first()).toBeVisible({ timeout: 30000 });
  await expect(page.locator('#page-tag')).toHaveText('IIIF');
  await expect(page.locator('#fac-credit')).toContainText('Universidad Complutense de Madrid');
  await expect.poll(() => urls.some(u => /archive\.org\/download\/baldus-consilia-venetiis-1575-vol3[^/]*\/page\/n\d+\.jpg/.test(u)), { timeout: 30000 }).toBe(true);
  expect(urls.some(u => /\/img\/page_\d+\.avif$/.test(u))).toBe(false);   // no local derivatives any more
  const regions = await page.evaluate(() => {          // the consilium's regions as an annotation page in Mirador's store
    const st = fac.mirador.store.getState().annotations || {};
    return Object.values(st).flatMap(byId => Object.values(byId)).filter(a => a.json && /consilium-regions/.test(a.id)).reduce((n, a) => n + (a.json.items || []).length, 0);
  });
  expect(regions).toBeGreaterThan(0);
});

test('Panormitanus pages come from the Harvard IIIF service (needs network)', async ({ page }) => {
  test.setTimeout(90000);
  await page.goto('/index.html?id=consilium-panormitanus_cons_print_venice_1571_v1-p1-1');
  await expect(page.locator('#page-tag')).toHaveText('IIIF', { timeout: 60000 });
  await expect(page.locator('#fac-credit')).toContainText('seq. 311');
});

// ---- search --------------------------------------------------------------------------

test('fulltext result opens the matching consilium in the reader', async ({ page }) => {
  const errors = watchConsole(page);
  await page.goto('/search.html');
  await page.fill('#search-input', 'abortu');
  const first = page.locator('#results-list a.hit').first();
  await expect(first).toBeVisible();
  const href = await first.getAttribute('href');
  expect(href).toMatch(/^index\.html\?id=/);
  const id = decodeURIComponent(href.split('id=')[1].split('&')[0]);
  await first.click();
  await expect(page).toHaveURL(/index\.html\?id=/);
  // the reader loads the text shard and the lemma list first, then jumps to the consilium
  await expect.poll(async () => { const c = await activeCons(page); return (await c.locator('.cons-id').count()) ? c.locator('.cons-id').textContent() : null; }, { timeout: 30000 }).toBe(id);
  expect(errors).toEqual([]);
});

test('subjects: the count on a chip equals the number of listed consilia', async ({ page }) => {
  const errors = watchConsole(page);
  await page.goto('/search.html');
  await page.click('#mode-tabs button[data-mode="index"]');
  const chip = page.locator('#index-cloud .chip').first();
  await expect(chip).toBeVisible();
  const n = parseInt(await chip.locator('.num').textContent(), 10);
  await chip.click();
  await expect(page.locator('#results-list a.hit').first()).toBeVisible();
  while (await page.locator('#results-list .more').count()) await page.locator('#results-list .more').click();   // results come in pages of 30
  expect(await page.locator('#results-list a.hit').count()).toBe(n);
  expect(errors).toEqual([]);
});

test('the map is reproducible: no random seeding in the search code', async ({ request }) => {
  const src = await (await request.get('/assets/search-app.js')).text();
  expect(src).not.toContain('Math.random');
});

test('Similar mode works from the compact float16 embeddings and shows the matched passage', async ({ page, request }) => {
  const errors = watchConsole(page);
  await page.goto(`/search.html?similar=${encodeURIComponent(P + 'v4-62')}`);
  await expect(page.locator('#results-list a.hit').first()).toBeVisible({ timeout: 90000 });
  expect(await page.locator('#results-list a.hit').count()).toBe(20);
  await expect(page.locator('#results-list a.hit .snip .ev').first()).toContainText(/passage/i);
  await expect(page.locator('#dist-bars .drow')).toHaveCount(await nPrints(request));
  expect(errors).toEqual([]);
});

test('lemma search: "heredem" also finds and marks the other forms of heres', async ({ page }) => {
  const errors = watchConsole(page);
  await page.goto('/search.html');
  await page.fill('#search-input', 'heredem');
  await expect(page.locator('#results-list a.hit').first()).toBeVisible({ timeout: 60000 });
  const other = t => { t = t.toLowerCase().replace(/[æę]|ae/g, 'e'); return /^h(e)?re/.test(t) && t !== 'heredem'; };   // heres, heredes, heredis …
  await expect.poll(async () => (await page.locator('#results-list mark').allTextContents()).some(other), { timeout: 30000 }).toBe(true);
  expect(errors).toEqual([]);
});

test('a word in quotation marks is searched in that form only', async ({ page }) => {
  await page.goto('/search.html');
  await page.fill('#search-input', '"heredem"');
  await expect(page.locator('#results-list a.hit').first()).toBeVisible({ timeout: 60000 });
  const marks = (await page.locator('#results-list mark').allTextContents()).map(t => t.toLowerCase().replace(/[æę]|ae/g, 'e'));
  expect(marks.length).toBeGreaterThan(0);
  expect(marks.every(t => t === 'heredem')).toBe(true);
});

test('several words must all occur, OR and -word work, commas compare terms in the chart', async ({ page }) => {
  const errors = watchConsole(page);
  const count = async q => {
    await page.fill('#search-input', q);
    await expect(page.locator('#result-summary')).toContainText(q.includes(',') ? 'feudum' : `“${q}”`, { timeout: 60000 });
    await expect(page.locator('#index-progress')).toHaveText('', { timeout: 60000 });
    await page.fill('#search-input', ''); await page.fill('#search-input', q);
    await expect(page.locator('#result-summary')).toContainText(q.includes(',') ? 'feudum' : `“${q}”`);
    return Number(await page.locator('#stat-count').textContent());
  };
  await page.goto('/search.html');
  const both = await count('feudum vasallus'), either = await count('feudum OR vasallus'), only = await count('feudum -vasallus'), one = await count('feudum');
  expect(both).toBeGreaterThan(0);
  expect(both).toBeLessThan(one);
  expect(either).toBeGreaterThan(one);
  expect(only + both).toBe(one);
  await count('feudum, vasallus');
  await expect(page.locator('#trend-chart polyline')).toHaveCount(2);
  await expect(page.locator('#trend-legend span:has(i)')).toHaveCount(2);
  await page.locator('#trend-metric button[data-metric="per10k"]').click();
  await expect(page.locator('#trend-note')).toContainText('per 10,000 words');
  await expect(page.locator('#results-list a.hit .tdots i').first()).toBeVisible();
  expect(errors).toEqual([]);
});

test('Ask tools: full-text syntax and term_frequency give the numbers of the search page', async ({ page }) => {
  const errors = watchConsole(page);
  await page.goto('/chat.html');
  await expect.poll(() => page.evaluate(() => typeof textsReady !== 'undefined' && textsReady !== null), { timeout: 30000 }).toBe(true);
  const r = await page.evaluate(async () => {
    await textsReady;
    const a = await TOOLS.search_fulltext.run({ query: 'feudum vasallus', limit: 3 });
    const b = await TOOLS.search_fulltext.run({ query: 'feudum -vasallus', limit: 3 });
    const t = await TOOLS.term_frequency.run({ terms: ['feudum', 'vasallus'], sections: true });
    return { a: a.total, b: b.total, hits: a.hits.length, terms: t.terms, prints: t.prints.length, sections: t.sections.length, chart: t.chart };
  });
  expect(r.a).toBe(58);
  expect(r.b).toBe(130);
  expect(r.hits).toBe(3);
  expect(r.terms.map(x => x.consilia)).toEqual([188, 70]);
  expect(r.terms[0].per_10k_words).toBeGreaterThan(0);
  expect(r.sections).toBeGreaterThan(20);
  expect(r.chart).toContain('search.html?mode=keyword&q=feudum%2C%20vasallus');
  expect(errors).toEqual([]);
});

test('every search mode explains itself', async ({ page }) => {
  await page.goto('/search.html');
  await page.locator('#mode-help summary').click();
  await expect(page.locator('#mode-help-body')).toContainText('BM25');
  await page.locator('#mode-tabs button[data-mode="semantic"]').click();
  await expect(page.locator('#mode-help-body')).toContainText('not a complete list');
});

test('fulltext runs in a worker and reports indexing progress', async ({ page }) => {
  await page.goto('/search.html');
  await page.fill('#search-input', 'usufructu');
  await expect(page.locator('#results-list a.hit').first()).toBeVisible();
  await expect(page.locator('#index-progress')).toHaveText('', { timeout: 60000 });
});

test('search state is in the URL (mode, query, scope)', async ({ page }) => {
  await page.goto('/search.html');
  await page.fill('#search-input', 'dos');
  await page.selectOption('#scope-select', `vol:${VOLS[2]}`);
  await expect(page).toHaveURL(/mode=keyword/);
  await expect(page).toHaveURL(/q=dos/);
  await expect(page).toHaveURL(/scope=vol%3ABaldo_Cons_Print_Venice_1575_v3/);
  await expect(page.locator('#results-list a.hit .vol').first()).toHaveText('Bald. 3');
});

// ---- about / chat ------------------------------------------------------------------------

test('about page shows live corpus numbers; chat page loads without errors', async ({ page, request }) => {
  const errors = watchConsole(page);
  await page.goto('/about.html');
  const man = await (await request.get('/site-data/manifest.json')).json();
  await expect(page.locator('[data-fill="n-consilia"]')).toHaveText(String(man.n_consilia));
  // the page is bilingual: one prints table per language block, only the active one is visible
  await expect(page.locator('.essay-block:not(.hidden) [data-fill="prints-table"] tbody tr')).toHaveCount(await nPrints(request));
  await page.locator('#essay-lang button[data-lang="de"]').click();
  await expect(page.locator('.essay-block:not(.hidden) [data-fill="prints-table"] th').first()).toHaveText('Band');
  await page.locator('#essay-lang button[data-lang="en"]').click();
  await page.goto('/chat.html');
  await expect(page.locator('#log .msg').first()).toContainText('Corpus loaded');
  expect(errors).toEqual([]);
});

// ---- preview gate -----------------------------------------------------------------------
test.describe('preview password gate', () => {
  test.use({ storageState: { cookies: [], origins: [] } });
  test('without the password only the gate is shown; the right password opens the site', async ({ page, request }) => {
    const cfg = await (await request.get('/assets/gate-config.js')).text();
    test.skip(!/"hash"/.test(cfg), 'no preview password set');
    const pw = process.env.CONSILIA_GATE_TEST_PASSWORD;
    await page.goto('/about.html');
    await expect(page.locator('#gate input[type=password]')).toBeVisible();
    await expect(page.locator('main')).toBeHidden();
    await page.fill('#gate input', 'wrong password');
    await page.click('#gate button');
    await expect(page.locator('#gate .err')).toHaveText('Wrong password.');
    test.skip(!pw, 'set CONSILIA_GATE_TEST_PASSWORD to check the right password');
    await page.fill('#gate input', pw);
    await page.click('#gate button');
    await expect(page.locator('#gate')).toHaveCount(0);
    await expect(page.locator('main')).toBeVisible();
    await page.goto('/search.html');
    await expect(page.locator('#gate')).toHaveCount(0);           // remembered in this browser
  });
});

// ---- bibliography -----------------------------------------------------------------------
test('bibliography: a footnote link opens its entry, tags filter the list', async ({ page }) => {
  const errors = watchConsole(page);
  await page.goto('/introduction.html');
  const link = page.locator('#n1 .fn-bib a').first();
  const href = await link.getAttribute('href');
  await link.click();
  await expect(page.locator('#bib-list li.hit')).toHaveCount(1);
  expect(await page.locator('#bib-list li.hit').getAttribute('id')).toBe(decodeURIComponent(href.split('#')[1]));
  await page.goto('/bibliography.html?tag=baldus');
  await expect(page.locator('#bib-count')).toContainText(' of ');
  const n = await page.locator('#bib-list .bib-entries > li').count();
  expect(n).toBeGreaterThan(5);
  expect(errors).toEqual([]);
});
