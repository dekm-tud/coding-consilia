const { test, expect } = require('@playwright/test');
test('reader shows subjects and body annotations for v1 n.17', async ({ page }) => {
  const errs = []; page.on('pageerror', e => errs.push(String(e)));
  await page.goto('/index.html?id=consilium-baldo_cons_print_venice_1575_v1-17');
  await page.waitForSelector('article.cons h2');
  const txt = (await page.locator('article.cons', { hasText: 'consilium-baldo_cons_print_venice_1575_v1-17' }).first().innerText()).toLowerCase();   // headings are upper-cased by CSS
  expect(txt).toContain('subjects'); expect(txt).toContain('testamentary capacity');
  expect(txt).toContain('body and sexuality'); expect(txt).toContain('adultery');
  expect(errs).toEqual([]);
});
test('chat tools answer from the layers without an LLM', async ({ page }) => {
  const errs = []; page.on('pageerror', e => errs.push(String(e)));
  await page.goto('/chat.html');
  await page.waitForFunction(() => typeof site !== 'undefined' && site && site.annotations && site.annotations.subjects && site.annotations.subjects.concepts);
  const r = await page.evaluate(async () => ({
    subj: (await TOOLS.list_subjects.run({ contains: 'dowry' })).concepts.slice(0, 2),
    bys: (await TOOLS.consilia_by_subject.run({ concept: 'fam_dowry_restitution', limit: 2 })),
    codes: (await TOOLS.list_body_codes.run({})).codes.slice(0, 2),
    byc: (await TOOLS.consilia_by_body_code.run({ code: 'adultery', limit: 2 })),
  }));
  expect(r.subj.length).toBe(2); expect(r.bys.total).toBeGreaterThan(100); expect(r.bys.hits[0].assignment).toBe('central');
  expect(r.codes[0].consilia_counted).toBeGreaterThan(0); expect(r.byc.total).toBeGreaterThan(10); expect(r.byc.hits[0].occurrences[0].evidence_la.length).toBeGreaterThan(5);
  expect(errs).toEqual([]);
});
test('chat: similar_consilia works from the stored vectors without a query model', async ({ page }) => {
  const errs = []; page.on('pageerror', e => errs.push(String(e)));
  await page.goto('/chat.html');
  await page.waitForFunction(() => typeof vecStore !== 'undefined' && vecStore && vecTag);
  const r = await page.evaluate(async () => TOOLS.similar_consilia.run({ id: 'consilium-baldo_cons_print_venice_1575_v4-62', limit: 3 }));
  expect(r.error).toBeUndefined(); expect(r.hits.length).toBe(3); expect(r.hits[0].passage.length).toBeGreaterThan(20);
  const s = await page.evaluate(async () => TOOLS.search_semantic.run({ query: 'dowry' }));
  expect(String(s.error || '')).toContain('not loaded');
  expect(errs).toEqual([]);
});
test('reader marks citations in the text and lists the cited works (v1 n.1)', async ({ page }) => {
  const errs = []; page.on('pageerror', e => errs.push(String(e)));
  await page.goto('/index.html?id=consilium-baldo_cons_print_venice_1575_v1-1');
  await page.waitForSelector('article.cons h2');
  const art = page.locator('article.cons', { hasText: 'consilium-baldo_cons_print_venice_1575_v1-1' }).first();
  expect(await art.locator('mark.sp').count()).toBeGreaterThan(5);
  expect((await art.innerText()).toLowerCase()).toContain('citations');
  expect(await art.locator('.cits .cit').count()).toBeGreaterThan(3); expect(await art.locator('a.chip[href*="layer=jurists"]').count()).toBeGreaterThan(0); expect(await art.locator('a.chip[href*="layer=references"]').count()).toBeGreaterThan(0);
  expect(errs).toEqual([]);
});
test('search index: a jurist lists exactly the consilia of the index entry', async ({ page }) => {
  const errs = []; page.on('pageerror', e => errs.push(String(e)));
  await page.goto('/search.html?mode=index&layer=jurists&keyword=Cynus%20de%20Pistorio');
  await page.waitForFunction(() => typeof layers !== 'undefined' && layers.jurists && document.querySelectorAll('#results-list .hit').length > 0, null, { timeout: 60000 });
  while (await page.locator('#results-list .more').count()) await page.locator('#results-list .more').click();   // results come in pages of 30
  const n = await page.evaluate(() => ({ listed: document.querySelectorAll('#results-list .hit').length, index: layers.jurists.terms['Cynus de Pistorio'].ids.length }));
  expect(n.listed).toBe(n.index); expect(n.index).toBeGreaterThan(100);
  expect(errs).toEqual([]);
});
test('chat tools answer from the citation layer', async ({ page }) => {
  await page.goto('/chat.html');
  await page.waitForFunction(() => typeof site !== 'undefined' && site && site.annotations);
  const r = await page.evaluate(async () => ({ by: await TOOLS.consilia_by_jurist.run({ name: 'Bartolus de Saxoferrato', limit: 2 }), jur: (await TOOLS.list_jurists.run({ contains: 'bartolus' })).jurists[0], c: await TOOLS.get_consilium.run({ id: 'consilium-baldo_cons_print_venice_1575_v1-2', with_translation: false }), refs: (await TOOLS.list_references.run({ contains: 'pactis' })).refs[0], byref: await TOOLS.consilia_by_reference.run({ key: 'D. 2.14', limit: 2 }) }));
  expect(r.by.total).toBeGreaterThan(100); expect(r.by.hits.length).toBe(2); expect(r.jur.name).toContain('Bartolus'); expect(r.c.citations_as_printed.length).toBeGreaterThan(3); expect(r.refs.key).toBe('D. 2.14'); expect(r.byref.total).toBeGreaterThan(100);
});
