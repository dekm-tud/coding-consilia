/* Fulltext index in a Web Worker: the main thread never blocks while the text of the prints is
   indexed. Messages in:  {type:'add', docs:[{id,title,summary,casus,body}]} or {type:'search', q, seq}
   Messages out: {type:'added', n} or {type:'results', seq, results:[{id,score,match}]} */
importScripts('minisearch.min.js', 'common.js');

// lemmas first (form + lemma per word in the index, the lemma of each query word), when the page sends
// {type:'lemmas', url} before the documents; without them, word forms only
let ready = Promise.resolve();
const ms = new MiniSearch({
  fields: ['title', 'summary', 'casus', 'body'],
  idField: 'id',
  processTerm: indexTerm,
  searchOptions: SEARCH_OPTIONS,
});
let n = 0;

onmessage = async e => {
  const m = e.data;
  if (m.type === 'lemmas') { ready = loadLemmas(m.url); return; }
  await ready;
  try {
    if (m.type === 'add') {
      ms.addAll(m.docs); n += m.docs.length;
      postMessage({ type: 'added', n });
    } else if (m.type === 'search') {
      const results = ms.search(buildQuery(m.q)).map(r => ({ id: r.id, score: r.score, match: r.match }));
      postMessage({ type: 'results', seq: m.seq, results });
    }
  } catch (err) {
    postMessage({ type: 'error', seq: m.seq, message: String(err) });
  }
};
