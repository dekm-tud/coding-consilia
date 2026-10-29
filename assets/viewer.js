/* Facsimile viewer: Mirador 4 (vendored, assets/vendor/mirador.min.js), one window, driven by the reader.
   Sources (site-data/iiif_manifests.json):
     - Internet Archive items (Baldus): the IIIF server of the Archive is unreliable (info.json and the manifest
       time out or answer 502), so the reader builds the Presentation 3 manifest itself: one canvas per page of the
       print, sized from site-data/image_dims.json, painted with the item's static page image
       (archive.org/download/<item>/page/n<idx>.jpg). Mirador shows it as a single image, no tile service needed.
     - other IIIF providers (Harvard, Panormitanus): the provider's manifest, fetched once and handed to Mirador.
   Local page N (PageXML stem) is canvas index canvas_map[stem] or N-1+canvas_offset.
   Regions of the active consilium (PageXML boxes, index.json `pages[].regions`) become an annotation page on the
   canvas (motivation commenting), drawn by Mirador's annotation overlay; the view is fitted to the first region. */
class Facsimile {
  constructor(el, { iiif = {}, images = {} } = {}) {
    this.el = el; this.iiif = iiif; this.images = images;
    this.mirador = null; this.windowId = 'facsimile'; this.manifests = {}; this.dims = null; this.current = null; this.page = null;
    if (!el.id) el.id = 'facsimile-' + Math.random().toString(36).slice(2);
  }
  _ensure() {
    if (this.mirador) return this.mirador;
    const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#175d78';
    this.mirador = Mirador.viewer({
      id: this.el.id,
      windows: [],
      workspaceControlPanel: { enabled: false },
      workspace: { type: 'mosaic', allowNewWindows: false, showZoomControls: true, draggingEnabled: false },
      window: { allowClose: false, allowMaximize: false, allowFullscreen: true, allowTopMenuButton: false, allowWindowSideBar: false,
                hideWindowTitle: true, sideBarOpen: false, defaultView: 'single', views: [{ key: 'single' }],
                highlightAllAnnotations: true, forceDrawAnnotations: true },
      thumbnailNavigation: { defaultPosition: 'off' },
      annotations: { filteredMotivations: ['commenting', 'oa:commenting'] },
      osdConfig: { crossOriginPolicy: false, ajaxWithCredentials: false,      // archive.org page images carry no CORS header
                 gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: true } },
      theme: { palette: { annotations: {
        hidden: { globalAlpha: 0 },
        default: { strokeStyle: accent, lineWidth: 3, globalAlpha: 0.9 },
        hovered: { strokeStyle: accent, lineWidth: 4, globalAlpha: 1 },
        selected: { strokeStyle: accent, lineWidth: 4, globalAlpha: 1 },
      } } },
    });
    return this.mirador;
  }
  _dispatch(action) { this._ensure().store.dispatch(action); }
  async _dims() {
    if (!this.dims) this.dims = fetchJSON('site-data/image_dims.json').catch(() => ({}));
    return this.dims;
  }
  /* {manifestId, canvases: [{id, width, height, label}], ia: item name or null} for a print */
  async manifest(volume) {
    if (this.manifests[volume]) return this.manifests[volume];
    const entry = this.iiif[volume];
    if (!entry) throw new Error(`no IIIF source for ${volume}`);
    const ia = (entry.manifest.match(/iiif\.archive\.org\/iiif\/\d\/([^/]+)\/manifest/) || [])[1];
    const p = (async () => {
      if (ia) {
        const dims = (await this._dims())[volume] || {};
        const stems = Object.keys(dims).sort((a, b) => parseInt(a.replace(/\D/g, ''), 10) - parseInt(b.replace(/\D/g, ''), 10));
        const base = new URL(`iiif/${volume}`, location.href.replace(/[^/]*$/, '')).href;
        const canvases = stems.map(stem => {
          const d = dims[stem]; const idx = this._index(entry, stem);
          return { id: `${base}/canvas/${stem}`, width: d.w, height: d.h, label: `p. ${stem.replace(/^page_/, '')}`, stem, idx,
                   image: `https://archive.org/download/${ia}/page/n${idx}.jpg` };
        });
        const json = { '@context': 'http://iiif.io/api/presentation/3/context.json', id: `${base}/manifest`, type: 'Manifest',
          label: { none: [volume] }, requiredStatement: { label: { none: ['Source'] }, value: { none: [entry.source || ''] } },
          items: canvases.map(c => ({ id: c.id, type: 'Canvas', width: c.width, height: c.height, label: { none: [c.label] },
            items: [{ id: `${c.id}/page`, type: 'AnnotationPage', items: [{ id: `${c.id}/image`, type: 'Annotation', motivation: 'painting', target: c.id,
              body: { id: c.image, type: 'Image', format: 'image/jpeg', width: c.width, height: c.height } }] }] })) };
        this._dispatch(Mirador.receiveManifest(json.id, json));
        return { manifestId: json.id, canvases, ia, byStem: Object.fromEntries(canvases.map(c => [c.stem, c])) };
      }
      const json = await fetchJSON(entry.manifest);
      const items = json.items || ((json.sequences || [{}])[0].canvases) || [];
      const canvases = items.map(c => ({ id: c.id || c['@id'], width: c.width, height: c.height,
        label: typeof c.label === 'string' ? c.label : Object.values(c.label || {}).flat().join(' ') }));
      const id = json.id || json['@id'];
      this._dispatch(Mirador.receiveManifest(id, json));
      return { manifestId: id, canvases, ia: null, byStem: null };
    })();
    this.manifests[volume] = p.catch(e => { delete this.manifests[volume]; throw e; });
    return this.manifests[volume];
  }
  _index(entry, stem) {
    const n = parseInt(stem.replace(/\D/g, ''), 10);
    return entry.canvas_map && entry.canvas_map[stem] != null ? entry.canvas_map[stem] : (n - 1) + (entry.canvas_offset || 0);
  }
  _viewerSize() {
    const r = this.el.getBoundingClientRect();
    return { w: Math.max(200, r.width), h: Math.max(200, r.height - 60) };     // minus Mirador's navigation bar
  }
  _fit(canvas, rect) {       // rect in canvas coordinates [x, y, w, h]; Mirador viewport: centre in canvas px, zoom = 1 / visible width
    const { w: vw, h: vh } = this._viewerSize();
    const [x, y, w, h] = rect;
    const bw = Math.max(w, h * vw / vh) * 1.04;
    const viewH = bw * vh / vw;
    const cy = h > viewH ? y + viewH / 2 : y + h / 2;
    this._dispatch(Mirador.updateViewport(this.windowId, { x: x + w / 2, y: cy, zoom: 1 / bw }));
  }
  /* Show one page; `page` = {page: stem, w, h, regions: [[x,y,w,h], …]} from index.json. */
  async show(volume, author, page, { highlight = true } = {}) {
    this._ensure();
    const entry = this.iiif[volume];
    const man = await this.manifest(volume);
    const canvas = man.byStem ? man.byStem[page.page] : man.canvases[this._index(entry, page.page)];
    if (!canvas) throw new Error(`no canvas for page ${page.page}`);
    const state = this.mirador.store.getState();
    if (!state.windows[this.windowId]) {
      this._dispatch(Mirador.addWindow({ id: this.windowId, manifestId: man.manifestId, canvasId: canvas.id }));
    } else if (state.windows[this.windowId].manifestId !== man.manifestId) {
      this._dispatch(Mirador.updateWindow(this.windowId, { manifestId: man.manifestId, canvasId: canvas.id }));
    } else if (this.current !== canvas.id) {
      this._dispatch(Mirador.setCanvas(this.windowId, canvas.id));
    }
    this.current = canvas.id; this.page = { canvas, page };
    const sx = page.w && canvas.width ? canvas.width / page.w : 1;          // PageXML px -> canvas px
    const regions = highlight ? (page.regions || []).map(r => r.map(v => Math.round(v * sx))) : [];
    const ap = { id: `${canvas.id}/consilium-regions`, type: 'AnnotationPage',
      items: regions.map(([x, y, w, h], i) => ({ id: `${canvas.id}/consilium-regions/${i}`, type: 'Annotation', motivation: 'commenting',
        body: { type: 'TextualBody', value: 'text of this consilium', format: 'text/plain' }, target: `${canvas.id}#xywh=${x},${y},${w},${h}` })) };
    this._dispatch(Mirador.receiveAnnotation(canvas.id, ap.id, ap));
    if (regions.length) {
      // the page width with the first region's top at the top of the view (as the reader scrolls from the start)
      const r = regions[0];
      setTimeout(() => this._fit(canvas, [0, r[1], canvas.width, r[3]]), 60);
    } else {
      setTimeout(() => this.home(), 60);
    }
    const attribution = entry.source || (this.images[volume] || {}).attribution || '';
    const link = man.ia ? `https://archive.org/details/${man.ia}/page/n${canvas.idx}/mode/1up` : canvas.id;
    return { kind: 'iiif', label: canvas.label || '', link, attribution };
  }
  zoom(f) {
    if (!this.mirador) return;
    const v = this.mirador.store.getState().viewers[this.windowId];
    if (v && v.zoom) this._dispatch(Mirador.updateViewport(this.windowId, { zoom: v.zoom * f }));
  }
  home() {
    if (!this.mirador || !this.page) return;
    const c = this.page.canvas;
    this._fit(c, [0, 0, c.width, c.height]);
  }
}
