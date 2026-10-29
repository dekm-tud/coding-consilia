/* Query encoder in a module Web Worker (transformers.js, pinned). Runs the model of the picker
   entirely in the browser: no token, no server. Nothing is downloaded until the page sends 'load'
   (an explicit click).

   in:  {type:'load', cfg:{onnx_repo, onnx_revision, dtype:{wasm,webgpu}, pooling, normalize, query_prefix}}
        {type:'embed', id, text}
   out: {type:'progress', file, loaded, total}, {type:'ready', device, dtype}
        {type:'embedding', id, vec:Float32Array}, {type:'error', id?, message} */
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js';

env.allowLocalModels = false;
env.useBrowserCache = true;          // Cache API 'transformers-cache': weights are downloaded once

let extractor = null, cfg = null, runtime = null;

async function build(device, dtype) {
  return pipeline('feature-extraction', cfg.onnx_repo, {
    device, dtype, revision: cfg.onnx_revision || 'main',
    progress_callback: p => { if (p.status === 'progress' && p.total) postMessage({ type: 'progress', file: p.file, loaded: p.loaded, total: p.total }); },
  });
}

onmessage = async e => {
  const m = e.data;
  try {
    if (m.type === 'load') {
      cfg = m.cfg;
      let hasGpu = false;   // an adapter must really exist: navigator.gpu alone is present in browsers without a usable GPU
      try { hasGpu = !!(self.navigator && self.navigator.gpu && await self.navigator.gpu.requestAdapter()); } catch (err) { hasGpu = false; }
      const plans = [];
      if (hasGpu && cfg.dtype.webgpu) plans.push(['webgpu', cfg.dtype.webgpu]);
      if (cfg.dtype.wasm) plans.push(['wasm', cfg.dtype.wasm]);
      if (!plans.length) throw new Error('this model needs WebGPU, which this browser does not offer');
      let lastErr = null;
      for (const [device, dtype] of plans) {
        try { extractor = await build(device, dtype); runtime = { device, dtype }; break; }
        catch (err) { lastErr = err; extractor = null; }
      }
      if (!extractor) throw lastErr || new Error('model could not be loaded');
      postMessage({ type: 'ready', ...runtime });
    } else if (m.type === 'embed') {
      if (!extractor) throw new Error('model not loaded');
      const out = await extractor((cfg.query_prefix || '') + m.text, { pooling: cfg.pooling, normalize: cfg.normalize !== false });
      const vec = out.data instanceof Float32Array ? out.data : new Float32Array(out.data);
      postMessage({ type: 'embedding', id: m.id, vec }, [vec.buffer]);
    }
  } catch (err) {
    postMessage({ type: 'error', id: m.id, message: String((err && err.message) || err) });
  }
};
