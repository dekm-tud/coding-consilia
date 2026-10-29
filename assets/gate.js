/* Password gate for the preview phase: a first blocker against casual visitors and search engines, not access
   control. The data files stay reachable to anyone who requests them directly. The password itself is not stored;
   assets/gate-config.js (written by tools/set_site_password.py) holds a salted PBKDF2 hash, and after a correct
   entry the browser keeps that hash in localStorage. Without a configured hash the gate is off. */
(function () {
  const cfg = window.CONSILIA_GATE;
  if (!cfg || !cfg.hash) return;
  const KEY = 'consilia-gate';
  try { if (localStorage.getItem(KEY) === cfg.hash) return; } catch (e) { /* storage blocked: ask every time */ }
  document.documentElement.classList.add('gated');
  const css = document.createElement('style');
  css.textContent = 'html.gated body > *:not(#gate){display:none !important}' +
    '#gate{max-width:22rem;margin:18vh auto 0;padding:1.5rem;font:1rem/1.4 system-ui,sans-serif;color:#1c3d5a}' +
    '#gate h1{font-size:1.25rem;margin:0 0 .4rem}#gate p{margin:0 0 1rem;color:#555}' +
    '#gate input{width:100%;box-sizing:border-box;padding:.55rem .6rem;font:inherit;border:1px solid #bbb;border-radius:4px}' +
    '#gate button{margin-top:.7rem;padding:.5rem 1rem;font:inherit;border:0;border-radius:4px;background:#1c3d5a;color:#fff;cursor:pointer}' +
    '#gate .err{color:#a33;min-height:1.4em;margin-top:.5rem}';
  document.head.appendChild(css);
  async function derive(pw) {
    const enc = new TextEncoder();
    const key = await crypto.subtle.importKey('raw', enc.encode(pw), 'PBKDF2', false, ['deriveBits']);
    const bits = await crypto.subtle.deriveBits({ name: 'PBKDF2', hash: 'SHA-256', salt: enc.encode(cfg.salt), iterations: cfg.iterations }, key, 256);
    return Array.from(new Uint8Array(bits), b => b.toString(16).padStart(2, '0')).join('');
  }
  function show() {
    const box = document.createElement('form');
    box.id = 'gate';
    box.innerHTML = '<h1>Consilia</h1><p>Preview for colleagues. Please enter the password.</p>' +
      '<input type="password" autocomplete="current-password" aria-label="Password" autofocus>' +
      '<button type="submit">Enter</button><div class="err" role="alert"></div>';
    document.body.prepend(box);
    const input = box.querySelector('input');
    input.focus();
    box.addEventListener('submit', async ev => {
      ev.preventDefault();
      if (await derive(input.value) === cfg.hash) {
        try { localStorage.setItem(KEY, cfg.hash); } catch (e) { /* session only */ }
        box.remove(); document.documentElement.classList.remove('gated');
        window.dispatchEvent(new Event('resize'));          // the viewer and the map size themselves on resize
      } else {
        box.querySelector('.err').textContent = 'Wrong password.'; input.select();
      }
    });
  }
  if (document.body) show(); else document.addEventListener('DOMContentLoaded', show);
})();
