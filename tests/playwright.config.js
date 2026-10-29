// Run (Playwright is not a project dependency):
//   npx -y -p @playwright/test playwright test -c tests/playwright.config.js
// or, with a local install: NODE_PATH=<dir>/node_modules <dir>/node_modules/.bin/playwright test -c tests/playwright.config.js
// the preview password gate (assets/gate.js): the tests enter as if the password had been typed once
const fs = require('fs'), path = require('path');
const gate = (fs.readFileSync(path.join(__dirname, '..', 'assets', 'gate-config.js'), 'utf8').match(/"hash": "([0-9a-f]+)"/) || [])[1];
const storageState = { cookies: [], origins: gate ? [{ origin: 'http://localhost:8765', localStorage: [{ name: 'consilia-gate', value: gate }] }] : [] };

module.exports = {
  testDir: '.',
  testMatch: ['smoke.spec.js', 'layers.spec.js', 'models.spec.js'],
  timeout: 60000,
  reporter: 'list',
  webServer: [
    { command: 'python3 -m http.server 8765', cwd: '..', port: 8765, reuseExistingServer: true },
    // local model files for tests/models.spec.js (Hugging Face requests are redirected here)
    { command: 'python3 tests/model_server.py', cwd: '..', url: 'https://127.0.0.1:8766/', ignoreHTTPSErrors: true, reuseExistingServer: true },
  ],
  use: {
    baseURL: 'http://localhost:8765',
    storageState,
    ignoreHTTPSErrors: true,          // only the local model server (tests/model_server.py) uses a self-signed certificate
    // optional: PW_CHROMIUM=/path/to/chrome to use a browser Playwright did not download itself
    launchOptions: process.env.PW_CHROMIUM ? { executablePath: process.env.PW_CHROMIUM } : {},
  },
};
