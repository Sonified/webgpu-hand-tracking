#!/usr/bin/env node
/**
 * Headless browser test runner for the WebGPU Vision engine.
 * Launches Chrome with WebGPU enabled, loads the test page, captures console output.
 *
 * Usage: node engine/run-test.mjs
 */

import puppeteer from 'puppeteer';
import { createServer } from 'http';
import { readFileSync, existsSync } from 'fs';
import { join, extname } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { homedir } from 'os';
import { printBanner } from './session-timer.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');

// Simple static file server
const MIME = {
  '.html': 'text/html', '.js': 'application/javascript', '.mjs': 'application/javascript',
  '.wgsl': 'text/plain', '.json': 'application/json', '.bin': 'application/octet-stream',
  '.wasm': 'application/wasm', '.onnx': 'application/octet-stream', '.css': 'text/css',
};

const server = createServer((req, res) => {
  const urlPath = decodeURIComponent(req.url.split('?')[0]);

  // Virtual route: serve ~/.session-timer/ files for browser access
  if (urlPath.startsWith('/.session-timer/')) {
    const f = join(homedir(), urlPath);
    res.setHeader('Content-Type', 'text/plain');
    if (existsSync(f)) { res.writeHead(200); res.end(readFileSync(f, 'utf8')); }
    else { res.writeHead(404); res.end(''); }
    return;
  }

  let path = join(ROOT, urlPath);
  if (path.endsWith('/')) path += 'index.html';

  if (!existsSync(path)) {
    res.writeHead(404);
    res.end('Not found');
    return;
  }

  const ext = extname(path);
  const mime = MIME[ext] || 'application/octet-stream';

  // Add COEP/COOP headers for crossOriginIsolated
  res.setHeader('Cross-Origin-Embedder-Policy', 'credentialless');
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Content-Type', mime);

  res.writeHead(200);
  res.end(readFileSync(path));
});

const PORT = 9222;

async function run() {
  await printBanner();

  // Start server
  await new Promise(r => server.listen(PORT, r));
  console.log(`Server running on http://localhost:${PORT}`);

  // Launch Chrome with WebGPU
  const browser = await puppeteer.launch({
    headless: 'new',
    args: [
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan,UseSkiaRenderer',
      '--disable-gpu-sandbox',
      '--no-sandbox',
    ],
  });

  const page = await browser.newPage();

  // Capture all console output
  const logs = [];
  page.on('console', msg => {
    const text = msg.text();
    logs.push(text);
    // Print with color based on type
    const type = msg.type();
    if (type === 'error') console.log('\x1b[31m' + text + '\x1b[0m');
    else if (type === 'warning') console.log('\x1b[33m' + text + '\x1b[0m');
    else if (text.includes('✅')) console.log('\x1b[32m' + text + '\x1b[0m');
    else if (text.includes('❌')) console.log('\x1b[31m' + text + '\x1b[0m');
    else console.log(text);
  });

  page.on('pageerror', err => {
    console.log('\x1b[31mPAGE ERROR: ' + err.message + '\x1b[0m');
  });

  // Navigate to test page
  console.log(`\nLoading http://localhost:${PORT}/engine/test.html ...\n`);
  await page.goto(`http://localhost:${PORT}/engine/test.html`, {
    waitUntil: 'networkidle0',
    timeout: 60000,
  });

  // Wait for "Full pipeline test complete" or error
  await page.waitForFunction(
    () => document.getElementById('log')?.textContent?.includes('test complete') ||
          document.getElementById('log')?.textContent?.includes('ERROR'),
    { timeout: 120000 }
  );

  // Small delay for any final logs
  await new Promise(r => setTimeout(r, 1000));

  // Check results
  const hasMatch = logs.some(l => l.includes('MATCH'));
  const hasError = logs.some(l => l.includes('Large divergence') || l.includes('ERROR'));

  console.log('\n' + '='.repeat(50));
  if (hasMatch) {
    console.log('\x1b[32m✅ ENGINE OUTPUT MATCHES ORT\x1b[0m');
  } else if (hasError) {
    console.log('\x1b[31m❌ ENGINE OUTPUT DOES NOT MATCH\x1b[0m');
  } else {
    console.log('\x1b[33m⚠ Test completed but no match/error verdict found\x1b[0m');
  }
  console.log('='.repeat(50));

  await printBanner();

  await browser.close();
  server.close();
  process.exit(hasMatch ? 0 : 1);
}

run().catch(err => {
  console.error(err);
  server.close();
  process.exit(1);
});
