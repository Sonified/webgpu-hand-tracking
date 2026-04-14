#!/usr/bin/env node
/** Run engine/bench.html headless and print results. Usage: node engine/bench.mjs */
import puppeteer from 'puppeteer';
import { createServer } from 'http';
import { readFileSync, existsSync } from 'fs';
import { join, extname } from 'path';
import { fileURLToPath } from 'url';
import { homedir } from 'os';
import { printBanner } from './session-timer.mjs';
const ROOT = join(fileURLToPath(import.meta.url), '../..');
const MIME = {'.html':'text/html','.js':'application/javascript','.mjs':'application/javascript','.wgsl':'text/plain','.json':'application/json','.bin':'application/octet-stream','.wasm':'application/wasm','.onnx':'application/octet-stream'};
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
  let p = join(ROOT, urlPath);
  if (p.endsWith('/')) p += 'index.html';
  if (!existsSync(p)) { res.writeHead(404); res.end(); return; }
  res.setHeader('Cross-Origin-Embedder-Policy', 'credentialless');
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.writeHead(200, { 'Content-Type': MIME[extname(p)] || 'application/octet-stream' });
  res.end(readFileSync(p));
});
await printBanner();
server.listen(9444, async () => {
  const browser = await puppeteer.launch({ headless: 'new', args: ['--enable-unsafe-webgpu','--enable-features=Vulkan','--no-sandbox','--disable-gpu-sandbox'] });
  const page = await browser.newPage();
  page.on('console', m => console.log(m.text()));
  await page.goto('http://localhost:9444/engine/bench.html', { waitUntil: 'networkidle0', timeout: 60000 });
  await page.waitForFunction(() => document.getElementById('log')?.textContent?.includes('done'), { timeout: 120000 });
  await new Promise(r => setTimeout(r, 500));
  await printBanner();
  await browser.close();
  server.close();
});
