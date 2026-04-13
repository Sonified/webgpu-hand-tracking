#!/usr/bin/env python3
"""
Headless WebGPU engine test via Playwright + Chromium.
Much faster than Puppeteer for quick iteration.

Usage: python3 engine/run-test.py
"""

import subprocess, sys, os, threading, http.server, socketserver

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 9333

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)
    def end_headers(self):
        self.send_header('Cross-Origin-Embedder-Policy', 'credentialless')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        super().end_headers()
    def log_message(self, *a):
        pass  # silence request logs

# Start server in background thread
server = socketserver.TCPServer(('', PORT), Handler)
t = threading.Thread(target=server.serve_forever, daemon=True)
t.start()

# Find Chrome
CHROME_PATHS = [
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
]
chrome = next((p for p in CHROME_PATHS if os.path.exists(p)), None)
if not chrome:
    print("Chrome not found"); sys.exit(1)

# Run headless Chrome, dump console to stdout
# --dump-dom won't work for JS. Use --headless=new with remote debugging.
# Simplest: use Chrome's --headless --print-to-pdf? No.
# Best: use a tiny JS snippet that polls for completion.

import json, urllib.request, time

# Launch headless Chrome with WebGPU
proc = subprocess.Popen([
    chrome,
    '--headless=new',
    '--no-sandbox',
    '--disable-gpu-sandbox',
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan,UseSkiaRenderer',
    '--remote-debugging-port=9334',
    '--disable-extensions',
    '--remote-allow-origins=*',
    f'--user-data-dir=/tmp/chrome-webgpu-test-{os.getpid()}',
    'about:blank',
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Wait for DevTools
for _ in range(30):
    try:
        urllib.request.urlopen('http://localhost:9334/json/version')
        break
    except:
        time.sleep(0.2)
else:
    print("Chrome didn't start"); proc.kill(); sys.exit(1)

# Get the WebSocket URL for CDP
resp = json.loads(urllib.request.urlopen('http://localhost:9334/json/list').read())
ws_url = resp[0]['webSocketDebuggerUrl'] if resp else None

# Use CDP via websocket to navigate and capture console
import websocket

ws = websocket.create_connection(ws_url)
msg_id = 0

def send_cdp(method, params=None):
    global msg_id
    msg_id += 1
    ws.send(json.dumps({'id': msg_id, 'method': method, 'params': params or {}}))
    return msg_id

def recv_until(target_id=None, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        ws.settimeout(max(0.1, deadline - time.time()))
        try:
            msg = json.loads(ws.recv())
            yield msg
            if target_id and msg.get('id') == target_id:
                return
        except websocket.WebSocketTimeoutException:
            continue

# Enable console
send_cdp('Runtime.enable')
send_cdp('Console.enable')

# Navigate
nav_id = send_cdp('Page.navigate', {'url': f'http://localhost:{PORT}/engine/test.html'})

# Collect console messages until test completes
print(f"\nRunning test from http://localhost:{PORT}/engine/test.html\n")
done = False
start = time.time()

for msg in recv_until(timeout=120):
    # Console message
    if msg.get('method') == 'Runtime.consoleAPICalled':
        args = msg['params'].get('args', [])
        text = ' '.join(a.get('value', str(a.get('description', ''))) for a in args)
        if text:
            # Color output
            if '✅' in text:
                print(f'\033[32m{text}\033[0m')
            elif '❌' in text:
                print(f'\033[31m{text}\033[0m')
            elif 'error' in text.lower() or 'diverge' in text.lower():
                print(f'\033[31m{text}\033[0m')
            else:
                print(text)
            if 'test complete' in text.lower() or 'ERROR' in text:
                done = True
                break

elapsed = time.time() - start
print(f'\nCompleted in {elapsed:.1f}s')

ws.close()
proc.kill()
server.shutdown()
