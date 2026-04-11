/*! coi-serviceworker - enables crossOriginIsolated on hosts without COOP/COEP headers */
if (typeof window === 'undefined') {
  // Service worker context
  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

  self.addEventListener("fetch", (e) => {
    if (e.request.cache === "only-if-cached" && e.request.mode !== "same-origin") return;

    e.respondWith(
      fetch(e.request).then((res) => {
        if (res.status === 0) return res;

        const headers = new Headers(res.headers);
        headers.set("Cross-Origin-Embedder-Policy", "credentialless");
        headers.set("Cross-Origin-Opener-Policy", "same-origin");

        return new Response(res.body, {
          status: res.status,
          statusText: res.statusText,
          headers,
        });
      }).catch((err) => console.error(err))
    );
  });
} else {
  // Main thread: register SW, reload once it takes control
  (async () => {
    if (!("serviceWorker" in navigator)) return;

    const registration = await navigator.serviceWorker.register(
      new URL("coi-serviceworker.js", document.currentScript.src)
    );

    // If already controlling, nothing to do
    if (navigator.serviceWorker.controller) return;

    // Wait for the SW to become active
    const sw = registration.installing || registration.waiting || registration.active;
    if (!sw) return;

    if (sw.state === "activated") {
      window.location.reload();
    } else {
      sw.addEventListener("statechange", () => {
        if (sw.state === "activated") window.location.reload();
      });
    }
  })();
}
