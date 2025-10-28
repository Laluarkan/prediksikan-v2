// serviceworker.js

const CACHE_NAME = "prediksi-kan-cache-v1";
const urlsToCache = [
  "/",
  "/index/",
  "/static/css/style.css",
  "/static/css/home.css",
  "/static/css/index.css",
  "/static/images/prediksikan-logo1.png",
];

// Saat instalasi: cache file statis
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(urlsToCache);
    })
  );
  self.skipWaiting();
});

// Saat aktivasi: hapus cache lama
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((name) => {
          if (name !== CACHE_NAME) {
            return caches.delete(name);
          }
        })
      );
    })
  );
  self.clients.claim();
});

// Saat fetch: hanya intercept request ke domain sendiri
self.addEventListener("fetch", (event) => {
  const reqUrl = new URL(event.request.url);

  // Batasi hanya untuk domain ini (hindari phishing detection)
  if (reqUrl.origin !== self.location.origin) {
    return; // Biarkan request eksternal lewat langsung
  }

  event.respondWith(
    caches.match(event.request).then((response) => {
      // Ambil dari cache kalau ada, jika tidak ambil dari jaringan
      return (
        response ||
        fetch(event.request).catch(() => {
          // Jika offline dan tidak ada di cache, kembalikan halaman fallback
          if (event.request.mode === "navigate") {
            return caches.match("/");
          }
        })
      );
    })
  );
});
