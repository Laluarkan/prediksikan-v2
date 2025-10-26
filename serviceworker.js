// serviceworker.js (di root proyek)

// Tentukan versi cache Anda
var staticCacheName = "django-pwa-v1";

// Daftar file yang akan di-cache saat instalasi
var filesToCache = [
    '/',
    '/index/',
    '/static/css/style.css',
    '/static/css/home.css',
    '/static/css/index.css',
    '/static/images/prediksikan-logo.png', // Tambahkan ikon utama Anda
    // Tambahkan URL file JS penting lainnya jika ada
];

// Event Install: Menyimpan aset statis
self.addEventListener('install', function (e) {
    e.waitUntil(
        caches.open(staticCacheName).then(function (cache) {
            return cache.addAll(filesToCache);
        })
    );
});

// Event Activate: Membersihkan cache lama
self.addEventListener('activate', function (e) {
    e.waitUntil(
        caches.keys().then(function (cacheNames) {
            return Promise.all(
                cacheNames.filter(function (cacheName) {
                    return cacheName.startsWith('django-pwa-') && cacheName !== staticCacheName;
                }).map(function (cacheName) {
                    return caches.delete(cacheName);
                })
            );
        })
    );
});

// Event Fetch: Mengambil dari cache jika ada, jika tidak, ambil dari jaringan
self.addEventListener('fetch', function (e) {
    // Abaikan permintaan yang bukan HTTP/HTTPS (misal ekstensi Chrome)
    if (!e.request.url.startsWith('http')) {
        return;
    }

    e.respondWith(
        caches.match(e.request).then(function (response) {
            return response || fetch(e.request);
        })
    );
});