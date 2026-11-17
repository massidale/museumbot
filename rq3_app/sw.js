// Service worker dell'app RQ3: guscio dalla rete se c'e' (cache come riserva),
// immagini e audio sempre dalla cache, cosi' l'app funziona offline sull'iPad.

// File del guscio, precaricati all'install. I task successivi aggiungono qui i loro file.
const SHELL_FILES = [
  './',
  'index.html',
  'style.css',
  'app.js',
  'logic.js',
  'player.js',
  'config.js',
  'store.js',
  'swclient.js',
  'operator.js',
  'texts.json',
  'manifest.json',
  'app.webmanifest',
  'icon-512.png',
];

const SHELL_CACHE = 'rq3-shell';
const MEDIA_PREFIX = 'rq3-media-';
const NET_TIMEOUT_MS = 4000;  // oltre, il guscio viene dalla cache (wifi lento)

const abs = path => new URL(path, self.registration.scope).href;

// Ogni immagine e ogni audio del manifesto, senza doppioni.
// Uguale a mediaUrls di player.js: il worker non importa moduli.
function mediaUrls(manifest) {
  const out = new Set();
  for (const art of manifest.artworks || []) {
    if (art.image) out.add(art.image);
    for (const lang of Object.keys(art.audio || {}))
      for (const c of Object.keys(art.audio[lang])) out.add(art.audio[lang][c].src);
  }
  return [...out];
}

// Manifesto dalla cache del guscio (salvato all'install o all'ultimo accesso in rete).
async function cachedManifest() {
  const r = await caches.match(abs('manifest.json'), {cacheName: SHELL_CACHE});
  return r ? r.json() : null;
}

const mediaCacheName = m => MEDIA_PREFIX + m.version;

// Scarica solo i file che mancano: se si interrompe, al giro dopo riparte da li'.
async function fillMedia(manifest) {
  const cache = await caches.open(mediaCacheName(manifest));
  const todo = [];
  for (const u of mediaUrls(manifest)) if (!(await cache.match(abs(u)))) todo.push(u);
  const worker = async () => {
    while (todo.length) {
      const u = todo.shift();
      await cache.add(new Request(abs(u), {cache: 'reload'}));
    }
  };
  await Promise.all([worker(), worker(), worker(), worker()]);
}

self.addEventListener('install', event => {
  event.waitUntil((async () => {
    const shell = await caches.open(SHELL_CACHE);
    await shell.addAll(SHELL_FILES.map(f => new Request(abs(f), {cache: 'reload'})));
    await fillMedia(await cachedManifest());
    await self.skipWaiting();
  })());
});

// Cancella le cache dei media di altre versioni del manifesto.
async function pruneMedia(manifest) {
  const keep = manifest ? mediaCacheName(manifest) : null;
  for (const k of await caches.keys())
    if (k.startsWith(MEDIA_PREFIX) && k !== keep) await caches.delete(k);
}

self.addEventListener('activate', event => {
  event.waitUntil((async () => {
    await pruneMedia(await cachedManifest());
    await self.clients.claim();
  })());
});

// La pagina chiede di completare la cache (dopo un install interrotto o un nuovo manifesto).
self.addEventListener('message', event => {
  if (!event.data || event.data.type !== 'fill') return;
  event.waitUntil((async () => {
    let ok = true, error = null;
    try {
      const r = await fetch(abs('manifest.json'), {cache: 'no-cache'});
      const m = await r.clone().json();
      await (await caches.open(SHELL_CACHE)).put(abs('manifest.json'), r);
      await fillMedia(m);
      await pruneMedia(m);
    } catch (e) { ok = false; error = String(e); }
    if (event.source) event.source.postMessage({type: 'filled', ok, error});
  })());
});

// Media: cache-first, ma solo nella cache della versione corrente del manifesto
// (gli stessi percorsi possono avere audio diversi in un'altra versione).
async function mediaFirst(req) {
  const m = await cachedManifest();
  if (!m) return fetch(req);
  const cache = await caches.open(mediaCacheName(m));
  const hit = await cache.match(req, {ignoreSearch: true});
  if (hit) return hit;
  const r = await fetch(req);
  if (r.ok && r.status === 200) await cache.put(req, r.clone());
  return r;
}

// Guscio: rete (con timeout) e copia in cache; senza rete, la copia in cache.
async function networkFirst(event) {
  const req = event.request;
  const cache = await caches.open(SHELL_CACHE);
  const net = fetch(req, {cache: 'no-cache'}).then(r => {
    if (r.ok) cache.put(req, r.clone());
    return r;
  });
  net.catch(() => {});  // se vince la cache, l'errore di rete non serve
  const fromCache = async () => {
    const hit = await cache.match(req, {ignoreSearch: true});
    if (hit) return hit;
    if (req.mode === 'navigate') return cache.match(abs('index.html'));
    return undefined;
  };
  const timeout = new Promise(res => setTimeout(() => res('timeout'), NET_TIMEOUT_MS));
  try {
    const first = await Promise.race([net, timeout]);
    if (first !== 'timeout') return first;
    return (await fromCache()) || (await net);
  } catch (e) {
    const hit = await fromCache();
    if (hit) return hit;
    throw e;
  }
}

self.addEventListener('fetch', event => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  const scope = new URL(self.registration.scope);
  const rel = url.pathname.slice(scope.pathname.length);
  if (rel.startsWith('audio/') || rel.startsWith('img/')) event.respondWith(mediaFirst(req));
  else event.respondWith(networkFirst(event));
});
