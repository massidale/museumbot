// Player delle guide e controllo della cache dei media.
// L'audio si legge dalla Cache API come blob e si suona da un object URL:
// Safari non serve le richieste a intervalli (Range) da una risposta intera in cache.

const DEFAULT_LABELS = {play: 'Play', pause: 'Pause', back10: '-10 s', seek: 'Position', error: 'Error'};
const BACK = 10;      // s: il tasto indietro
const MAX_STEP = 2;   // s: fra due timeupdate di un ascolto continuo passa meno di cosi'

let playing = null;  // l'audio che sta suonando: ne suona uno alla volta

// ---- decisioni pure, testate ----

// Secondi ascoltati fra due timeupdate: i salti (avanti o indietro) non contano.
export const heardStep = (prev, t, seeking) => {
  const d = t - prev;
  return !seeking && d > 0 && d <= MAX_STEP ? d : 0;
};

// Indietro di BACK secondi, non prima dell'inizio.
export const backTo = t => Math.max(0, (Number.isFinite(t) ? t : 0) - BACK);

// m:ss; '--:--' se la durata non e' ancora nota.
export function fmtTime(sec) {
  if (!Number.isFinite(sec) || sec < 0) return '--:--';
  const s = Math.floor(sec);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

// Ogni immagine e ogni audio del manifesto, senza doppioni.
export function mediaUrls(manifest) {
  const out = new Set();
  for (const art of manifest.artworks || []) {
    if (art.image) out.add(art.image);
    for (const lang of Object.keys(art.audio || {}))
      for (const c of Object.keys(art.audio[lang])) out.add(art.audio[lang][c].src);
  }
  return [...out];
}

// Nome della cache dei media per una versione del manifesto (lo stesso di sw.js).
export const mediaCacheName = version => `rq3-media-${version}`;

// Cerca u solo nella cache della versione: un'altra versione puo' avere gli stessi
// percorsi con audio diversi.
async function matchMedia(u, version) {
  const name = mediaCacheName(version);
  if (!(await caches.has(name))) return undefined;
  return (await caches.open(name)).match(u);
}

// Quali media della versione del manifesto mancano dalla cache e quanti byte occupano gli altri.
export async function checkCache(manifest) {
  const missing = [];
  let bytes = 0;
  for (const u of mediaUrls(manifest)) {
    const r = await matchMedia(u, manifest.version);
    if (!r) { missing.push(u); continue; }
    const len = Number(r.headers.get('content-length'));
    bytes += Number.isFinite(len) && len > 0 ? len : (await r.blob()).size;
  }
  return {missing, bytes};
}

// Dalla cache della versione; se manca (o la versione non e' data), dalla rete,
// passando dal service worker, che usa la stessa cache.
async function fetchBlob(src, version) {
  const r = (version != null && (await matchMedia(src, version))) || (await fetch(src));
  if (!r.ok) throw new Error(`${src}: HTTP ${r.status}`);
  const blob = await r.blob();
  return blob.type ? blob : new Blob([blob], {type: 'audio/mpeg'});
}

// Blob gia' richiesti in anticipo (src -> promessa): si usano una volta sola.
const prefetched = new Map();

// Scarica in anticipo i file (le guide dell'opera dopo); un errore qui non conta:
// la voce sparisce e loadBlob ripiega sulla rete.
export function prefetch(srcs, version) {
  for (const src of srcs) {
    if (prefetched.has(src)) continue;
    const p = fetchBlob(src, version);
    prefetched.set(src, p);
    p.catch(() => { if (prefetched.get(src) === p) prefetched.delete(src); });
  }
}

// Usa prima il blob richiesto in anticipo (e lo toglie dalla memoria).
export function loadBlob(src, version) {
  const p = prefetched.get(src);
  if (p) {
    prefetched.delete(src);
    return p.catch(() => fetchBlob(src, version));
  }
  return fetchBlob(src, version);
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

// state = {plays, ended, heard}: plays conta gli avvii riusciti, ended resta vero dopo la
// prima volta che si arriva alla fine, heard sono i secondi ascoltati (senza i salti).
// version: versione del manifesto, per leggere l'audio dalla sua cache.
// onChange(state) a ogni avvio, pausa e fine; stats() da' lo stato in ogni momento.
// onError({src, message}): file non caricato o non riproducibile; il visitatore vede solo
// labels.error, mai il percorso (che contiene il nome della condizione).
export function createPlayer(container, src, label, onChange = () => {}, labels = {}, version = null,
                             onError = () => {}) {
  const L = {...DEFAULT_LABELS, ...labels};
  const state = {plays: 0, ended: false, heard: 0};
  let ready = false, destroyed = false, url = null, failed = false;
  let last = 0, dragging = false;

  const audio = new Audio();
  audio.preload = 'auto';

  const root = el('div', 'player');
  const title = el('div', 'player-label', label);
  const btnPlay = el('button', 'player-play', L.play);
  const btnBack = el('button', 'player-back secondary', L.back10);
  const time = el('div', 'player-time', `${fmtTime(0)} / --:--`);
  btnPlay.type = btnBack.type = 'button';
  const seek = el('input', 'player-seek');
  Object.assign(seek, {type: 'range', min: '0', max: '0', step: '0.1', value: '0'});
  seek.setAttribute('aria-label', L.seek);
  const controls = el('div', 'player-controls');
  controls.append(btnPlay, btnBack, time);
  root.append(title, controls, seek);
  container.append(root);

  const stats = () => ({...state, heard: Math.round(state.heard * 10) / 10});
  const notify = () => { if (!destroyed) onChange(stats()); };  // non dopo destroy (pausa finale)
  const refresh = () => {
    const can = ready && !destroyed && !failed;
    btnPlay.disabled = btnBack.disabled = seek.disabled = !can;
    btnPlay.textContent = audio.paused ? L.play : L.pause;
    root.classList.toggle('player-disabled', !can);
    const dur = audio.duration;
    if (Number.isFinite(dur)) seek.max = String(dur);
    const t = dragging ? Number(seek.value) : audio.currentTime;
    if (!dragging) seek.value = String(t);
    time.textContent = `${fmtTime(t)} / ${fmtTime(dur)}`;
  };

  const start = () => {
    if (!ready || destroyed) return;
    if (playing && playing !== audio) playing.pause();
    // si conta solo un avvio riuscito
    audio.play().then(() => { state.plays++; notify(); })
      .catch(e => console.warn('play non riuscito', e));
  };
  const jump = t => { audio.currentTime = t; last = t; refresh(); };

  btnPlay.addEventListener('click', () => (audio.paused ? start() : audio.pause()));
  btnBack.addEventListener('click', () => jump(backTo(audio.currentTime)));
  // Mentre si trascina si mostra solo il punto; si salta quando si lascia il cursore.
  seek.addEventListener('input', () => { dragging = true; refresh(); });
  seek.addEventListener('change', () => { dragging = false; jump(Number(seek.value)); });

  audio.addEventListener('play', () => { playing = audio; last = audio.currentTime; refresh(); });
  audio.addEventListener('pause', () => { if (playing === audio) playing = null; refresh(); notify(); });
  audio.addEventListener('timeupdate', () => {
    const t = audio.currentTime;
    if (!audio.paused) state.heard += heardStep(last, t, audio.seeking);
    last = t;
    refresh();
  });
  audio.addEventListener('seeked', () => { last = audio.currentTime; });
  audio.addEventListener('loadedmetadata', refresh);
  audio.addEventListener('durationchange', refresh);
  audio.addEventListener('ended', () => {
    state.ended = true;
    refresh();
    notify();
  });
  // Dettagli in console e all'app (che li registra); al visitatore un testo neutro.
  const fail = message => {
    if (destroyed || failed) return;
    failed = true;
    audio.pause();
    console.error('player:', src, message);
    root.append(el('div', 'player-error', L.error));
    refresh();
    onError({src, message});
  };
  audio.addEventListener('error', () => {
    if (!url) return;  // nessun file ancora (o player chiuso)
    const e = audio.error;
    fail(e ? `MediaError ${e.code}${e.message ? `: ${e.message}` : ''}` : 'errore audio');
  });

  refresh();
  loadBlob(src, version).then(blob => {
    if (destroyed) return;
    url = URL.createObjectURL(blob);
    audio.src = url;
    audio.load();
    ready = true;
    refresh();
  }).catch(e => fail(String((e && e.message) || e)));

  return {
    destroy() {
      destroyed = true;
      audio.pause();
      audio.removeAttribute('src');
      audio.load();
      if (url) URL.revokeObjectURL(url);
      url = null;
      if (playing === audio) playing = null;
      root.remove();
    },
    stats,
  };
}
