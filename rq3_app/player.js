// Player delle guide e controllo della cache dei media.
// L'audio si legge dalla Cache API come blob e si suona da un object URL:
// Safari non serve le richieste a intervalli (Range) da una risposta intera in cache.

const DEFAULT_LABELS = {play: 'Play', pause: 'Pause', restart: 'Restart', error: 'Error'};
const SEEK_SLACK = 0.5;   // s: un salto avanti entro questa tolleranza e' ammesso
const END_SLACK = 1;      // s: la fine conta se si e' ascoltato fino a qui dalla fine

let playing = null;  // l'audio che sta suonando: ne suona uno alla volta

// ---- niente salti avanti (decisioni pure, testate) ----

// Dove riportare un salto: null se e' lecito (indietro o entro SEEK_SLACK dal punto raggiunto).
export const seekTarget = (t, maxReached, slack = SEEK_SLACK) => (t > maxReached + slack ? maxReached : null);

// Punto piu' avanti raggiunto ascoltando: non cresce durante un salto.
export const reached = (maxReached, t, seeking) => (seeking ? maxReached : Math.max(maxReached, t));

// La fine vale come ascolto completo solo se ci si e' arrivati ascoltando.
export const endedCounts = (maxReached, duration) =>
  Number.isFinite(duration) && maxReached >= duration - END_SLACK;

// Comandi di Centro di controllo, schermata di blocco e cuffie: avanti e traccia
// successiva/precedente non fanno niente. Le azioni non supportate lanciano un errore.
let sessionLocked = false;
function lockMediaSession() {
  if (sessionLocked || typeof navigator === 'undefined' || !navigator.mediaSession) return;
  sessionLocked = true;
  for (const a of ['seekto', 'seekforward', 'nexttrack', 'previoustrack']) {
    try { navigator.mediaSession.setActionHandler(a, () => {}); } catch (e) { /* non supportata */ }
  }
}

// m:ss; '--:--' se la durata non e' ancora nota.
export function fmtTime(sec) {
  if (!Number.isFinite(sec) || sec < 0) return '--:--';
  const s = Math.floor(sec);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

// Ogni immagine e ogni audio del manifesto (prove del volume comprese), senza doppioni.
export function mediaUrls(manifest) {
  const out = new Set();
  for (const s of Object.values(manifest.sample || {})) if (s && s.src) out.add(s.src);
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
export async function loadBlob(src, version) {
  const r = (version != null && (await matchMedia(src, version))) || (await fetch(src));
  if (!r.ok) throw new Error(`${src}: HTTP ${r.status}`);
  const blob = await r.blob();
  return blob.type ? blob : new Blob([blob], {type: 'audio/mpeg'});
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

// state = {plays, ended}: plays conta gli avvii da capo, ended resta vero dopo la prima fine
// raggiunta ascoltando (non saltando avanti).
// version: versione del manifesto, per leggere l'audio dalla sua cache.
// onError({src, message}): file non caricato o non riproducibile; il visitatore vede solo
// labels.error, mai il percorso (che contiene il nome della condizione).
export function createPlayer(container, src, label, onChange = () => {}, labels = {}, version = null,
                             onError = () => {}) {
  const L = {...DEFAULT_LABELS, ...labels};
  const state = {plays: 0, ended: false};
  let enabled = true, ready = false, destroyed = false, fromStart = true, url = null;
  let maxReached = 0, failed = false;

  lockMediaSession();
  const audio = new Audio();
  audio.preload = 'auto';

  const root = el('div', 'player');
  const title = el('div', 'player-label', label);
  const btnPlay = el('button', 'player-play', L.play);
  const btnRestart = el('button', 'player-restart', L.restart);
  const time = el('div', 'player-time', `${fmtTime(0)} / --:--`);
  btnPlay.type = btnRestart.type = 'button';
  const controls = el('div', 'player-controls');
  controls.append(btnPlay, btnRestart, time);
  root.append(title, controls);
  container.append(root);

  const notify = () => onChange({...state});
  const refresh = () => {
    const can = ready && enabled && !destroyed && !failed;
    btnPlay.disabled = btnRestart.disabled = !can;
    btnPlay.textContent = audio.paused ? L.play : L.pause;
    root.classList.toggle('player-disabled', !can);
    time.textContent = `${fmtTime(audio.currentTime)} / ${fmtTime(audio.duration)}`;
  };

  const start = () => {
    if (!ready || !enabled || destroyed) return;
    if (playing && playing !== audio) playing.pause();
    // si conta solo un avvio riuscito
    audio.play().then(() => {
      if (fromStart) { state.plays++; fromStart = false; notify(); }
    }).catch(e => console.warn('play non riuscito', e));
  };

  btnPlay.addEventListener('click', () => (audio.paused ? start() : audio.pause()));
  btnRestart.addEventListener('click', () => {
    audio.pause();
    audio.currentTime = 0;
    fromStart = true;
    start();
  });

  audio.addEventListener('play', () => { playing = audio; refresh(); });
  audio.addEventListener('pause', () => { if (playing === audio) playing = null; refresh(); });
  audio.addEventListener('timeupdate', () => {
    maxReached = reached(maxReached, audio.currentTime, audio.seeking);
    refresh();
  });
  // Un salto oltre il punto raggiunto (Centro di controllo, cuffie, ...) torna indietro.
  audio.addEventListener('seeking', () => {
    const back = seekTarget(audio.currentTime, maxReached);
    if (back != null) audio.currentTime = back;
  });
  audio.addEventListener('loadedmetadata', refresh);
  audio.addEventListener('durationchange', refresh);
  audio.addEventListener('ended', () => {
    fromStart = true;
    if (!state.ended && endedCounts(maxReached, audio.duration)) { state.ended = true; notify(); }
    refresh();
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
    // Un player disattivato non si avvia; se stava suonando, si ferma.
    setEnabled(on) {
      enabled = Boolean(on);
      if (!enabled) audio.pause();
      refresh();
    },
  };
}
