// Player delle guide e controllo della cache dei media.
// L'audio si legge dalla Cache API come blob e si suona da un object URL:
// Safari non serve le richieste a intervalli (Range) da una risposta intera in cache.

const DEFAULT_LABELS = {play: 'Play', pause: 'Pause', restart: 'Restart'};

let playing = null;  // l'audio che sta suonando: ne suona uno alla volta

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

// Quali media mancano dalla cache e quanti byte occupano quelli presenti.
export async function checkCache(manifest) {
  const missing = [];
  let bytes = 0;
  for (const u of mediaUrls(manifest)) {
    const r = await caches.match(u);
    if (!r) { missing.push(u); continue; }
    const len = Number(r.headers.get('content-length'));
    bytes += Number.isFinite(len) && len > 0 ? len : (await r.blob()).size;
  }
  return {missing, bytes};
}

async function loadBlob(src) {
  const r = (await caches.match(src)) || (await fetch(src));
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

// state = {plays, ended}: plays conta gli avvii da capo, ended resta vero dopo la prima fine.
export function createPlayer(container, src, label, onChange = () => {}, labels = {}) {
  const L = {...DEFAULT_LABELS, ...labels};
  const state = {plays: 0, ended: false};
  let enabled = true, ready = false, destroyed = false, fromStart = true, url = null;

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
    const can = ready && enabled && !destroyed;
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
  audio.addEventListener('timeupdate', refresh);
  audio.addEventListener('loadedmetadata', refresh);
  audio.addEventListener('durationchange', refresh);
  audio.addEventListener('ended', () => {
    fromStart = true;
    if (!state.ended) { state.ended = true; notify(); }
    refresh();
  });

  refresh();
  loadBlob(src).then(blob => {
    if (destroyed) return;
    url = URL.createObjectURL(blob);
    audio.src = url;
    audio.load();
    ready = true;
    refresh();
  }).catch(e => {
    if (destroyed) return;
    root.append(el('div', 'player-error', String(e.message || e)));
  });

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
