// Lato pagina del service worker: registrazione, storage persistente, install fallito
// e completamento della cache dei media (lo usa anche la schermata dell'operatore).
import {checkCache} from './player.js';

const FILL_TIMEOUT_MS = 10 * 60 * 1000;  // oltre, lo scaricamento si considera fallito

let reg = null;
let report = () => {};
let pending = null;      // scaricamento in attesa: {resolve, reject, timer}
let listening = false;

// Chiude lo scaricamento in attesa, se c'e'.
function settle(err) {
  if (!pending) return;
  const p = pending;
  pending = null;
  clearTimeout(p.timer);
  if (err) p.reject(err); else p.resolve();
}

// Un worker in install che diventa 'redundant' e' un install fallito.
function onState(state) {
  if (state === 'redundant') {
    report('installazione fallita: alcuni file non sono stati scaricati; riprovare con "Scarica i mancanti"', true);
    settle(new Error('installazione del service worker fallita'));
    return;
  }
  report(`service worker: ${state}`, false);
  if (state === 'activated') settle();
}

function watch(w) {
  if (!w) return;
  onState(w.state);
  w.addEventListener('statechange', () => onState(w.state));
}

// Risposta del worker a una richiesta 'fill'.
function onMessage(ev) {
  if (!ev.data || ev.data.type !== 'filled') return;
  if (ev.data.ok) settle();
  else {
    report(`scaricamento non riuscito: ${ev.data.error}`, true);
    settle(new Error(`scaricamento non riuscito: ${ev.data.error}`));
  }
}

async function register() {
  reg = await navigator.serviceWorker.register('sw.js', {updateViaCache: 'none'});
  watch(reg.installing);
  reg.addEventListener('updatefound', () => watch(reg.installing));
  return reg;
}

// Chiede lo storage persistente e registra il worker; onStatus(msg, isError) a ogni cambio.
// Il worker non blocca la pagina: null se non c'e' o se la registrazione fallisce.
export async function registerServiceWorker(onStatus) {
  report = onStatus || (() => {});
  if (navigator.storage && navigator.storage.persist) {
    navigator.storage.persist().then(ok => report(`storage persistente: ${ok}`, !ok));
  } else report('storage persistente: non disponibile', true);

  if (!('serviceWorker' in navigator)) { report('service worker: non disponibile', true); return null; }
  if (!listening) {
    navigator.serviceWorker.addEventListener('message', onMessage);
    listening = true;
  }
  try {
    await register();
    if (reg.active && !reg.installing)
      report(`service worker: attivo (${navigator.serviceWorker.controller ? 'controlla la pagina' : 'non ancora in controllo'})`, false);
    return reg;
  } catch (e) {
    report(`service worker: ${e}`, true);
    return null;
  }
}

// Completa la cache dei media: col worker attivo gli si chiede di scaricare i mancanti;
// senza (install fallito) si registra di nuovo, che rifa' l'install da dove si era fermato.
// Si risolve a scaricamento concluso; rifiuta su errore o dopo FILL_TIMEOUT_MS.
export async function fillMissing(manifest) {
  if (!('serviceWorker' in navigator)) throw new Error('service worker non disponibile');
  if (manifest && (await checkCache(manifest)).missing.length === 0) return;
  if (pending) throw new Error('scaricamento gia\' in corso');
  const done = new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      report('nessuna risposta dal service worker: scaricamento non concluso', true);
      settle(new Error('nessuna risposta dal service worker'));
    }, FILL_TIMEOUT_MS);
    pending = {resolve, reject, timer};
  });
  if (reg && reg.active) reg.active.postMessage({type: 'fill'});
  else register().then(r => {
    // nessun install da seguire: il worker c'e' gia', gli si chiede di scaricare
    if (r.active && !r.installing && !r.waiting) r.active.postMessage({type: 'fill'});
  }).catch(e => { report(`service worker: ${e}`, true); settle(e); });
  return done;
}
