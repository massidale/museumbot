// Invio delle sessioni al database online: dopo ogni salvataggio, ogni 30 s e quando torna
// la rete. Un giro alla volta; 'sent' e 'rejected' (session_id -> updated_at) in IndexedDB,
// 'write_keys' (session_id -> chiave) per riconoscere la stessa sessione sul server.

import {getAllSessions, getSetting, setSetting} from './store.js';

// Record da inviare: non gia' inviati (o respinti) con lo stesso updated_at, i piu' vecchi prima.
export function pending(records, sent) {
  return records.filter(r => sent[r.session_id] !== r.updated_at)
    .sort((a, b) => (a.updated_at < b.updated_at ? -1 : a.updated_at > b.updated_at ? 1 : 0));
}

// 16 byte casuali -> 32 caratteri esadecimali.
export const newKey = bytes => Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('');

let endpoint = 'api/session';
let running = false;
let again = false;

// Chiave della sessione: quella salvata o una nuova, salvata prima del primo invio.
async function keyFor(id) {
  const keys = (await getSetting('write_keys')) || {};
  if (keys[id]) return keys[id];
  keys[id] = newKey(crypto.getRandomValues(new Uint8Array(16)));
  await setSetting('write_keys', keys);
  return keys[id];
}

// Aggiorna una mappa di impostazione leggendola di nuovo (altri giri non la perdono).
async function mark(name, id, at) {
  const m = (await getSetting(name)) || {};
  m[id] = at;
  await setSetting(name, m);
}

// Opzioni dell'invio: keepalive (parte anche a pagina chiusa), niente redirect (portali), timeout.
export const fetchInit = (rec, key) => ({method: 'POST', body: JSON.stringify(rec),
  headers: {'Content-Type': 'application/json', 'X-Write-Key': key},
  keepalive: true, redirect: 'error', signal: AbortSignal.timeout(15000)});

// Esito di una risposta: 'ok' (200 con {ok:true}), 'rejected' (400/403, non si riprova),
// 'skip' (429: questo record per ora, si va avanti) o 'stop' (rete, portale, altro).
export async function classify(res) {
  if (res.status === 200) {
    try { return (await res.json()).ok === true ? 'ok' : 'stop'; } catch (e) { return 'stop'; }
  }
  if (res.status === 429) return 'skip';
  if (res.status === 400 || res.status === 403) {
    let err = '';
    try { err = (await res.json()).error || ''; } catch (e) { /* corpo non JSON */ }
    console.warn(`invio respinto (${res.status}): ${err}`);
    return 'rejected';
  }
  return 'stop';
}

async function sendOne(rec) {
  const key = await keyFor(rec.session_id);
  let res;
  try { res = await fetch(endpoint, fetchInit(rec, key)); } catch (e) { return 'stop'; }
  return classify(res);
}

async function round() {
  const [all, sent, rejected] = await Promise.all(
    [getAllSessions(), getSetting('sent'), getSetting('rejected')]);
  for (const rec of pending(all, {...(sent || {}), ...(rejected || {})})) {
    const out = await sendOne(rec);
    if (out === 'stop') return;
    if (out === 'skip') continue;
    await mark(out === 'ok' ? 'sent' : 'rejected', rec.session_id, rec.updated_at);
  }
}

// Chiede un giro; se uno e' in corso ne parte un altro alla fine, mai due insieme.
export async function syncSoon() {
  if (running) { again = true; return; }
  running = true;
  try {
    do {
      again = false;
      try { await round(); } catch (e) { console.error('invio non riuscito', e); }
    } while (again);
  } finally { running = false; }
}

export function startSync(opts = {}) {
  endpoint = opts.endpoint || 'api/session';
  setInterval(syncSoon, opts.every || 30000);
  window.addEventListener('online', syncSoon);
  window.addEventListener('pagehide', syncSoon);
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') syncSoon();
  });
  syncSoon();
}

// Sessioni del dispositivo inviate (con l'ultimo updated_at) sul totale.
export async function syncState() {
  const [all, sent] = await Promise.all([getAllSessions(), getSetting('sent')]);
  const s = sent || {};
  return {sent: all.filter(r => s[r.session_id] === r.updated_at).length, total: all.length};
}
