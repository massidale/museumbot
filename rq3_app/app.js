// Per ora: pagina di prova del guscio offline e del player (per lo sviluppatore).
import {validateManifest} from './logic.js';
import {createPlayer, checkCache} from './player.js';
import {APP_VERSION} from './config.js';

const app = document.getElementById('app');
const mb = b => (b / 1e6).toFixed(1);
const FILL_TIMEOUT_MS = 10 * 60 * 1000;  // oltre, lo scaricamento si considera fallito

function line(parent, text, cls) {
  const p = document.createElement('p');
  if (cls) p.className = cls;
  p.textContent = text;
  parent.append(p);
  return p;
}

function section(title) {
  const s = document.createElement('section');
  const h = document.createElement('h2');
  h.textContent = title;
  s.append(h);
  app.append(s);
  return s;
}

// Registra il service worker e segue ogni install: onState(w.state) a ogni cambio.
// Un worker in install che diventa 'redundant' e' un install fallito.
async function registerWorker(onState) {
  const reg = await navigator.serviceWorker.register('sw.js', {updateViaCache: 'none'});
  const watch = w => {
    if (!w) return;
    onState(w.state);
    w.addEventListener('statechange', () => onState(w.state));
  };
  watch(reg.installing);
  reg.addEventListener('updatefound', () => watch(reg.installing));
  return reg;
}

async function showCache(box, manifest) {
  box.className = '';
  box.textContent = 'controllo della cache…';
  const {missing, bytes} = await checkCache(manifest);
  box.className = missing.length ? 'err' : 'ok';
  box.textContent = `cache: ${missing.length} mancanti, ${mb(bytes)} MB`
    + (missing.length ? ` (${missing.slice(0, 3).join(', ')}${missing.length > 3 ? ', …' : ''})` : '');
  return missing;
}

async function showStorage(box) {
  if (!navigator.storage || !navigator.storage.estimate) { box.textContent = 'estimate non disponibile'; return; }
  const {usage, quota} = await navigator.storage.estimate();
  const persisted = navigator.storage.persisted ? await navigator.storage.persisted() : null;
  box.textContent = `storage: ${mb(usage)} MB usati su ${mb(quota)} MB; persistente: ${persisted}`;
}

function showPlayers(sec, art, version) {
  const lang = 'en';
  const conds = ['flat', 'explorer'];
  const states = conds.map(() => line(sec, ''));
  const show = (i, s) => { states[i].textContent = `Guida ${i + 1} (${conds[i]}): plays=${s.plays}, ended=${s.ended}`; };
  const players = [];
  conds.forEach((c, i) => {
    const onChange = s => {
      show(i, s);
      if (i === 0 && s.ended) players[1].setEnabled(true);  // Guida 2 dopo la fine della Guida 1
    };
    players.push(createPlayer(sec, art.audio[lang][c].src, `Guida ${i + 1}`, onChange,
                              {play: 'Play', pause: 'Pausa', restart: 'Da capo'}, version));
    show(i, {plays: 0, ended: false});
  });
  players[1].setEnabled(false);
  window.rq3Players = players;  // per le prove da console
}

async function main() {
  const h = document.createElement('h1');
  h.textContent = `Prova del guscio offline (app ${APP_VERSION})`;
  app.append(h);

  const status = section('Stato');
  const persistBox = line(status, 'persist: …');
  const swBox = line(status, 'service worker: …');
  const valBox = line(status, 'manifesto: …');
  const cacheBox = line(status, 'cache: …');
  const storageBox = line(status, 'storage: …');
  const errBox = line(status, '', 'err');

  if (navigator.storage && navigator.storage.persist) {
    navigator.storage.persist().then(ok => { persistBox.textContent = `persist: ${ok}`; });
  } else persistBox.textContent = 'persist: non disponibile';

  const manifest = await (await fetch('manifest.json')).json();
  const errs = validateManifest(manifest);
  valBox.className = errs.length ? 'err' : 'ok';
  valBox.textContent = errs.length
    ? `manifesto ${manifest.version}: ${errs.length} errori — ${errs.slice(0, 3).join('; ')}`
    : `manifesto ${manifest.version}: valido, ${manifest.artworks.length} opere`;

  const refresh = async () => { await showCache(cacheBox, manifest); await showStorage(storageBox); };

  // Attesa di uno scaricamento (messaggio 'filled' o fine di un install), con timeout.
  let fillTimer = null;
  const fillDone = () => { clearTimeout(fillTimer); fillTimer = null; };

  // Il service worker non blocca la pagina: a ogni cambio di stato si ricontrolla la cache.
  let reg = null;
  const onState = state => {
    swBox.className = state === 'redundant' ? 'err' : '';
    swBox.textContent = `service worker: ${state}`;
    if (state === 'redundant') errBox.textContent = 'installazione fallita: alcuni file non sono stati scaricati (vedi i mancanti); riprovare con "Scarica i mancanti"';
    if (state === 'activated' || state === 'redundant') { fillDone(); refresh(); }
  };
  const register = () => registerWorker(onState).then(r => { reg = r; return r; });
  if ('serviceWorker' in navigator) {
    try {
      await register();
      if (reg.active && !reg.installing) swBox.textContent = `service worker: attivo (${navigator.serviceWorker.controller ? 'controlla la pagina' : 'non ancora in controllo'})`;
    } catch (e) {
      swBox.className = 'err';
      swBox.textContent = `service worker: ${e}`;
    }
  } else swBox.textContent = 'service worker: non disponibile';

  refresh();

  // Completa la cache: col worker attivo gli si chiede di scaricare i mancanti; senza
  // (install fallito) si registra di nuovo, che rifa' l'install da dove si era fermato.
  const fill = () => {
    if (!('serviceWorker' in navigator)) return;
    errBox.textContent = '';
    cacheBox.className = '';
    cacheBox.textContent = 'scaricamento in corso…';
    fillTimer = setTimeout(() => {
      fillTimer = null;
      errBox.textContent = 'nessuna risposta dal service worker: scaricamento non concluso';
      refresh();
    }, FILL_TIMEOUT_MS);
    if (reg && reg.active) reg.active.postMessage({type: 'fill'});
    else register().catch(e => { fillDone(); errBox.textContent = `service worker: ${e}`; refresh(); });
  };
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.addEventListener('message', ev => {
      if (!ev.data || ev.data.type !== 'filled') return;
      fillDone();
      if (!ev.data.ok) errBox.textContent = `scaricamento non riuscito: ${ev.data.error}`;
      refresh();
    });
  }

  const btns = document.createElement('p');
  const again = document.createElement('button');
  again.className = 'secondary';
  again.textContent = 'Ricontrolla';
  again.onclick = refresh;
  const fillBtn = document.createElement('button');
  fillBtn.className = 'secondary';
  fillBtn.textContent = 'Scarica i mancanti';
  fillBtn.onclick = fill;
  btns.append(again, ' ', fillBtn);
  status.append(btns);

  const art = manifest.artworks[0];
  const sec = section(`Player: ${art.title.en} (${art.id})`);
  showPlayers(sec, art, manifest.version);
}

main().catch(e => line(app, `errore: ${e.message || e}`, 'err'));
