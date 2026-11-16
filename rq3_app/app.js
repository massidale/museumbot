// Per ora: pagina di prova del guscio offline e del player (per lo sviluppatore).
import {validateManifest} from './logic.js';
import {createPlayer, checkCache} from './player.js';
import {APP_VERSION} from './config.js';

const app = document.getElementById('app');
const mb = b => (b / 1e6).toFixed(1);

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

async function registerWorker() {
  if (!('serviceWorker' in navigator)) return null;
  await navigator.serviceWorker.register('sw.js', {updateViaCache: 'none'});
  return navigator.serviceWorker.ready;  // si risolve quando l'install e' finito
}

async function showCache(box, manifest) {
  box.textContent = 'controllo della cache…';
  const {missing, bytes} = await checkCache(manifest);
  box.className = missing.length ? 'err' : 'ok';
  box.textContent = `cache: ${missing.length} mancanti, ${mb(bytes)} MB`;
  return missing;
}

async function showStorage(box) {
  if (!navigator.storage || !navigator.storage.estimate) { box.textContent = 'estimate non disponibile'; return; }
  const {usage, quota} = await navigator.storage.estimate();
  const persisted = navigator.storage.persisted ? await navigator.storage.persisted() : null;
  box.textContent = `storage: ${mb(usage)} MB usati su ${mb(quota)} MB; persistente: ${persisted}`;
}

function showPlayers(sec, art) {
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
                              {play: 'Play', pause: 'Pausa', restart: 'Da capo'}));
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
  const cacheBox = line(status, 'cache: in attesa del service worker…');
  const storageBox = line(status, 'storage: …');

  if (navigator.storage && navigator.storage.persist) {
    navigator.storage.persist().then(ok => { persistBox.textContent = `persist: ${ok}`; });
  } else persistBox.textContent = 'persist: non disponibile';

  const manifest = await (await fetch('manifest.json')).json();
  const errs = validateManifest(manifest);
  valBox.className = errs.length ? 'err' : 'ok';
  valBox.textContent = errs.length
    ? `manifesto ${manifest.version}: ${errs.length} errori — ${errs.slice(0, 3).join('; ')}`
    : `manifesto ${manifest.version}: valido, ${manifest.artworks.length} opere`;

  const reg = await registerWorker().catch(e => { swBox.className = 'err'; swBox.textContent = `service worker: ${e}`; });
  if (reg) swBox.textContent = `service worker: attivo (${navigator.serviceWorker.controller ? 'controlla la pagina' : 'non ancora in controllo'})`;

  await showCache(cacheBox, manifest);
  await showStorage(storageBox);

  const btns = document.createElement('p');
  const again = document.createElement('button');
  again.className = 'secondary';
  again.textContent = 'Ricontrolla';
  again.onclick = async () => { await showCache(cacheBox, manifest); await showStorage(storageBox); };
  const fill = document.createElement('button');
  fill.className = 'secondary';
  fill.textContent = 'Scarica i mancanti';
  fill.onclick = () => {
    if (!reg || !reg.active) return;
    cacheBox.textContent = 'scaricamento in corso…';
    reg.active.postMessage({type: 'fill'});
  };
  btns.append(again, ' ', fill);
  status.append(btns);
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.addEventListener('message', async ev => {
      if (ev.data && ev.data.type === 'filled') {
        await showCache(cacheBox, manifest);
        await showStorage(storageBox);
        if (!ev.data.ok) line(status, `errore: ${ev.data.error}`, 'err');
      }
    });
  }

  const art = manifest.artworks[0];
  const sec = section(`Player: ${art.title.en} (${art.id})`);
  showPlayers(sec, art);
}

main().catch(e => line(app, `errore: ${e.message || e}`, 'err'));
