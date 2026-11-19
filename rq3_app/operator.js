// Schermata dell'operatore (solo per chi conduce lo studio, in inglese). Senza PIN: solo
// "Nuova sessione" e l'area d'angolo; col PIN: nome del dispositivo, conteggi, controllo
// del materiale, export delle sessioni e cancellazione di quelle esportate.
import {toJson, deletableIds, validateManifest, tourOrder, moveItem} from './logic.js';
import {getAllSessions, deleteSessions, getSetting, setSetting, getSaveFailures} from './store.js';
import {checkCache, mediaUrls} from './player.js';
import {fillMissing} from './swclient.js';
import {syncState} from './sync.js';
import {APP_VERSION, PIN} from './config.js';

export const DEFAULT_DEVICE = 'non-impostato';
export const HOLD_MS = 2000;  // tocco lungo per il PIN
const HOLD_SLOP = 12;      // px: oltre, il dito si e' spostato e il tocco lungo si annulla
const MAX_LISTED = 20;     // file mancanti mostrati per esteso

// ---- utilita' ----

function h(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v == null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) e.append(c);
  return e;
}

const pad = n => String(n).padStart(2, '0');

// rq3_<device>_<AAAAMMGG-HHMM>.json, ora locale.
export function exportFileName(device, d = new Date()) {
  const stamp = `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}`;
  return `rq3_${device}_${stamp}.json`;
}

// Il nome finisce nel nome del file: solo lettere, cifre, '-' e '_'.
export const validDevice = s => /^[A-Za-z0-9_-]{1,40}$/.test(s) && s !== DEFAULT_DEVICE;

const errText = e => String((e && e.message) || e);

// Senza un service worker che controlla la pagina l'app non funziona offline.
const swControls = () => 'serviceWorker' in navigator && navigator.serviceWorker.controller != null;

// Tocco lungo (touch e mouse): onFire dopo ms; si annulla se il dito si alza o si sposta.
// Restituisce la funzione che annulla il tocco in corso.
export function longPress(el, ms, onFire) {
  let timer = null, x = 0, y = 0;
  const cancel = () => {
    clearTimeout(timer);
    timer = null;
    el.classList.remove('holding');
  };
  el.addEventListener('pointerdown', e => {
    cancel();
    x = e.clientX; y = e.clientY;
    el.classList.add('holding');
    timer = setTimeout(() => { cancel(); onFire(); }, ms);
  });
  el.addEventListener('pointermove', e => {
    if (timer && Math.hypot(e.clientX - x, e.clientY - y) > HOLD_SLOP) cancel();
  });
  for (const t of ['pointerup', 'pointercancel', 'pointerleave']) el.addEventListener(t, cancel);
  el.addEventListener('contextmenu', e => e.preventDefault());
  return cancel;
}

// Form del PIN (nascosto): onOk dopo il PIN giusto; onCancel se c'e' "Annulla".
function pinForm(onOk, onCancel = null) {
  const input = h('input', {type: 'password', inputmode: 'numeric', autocomplete: 'off', 'aria-label': 'PIN'});
  const msg = h('p', {class: 'err'});
  const cancel = onCancel && h('button', {type: 'button', class: 'secondary'}, 'Cancel');
  const form = h('form', {class: 'op-row op-pin', hidden: true}, input, h('button', {type: 'submit'}, 'Open'), cancel, msg);
  const reset = () => { input.value = ''; msg.textContent = ''; form.hidden = true; };
  form.addEventListener('submit', e => {
    e.preventDefault();
    if (input.value === PIN) { reset(); onOk(); }
    else { input.value = ''; msg.textContent = 'Wrong PIN.'; }
  });
  if (cancel) cancel.addEventListener('click', () => { reset(); onCancel(); });
  const open = () => { form.hidden = false; input.focus(); };
  return {form, input, open, reset};
}

// Angolo dell'operatore sulle schermate del visitatore: tocco lungo, PIN, conferma,
// poi onAbort. Gli elementi stanno fuori da #app (le schermate non li toccano).
// Restituisce la funzione che li toglie.
export function mountAbortGate(onAbort) {
  const corner = h('div', {class: 'op-corner', id: 'op-abort-corner', 'aria-label': 'Operator area'});
  const box = h('div', {class: 'op-overlay', id: 'op-abort', lang: 'en', hidden: true}, h('p', {}, 'Operator: PIN'));
  const close = () => { pin.reset(); box.hidden = true; };
  const pin = pinForm(() => {
    box.hidden = true;
    if (window.confirm('Stop this session?')) onAbort();
  }, close);
  box.append(pin.form);
  const cancelHold = longPress(corner, HOLD_MS, () => { box.hidden = false; pin.open(); });
  document.body.append(corner, box);
  return () => { cancelHold(); corner.remove(); box.remove(); };
}

// ---- stato del service worker (registrato da app.js) ----

let swStatus = [];          // [messaggio, errore?]
let swListener = () => {};  // la schermata aperta, se c'e'

// Un messaggio per tipo (persist, service worker, ...): il nuovo sostituisce il vecchio.
export function onSwStatus(msg, isError) {
  const kind = msg.split(':')[0];
  swStatus = swStatus.filter(([m]) => m.split(':')[0] !== kind);
  swStatus.push([msg, isError]);
  swListener(msg);
}

// Avviso in rosso se dei salvataggi non sono riusciti: quei dati potrebbero mancare.
function saveWarning() {
  const f = getSaveFailures();
  if (!f.length) return null;
  const ids = [...new Set(f.map(x => x.session_id))].join(', ');
  return h('p', {class: 'err', id: 'op-save-errors'},
    `Warning: ${f.length} saves failed (sessions ${ids}); `
    + `the last one at step ${f[f.length - 1].last_step}: ${f[f.length - 1].error}`);
}

// ---- schermata ----

// Disegna la schermata in root; onStart avvia la sessione del visitatore.
// Restituisce la funzione che la chiude (timer, object URL).
export function showOperator(root, {onStart = () => {}, message = ''} = {}) {
  let closed = false;
  let device = null;                  // nome salvato; null se non impostato
  let check = {state: 'pending'};     // controllo del materiale: pending | ok | error
  let checkSeq = 0;
  let prepared = null;                // {text, ids}: tutte le sessioni, pronte da esportare
  let blobUrl = null;

  // -- nuova sessione --
  const start = h('button', {type: 'button', id: 'op-start', disabled: true}, 'New session');
  const startBar = h('div', {class: 'actions'}, start);   // a fondo pagina, come nelle altre schermate
  const startNote = h('p', {class: 'muted', id: 'op-start-note'});
  start.addEventListener('click', () => { if (canStart()) onStart(); });
  const canStart = () => device != null && check.state === 'ok';
  function updateStart() {
    start.disabled = !canStart();
    const why = [];
    if (device == null) why.push('device name not set');
    if (check.state === 'pending') why.push('checking the media');
    if (check.state === 'error') why.push(check.sw === false ? 'media incomplete or service worker not active'
                                                              : 'media incomplete or not valid');
    startNote.textContent = why.length ? `New session disabled: ${why.join('; ')}.` : '';
    startNote.className = check.state === 'error' ? 'err' : 'muted';
  }

  // -- PIN --
  const corner = h('div', {class: 'op-corner', id: 'op-corner', 'aria-label': 'Operator area'});
  const pin = pinForm(() => openPanel());
  pin.input.id = 'op-pin';
  const cancelHold = longPress(corner, HOLD_MS, () => { if (panel.hidden) pin.open(); });

  // -- dispositivo --
  const devInput = h('input', {type: 'text', id: 'op-device', autocomplete: 'off', autocapitalize: 'off',
                               spellcheck: 'false', placeholder: 'e.g. ipad-1'});
  const devMsg = h('p', {class: 'muted'});
  const devForm = h('form', {class: 'op-row'}, devInput, h('button', {type: 'submit'}, 'Save name'));
  devForm.addEventListener('submit', async e => {
    e.preventDefault();
    const v = devInput.value.trim();
    if (!validDevice(v)) {
      say(devMsg, 'Invalid name: only letters, digits, "-" and "_" (e.g. ipad-1).', true);
      return;
    }
    try {
      await setSetting('device', v);
      say(devMsg, `Name saved: ${v}.`);
      await refresh();
    } catch (err) { say(devMsg, `Name not saved: ${errText(err)}`, true); }
  });

  // -- ordine del giro: casuale, o fisso per tutte le nuove sessioni di questo iPad --
  let tourIds = null;   // ordine mostrato (id del manifesto)
  let titles = {};
  const tourFixed = h('input', {type: 'checkbox', id: 'op-tour-fixed'});
  const tourList = h('ol', {class: 'op-tour', id: 'op-tour'});
  const tourMsg = h('p', {class: 'muted', id: 'op-tour-msg'});
  tourFixed.addEventListener('change', saveTour);

  async function loadTour() {
    try {
      const [manifest, saved] = await Promise.all(
        [fetch('manifest.json').then(r => r.json()), getSetting('tour_order')]);
      if (closed) return;
      titles = Object.fromEntries(manifest.artworks.map(a => [a.id, a.title.en]));
      tourFixed.checked = Array.isArray(saved) && saved.length > 0;
      tourIds = tourOrder(manifest.artworks.map(a => a.id), tourFixed.checked ? saved : null);
      renderTour();
    } catch (e) { say(tourMsg, `Tour order not read: ${errText(e)}`, true); }
  }

  function renderTour() {
    if (!tourIds) return;
    const on = tourFixed.checked, last = tourIds.length - 1;
    const arrow = (i, d, label) => h('button', {type: 'button', class: 'secondary',
      'aria-label': `Move ${titles[tourIds[i]]} ${label}`, disabled: !on || i + d < 0 || i + d > last,
      onclick: () => { tourIds = moveItem(tourIds, i, d); saveTour(); }}, d < 0 ? '↑' : '↓');
    tourList.classList.toggle('off', !on);
    tourList.replaceChildren(...tourIds.map((id, i) =>
      h('li', {}, h('span', {}, `${i + 1}. ${titles[id] || id}`), arrow(i, -1, 'up'), arrow(i, 1, 'down'))));
  }

  async function saveTour() {
    if (!tourIds) return;
    renderTour();
    try {
      await setSetting('tour_order', tourFixed.checked ? tourIds : null);
      say(tourMsg, tourFixed.checked ? 'Fixed order saved: new sessions follow it.'
                                     : 'Random order: each new session shuffles the paintings.');
    } catch (e) { say(tourMsg, `Tour order not saved: ${errText(e)}`, true); }
  }

  // -- sessioni, export, cancellazione --
  const counts = h('p', {id: 'op-counts'}, 'Reading sessions...');
  const online = h('p', {class: 'muted', id: 'op-online'});
  const exportBtn = h('button', {type: 'button', id: 'op-export'}, 'Download');
  const deleteBtn = h('button', {type: 'button', class: 'secondary', id: 'op-delete'}, 'Delete exported sessions');
  const exportMsg = h('p', {class: 'muted', id: 'op-export-msg'});
  const linkBox = h('div', {id: 'op-download'});
  const deleteMsg = h('p', {class: 'muted', id: 'op-delete-msg'});
  exportBtn.addEventListener('click', doDownload);
  deleteBtn.addEventListener('click', doDelete);

  // -- materiale --
  const checkBtn = h('button', {type: 'button', id: 'op-check'}, 'Check media');
  const fillBtn = h('button', {type: 'button', class: 'secondary', id: 'op-fill', hidden: true}, 'Download missing files');
  const checkBox = h('div', {class: 'op-check', id: 'op-check-result'});
  checkBtn.addEventListener('click', () => runCheck());
  fillBtn.addEventListener('click', doFill);

  // -- service worker --
  const swBox = h('div', {class: 'op-status', id: 'op-status'});
  const renderSw = () =>
    swBox.replaceChildren(...swStatus.map(([m, isErr]) => h('p', {class: isErr ? 'err' : 'muted'}, m)));
  swListener = msg => {
    if (closed) return;
    renderSw();
    if (msg === 'service worker: activated') runCheck();  // install concluso: la cache e' cambiata
  };

  const panel = h('section', {class: 'op-panel', id: 'op-panel', hidden: true},
    h('h2', {}, 'Device'),
    devForm, devMsg,
    h('h2', {}, 'Tour order'),
    h('label', {class: 'op-toggle'}, tourFixed, 'Use a fixed order (otherwise random for each session)'),
    tourList, tourMsg,
    h('h2', {}, 'Sessions'),
    counts,
    online,
    h('div', {class: 'op-row'}, exportBtn, deleteBtn),
    exportMsg, linkBox, deleteMsg,
    h('p', {class: 'note'}, 'Delete only after checking that the file has reached the Mac.'),
    h('h2', {}, 'Media'),
    h('div', {class: 'op-row'}, checkBtn, fillBtn),
    checkBox,
    h('h2', {}, 'Service worker'),
    swBox,
    h('div', {class: 'actions'},
      h('button', {type: 'button', class: 'secondary', onclick: closePanel}, 'Close panel')));

  function say(el, text, isErr = false) {
    el.textContent = text;
    el.className = isErr ? 'err' : 'muted';
  }

  function openPanel() {
    panel.hidden = false;
    corner.hidden = true;
    startBar.hidden = startNote.hidden = true;
    renderSw();
    refresh();
    runCheck();
    loadTour();
  }

  function closePanel() {
    panel.hidden = true;
    corner.hidden = false;
    startBar.hidden = startNote.hidden = false;
    clearLink();
    for (const el of [devMsg, tourMsg, exportMsg, deleteMsg]) el.textContent = '';
    window.scrollTo(0, 0);
  }

  // Nome del dispositivo, conteggi e testo dell'export, dal database.
  async function refresh() {
    try {
      const [all, d, exported] = await Promise.all(
        [getAllSessions(), getSetting('device'), getSetting('last_export_ids')]);
      if (closed) return;
      device = d && d !== DEFAULT_DEVICE ? d : null;
      if (document.activeElement !== devInput) devInput.value = device || '';
      const complete = all.filter(r => r.completed === true).length;
      const inExport = deletableIds(all, exported || []).length;
      prepared = {text: toJson(all), ids: all.map(r => r.session_id)};
      counts.textContent = `${all.length} sessions on this device: ${complete} complete, `
        + `${all.length - complete} incomplete. Already exported (last export): ${inExport}.`;
      exportBtn.disabled = all.length === 0;
      const st = await syncState();
      if (closed) return;
      online.textContent = `Online: ${st.sent} of ${st.total} sessions sent`;
    } catch (e) {
      console.error(e);
      if (closed) return;
      prepared = null;
      say(counts, `Could not read sessions: ${errText(e)}`, true);
    }
    updateStart();
  }

  // validateManifest + checkCache; i risultati vecchi (un controllo piu' recente) si scartano.
  async function runCheck() {
    const seq = ++checkSeq;
    check = {state: 'pending'};
    checkBtn.disabled = true;
    checkBox.replaceChildren(h('p', {class: 'muted'}, 'Checking...'));
    updateStart();
    let next;
    try {
      const manifest = await (await fetch('manifest.json')).json();
      const errors = validateManifest(manifest);
      const total = errors.length ? 0 : mediaUrls(manifest).length;
      const {missing, bytes} = errors.length ? {missing: [], bytes: 0} : await checkCache(manifest);
      const sw = swControls();
      next = {state: errors.length || missing.length || !sw ? 'error' : 'ok', manifest, errors, missing, bytes, total, sw};
    } catch (e) {
      console.error(e);
      next = {state: 'error', manifest: null, errors: [`check failed: ${errText(e)}`], missing: [], sw: swControls()};
    }
    if (closed || seq !== checkSeq) return;
    check = next;
    checkBtn.disabled = false;
    renderCheck();
    updateStart();
  }

  function renderCheck() {
    const {state, manifest, errors, missing, bytes, total, sw} = check;
    const out = [];
    if (!sw) {
      out.push(h('p', {class: 'err', id: 'op-sw-missing'}, 'The service worker does not control the page: '
        + 'without it the app does not work offline. Reload the page with a connection and check again.'));
    }
    if (state === 'ok') {
      out.push(h('p', {class: 'ok'}, `Media complete: manifest ${manifest.version}, `
        + `${total} files cached (${(bytes / 1e6).toFixed(1)} MB).`));
    }
    if (errors.length) {
      out.push(h('p', {class: 'err'}, `Manifest not valid (${errors.length} errors):`),
        h('ul', {class: 'op-list err'}, errors.slice(0, MAX_LISTED).map(s => h('li', {}, s))));
    }
    if (missing.length) {
      out.push(h('p', {class: 'err'}, `${missing.length} of ${total} files missing `
        + `(manifest ${manifest.version}):`),
        h('ul', {class: 'op-list err', id: 'op-missing'}, missing.slice(0, MAX_LISTED).map(s => h('li', {}, s))));
      if (missing.length > MAX_LISTED) out.push(h('p', {class: 'err'}, `... and ${missing.length - MAX_LISTED} more.`));
    }
    if (errors.length > MAX_LISTED) out.push(h('p', {class: 'err'}, `... and ${errors.length - MAX_LISTED} more errors.`));
    checkBox.replaceChildren(...out);
    fillBtn.hidden = !(missing.length && manifest);
  }

  async function doFill() {
    if (!check.manifest) return;
    fillBtn.disabled = checkBtn.disabled = true;
    checkBox.prepend(h('p', {class: 'muted'}, 'Downloading...'));
    try {
      await fillMissing(check.manifest);
    } catch (e) {
      console.error(e);
      if (!closed) checkBox.prepend(h('p', {class: 'err'}, `Download failed: ${errText(e)}`));
    }
    fillBtn.disabled = false;
    if (!closed) await runCheck();
  }

  function clearLink() {
    linkBox.replaceChildren();
    if (blobUrl) URL.revokeObjectURL(blobUrl);
    blobUrl = null;
  }

  // Le sessioni del file risultano esportate solo da qui in poi.
  async function markExported(ids, name) {
    try {
      await setSetting('last_export_ids', ids);
      say(exportMsg, `Exported ${ids.length} sessions to ${name}.`);
    } catch (e) {
      say(exportMsg, `File created, but the export was not recorded (${errText(e)}): `
        + 'the sessions will not be deleted.', true);
    }
    await refresh();
  }

  function offerDownload(file, ids) {
    clearLink();
    blobUrl = URL.createObjectURL(file);
    const a = h('a', {href: blobUrl, download: file.name, class: 'op-download'},
      `Download ${file.name} (${ids.length} sessions)`);
    a.addEventListener('click', () => markExported(ids, file.name));
    linkBox.append(a);
  }

  // Il file dell'export: tutte le sessioni del dispositivo, preparate in refresh().
  function exportFile() {
    if (!prepared || prepared.ids.length === 0) { say(exportMsg, 'No sessions to export.'); return null; }
    return {file: new File([prepared.text], exportFileName(device || DEFAULT_DEVICE), {type: 'application/json'}),
            ids: prepared.ids};
  }

  // Download normale: sul Mac in Download, sull'iPad in File > Download. Il link resta
  // visibile se il browser non fa partire il download da solo.
  function doDownload() {
    clearLink();
    const ex = exportFile();
    if (!ex) return;
    offerDownload(ex.file, ex.ids);
    const a = linkBox.querySelector('a');
    a.click();
  }

  // Cancella solo le sessioni dell'ultimo export ancora presenti; poi l'export si azzera.
  async function doDelete() {
    deleteBtn.disabled = true;
    try {
      const [all, exported] = await Promise.all([getAllSessions(), getSetting('last_export_ids')]);
      const ids = deletableIds(all, exported || []);
      if (ids.length === 0) {
        say(deleteMsg, 'No exported sessions to delete: export first.');
        return;
      }
      const left = all.length - ids.length;
      if (!window.confirm(`Permanently delete ${ids.length} exported sessions? `
        + `${left} not exported will remain. This cannot be undone.`)) {
        say(deleteMsg, 'Deletion cancelled.');
        return;
      }
      await deleteSessions(ids);
      await setSetting('last_export_ids', []);
      say(deleteMsg, `Deleted ${ids.length} sessions.`);
    } catch (e) {
      console.error(e);
      say(deleteMsg, `Deletion failed: ${errText(e)}`, true);
    } finally {
      if (!closed) {
        deleteBtn.disabled = false;
        await refresh();
      }
    }
  }

  root.replaceChildren(...[
    corner,
    h('h1', {}, 'Audio guides: operator'),
    h('p', {class: 'muted'}, `app ${APP_VERSION}`),
    message && h('p', {class: 'err'}, message),
    saveWarning(),
    pin.form,
    panel,
    startNote,
    startBar,
  ].filter(Boolean));

  // Anche senza PIN: "Nuova sessione" dipende dal nome e dal materiale.
  refresh();
  runCheck();
  // un worker che prende il controllo cambia l'esito del controllo
  const onController = () => { if (!closed) runCheck(); };
  if ('serviceWorker' in navigator) navigator.serviceWorker.addEventListener('controllerchange', onController);

  return () => {
    closed = true;
    if ('serviceWorker' in navigator) navigator.serviceWorker.removeEventListener('controllerchange', onController);
    cancelHold();
    clearLink();
    swListener = () => {};
  };
}
