// Schermata dell'operatore (solo per chi conduce lo studio, in italiano). Senza PIN: solo
// "Nuova sessione" e l'area d'angolo; col PIN: nome del dispositivo, conteggi, controllo
// del materiale, export delle sessioni e cancellazione di quelle esportate.
import {toJson, deletableIds, validateManifest} from './logic.js';
import {getAllSessions, deleteSessions, getSetting, setSetting, getSaveFailures} from './store.js';
import {checkCache, mediaUrls} from './player.js';
import {fillMissing} from './swclient.js';
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
  const cancel = onCancel && h('button', {type: 'button', class: 'secondary'}, 'Annulla');
  const form = h('form', {class: 'op-row op-pin', hidden: true}, input, h('button', {type: 'submit'}, 'Apri'), cancel, msg);
  const reset = () => { input.value = ''; msg.textContent = ''; form.hidden = true; };
  form.addEventListener('submit', e => {
    e.preventDefault();
    if (input.value === PIN) { reset(); onOk(); }
    else { input.value = ''; msg.textContent = 'PIN errato.'; }
  });
  if (cancel) cancel.addEventListener('click', () => { reset(); onCancel(); });
  const open = () => { form.hidden = false; input.focus(); };
  return {form, input, open, reset};
}

// Angolo dell'operatore sulle schermate del visitatore: tocco lungo, PIN, conferma,
// poi onAbort. Gli elementi stanno fuori da #app (le schermate non li toccano).
// Restituisce la funzione che li toglie.
export function mountAbortGate(onAbort) {
  const corner = h('div', {class: 'op-corner', id: 'op-abort-corner', 'aria-label': 'Area operatore'});
  const box = h('div', {class: 'op-overlay', id: 'op-abort', lang: 'it', hidden: true}, h('p', {}, 'Operatore: PIN'));
  const close = () => { pin.reset(); box.hidden = true; };
  const pin = pinForm(() => {
    box.hidden = true;
    if (window.confirm('Interrompere la sessione?')) onAbort();
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
    `Attenzione: ${f.length} salvataggi non riusciti (sessioni ${ids}); `
    + `l'ultimo al passo ${f[f.length - 1].last_step}: ${f[f.length - 1].error}`);
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
  const start = h('button', {type: 'button', class: 'big', id: 'op-start', disabled: true}, 'Nuova sessione');
  const startNote = h('p', {class: 'muted', id: 'op-start-note'});
  start.addEventListener('click', () => { if (canStart()) onStart(); });
  const canStart = () => device != null && check.state === 'ok';
  function updateStart() {
    start.disabled = !canStart();
    const why = [];
    if (device == null) why.push('nome del dispositivo non impostato');
    if (check.state === 'pending') why.push('controllo del materiale in corso');
    if (check.state === 'error') why.push(check.sw === false ? 'materiale incompleto o service worker non attivo'
                                                              : 'materiale incompleto o non valido');
    startNote.textContent = why.length ? `Nuova sessione disattivata: ${why.join('; ')}.` : '';
    startNote.className = check.state === 'error' ? 'err' : 'muted';
  }

  // -- PIN --
  const corner = h('div', {class: 'op-corner', id: 'op-corner', 'aria-label': 'Area operatore'});
  const pin = pinForm(() => openPanel());
  pin.input.id = 'op-pin';
  const cancelHold = longPress(corner, HOLD_MS, () => { if (panel.hidden) pin.open(); });

  // -- dispositivo --
  const devInput = h('input', {type: 'text', id: 'op-device', autocomplete: 'off', autocapitalize: 'off',
                               spellcheck: 'false', placeholder: 'es. ipad-1'});
  const devMsg = h('p', {class: 'muted'});
  const devForm = h('form', {class: 'op-row'}, devInput, h('button', {type: 'submit'}, 'Salva nome'));
  devForm.addEventListener('submit', async e => {
    e.preventDefault();
    const v = devInput.value.trim();
    if (!validDevice(v)) {
      say(devMsg, 'Nome non valido: solo lettere, cifre, "-" e "_" (es. ipad-1).', true);
      return;
    }
    try {
      await setSetting('device', v);
      say(devMsg, `Nome salvato: ${v}.`);
      await refresh();
    } catch (err) { say(devMsg, `Nome non salvato: ${errText(err)}`, true); }
  });

  // -- sessioni, export, cancellazione --
  const counts = h('p', {id: 'op-counts'}, 'Lettura delle sessioni...');
  const exportBtn = h('button', {type: 'button', id: 'op-export'}, 'Scarica');
  const shareBtn = h('button', {type: 'button', class: 'secondary', id: 'op-share'}, 'Condividi');
  const deleteBtn = h('button', {type: 'button', class: 'secondary', id: 'op-delete'}, 'Cancella le sessioni esportate');
  const exportMsg = h('p', {class: 'muted', id: 'op-export-msg'});
  const linkBox = h('div', {id: 'op-download'});
  const deleteMsg = h('p', {class: 'muted', id: 'op-delete-msg'});
  exportBtn.addEventListener('click', doDownload);
  shareBtn.addEventListener('click', doShare);
  deleteBtn.addEventListener('click', doDelete);

  // -- materiale --
  const checkBtn = h('button', {type: 'button', id: 'op-check'}, 'Controlla materiale');
  const fillBtn = h('button', {type: 'button', class: 'secondary', id: 'op-fill', hidden: true}, 'Scarica i mancanti');
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
    h('h2', {}, 'Dispositivo'),
    devForm, devMsg,
    h('h2', {}, 'Sessioni'),
    counts,
    h('div', {class: 'op-row'}, exportBtn, shareBtn, deleteBtn),
    exportMsg, linkBox, deleteMsg,
    h('p', {class: 'note'}, 'Cancellare solo dopo aver controllato che il file sia arrivato sul Mac.'),
    h('h2', {}, 'Materiale'),
    h('div', {class: 'op-row'}, checkBtn, fillBtn),
    checkBox,
    h('h2', {}, 'Service worker'),
    swBox,
    h('div', {class: 'actions'},
      h('button', {type: 'button', class: 'secondary', onclick: closePanel}, 'Chiudi pannello')));

  function say(el, text, isErr = false) {
    el.textContent = text;
    el.className = isErr ? 'err' : 'muted';
  }

  function openPanel() {
    panel.hidden = false;
    corner.hidden = true;
    renderSw();
    refresh();
    runCheck();
  }

  function closePanel() {
    panel.hidden = true;
    corner.hidden = false;
    clearLink();
    for (const el of [devMsg, exportMsg, deleteMsg]) el.textContent = '';
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
      counts.textContent = `${all.length} sessioni nel dispositivo: ${complete} complete, `
        + `${all.length - complete} interrotte. Gia' esportate (ultimo export): ${inExport}.`;
      exportBtn.disabled = shareBtn.disabled = all.length === 0;
    } catch (e) {
      console.error(e);
      if (closed) return;
      prepared = null;
      say(counts, `Lettura delle sessioni non riuscita: ${errText(e)}`, true);
    }
    updateStart();
  }

  // validateManifest + checkCache; i risultati vecchi (un controllo piu' recente) si scartano.
  async function runCheck() {
    const seq = ++checkSeq;
    check = {state: 'pending'};
    checkBtn.disabled = true;
    checkBox.replaceChildren(h('p', {class: 'muted'}, 'Controllo in corso...'));
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
      next = {state: 'error', manifest: null, errors: [`controllo non riuscito: ${errText(e)}`], missing: [], sw: swControls()};
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
      out.push(h('p', {class: 'err', id: 'op-sw-missing'}, 'Il service worker non controlla la pagina: '
        + 'senza, l\'app non funziona offline. Ricaricare la pagina con la rete e ripetere il controllo.'));
    }
    if (state === 'ok') {
      out.push(h('p', {class: 'ok'}, `Materiale completo: manifesto ${manifest.version}, `
        + `${total} file in cache (${(bytes / 1e6).toFixed(1)} MB).`));
    }
    if (errors.length) {
      out.push(h('p', {class: 'err'}, `Manifesto non valido (${errors.length} errori):`),
        h('ul', {class: 'op-list err'}, errors.slice(0, MAX_LISTED).map(s => h('li', {}, s))));
    }
    if (missing.length) {
      out.push(h('p', {class: 'err'}, `Mancano ${missing.length} file su ${total} `
        + `(manifesto ${manifest.version}):`),
        h('ul', {class: 'op-list err', id: 'op-missing'}, missing.slice(0, MAX_LISTED).map(s => h('li', {}, s))));
      if (missing.length > MAX_LISTED) out.push(h('p', {class: 'err'}, `... e altri ${missing.length - MAX_LISTED}.`));
    }
    if (errors.length > MAX_LISTED) out.push(h('p', {class: 'err'}, `... e altri ${errors.length - MAX_LISTED} errori.`));
    checkBox.replaceChildren(...out);
    fillBtn.hidden = !(missing.length && manifest);
  }

  async function doFill() {
    if (!check.manifest) return;
    fillBtn.disabled = checkBtn.disabled = true;
    checkBox.prepend(h('p', {class: 'muted'}, 'Scaricamento in corso...'));
    try {
      await fillMissing(check.manifest);
    } catch (e) {
      console.error(e);
      if (!closed) checkBox.prepend(h('p', {class: 'err'}, `Scaricamento non riuscito: ${errText(e)}`));
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
      say(exportMsg, `Esportate ${ids.length} sessioni in ${name}.`);
    } catch (e) {
      say(exportMsg, `File creato, ma l'export non e' stato registrato (${errText(e)}): `
        + 'le sessioni non verranno cancellate.', true);
    }
    await refresh();
  }

  function offerDownload(file, ids) {
    clearLink();
    blobUrl = URL.createObjectURL(file);
    const a = h('a', {href: blobUrl, download: file.name, class: 'op-download'},
      `Scarica ${file.name} (${ids.length} sessioni)`);
    a.addEventListener('click', () => markExported(ids, file.name));
    linkBox.append(a);
  }

  // Il file dell'export: tutte le sessioni del dispositivo, preparate in refresh().
  function exportFile() {
    if (!prepared || prepared.ids.length === 0) { say(exportMsg, 'Nessuna sessione da esportare.'); return null; }
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

  // Menu di condivisione (AirDrop, mail, Salva su File): share() va chiamato subito, nel tocco.
  function doShare() {
    clearLink();
    const ex = exportFile();
    if (!ex) return;
    const {file, ids} = ex;
    let canShare = false;
    try { canShare = !!(navigator.canShare && navigator.canShare({files: [file]})); } catch (e) { canShare = false; }
    if (!canShare) {
      say(exportMsg, 'Condivisione non disponibile: scaricare il file dal link.');
      offerDownload(file, ids);
      return;
    }
    shareBtn.disabled = true;
    say(exportMsg, 'Condivisione in corso...');
    navigator.share({files: [file]}).then(
      () => markExported(ids, file.name),
      e => {
        if (closed) return;
        if (e && e.name === 'AbortError') {
          say(exportMsg, 'Condivisione annullata: le sessioni non risultano esportate.');
        } else {
          console.error(e);
          say(exportMsg, `Condivisione non riuscita (${errText(e)}): scaricare il file dal link.`, true);
          offerDownload(file, ids);
        }
      }).finally(() => { if (!closed) shareBtn.disabled = prepared == null || prepared.ids.length === 0; });
  }

  // Cancella solo le sessioni dell'ultimo export ancora presenti; poi l'export si azzera.
  async function doDelete() {
    deleteBtn.disabled = true;
    try {
      const [all, exported] = await Promise.all([getAllSessions(), getSetting('last_export_ids')]);
      const ids = deletableIds(all, exported || []);
      if (ids.length === 0) {
        say(deleteMsg, 'Nessuna sessione esportata da cancellare: prima esportare.');
        return;
      }
      const left = all.length - ids.length;
      if (!window.confirm(`Cancellare definitivamente ${ids.length} sessioni esportate? `
        + `Ne restano ${left} non esportate. L'operazione non si puo' annullare.`)) {
        say(deleteMsg, 'Cancellazione annullata.');
        return;
      }
      await deleteSessions(ids);
      await setSetting('last_export_ids', []);
      say(deleteMsg, `Cancellate ${ids.length} sessioni.`);
    } catch (e) {
      console.error(e);
      say(deleteMsg, `Cancellazione non riuscita: ${errText(e)}`, true);
    } finally {
      if (!closed) {
        deleteBtn.disabled = false;
        await refresh();
      }
    }
  }

  root.replaceChildren(...[
    corner,
    h('h1', {}, 'Audioguide: operatore'),
    h('p', {class: 'muted'}, `app ${APP_VERSION}`),
    message && h('p', {class: 'err'}, message),
    saveWarning(),
    start, startNote,
    pin.form,
    panel,
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
