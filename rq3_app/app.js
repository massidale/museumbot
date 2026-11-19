// Percorso del visitatore: dalla lingua ai ringraziamenti, con il record della sessione
// salvato in IndexedDB dopo ogni passo. Tutti i testi per il visitatore vengono da texts.json.
import {
  LANGS, validateManifest, planSession, newSession, buildTrial,
  answerComplete, fitComplete, answer, markStep, abortSession, nextTime,
} from './logic.js';
import {createPlayer, prefetch} from './player.js';
import {isQrMode, nextCount} from './qr.js';
import {APP_VERSION} from './config.js';
import {putSession, getSetting} from './store.js';
import {registerServiceWorker} from './swclient.js';
import {startSync, syncSoon} from './sync.js';
import {MAX_WHY} from './validate.js';
import {showOperator as operatorScreen, onSwStatus, mountAbortGate, DEFAULT_DEVICE} from './operator.js';

const MAX_ARTWORKS = 4;
const RETURN_MS = 30 * 1000;    // dopo i saluti si torna da soli all'operatore

const app = document.getElementById('app');
let texts = null;      // texts.json intero
let T = null;          // testi della lingua scelta
let S = null;          // sessione in corso: manifest, plan, record, ...
let cleanups = [];     // da chiamare quando si cambia schermata
let unmountGate = null;  // angolo dell'operatore sulle schermate del visitatore

let lastMs = 0;
const now = () => { lastMs = nextTime(lastMs, Date.now()); return new Date(lastMs).toISOString(); };

// Modo QR (telefono del visitatore): si attiva con ?m=qr e resta per la scheda.
const QR_SESSIONS_KEY = 'rq3_qr_sessions';
function detectQr() {
  let stored = null;
  try { stored = sessionStorage.getItem('rq3_mode'); } catch (e) { /* niente sessionStorage */ }
  const qr = isQrMode(location.search, stored);
  if (qr) { try { sessionStorage.setItem('rq3_mode', 'qr'); } catch (e) { /* ignorato */ } }
  return qr;
}
const QR = detectQr();

// Sessioni precedenti di questo telefono; poi si salva count + 1.
function earlierSessions() {
  try {
    const count = nextCount(localStorage.getItem(QR_SESSIONS_KEY));
    localStorage.setItem(QR_SESSIONS_KEY, String(count + 1));
    return count;
  } catch (e) { return 0; }
}

// ---- utilita' del DOM ----

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

// Sostituisce {n}, {i}, ... con i valori dati.
const fmt = (s, vars) => s.replace(/\{(\w+)\}/g, (m, k) => (k in vars ? String(vars[k]) : m));

// Id della sessione in fondo a ogni schermata del visitatore (per segnare errori e esclusioni).
const sessionTag = cls => (cls !== 'operator' && S && S.sessionId
  ? h('p', {class: 'session-id', lang: 'en'}, `ID ${S.sessionId}`) : null);

// Nuova schermata: chiude la precedente (audio, timer) e torna in cima.
function screen(cls, ...kids) {
  for (const f of cleanups) { try { f(); } catch (e) { console.error(e); } }
  cleanups = [];
  app.className = cls || '';
  app.replaceChildren(...kids.flat().filter(Boolean), ...[sessionTag(cls)].filter(Boolean));
  window.scrollTo(0, 0);
}

const paras = list => list.map(s => h('p', {}, s));

// Pulsante a scelta (aria-pressed); onPick al tocco.
const option = (label, onPick, cls = 'opt') =>
  h('button', {type: 'button', class: cls, 'aria-pressed': 'false', onclick: onPick}, label);

function press(buttons, active) {
  for (const b of buttons) b.setAttribute('aria-pressed', String(b === active));
}

// ---- salvataggio ----

// Le scritture vanno in coda, nell'ordine; una copia per non salvare modifiche successive.
let saving = Promise.resolve();
function save() {
  const copy = structuredClone(S.record);
  saving = saving.then(() => putSession(copy)).then(() => { syncSoon(); }).catch(e => console.error('salvataggio non riuscito', e));
  return saving;
}

function step(name, patch = {}) {
  S.record = markStep({...S.record, ...patch}, name, now());
  return save();
}

function putTrial(trial) {
  const trials = [...S.record.trials];
  trials[trial.index] = trial;
  S.record = {...S.record, trials, updated_at: now()};
  return save();
}

// Errore di un audio: dettagli nel record (e in console dal player); al visitatore solo
// il testo neutro.
function playerError(trialIndex, position, {src, message}) {
  if (!S || !S.record) return;
  const entry = {trial_index: trialIndex, position, src, message, at: now()};
  S.record = {...S.record, player_errors: [...(S.record.player_errors || []), entry], updated_at: now()};
  save();
}

// ---- identificativi ----

function randomU32() {
  return crypto.getRandomValues(new Uint32Array(1))[0];
}

// 6 caratteri base36.
function newSessionId() {
  return Array.from(crypto.getRandomValues(new Uint32Array(6)), v => (v % 36).toString(36)).join('');
}

// ---- operatore (operator.js) ----

function showOperator(message = '') {
  S = null;
  if (unmountGate) { unmountGate(); unmountGate = null; }
  document.documentElement.lang = 'it';
  screen('operator');
  cleanups.push(operatorScreen(app, {onStart: startSession, message}));
}

// L'operatore interrompe la sessione (tocco lungo, PIN, conferma). screen() in
// showOperator ferma l'audio; il record resta incompleto, col passo a cui si era fermato.
async function abortByOperator() {
  if (QR) return;
  if (S && S.record && !S.record.completed) {
    S.record = abortSession(S.record, now());
    await save();
  }
  showOperator();
}

// ---- percorso del visitatore ----

export async function startSession() {
  try {
    const manifest = await (await fetch('manifest.json')).json();
    const errs = validateManifest(manifest);
    if (errs.length) throw new Error(`manifest not valid: ${errs.slice(0, 3).join('; ')}`);
    if (!texts) texts = await (await fetch('texts.json')).json();
    let device = 'qr';
    if (!QR) {
      device = DEFAULT_DEVICE;
      try { device = (await getSetting('device')) || DEFAULT_DEVICE; } catch (e) { console.error(e); }
    }
    const seed = randomU32();
    const plan = planSession(manifest, seed);
    S = {
      manifest, seed, plan, device, sessionId: newSessionId(), record: null,
      total: Math.min(MAX_ARTWORKS, plan.artworkOrder.length), categories: null,
    };
    if (!QR && !unmountGate) unmountGate = mountAbortGate(abortByOperator);
    showLanguage();
  } catch (e) {
    console.error(e);
    const message = `Could not start the session: ${e.message || e}`;
    if (QR) screen('', h('p', {class: 'lead'}, message));   // dal telefono niente operatore
    else showOperator(message);
  }
}

function showLanguage() {
  document.documentElement.lang = 'en';
  const pick = lang => {
    S.lang = lang;
    T = texts[lang];
    document.documentElement.lang = lang;
    // Telefono: il flat della prima opera si scarica subito (e' il confronto per tutti).
    if (QR) prefetch([artwork(0).audio[lang].flat.src], S.manifest.version);
    showInfo();
  };
  // La parete della galleria con un cartellino per lingua.
  screen('lang',
    h('div', {class: 'lang-head'}, LANGS.map(l => h('p', {class: 'lang-prompt', lang: l}, texts[l].choose_language))),
    h('div', {class: 'lang-labels'}, LANGS.map(l =>
      h('button', {type: 'button', class: 'lang-label', lang: l, onclick: () => pick(l)},
        h('span', {class: 'lang-name'}, texts[l].lang_name),
        h('span', {class: 'lang-go'}, texts[l].lang_go)))));
}

// Informativa: l'accettazione fissa l'ora del consenso.
function showInfo() {
  screen('',
    h('h1', {}, T.info_title),
    paras(T.info_body),
    h('div', {class: 'actions'}, h('button', {type: 'button', onclick: () => {
      S.consentAt = now();
      showIntro();
    }}, T.info_accept)));
}

function showIntro() {
  screen('',
    h('h1', {}, T.intro_title),
    paras(T.intro_body),
    h('h2', {}, T.intro_steps_title),
    h('ol', {class: 'steps'}, T.intro_steps.map(s => h('li', {}, s))),
    h('p', {class: 'note'}, T.intro_note),
    h('div', {class: 'actions'}, h('button', {type: 'button', onclick: begin}, T.continue)));
}

// Il record nasce quando il visitatore comincia (dopo l'introduzione).
function begin() {
  const record = {
    ...newSession({
      sessionId: S.sessionId, device: S.device, appVersion: APP_VERSION, manifestVersion: S.manifest.version,
      seed: S.seed, lang: S.lang, now: now(), consentAt: S.consentAt,
    }),
    description_order: [...S.plan.descriptionOrder],
  };
  if (QR) record.earlier_sessions = earlierSessions();
  S.record = record;
  save();
  showClassification();
}

// Descrizioni come pulsanti; il nome della categoria non compare mai.
function descriptionCards(cats, onToggle) {
  return cats.map(c => {
    const b = option(T.falk[c], () => onToggle(c, b), 'card');
    b.dataset.cat = c;
    return b;
  });
}

function showClassification() {
  const selected = [];
  const counter = h('p', {class: 'counter'});
  const next = h('button', {type: 'button'}, T.continue);
  const cards = descriptionCards(S.plan.descriptionOrder, c => {
    const k = selected.indexOf(c);
    if (k >= 0) selected.splice(k, 1);
    else if (selected.length < 2) selected.push(c);
    refresh();
  });
  const refresh = () => {
    for (const b of cards) {
      const on = selected.includes(b.dataset.cat);
      b.setAttribute('aria-pressed', String(on));
      b.disabled = !on && selected.length >= 2;
    }
    counter.textContent = fmt(T.class_selected, {n: selected.length});
    next.disabled = selected.length === 0;
  };
  next.addEventListener('click', () => {
    // le scelte nell'ordine in cui sono mostrate
    const chosen = S.plan.descriptionOrder.filter(c => selected.includes(c));
    step('classification', {chosen: [...selected]});  // nel record: nell'ordine in cui sono state toccate
    showFit(chosen);
  });
  refresh();
  screen('',
    h('h1', {}, T.class_title),
    h('p', {class: 'lead'}, T.class_hint),
    h('div', {class: 'stack'}, cards),
    h('div', {class: 'actions'}, counter, next));
}

// Quanto descrive il visitatore ogni descrizione scelta (1-7); nessuna principale.
function showFit(chosen) {
  const fit = {};
  const next = h('button', {type: 'button', disabled: true}, T.continue);
  const items = chosen.map(c => h('div', {class: 'question'},
    h('p', {class: 'fit-desc'}, T.falk[c]),
    scale(v => { fit[c] = v; next.disabled = !fitComplete(chosen, fit); }, T.fit_low, T.fit_high)));
  next.addEventListener('click', () => {
    if (!fitComplete(chosen, fit)) return;
    S.categories = chosen;
    step('fit', {categories: [...chosen], category_fit: {...fit}});
    // Telefono: ora si sa la personalizzata della prima opera (il flat e' gia' richiesto).
    if (QR) prefetch(buildTrial(0, artwork(0), S.lang, S.categories, S.plan.firsts[0]).audio, S.manifest.version);
    showTask();
  });
  screen('',
    h('h1', {}, chosen.length === 1 ? T.fit_title_one : T.fit_title_two),
    h('div', {class: 'stack'}, items),
    h('div', {class: 'actions'}, next));
}

function showTask() {
  screen('',
    h('h1', {}, T.task_title),
    h('p', {class: 'lead'}, T.task_body),
    h('div', {class: 'actions'}, h('button', {type: 'button', onclick: () => showTrial(0)}, T.continue)));
}

const artwork = i => S.manifest.artworks.find(a => a.id === S.plan.artworkOrder[i]);

// Pulsanti a scelta singola: items = [[valore, etichetta], ...].
function choiceButtons(items, onPick, cls = 'choices') {
  const buttons = items.map(([value, label]) => {
    const b = option(label, () => { press(buttons, b); onPick(value); });
    return b;
  });
  return h('div', {class: cls}, buttons);
}

// Scala 1-7 con le etichette agli estremi.
const scale = (onPick, low, high) => h('div', {},
  choiceButtons([1, 2, 3, 4, 5, 6, 7].map(n => [n, String(n)]), onPick, 'scale'),
  h('div', {class: 'scale-ends'}, h('span', {}, low), h('span', {}, high)));


function showTrial(i) {
  const art = artwork(i);
  let trial = {...buildTrial(i, art, S.lang, S.categories, S.plan.firsts[i]), shown_at: now()};
  S.record = markStep({...S.record, trials: [...S.record.trials.slice(0, i), trial]}, `trial_${i}`, now());
  save();

  const ans = {ratings: [null, null]};
  const labels = {play: T.play, pause: T.pause, back10: T.back10, seek: T.seek, error: T.player_error};
  const guideLabel = n => fmt(T.guide_label, {n});

  // Guide: ascolto libero (avanti, indietro, cursore); si registra quanto si e' sentito.
  const guides = h('div', {class: 'guides'});
  const players = [];
  const withStats = (t, pos, s) => {
    const upd = (arr, v) => arr.map((x, k) => (k === pos ? v : x));
    return {...t, plays: upd(t.plays, s.plays), reached_end: upd(t.reached_end, t.reached_end[pos] || s.ended),
            heard_sec: upd(t.heard_sec, s.heard)};
  };
  // Il voto di ogni guida subito sotto il suo player.
  const rating = pos => h('div', {class: 'question'},
    h('p', {class: 'q'}, fmt(T.q_rating, {guide: guideLabel(pos + 1)})),
    scale(v => { ans.ratings[pos] = v; refresh(); }, T.rating_low, T.rating_high));
  [0, 1].forEach(pos => {
    const box = h('div', {class: 'guide'});
    guides.append(box);
    players.push(createPlayer(box, trial.audio[pos], guideLabel(pos + 1),
                              s => { trial = withStats(trial, pos, s); putTrial(trial); }, labels,
                              S.manifest.version, err => playerError(i, pos + 1, err)));
    box.append(rating(pos));
  });

  if (i + 1 < S.total) {   // le guide dell'opera dopo si scaricano mentre si risponde
    prefetch(buildTrial(i + 1, artwork(i + 1), S.lang, S.categories, S.plan.firsts[i + 1]).audio, S.manifest.version);
  }

  const why = h('textarea', {rows: '3', maxlength: String(MAX_WHY)});
  const next = h('button', {type: 'button'}, T.next);
  const questions = h('div', {class: 'questions'},
    h('div', {class: 'question'}, h('label', {class: 'q'}, T.q_why_text, why)));

  const refresh = () => { next.disabled = !answerComplete(ans); };

  next.addEventListener('click', () => {
    if (!answerComplete(ans)) return;
    // lo stato dei player in questo momento (un audio puo' stare ancora suonando)
    players.forEach((p, pos) => { trial = withStats(trial, pos, p.stats()); });
    trial = answer(trial, {...ans, why: why.value.trim(), now: now()});
    putTrial(trial);
    if (i + 1 < S.total) showMore(i);
    else finish();
  });

  screen('wide',
    h('div', {class: 'trial'},
      h('figure', {class: 'painting'},
        h('p', {class: 'kicker'}, fmt(T.painting_of, {i: i + 1, n: S.total})),
        h('img', {src: art.image, alt: art.title[S.lang]}),
        h('figcaption', {}, h('h1', {}, art.title[S.lang]), h('p', {class: 'artist'}, art.artist))),
      h('div', {class: 'panel'}, guides, questions)),
    h('div', {class: 'actions'}, next));
  cleanups.push(() => players.forEach(p => p.destroy()));  // dopo screen()
  refresh();
}

function showMore(i) {
  screen('',
    h('h1', {}, T.more_title),
    h('div', {class: 'stack bottom'},
      h('button', {type: 'button', class: 'big', onclick: () => { step(`more_${i}`); showTrial(i + 1); }}, T.more_yes),
      h('button', {type: 'button', class: 'big secondary', onclick: () => { step(`more_${i}`); finish(); }}, T.more_no)));
}

// Fine della sessione, dopo l'ultima opera o quando il visitatore dice basta.
function finish() {
  step('end', {completed: true, finished_at: now()});
  goodbye('end_title', 'end_body');
}

// Saluti; poi, a un tocco o da soli, si torna all'operatore.
function goodbye(titleKey, bodyKey) {
  if (QR) { screen('', h('h1', {}, T[titleKey]), h('p', {class: 'lead'}, T[bodyKey])); return; }
  let armed = false;
  const back = () => { if (armed) showOperator(); };
  const timer = setTimeout(() => { armed = true; back(); }, RETURN_MS);
  const arm = setTimeout(() => { armed = true; }, 1500);
  document.addEventListener('click', back);
  screen('', h('h1', {}, T[titleKey]), h('p', {class: 'lead'}, T[bodyKey]));
  cleanups.push(() => { clearTimeout(timer); clearTimeout(arm); document.removeEventListener('click', back); });
}

if (QR) {
  startSync();
  startSession();
} else {
  registerServiceWorker(onSwStatus);
  startSync();
  showOperator();
}
