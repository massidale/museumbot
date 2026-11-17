// Percorso del visitatore: dalla lingua ai ringraziamenti, con il record della sessione
// salvato in IndexedDB dopo ogni passo. Tutti i testi per il visitatore vengono da texts.json.
import {
  LANGS, REASONS, validateManifest, planSession, newSession, buildTrial,
  answerComplete, answer, markStep, abortSession,
} from './logic.js';
import {createPlayer, loadBlob} from './player.js';
import {APP_VERSION} from './config.js';
import {putSession, getSetting} from './store.js';
import {registerServiceWorker} from './swclient.js';
import {showOperator as operatorScreen, onSwStatus, mountAbortGate, DEFAULT_DEVICE} from './operator.js';

const MAX_ARTWORKS = 4;
const RETURN_MS = 30 * 1000;    // dopo i saluti si torna da soli all'operatore

const app = document.getElementById('app');
let texts = null;      // texts.json intero
let T = null;          // testi della lingua scelta
let S = null;          // sessione in corso: manifest, plan, record, ...
let cleanups = [];     // da chiamare quando si cambia schermata
let unmountGate = null;  // angolo dell'operatore sulle schermate del visitatore

const now = () => new Date().toISOString();

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

// Nuova schermata: chiude la precedente (audio, timer) e torna in cima.
function screen(cls, ...kids) {
  for (const f of cleanups) { try { f(); } catch (e) { console.error(e); } }
  cleanups = [];
  app.className = cls || '';
  app.replaceChildren(...kids.flat().filter(Boolean));
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
  saving = saving.then(() => putSession(copy)).catch(e => console.error('salvataggio non riuscito', e));
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
// il testo neutro. trialIndex e position sono null per la prova del volume.
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
    if (errs.length) throw new Error(`manifesto non valido: ${errs.slice(0, 3).join('; ')}`);
    if (!texts) texts = await (await fetch('texts.json')).json();
    let device = DEFAULT_DEVICE;
    try { device = (await getSetting('device')) || DEFAULT_DEVICE; } catch (e) { console.error(e); }
    const seed = randomU32();
    const plan = planSession(manifest, seed);
    S = {
      manifest, seed, plan, device, sessionId: newSessionId(), record: null,
      total: Math.min(MAX_ARTWORKS, plan.artworkOrder.length), categories: null,
    };
    if (!unmountGate) unmountGate = mountAbortGate(abortByOperator);
    showLanguage();
  } catch (e) {
    console.error(e);
    showOperator(`Impossibile avviare la sessione: ${e.message || e}`);
  }
}

function showLanguage() {
  document.documentElement.lang = 'en';
  const pick = lang => {
    S.lang = lang;
    T = texts[lang];
    document.documentElement.lang = lang;
    showIntro();
  };
  screen('',
    LANGS.map(l => h('p', {class: 'lead', lang: l}, texts[l].choose_language)),
    h('div', {class: 'stack bottom'},
      LANGS.map(l => h('button', {type: 'button', class: 'big', lang: l, onclick: () => pick(l)}, texts[l].lang_name))));
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
  S.record = {
    ...newSession({
      sessionId: S.sessionId, device: S.device, appVersion: APP_VERSION, manifestVersion: S.manifest.version,
      seed: S.seed, lang: S.lang, now: now(),
    }),
    description_order: [...S.plan.descriptionOrder],
  };
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
    if (chosen.length === 1) setCategories(chosen[0], null);
    else showMain(chosen);
  });
  refresh();
  screen('',
    h('h1', {}, T.class_title),
    h('p', {class: 'lead'}, T.class_hint),
    h('div', {class: 'stack'}, cards),
    h('div', {class: 'actions'}, counter, next));
}

function showMain(chosen) {
  let main = null;
  const next = h('button', {type: 'button', disabled: true}, T.continue);
  const cards = descriptionCards(chosen, (c, b) => {
    main = c;
    press(cards, b);
    next.disabled = false;
  });
  next.addEventListener('click', () => setCategories(main, chosen.find(c => c !== main)));
  screen('',
    h('h1', {}, T.main_title),
    h('div', {class: 'stack'}, cards),
    h('div', {class: 'actions'}, next));
}

function setCategories(main, secondary) {
  S.categories = secondary ? [main, secondary] : [main];
  step('main', {categories: {main, secondary}});
  showVolume();
}

// Audio come blob, come nel player (Safari e le richieste a intervalli).
const blobUrl = async (src, version) => URL.createObjectURL(await loadBlob(src, version));

// Prova del volume: un audio neutro per lingua (manifest.sample), non una delle guide.
function showVolume() {
  const sample = S.manifest.sample[S.lang].src;
  const audio = new Audio();
  let url = null, closed = false, failed = false;
  const play = h('button', {type: 'button', class: 'big secondary', disabled: true}, T.volume_play);
  const errBox = h('p', {class: 'player-error', hidden: true}, T.player_error);
  const fail = message => {
    if (closed || failed) return;
    failed = true;
    play.disabled = true;
    errBox.hidden = false;
    console.error('prova del volume:', sample, message);
    playerError(null, null, {src: sample, message});
  };
  play.addEventListener('click', () => {
    audio.pause();
    audio.currentTime = 0;
    audio.play().catch(e => console.warn('play non riuscito', e));
  });
  audio.addEventListener('error', () => {
    if (!url) return;
    const e = audio.error;
    fail(e ? `MediaError ${e.code}${e.message ? `: ${e.message}` : ''}` : 'errore audio');
  });
  blobUrl(sample, S.manifest.version).then(u => {
    if (closed) { URL.revokeObjectURL(u); return; }
    url = u;
    audio.src = u;
    audio.load();
    play.disabled = false;
  }).catch(e => fail(String((e && e.message) || e)));
  const ok = h('button', {type: 'button', class: 'big', onclick: () => { step('volume'); showTask(); }}, T.volume_ok);
  screen('',
    h('h1', {}, T.volume_title),
    h('p', {class: 'lead'}, T.volume_body),
    h('div', {class: 'stack bottom'}, play, errBox, ok));
  // dopo screen(), che chiude la schermata precedente
  cleanups.push(() => {
    closed = true;
    audio.pause();
    const u = url;
    url = null;  // prima di togliere src: l'eventuale 'error' che segue non conta
    audio.removeAttribute('src');
    audio.load();
    if (u) URL.revokeObjectURL(u);
  });
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

// Gli stessi con la loro domanda.
const choiceGroup = (question, items, onPick, cls) =>
  h('div', {class: 'question'}, h('p', {class: 'q'}, question), choiceButtons(items, onPick, cls));

function showTrial(i) {
  const art = artwork(i);
  let trial = {...buildTrial(i, art, S.lang, S.categories, S.plan.firsts[i]), shown_at: now()};
  S.record = markStep({...S.record, trials: [...S.record.trials.slice(0, i), trial]}, `trial_${i}`, now());
  save();

  const ans = {position: null, strength: null, reasons: [], familiar: null};
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
  [0, 1].forEach(pos => {
    const box = h('div', {class: 'guide'});
    guides.append(box);
    players.push(createPlayer(box, trial.audio[pos], guideLabel(pos + 1),
                              s => { trial = withStats(trial, pos, s); putTrial(trial); }, labels,
                              S.manifest.version, err => playerError(i, pos + 1, err)));
  });

  const why = h('textarea', {rows: '3'});
  const next = h('button', {type: 'button'}, T.next);
  const reasonBoxes = REASONS.map(r => {
    const box = h('input', {type: 'checkbox', value: r});
    box.addEventListener('change', () => {
      ans.reasons = REASONS.filter(id => reasonBoxes.some(b => b.value === id && b.checked));
      refresh();
    });
    return box;
  });
  const questions = h('div', {class: 'questions'},
    choiceGroup(T.q_choice, [[1, guideLabel(1)], [2, guideLabel(2)]], v => { ans.position = v; refresh(); }),
    h('div', {class: 'question'},
      h('p', {class: 'q'}, T.q_strength),
      choiceButtons([1, 2, 3, 4, 5, 6, 7].map(n => [n, String(n)]), v => { ans.strength = v; refresh(); }, 'scale'),
      h('div', {class: 'scale-ends'}, h('span', {}, T.strength_low), h('span', {}, T.strength_high))),
    h('div', {class: 'question'},
      h('p', {class: 'q'}, T.q_reasons),
      h('div', {class: 'reasons'}, reasonBoxes.map(b => h('label', {class: 'reason'}, b, h('span', {}, T.reasons[b.value]))))),
    h('div', {class: 'question'}, h('label', {class: 'q'}, T.q_why_text, why)),
    choiceGroup(T.q_familiar, [[true, T.yes], [false, T.no]], v => { ans.familiar = v; refresh(); }));

  const refresh = () => { next.disabled = !answerComplete(ans); };

  next.addEventListener('click', () => {
    if (!answerComplete(ans)) return;
    // lo stato dei player in questo momento (un audio puo' stare ancora suonando)
    players.forEach((p, pos) => { trial = withStats(trial, pos, p.stats()); });
    trial = answer(trial, {...ans, why: why.value.trim(), now: now()});
    putTrial(trial);
    if (i + 1 < S.total) showMore(i);
    else showFinal();
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
      h('button', {type: 'button', class: 'big secondary', onclick: () => { step(`more_${i}`); showFinal(); }}, T.more_no)));
}

// Eta', frequenza di visita, lingua madre: si salvano gli id delle opzioni.
function showFinal() {
  const keys = Object.keys(T.final);
  const final = Object.fromEntries(keys.map(k => [k, null]));
  const finish = h('button', {type: 'button', disabled: true}, T.finish);
  const groups = keys.map(k => choiceGroup(T.final[k].label, Object.entries(T.final[k].options), v => {
    final[k] = v;
    finish.disabled = !keys.every(q => final[q] != null);
  }, 'choices wrap'));
  finish.addEventListener('click', () => {
    if (!keys.every(q => final[q] != null)) return;
    step('final', {final: {...final}});
    step('end', {completed: true, finished_at: now()});
    goodbye('end_title', 'end_body');
  });
  screen('', h('h1', {}, T.final_title), groups, h('div', {class: 'actions'}, finish));
}

// Saluti; poi, a un tocco o da soli, si torna all'operatore.
function goodbye(titleKey, bodyKey) {
  let armed = false;
  const back = () => { if (armed) showOperator(); };
  const timer = setTimeout(() => { armed = true; back(); }, RETURN_MS);
  const arm = setTimeout(() => { armed = true; }, 1500);
  document.addEventListener('click', back);
  screen('', h('h1', {}, T[titleKey]), h('p', {class: 'lead'}, T[bodyKey]));
  cleanups.push(() => { clearTimeout(timer); clearTimeout(arm); document.removeEventListener('click', back); });
}

registerServiceWorker(onSwStatus);
showOperator();
