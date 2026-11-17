// Logica pura dell'app RQ3: coppie di guide, piano della sessione, record.

export const CATEGORIES = ['explorer', 'facilitator', 'experience_seeker',
                           'professional_hobbyist', 'recharger'];
export const LANGS = ['en', 'de'];

// Nome di un misto: le due categorie nell'ordine di Falk, unite da '+'.
export function mixKey(a, b) {
  const ia = CATEGORIES.indexOf(a), ib = CATEGORIES.indexOf(b);
  if (ia < 0 || ib < 0) throw new Error(`categoria ignota: ${ia < 0 ? a : b}`);
  if (ia === ib) throw new Error(`categorie uguali: ${a}`);
  return ia < ib ? `${a}+${b}` : `${b}+${a}`;
}

// flat, 5 pure, 10 miste (in ordine).
export function allConditions() {
  const out = ['flat', ...CATEGORIES];
  for (let i = 0; i < CATEGORIES.length; i++)
    for (let j = i + 1; j < CATEGORIES.length; j++) out.push(mixKey(CATEGORIES[i], CATEGORIES[j]));
  return out;
}

// [k] -> k contro flat; [k, j] (principale prima) -> misto contro k.
export function pairFor(categories) {
  if (!Array.isArray(categories) || categories.length < 1 || categories.length > 2)
    throw new Error('servono una o due categorie');
  for (const c of categories)
    if (!CATEGORIES.includes(c)) throw new Error(`categoria ignota: ${c}`);
  if (categories.length === 1) return {personal: categories[0], comparison: 'flat'};
  return {personal: mixKey(categories[0], categories[1]), comparison: categories[0]};
}

// mulberry32: valori in [0, 1).
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Fisher-Yates su una copia.
export function shuffle(arr, rand) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// Un errore leggibile per ogni cosa che manca.
export function validateManifest(m) {
  const errs = [];
  if (!m || !Array.isArray(m.artworks) || m.artworks.length === 0) {
    return ['manifesto senza elenco di opere (artworks)'];
  }
  const conds = allConditions();
  for (const art of m.artworks) {
    for (const lang of LANGS) {
      const la = art.audio && art.audio[lang];
      if (!la) { errs.push(`opera ${art.id}: lingua ${lang} mancante`); continue; }
      for (const c of conds)
        if (!la[c] || !la[c].src) errs.push(`opera ${art.id}, lingua ${lang}: audio mancante per ${c}`);
    }
  }
  return errs;
}

// Un solo flusso rng(seed): descrizioni, opere, poi un 'first' per opera.
export function planSession(manifest, seed) {
  const rand = rng(seed);
  const descriptionOrder = shuffle(CATEGORIES, rand);
  const artworkOrder = shuffle(manifest.artworks.map(a => a.id), rand);
  const firsts = artworkOrder.map(() => (rand() < 0.5 ? 'personal' : 'comparison'));
  return {descriptionOrder, artworkOrder, firsts};
}

export function newSession({sessionId, device, appVersion, manifestVersion, seed, lang, now}) {
  return {
    session_id: sessionId, device, app_version: appVersion, manifest_version: manifestVersion,
    seed, lang, started_at: now, updated_at: now, finished_at: null,
    last_step: 'start', completed: false,
    categories: null, trials: [],
  };
}

export function buildTrial(index, artwork, lang, categories, first) {
  const {personal, comparison} = pairFor(categories);
  const src = c => artwork.audio[lang][c].src;
  const audio = first === 'personal' ? [src(personal), src(comparison)]
                                     : [src(comparison), src(personal)];
  return {
    index, artwork_id: artwork.id, personal, comparison, first, audio,
    plays: [0, 0], reached_end: [false, false], heard_sec: [0, 0],
    choice: null, choice_position: null, strength: null,
    why: '', shown_at: null, answered_at: null,
  };
}

// Il perche' e' una domanda aperta facoltativa: non conta.
export function answerComplete({position, strength}) {
  return (position === 1 || position === 2)
    && Number.isInteger(strength) && strength >= 1 && strength <= 7;
}

// La posizione (1|2) diventa la condizione scelta, secondo l'ordine di ascolto.
export function answer(trial, {position, strength, why, now}) {
  const second = trial.first === 'personal' ? 'comparison' : 'personal';
  const choice = position === 1 ? trial.first : second;
  return {
    ...trial, choice, choice_position: position, strength, why: why || '', answered_at: now,
  };
}

export const markStep = (session, step, now) => ({...session, last_step: step, updated_at: now});

// Sessione chiusa dall'operatore: resta incompleta, col passo a cui si era fermata.
export const abortSession = (session, now) =>
  ({...session, completed: false, aborted_by_operator: true, updated_at: now});

// ---- export in CSV: una riga per opera, coi dati della sessione ripetuti ----
// Le stesse colonne le legge rq3_visitors/collect.py.

export const CSV_COLUMNS = [
  'session_id', 'device', 'app_version', 'manifest_version', 'seed', 'lang',
  'started_at', 'updated_at', 'finished_at', 'completed', 'last_step', 'aborted_by_operator',
  'chosen', 'description_order', 'category_main', 'category_secondary', 'player_errors',
  'trial_index', 'artwork_id', 'personal', 'comparison', 'first', 'audio_1', 'audio_2',
  'plays_1', 'plays_2', 'heard_sec_1', 'heard_sec_2', 'reached_end_1', 'reached_end_2',
  'choice', 'choice_position', 'strength', 'why', 'shown_at', 'answered_at',
];

// Un valore della cella: vuoto per null, liste unite da '|', testo tra virgolette se serve.
export function csvCell(v) {
  if (v == null) return '';
  const s = Array.isArray(v) ? v.join('|') : String(v);
  return /[",\r\n]|^\s|\s$/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

// Le righe di una sessione; una senza opere da' una riga con le colonne dell'opera vuote.
export function sessionRows(r) {
  const cats = r.categories || {};
  const errs = r.player_errors && r.player_errors.length ? JSON.stringify(r.player_errors) : null;
  const base = {
    session_id: r.session_id, device: r.device, app_version: r.app_version,
    manifest_version: r.manifest_version, seed: r.seed, lang: r.lang,
    started_at: r.started_at, updated_at: r.updated_at, finished_at: r.finished_at,
    completed: r.completed, last_step: r.last_step, aborted_by_operator: r.aborted_by_operator || false,
    chosen: r.chosen, description_order: r.description_order,
    category_main: cats.main, category_secondary: cats.secondary, player_errors: errs,
  };
  const pair = (row, name, arr) => { row[`${name}_1`] = arr ? arr[0] : null; row[`${name}_2`] = arr ? arr[1] : null; };
  if (!r.trials || r.trials.length === 0) return [base];
  return r.trials.map(t => {
    const row = {...base, trial_index: t.index, artwork_id: t.artwork_id, personal: t.personal,
      comparison: t.comparison, first: t.first, choice: t.choice, choice_position: t.choice_position,
      strength: t.strength, why: t.why, shown_at: t.shown_at, answered_at: t.answered_at};
    pair(row, 'audio', t.audio);
    pair(row, 'plays', t.plays);
    pair(row, 'heard_sec', t.heard_sec);
    pair(row, 'reached_end', t.reached_end);
    return row;
  });
}

// CSV con BOM (Excel lo legge come UTF-8), virgola e righe CRLF.
export function toCsv(records) {
  const lines = [CSV_COLUMNS.join(',')];
  for (const r of records)
    for (const row of sessionRows(r)) lines.push(CSV_COLUMNS.map(c => csvCell(row[c])).join(','));
  return '﻿' + lines.join('\r\n') + '\r\n';
}

export function deletableIds(records, exportedIds) {
  const present = new Set(records.map(r => r.session_id));
  return exportedIds.filter(id => present.has(id));
}
