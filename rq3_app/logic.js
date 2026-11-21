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

// [k] -> k contro flat; [k, j] (in qualsiasi ordine) -> misto contro flat.
export function pairFor(categories) {
  if (!Array.isArray(categories) || categories.length < 1 || categories.length > 2)
    throw new Error('servono una o due categorie');
  for (const c of categories)
    if (!CATEGORIES.includes(c)) throw new Error(`categoria ignota: ${c}`);
  if (categories.length === 1) return {personal: categories[0], comparison: 'flat'};
  return {personal: mixKey(categories[0], categories[1]), comparison: 'flat'};
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

// Ordine del giro: quello fissato dall'operatore (id del manifesto) e in coda le opere che non
// ci sono, nell'ordine casuale; senza ordine fissato, quello casuale.
export function tourOrder(randomOrder, fixed) {
  if (!Array.isArray(fixed) || !fixed.length) return [...randomOrder];
  const head = fixed.filter((id, i) => randomOrder.includes(id) && fixed.indexOf(id) === i);
  return [...head, ...randomOrder.filter(id => !head.includes(id))];
}

// Sposta l'elemento i di d posti (-1 su, +1 giu'); fuori dai bordi la lista resta uguale.
export function moveItem(list, i, d) {
  const j = i + d, out = [...list];
  if (i < 0 || i >= list.length || j < 0 || j >= list.length) return out;
  [out[i], out[j]] = [out[j], out[i]];
  return out;
}

export function newSession({sessionId, device, appVersion, manifestVersion, seed, lang, now, consentAt = null}) {
  return {
    session_id: sessionId, device, app_version: appVersion, manifest_version: manifestVersion,
    seed, lang, started_at: now, updated_at: now, finished_at: null,
    last_step: 'start', completed: false, consent_at: consentAt,
    categories: null, category_fit: null, trials: [],
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
    ratings: [null, null], voice: null,
    shown_at: null, answered_at: null,
  };
}

const isScore = v => Number.isInteger(v) && v >= 1 && v <= 7;

// Un voto 1-7 per guida; la registrazione e' facoltativa.
export const answerComplete = ({ratings}) =>
  Array.isArray(ratings) && ratings.length === 2 && ratings.every(isScore);

// Un voto 1-7 per ogni categoria scelta.
export const fitComplete = (chosen, fit) => chosen.every(c => isScore(fit[c]));

// I voti nell'ordine di ascolto (Guida 1, Guida 2), come plays e heard_sec; voice come quella
// finale: {sec, mime, recorded_at} o {skipped: true[, no_mic: true]}.
export const answer = (trial, {ratings, voice, now}) =>
  ({...trial, ratings: [...ratings], voice: voice ?? null, answered_at: now});

// Opera saltata dal visitatore: niente voti, si passa oltre.
export const skipTrial = (trial, now) =>
  ({...trial, ratings: [null, null], voice: null, skipped: true, skipped_at: now});

export const markStep = (session, step, now) => ({...session, last_step: step, updated_at: now});

// Sessione chiusa dall'operatore: resta incompleta, col passo a cui si era fermata.
export const abortSession = (session, now) =>
  ({...session, completed: false, aborted_by_operator: true, updated_at: now});

// Export: le sessioni come lista JSON leggibile (un record per sessione, rientrato).
export const toJson = records => JSON.stringify(records, null, 2) + '\n';

export function deletableIds(records, exportedIds) {
  const present = new Set(records.map(r => r.session_id));
  return exportedIds.filter(id => present.has(id));
}

// Orologio strettamente crescente: ogni salvataggio ha un updated_at diverso.
export const nextTime = (lastMs, nowMs) => Math.max(nowMs, lastMs + 1);
