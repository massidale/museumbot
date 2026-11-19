// Controllo del record di sessione, in comune fra app e server (nessun import).
// REQUIRED come in rq3_visitors/collect.py.
export const REQUIRED = ['session_id', 'device', 'app_version', 'manifest_version', 'seed',
  'lang', 'started_at', 'updated_at', 'completed', 'last_step', 'trials'];
export const MAX_BYTES = 65536;   // lo controlla il server sul corpo
export const MAX_TRIALS = 4;
export const MAX_WHY = 2000;
export const SESSION_ID_RE = /^[0-9a-z]{6}$/;
const LANGS = ['en', 'de'];
const CHOICES = ['personal', 'comparison'];
const isScore = v => Number.isInteger(v) && v >= 1 && v <= 7;
const DEVICE_RE = /^[A-Za-z0-9_-]{1,40}$/;   // come validDevice di operator.js

function checkTrial(t, i) {
  if (t === null || typeof t !== 'object' || Array.isArray(t)) return [`prova ${i}: non e' un oggetto`];
  const out = [];
  if (t.choice != null && !CHOICES.includes(t.choice)) out.push(`prova ${i}: choice non ammessa`);
  const r = t.ratings;
  if (r != null && !(Array.isArray(r) && r.length === 2 && r.every(v => v === null || isScore(v))))
    out.push(`prova ${i}: ratings non ammessi`);
  if (t.why != null && (typeof t.why !== 'string' || t.why.length > MAX_WHY))
    out.push(`prova ${i}: why non ammesso`);
  return out;
}

// Problemi del record, uno per messaggio; vuoto se va bene.
export function validateRecord(rec) {
  if (rec === null || typeof rec !== 'object' || Array.isArray(rec)) return ['il record non e\' un oggetto'];
  const out = REQUIRED.filter(k => !(k in rec)).map(k => `manca ${k}`);
  if ('session_id' in rec && !(typeof rec.session_id === 'string' && SESSION_ID_RE.test(rec.session_id)))
    out.push('session_id non ammesso');
  if ('completed' in rec && typeof rec.completed !== 'boolean') out.push('completed non ammesso');
  for (const k of ['started_at', 'updated_at'])
    if (k in rec && typeof rec[k] !== 'string') out.push(`${k} non ammesso`);
  if ('device' in rec && !(typeof rec.device === 'string' && DEVICE_RE.test(rec.device)))
    out.push('device non ammesso');
  if ('lang' in rec && !LANGS.includes(rec.lang)) out.push('lang non ammessa');
  const fit = rec.category_fit;
  if (fit != null && !(typeof fit === 'object' && !Array.isArray(fit) && Object.values(fit).every(isScore)))
    out.push('category_fit non ammesso');
  if ('trials' in rec) {
    if (!Array.isArray(rec.trials)) out.push('trials non e\' una lista');
    else if (rec.trials.length > MAX_TRIALS) out.push(`trials: piu' di ${MAX_TRIALS} prove`);
    else rec.trials.forEach((t, i) => out.push(...checkTrial(t, i)));
  }
  return out;
}

// Prove con una scelta.
export const answered = rec =>
  (Array.isArray(rec.trials) ? rec.trials : []).filter(t => t && t.choice != null).length;

// Come collect._rank: completo, piu' risposte, piu' recente.
export const rank = rec => [rec.completed ? 1 : 0, answered(rec), rec.updated_at || ''];

// true se next non e' piu' indietro di prev (ordine lessicografico).
export function notBehind(next, prev) {
  const a = rank(next), b = rank(prev);
  for (let i = 0; i < a.length; i++) {
    if (a[i] > b[i]) return true;
    if (a[i] < b[i]) return false;
  }
  return true;
}
