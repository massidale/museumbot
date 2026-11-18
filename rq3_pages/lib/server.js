// Decisioni del server per le sessioni online: pure, senza D1 ne' Request.
import {validateRecord, notBehind, answered, MAX_BYTES} from '../../rq3_app/validate.js';

export const LIMITS = {newPerHour: 300, writesPerSession: 300};
export const KEY_RE = /^[0-9a-f]{32}$/;

const no = (status, error) => ({status, write: null, error});
const byteLength = s => new TextEncoder().encode(s).length;

// Esito dell'invio: scrittura (riga da salvare) o niente da scrivere.
export function decideWrite({body, key, existing, newLastHour, now}) {
  if (byteLength(body) > MAX_BYTES) return no(400, 'corpo troppo lungo');
  let rec;
  try { rec = JSON.parse(body); } catch { return no(400, 'JSON non valido'); }
  const problems = validateRecord(rec);
  if (problems.length) return no(400, problems.join('; '));
  if (typeof key !== 'string' || !KEY_RE.test(key)) return no(400, 'chiave non valida');
  if (existing && existing.write_key !== key) return no(403, 'chiave diversa');
  if (!existing && newLastHour >= LIMITS.newPerHour) return no(429, 'troppe sessioni nuove');
  if (existing && existing.writes >= LIMITS.writesPerSession) return no(429, 'troppi invii');
  if (existing && !notBehind(rec, JSON.parse(existing.record))) return {status: 200, write: null};
  return {status: 200, write: {
    session_id: rec.session_id, write_key: key, device: rec.device,
    completed: rec.completed ? 1 : 0, answered: answered(rec), updated_at: rec.updated_at,
    created_at: existing ? existing.created_at : now, received_at: now,
    writes: existing ? existing.writes + 1 : 1, record: JSON.stringify(rec)}};
}

// Esportazione: serve un token impostato e l'intestazione identica.
export const authorized = (header, token) =>
  typeof token === 'string' && token !== '' && header === 'Bearer ' + token;
