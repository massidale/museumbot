// POST /api/session: riceve il record di una sessione e lo salva se non e' piu' indietro.
import {decideWrite} from '../../lib/server.js';

const json = (obj, status) => new Response(JSON.stringify(obj), {
  status, headers: {'content-type': 'application/json'}});

// session_id dal corpo, se si legge; altrimenti null.
function sessionIdOf(body) {
  try {
    const id = JSON.parse(body).session_id;
    return typeof id === 'string' ? id : null;
  } catch { return null; }
}

const UPSERT = `INSERT INTO sessions (session_id, write_key, device, completed, answered,
  updated_at, created_at, received_at, writes, record) VALUES (?,?,?,?,?,?,?,?,?,?)
  ON CONFLICT(session_id) DO UPDATE SET write_key=excluded.write_key, device=excluded.device,
  completed=excluded.completed, answered=excluded.answered, updated_at=excluded.updated_at,
  created_at=excluded.created_at, received_at=excluded.received_at, writes=excluded.writes,
  record=excluded.record`;

export async function onRequestPost({request, env}) {
  const body = await request.text();
  const key = request.headers.get('X-Write-Key');
  const now = new Date().toISOString();
  const id = sessionIdOf(body);
  const existing = id
    ? await env.DB.prepare('SELECT * FROM sessions WHERE session_id = ?').bind(id).first() : null;
  const hourAgo = new Date(Date.now() - 3600 * 1000).toISOString();
  const recent = await env.DB.prepare('SELECT COUNT(*) AS n FROM sessions WHERE created_at > ?')
    .bind(hourAgo).first();
  const out = decideWrite({body, key, existing, newLastHour: recent.n, now});
  if (out.write) {
    const w = out.write;
    await env.DB.prepare(UPSERT).bind(w.session_id, w.write_key, w.device, w.completed,
      w.answered, w.updated_at, w.created_at, w.received_at, w.writes, w.record).run();
  }
  return out.status === 200 ? json({ok: true}, 200) : json({ok: false, error: out.error}, out.status);
}
