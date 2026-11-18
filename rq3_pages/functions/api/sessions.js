// GET /api/sessions: esporta i record (solo con il token).
import {authorized} from '../../lib/server.js';

const json = (obj, status) => new Response(JSON.stringify(obj), {
  status, headers: {'content-type': 'application/json'}});

export async function onRequestGet({request, env}) {
  if (!authorized(request.headers.get('Authorization'), env.EXPORT_TOKEN))
    return json({ok: false, error: 'non autorizzato'}, 401);
  const {results} = await env.DB.prepare('SELECT record FROM sessions ORDER BY created_at').all();
  return json(results.map(r => JSON.parse(r.record)), 200);
}
