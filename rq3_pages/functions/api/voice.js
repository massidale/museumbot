// PUT /api/voice?id=<session_id>: registrazione di fine giro, in R2 (chiave = session_id).
// GET /api/voice (solo col token): elenco; GET /api/voice?id=<session_id>: il file.
import {precheckVoice, decideVoice, authorized} from '../../lib/server.js';

const json = (obj, status) => new Response(JSON.stringify(obj), {
  status, headers: {'content-type': 'application/json'}});

export async function onRequestPut({request, env}) {
  const id = new URL(request.url).searchParams.get('id');
  const key = request.headers.get('X-Write-Key');
  const type = request.headers.get('Content-Type');
  const length = request.headers.get('Content-Length');
  const bad = precheckVoice({id, key, type, size: length == null ? null : Number(length)});
  if (bad) return json({ok: false, error: bad.error}, bad.status);
  const existing = await env.DB.prepare('SELECT write_key, record FROM sessions WHERE session_id = ?')
    .bind(id).first();
  const out = decideVoice({existing, key});
  if (out.status !== 200) return json({ok: false, error: out.error}, out.status);
  const body = await request.arrayBuffer();
  const late = precheckVoice({id, key, type, size: body.byteLength});
  if (late) return json({ok: false, error: late.error}, late.status);
  await env.VOICE.put(id, body, {httpMetadata: {contentType: type},
                                 customMetadata: {received_at: new Date().toISOString()}});
  return json({ok: true}, 200);
}

export async function onRequestGet({request, env}) {
  if (!authorized(request.headers.get('Authorization'), env.EXPORT_TOKEN))
    return json({ok: false, error: 'non autorizzato'}, 401);
  const id = new URL(request.url).searchParams.get('id');
  if (id) {
    const obj = await env.VOICE.get(id);
    if (!obj) return json({ok: false, error: 'non trovata'}, 404);
    return new Response(obj.body, {headers: {'content-type': obj.httpMetadata?.contentType || 'application/octet-stream'}});
  }
  const out = [];
  let cursor;
  do {
    const page = await env.VOICE.list({cursor, include: ['httpMetadata']});
    for (const o of page.objects)
      out.push({session_id: o.key, size: o.size, type: o.httpMetadata?.contentType || '', uploaded: o.uploaded});
    cursor = page.truncated ? page.cursor : undefined;
  } while (cursor);
  return json(out, 200);
}
