import test from 'node:test';
import assert from 'node:assert/strict';
import {pending, newKey} from '../sync.js';

const rec = (id, at) => ({session_id: id, updated_at: at});

test('pending: record mai inviato', () => {
  assert.deepEqual(pending([rec('a', '2026-01-01T10:00:00Z')], {}).map(r => r.session_id), ['a']);
});

test('pending: stesso updated_at gia\' inviato e\' escluso', () => {
  const r = rec('a', '2026-01-01T10:00:00Z');
  assert.deepEqual(pending([r], {a: r.updated_at}), []);
});

test('pending: inviato con updated_at vecchio e\' incluso', () => {
  const r = rec('a', '2026-01-01T10:05:00Z');
  assert.equal(pending([r], {a: '2026-01-01T10:00:00Z'}).length, 1);
});

test('pending: ordinati per updated_at', () => {
  const out = pending([rec('c', '2026-01-03'), rec('a', '2026-01-01'), rec('b', '2026-01-02')], {});
  assert.deepEqual(out.map(r => r.session_id), ['a', 'b', 'c']);
});

test('pending: non modifica l\'ingresso', () => {
  const recs = [rec('b', '2'), rec('a', '1')];
  pending(recs, {});
  assert.equal(recs[0].session_id, 'b');
});

test('newKey: 32 caratteri esadecimali minuscoli', () => {
  const k = newKey(new Uint8Array(16).map((_, i) => i * 17));
  assert.match(k, /^[0-9a-f]{32}$/);
  assert.equal(newKey(new Uint8Array(16)), '0'.repeat(32));
  assert.equal(newKey(new Uint8Array(16).fill(255)), 'f'.repeat(32));
});

import {fetchInit, classify, voiceInit} from '../sync.js';

test('fetchInit: keepalive, nessun redirect, timeout', () => {
  const i = fetchInit({a: 1}, 'k'.repeat(32));
  assert.equal(i.method, 'POST');
  assert.equal(i.keepalive, true);
  assert.equal(i.redirect, 'error');
  assert.ok(i.signal instanceof AbortSignal);
  assert.equal(i.headers['X-Write-Key'], 'k'.repeat(32));
  assert.equal(i.body, '{"a":1}');
});

const resp = (body, status) => new Response(body, {status});

test('classify: 200 con ok true e\' inviato', async () => {
  assert.equal(await classify(resp('{"ok":true}', 200)), 'ok');
});
test('classify: 200 HTML (portale) o senza ok e\' rete assente', async () => {
  assert.equal(await classify(resp('<html></html>', 200)), 'stop');
  assert.equal(await classify(resp('{"ok":false}', 200)), 'stop');
  assert.equal(await classify(resp('null', 200)), 'stop');
});
test('classify: 400 e 403 respinti, 429 saltato, altro stop', async () => {
  assert.equal(await classify(resp('{"error":"x"}', 400)), 'rejected');
  assert.equal(await classify(resp('{}', 403)), 'rejected');
  assert.equal(await classify(resp('{}', 429)), 'skip');
  assert.equal(await classify(resp('', 500)), 'stop');
});
test('classify: 404 e 409 (registrazione prima del suo record) saltati', async () => {
  assert.equal(await classify(resp('{}', 404)), 'skip');
  assert.equal(await classify(resp('{}', 409)), 'skip');
});
test('voiceInit: PUT col tipo e la chiave, senza keepalive', () => {
  const data = new ArrayBuffer(8);
  const i = voiceInit({data, mime: 'audio/mp4'}, 'k'.repeat(32));
  assert.equal(i.method, 'PUT'); assert.equal(i.body, data);
  assert.equal(i.headers['Content-Type'], 'audio/mp4');
  assert.equal(i.headers['X-Write-Key'], 'k'.repeat(32));
  assert.equal('keepalive' in i, false); assert.equal(i.redirect, 'error');
});
