import test from 'node:test';
import assert from 'node:assert/strict';
import {LIMITS, KEY_RE, decideWrite, authorized, precheck} from '../lib/server.js';
import {MAX_BYTES} from '../../rq3_app/validate.js';
import {newSession, buildTrial, answer, allConditions, CATEGORIES} from '../../rq3_app/logic.js';

const KEY = 'a'.repeat(32);
const OTHER = 'b'.repeat(32);
const NOW = '2026-11-18T12:00:00Z';

function artwork() {
  const audio = {};
  for (const l of ['en', 'de']) {
    audio[l] = {};
    for (const c of allConditions()) audio[l][c] = {src: `audio/${l}/${c}.mp3`, sec: 60};
  }
  return {id: 'a1', image: 'img/a1.jpg', title: {en: 'T', de: 'T'}, artist: 'X', audio};
}

// Record con una prova risposta.
function rec(patch = {}) {
  const s = newSession({sessionId: 'abc123', device: 'qr', appVersion: '0.2.0',
    manifestVersion: 'v', seed: 1, lang: 'en', now: '2026-11-18T10:00:00Z'});
  const t = answer(buildTrial(0, artwork(), 'en', [CATEGORIES[0]], 'personal'),
    {position: 1, strength: 4, why: 'perche', now: '2026-11-18T10:01:00Z'});
  return {...s, trials: [t], ...patch};
}
const call = (r, over = {}) =>
  decideWrite({body: JSON.stringify(r), key: KEY, existing: null, newLastHour: 0, now: NOW, ...over});
// Riga come la leggerebbe D1.
const rowOf = (r, over = {}) => ({session_id: r.session_id, write_key: KEY, device: r.device,
  completed: r.completed ? 1 : 0, answered: 1, updated_at: r.updated_at,
  created_at: '2026-11-18T10:00:30Z', received_at: '2026-11-18T10:02:00Z', writes: 5,
  record: JSON.stringify(r), ...over});

test('costanti', () => {
  assert.deepEqual(LIMITS, {newPerHour: 300, writesPerSession: 300});
  assert.ok(KEY_RE.test(KEY)); assert.ok(!KEY_RE.test('A'.repeat(32))); assert.ok(!KEY_RE.test('a'));
});
test('corpo troppo lungo: 400', () => {
  const big = JSON.stringify(rec()) + ' '.repeat(MAX_BYTES);
  assert.equal(decideWrite({body: big, key: KEY, existing: null, newLastHour: 0, now: NOW}).status, 400);
});
test('lunghezza contata in byte, non in caratteri', () => {
  const body = JSON.stringify(rec()).replace(/\}$/, ',"x":"' + 'é'.repeat(MAX_BYTES / 2) + '"}');
  assert.ok(body.length < MAX_BYTES);
  assert.equal(decideWrite({body, key: KEY, existing: null, newLastHour: 0, now: NOW}).status, 400);
});
test('JSON non valido: 400', () => {
  const d = decideWrite({body: '{no', key: KEY, existing: null, newLastHour: 0, now: NOW});
  assert.equal(d.status, 400); assert.equal(d.write, null); assert.ok(d.error);
});
test('record non valido: 400', () => {
  assert.equal(call(rec({lang: 'fr'})).status, 400);
});
test('chiave con formato sbagliato o assente: 400', () => {
  assert.equal(call(rec(), {key: 'xyz'}).status, 400);
  assert.equal(call(rec(), {key: null}).status, 400);
  assert.equal(call(rec(), {key: 'A'.repeat(32)}).status, 400);
});
test('riga esistente con chiave diversa: 403', () => {
  const r = rec();
  const d = call(r, {existing: rowOf(r, {write_key: OTHER})});
  assert.equal(d.status, 403); assert.equal(d.write, null);
});
test('riga nuova oltre il limite orario: 429', () => {
  assert.equal(call(rec(), {newLastHour: 300}).status, 429);
  assert.equal(call(rec(), {newLastHour: 299}).status, 200);
});
test('il limite orario non vale per una riga esistente', () => {
  const r = rec();
  assert.equal(call(r, {existing: rowOf(r), newLastHour: 999}).status, 200);
});
test('riga esistente oltre i 300 invii: 429', () => {
  const r = rec();
  assert.equal(call(r, {existing: rowOf(r, {writes: 300})}).status, 429);
  assert.equal(call(r, {existing: rowOf(r, {writes: 299})}).status, 200);
});
test('invio piu\' vecchio dopo uno piu\' avanti: 200 senza scrittura', () => {
  const base = rec();
  const ahead = {...base, trials: [base.trials[0], base.trials[0]]};
  const d = call(base, {existing: rowOf(ahead)});
  assert.equal(d.status, 200); assert.equal(d.write, null);
  const older = {...base, updated_at: '2026-11-18T09:00:00Z'};
  assert.equal(call(older, {existing: rowOf(base)}).write, null);
});
test('riga nuova: scrittura con writes 1 e created_at ora', () => {
  const r = rec({completed: true});
  const d = call(r);
  assert.equal(d.status, 200);
  assert.deepEqual(d.write, {session_id: 'abc123', write_key: KEY, device: 'qr', completed: 1,
    answered: 1, updated_at: r.updated_at, created_at: NOW, received_at: NOW, writes: 1,
    record: JSON.stringify(r)});
});
test('riga esistente: created_at resta, writes cresce di 1', () => {
  const base = rec();
  const next = {...base, updated_at: '2026-11-18T10:05:00Z'};
  const d = call(next, {existing: rowOf(base)});
  assert.equal(d.status, 200);
  assert.equal(d.write.created_at, '2026-11-18T10:00:30Z');
  assert.equal(d.write.received_at, NOW);
  assert.equal(d.write.writes, 6);
  assert.equal(d.write.updated_at, next.updated_at);
});
test('record riletto con JSON.parse uguale al record inviato', () => {
  const r = rec({consent_at: '2026-11-18T10:00:00Z', earlier_sessions: 2});
  assert.deepEqual(JSON.parse(call(r).write.record), r);
});
test('la riga ha solo i campi attesi, nessun dato del visitatore oltre al record', () => {
  const d = call(rec());
  assert.deepEqual(Object.keys(d.write).sort(), ['answered', 'completed', 'created_at', 'device',
    'received_at', 'record', 'session_id', 'updated_at', 'write_key', 'writes']);
});
test('authorized: solo con token impostato e intestazione identica', () => {
  assert.equal(authorized('Bearer t0k', 't0k'), true);
  assert.equal(authorized('Bearer x', 't0k'), false);
  assert.equal(authorized(null, 't0k'), false);
  assert.equal(authorized('Bearer undefined', undefined), false);
  assert.equal(authorized('Bearer ', ''), false);
});

test('precheck: record e chiave buoni', () => {
  const out = precheck(JSON.stringify(rec()), KEY);
  assert.equal(out.rec.session_id, rec().session_id);
});
test('precheck: stessi errori 400 di decideWrite', () => {
  assert.equal(precheck('x'.repeat(MAX_BYTES + 1), KEY).error, 'corpo troppo lungo');
  assert.equal(precheck('{non json', KEY).error, 'JSON non valido');
  assert.equal(precheck('{}', KEY).status, 400);
  assert.equal(precheck(JSON.stringify(rec()), 'zz').error, 'chiave non valida');
  assert.equal(precheck(JSON.stringify(rec()), null).status, 400);
});
