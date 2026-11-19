import test from 'node:test';
import assert from 'node:assert/strict';
import {
  REQUIRED, MAX_BYTES, MAX_TRIALS, MAX_WHY, SESSION_ID_RE,
  validateRecord, answered, rank, notBehind,
} from '../validate.js';
import {newSession, buildTrial, answer, allConditions, CATEGORIES} from '../logic.js';

function artwork() {
  const audio = {};
  for (const l of ['en', 'de']) {
    audio[l] = {};
    for (const c of allConditions()) audio[l][c] = {src: `audio/${l}/${c}.mp3`, sec: 60};
  }
  return {id: 'a1', image: 'img/a1.jpg', title: {en: 'T', de: 'T'}, artist: 'X', audio};
}

// Record valido: una prova risposta, come nella 0.1.0 (senza consent_at).
function valid() {
  const s = newSession({sessionId: 'abc123', device: 'qr', appVersion: '0.1.0',
    manifestVersion: 'v', seed: 1, lang: 'en', now: '2026-11-18T10:00:00Z'});
  const t = answer(buildTrial(0, artwork(), 'en', [CATEGORIES[0]], 'personal'),
    {ratings: [4, 6], why: 'perche', now: '2026-11-18T10:01:00Z'});
  return {...s, trials: [t]};
}
const withTrial = (rec, patch) => ({...rec, trials: [{...rec.trials[0], ...patch}]});

test('costanti', () => {
  assert.equal(MAX_BYTES, 65536); assert.equal(MAX_TRIALS, 5); assert.equal(MAX_WHY, 2000);
  assert.ok(SESSION_ID_RE.test('a1b2c3'));
  assert.equal(REQUIRED.length, 11);
});
test('il record valido non ha problemi, anche senza campi nuovi', () => {
  assert.deepEqual(validateRecord(valid()), []);
  const r = {...valid(), consent_at: '2026-11-18T10:00:00Z', earlier_sessions: 1};
  assert.deepEqual(validateRecord(r), []);
});
test('un problema per ogni errore', () => {
  const r = valid();
  const {seed, ...noSeed} = r;
  assert.equal(validateRecord(noSeed).length, 1);
  assert.equal(validateRecord({...r, session_id: 'ABC123'}).length, 1);
  assert.equal(validateRecord({...r, lang: 'fr'}).length, 1);
  assert.equal(validateRecord({...r, trials: Array(6).fill(r.trials[0])}).length, 1);
  for (const ratings of [[8, 1], [2.5, 3], [1], 'x'])
    assert.equal(validateRecord(withTrial(r, {ratings})).length, 1, String(ratings));
  for (const category_fit of [{explorer: 9}, {explorer: 2.5}, 'x'])
    assert.equal(validateRecord({...r, category_fit}).length, 1, JSON.stringify(category_fit));
  assert.equal(validateRecord(withTrial(r, {why: 'a'.repeat(2001)})).length, 1);
  assert.equal(validateRecord({...r, trials: 'no'}).length, 1);
});
test('prova non risposta: voti e answered_at null vanno bene', () => {
  const r = withTrial(valid(), {ratings: [null, null], answered_at: null});
  assert.deepEqual(validateRecord(r), []);
  assert.deepEqual(validateRecord({...valid(), category_fit: null}), []);
  assert.deepEqual(validateRecord({...valid(), category_fit: {explorer: 7, recharger: 1}}), []);
  assert.equal(answered(r), 0); assert.equal(answered(valid()), 1);
});
test('record non oggetto', () => {
  assert.ok(validateRecord(null).length > 0);
  assert.ok(validateRecord([]).length > 0);
});
test('notBehind: completo, poi risposte, poi updated_at', () => {
  const base = valid();
  const done = {...base, completed: true, trials: []};
  const more = {...base, trials: [base.trials[0], base.trials[0]]};
  assert.deepEqual(rank(base), [0, 1, base.updated_at]);
  assert.equal(notBehind(done, more), true);
  assert.equal(notBehind(more, done), false);
  assert.equal(notBehind(more, base), true);
  assert.equal(notBehind(base, more), false);
  const later = {...base, updated_at: '2026-11-18T11:00:00Z'};
  assert.equal(notBehind(later, base), true);
  assert.equal(notBehind(base, later), false);
  assert.equal(notBehind(base, {...base}), true);
});

test('completed deve essere booleano', () => {
  assert.deepEqual(validateRecord({...valid(), completed: 'si'}), ['completed non ammesso']);
});
test('started_at e updated_at devono essere stringhe', () => {
  assert.deepEqual(validateRecord({...valid(), started_at: 5}), ['started_at non ammesso']);
  assert.deepEqual(validateRecord({...valid(), updated_at: null}), ['updated_at non ammesso']);
});
test('device: stringa con il formato dell\'operatore', () => {
  assert.deepEqual(validateRecord({...valid(), device: 'qr'}), []);
  assert.deepEqual(validateRecord({...valid(), device: 'ipad-1_A'}), []);
  assert.deepEqual(validateRecord({...valid(), device: 7}), ['device non ammesso']);
  assert.deepEqual(validateRecord({...valid(), device: 'a b'}), ['device non ammesso']);
  assert.deepEqual(validateRecord({...valid(), device: 'x'.repeat(41)}), ['device non ammesso']);
});
