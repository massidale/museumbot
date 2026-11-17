import test from 'node:test';
import assert from 'node:assert/strict';
import {
  CATEGORIES, mixKey, allConditions, pairFor, rng, shuffle, validateManifest,
  planSession, newSession, buildTrial, answerComplete,
  answer, markStep, abortSession, toJsonl, deletableIds,
} from '../logic.js';

// Manifesto finto di 4 opere, stessa forma del Task 2.
function fakeManifest() {
  const audio = {};
  for (const l of ['en', 'de']) {
    audio[l] = {};
    for (const c of allConditions()) audio[l][c] = {src: `audio/${l}/${c}.mp3`, sec: 60};
  }
  return {
    version: 'v1', categories: CATEGORIES,
    sample: {en: {src: 'audio/en/sample.mp3', sec: 8}, de: {src: 'audio/de/sample.mp3', sec: 8}},
    artworks: [1, 2, 3, 4].map(i => ({
      id: `a${i}`, image: `img/a${i}.jpg`, title: {en: `T${i}`, de: `T${i}`},
      artist: 'X', audio: structuredClone(audio),
    })),
  };
}
const M = fakeManifest();

test('mixKey segue l\'ordine di Falk', () => {
  assert.equal(mixKey('recharger', 'professional_hobbyist'), 'professional_hobbyist+recharger');
  assert.equal(mixKey('explorer', 'facilitator'), 'explorer+facilitator');
  assert.throws(() => mixKey('explorer', 'explorer'));
  assert.throws(() => mixKey('explorer', 'tourist'));
});
test('16 condizioni', () => {
  const c = allConditions();
  assert.equal(c.length, 16); assert.equal(new Set(c).size, 16);
  assert.ok(c.includes('flat') && c.includes('experience_seeker+recharger'));
});
test('pairFor: una categoria contro flat, due contro la principale', () => {
  assert.deepEqual(pairFor(['recharger']), {personal: 'recharger', comparison: 'flat'});
  assert.deepEqual(pairFor(['recharger', 'explorer']),
                   {personal: 'explorer+recharger', comparison: 'recharger'});
  for (const bad of [[], ['explorer', 'explorer'], ['a'], ['explorer', 'facilitator', 'recharger']])
    assert.throws(() => pairFor(bad));
});
test('planSession e\' deterministica e copre le 4 opere', () => {
  const a = planSession(M, 42), b = planSession(M, 42);
  assert.deepEqual(a, b);
  assert.deepEqual([...a.artworkOrder].sort(), M.artworks.map(x => x.id).sort());
  assert.deepEqual([...a.descriptionOrder].sort(), [...CATEGORIES].sort());
  assert.equal(a.firsts.length, 4);
});
test('planSession: entrambi gli ordini delle guide escono su molti seed', () => {
  const f = Array.from({length: 200}, (_, s) => planSession(M, s).firsts[0]);
  const p = f.filter(x => x === 'personal').length;
  assert.ok(p > 70 && p < 130);
});
test('rng e shuffle', () => {
  const r = rng(7);
  for (let i = 0; i < 100; i++) { const v = r(); assert.ok(v >= 0 && v < 1); }
  const arr = [1, 2, 3, 4, 5];
  const s = shuffle(arr, rng(1));
  assert.deepEqual(arr, [1, 2, 3, 4, 5]);
  assert.deepEqual([...s].sort(), arr);
});
test('buildTrial mette gli audio nell\'ordine di ascolto', () => {
  const t = buildTrial(0, M.artworks[0], 'de', ['explorer', 'facilitator'], 'comparison');
  assert.equal(t.personal, 'explorer+facilitator'); assert.equal(t.comparison, 'explorer');
  assert.deepEqual(t.audio, [M.artworks[0].audio.de.explorer.src,
                             M.artworks[0].audio.de['explorer+facilitator'].src]);
});
test('buildTrial parte con contatori di ascolto a zero', () => {
  const t = buildTrial(0, M.artworks[0], 'en', ['recharger'], 'personal');
  assert.deepEqual(t.plays, [0, 0]); assert.deepEqual(t.reached_end, [false, false]);
  assert.deepEqual(t.heard_sec, [0, 0]);
  assert.equal('listened_full' in t, false);
});
test('newSession: niente campo consent, primo passo start', () => {
  const s = newSession({sessionId: 'x', device: 'ipad-1', appVersion: '0.1.0',
    manifestVersion: 'v', seed: 1, lang: 'en', now: 'A'});
  assert.equal('consent' in s, false); assert.equal(s.last_step, 'start');
});
test('answer traduce la posizione nella condizione', () => {
  const t = buildTrial(0, M.artworks[0], 'en', ['recharger'], 'comparison');
  const r = answer(t, {position: 2, strength: 5, reasons: ['suited'], familiar: false, why: '', now: 'T'});
  assert.equal(r.choice, 'personal'); assert.equal(r.choice_position, 2);
  assert.deepEqual(r.reasons, ['suited']); assert.equal(r.answered_at, 'T');
  assert.equal(t.choice, null);
});
test('answerComplete chiede posizione, forza, almeno un motivo e familiarita\'', () => {
  const ok = {position: 1, strength: 7, reasons: ['tone'], familiar: true};
  assert.equal(answerComplete(ok), true);
  assert.equal(answerComplete({...ok, reasons: []}), false);
  assert.equal(answerComplete({...ok, reasons: ['boh']}), false);
  assert.equal(answerComplete({...ok, strength: 0}), false);
  assert.equal(answerComplete({...ok, familiar: null}), false);
});
test('markStep aggiorna passo e ora', () => {
  const s0 = newSession({sessionId: 'x', device: 'ipad-1', appVersion: '0.1.0',
    manifestVersion: 'v', seed: 1, lang: 'en', now: 'A'});
  const s = markStep(s0, 'classification', 'B');
  assert.equal(s.last_step, 'classification'); assert.equal(s.updated_at, 'B');
  assert.equal(s.completed, false);
  assert.equal(s0.last_step, 'start');
});
test('validateManifest segnala l\'audio mancante', () => {
  const bad = structuredClone(M); delete bad.artworks[2].audio.de['explorer+recharger'];
  const e = validateManifest(bad);
  assert.equal(e.length, 1); assert.match(e[0], /explorer\+recharger/);
  assert.deepEqual(validateManifest(M), []);
  assert.ok(validateManifest({version: 'v'}).length > 0);
});
test('validateManifest chiede la prova del volume in ogni lingua', () => {
  const noDe = structuredClone(M); delete noDe.sample.de;
  const e = validateManifest(noDe);
  assert.equal(e.length, 1); assert.match(e[0], /prova del volume/); assert.match(e[0], /de/);
  const none = structuredClone(M); delete none.sample;
  assert.equal(validateManifest(none).length, 2);
});
test('abortSession tiene il passo e segna l\'interruzione', () => {
  const s0 = markStep(newSession({sessionId: 'x', device: 'ipad-1', appVersion: '0.1.0',
    manifestVersion: 'v', seed: 1, lang: 'en', now: 'A'}), 'trial_1', 'B');
  const s = abortSession(s0, 'C');
  assert.equal(s.last_step, 'trial_1'); assert.equal(s.completed, false);
  assert.equal(s.aborted_by_operator, true); assert.equal(s.updated_at, 'C');
  assert.equal(s0.aborted_by_operator, undefined);
});
test('deletableIds tocca solo gli esportati presenti', () => {
  assert.deepEqual(deletableIds([{session_id: 'a'}, {session_id: 'b'}], ['b', 'z']), ['b']);
  assert.deepEqual(deletableIds([{session_id: 'a'}], []), []);
});
test('toJsonl', () => {
  assert.equal(toJsonl([{a: 1}, {b: 2}]), '{"a":1}\n{"b":2}\n');
});
