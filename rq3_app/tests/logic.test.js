import test from 'node:test';
import assert from 'node:assert/strict';
import {
  CATEGORIES, mixKey, allConditions, pairFor, rng, shuffle, validateManifest,
  planSession, newSession, buildTrial, answerComplete,
  answer, fitComplete, markStep, abortSession, toJson, deletableIds, tourOrder, moveItem,
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
test('pairFor: la personalizzata (pura o mista) sempre contro flat', () => {
  assert.deepEqual(pairFor(['recharger']), {personal: 'recharger', comparison: 'flat'});
  assert.deepEqual(pairFor(['recharger', 'explorer']),
                   {personal: 'explorer+recharger', comparison: 'flat'});
  assert.deepEqual(pairFor(['explorer', 'recharger']), pairFor(['recharger', 'explorer']));
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
  assert.equal(t.personal, 'explorer+facilitator'); assert.equal(t.comparison, 'flat');
  assert.deepEqual(t.audio, [M.artworks[0].audio.de.flat.src,
                             M.artworks[0].audio.de['explorer+facilitator'].src]);
});
test('buildTrial parte con contatori di ascolto a zero', () => {
  const t = buildTrial(0, M.artworks[0], 'en', ['recharger'], 'personal');
  assert.deepEqual(t.plays, [0, 0]); assert.deepEqual(t.reached_end, [false, false]);
  assert.deepEqual(t.heard_sec, [0, 0]);
  assert.deepEqual(t.ratings, [null, null]);
  assert.equal('listened_full' in t, false); assert.equal('strength' in t, false);
  assert.equal('choice' in t, false); assert.equal('choice_position' in t, false);
});
test('newSession: niente campo consent, primo passo start', () => {
  const s = newSession({sessionId: 'x', device: 'ipad-1', appVersion: '0.1.0',
    manifestVersion: 'v', seed: 1, lang: 'en', now: 'A'});
  assert.equal('consent' in s, false); assert.equal('final' in s, false);
  assert.equal(s.last_step, 'start');
});
test('newSession scrive consent_at se dato, altrimenti null', () => {
  const base = {sessionId: 'x', device: 'ipad-1', appVersion: '0.2.0',
    manifestVersion: 'v', seed: 1, lang: 'en', now: 'A'};
  assert.equal(newSession({...base, consentAt: 'C'}).consent_at, 'C');
  assert.equal(newSession(base).consent_at, null);
});
test('answer salva i due voti, il perche\' e l\'ora', () => {
  const t = buildTrial(0, M.artworks[0], 'en', ['recharger'], 'comparison');
  const r = answer(t, {ratings: [3, 6], why: 'piu\' vivace', now: 'T'});
  assert.deepEqual(r.ratings, [3, 6]); assert.equal('strength' in r, false);
  assert.equal('choice' in r, false);
  assert.equal(r.why, 'piu\' vivace'); assert.equal(r.answered_at, 'T');
  assert.equal('reasons' in r, false); assert.equal('familiar' in r, false);
  assert.deepEqual(t.ratings, [null, null]);
});
test('answerComplete chiede i due voti; il perche\' e\' facoltativo', () => {
  const ok = {ratings: [4, 7]};
  assert.equal(answerComplete(ok), true);
  assert.equal(answerComplete({ratings: [4, 4]}), true);
  for (const ratings of [[4, null], [0, 7], [4, 7.5], [8, 1], [4], undefined])
    assert.equal(answerComplete({...ok, ratings}), false, String(ratings));
});
test('fitComplete chiede un voto 1-7 per ogni categoria scelta', () => {
  assert.equal(fitComplete(['explorer'], {explorer: 5}), true);
  assert.equal(fitComplete(['explorer', 'recharger'], {explorer: 5, recharger: 1}), true);
  assert.equal(fitComplete(['explorer', 'recharger'], {explorer: 5}), false);
  assert.equal(fitComplete(['explorer'], {explorer: 0}), false);
  assert.equal(fitComplete(['explorer'], {explorer: 2.5}), false);
  assert.equal(fitComplete(['explorer'], {}), false);
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
test('toJson: lista JSON rientrata che si rilegge uguale', () => {
  const recs = [{session_id: 'a', trials: [{why: 'Lebendiger, "kürzer"'}]}, {session_id: 'b', trials: []}];
  const text = toJson(recs);
  assert.deepEqual(JSON.parse(text), recs);
  assert.ok(text.startsWith('[\n  {') && text.endsWith(']\n'));
});

test('nextTime: orologio strettamente crescente', async () => {
  const {nextTime} = await import('../logic.js');
  assert.equal(nextTime(0, 1000), 1000);
  assert.equal(nextTime(1000, 1000), 1001);
  assert.equal(nextTime(1005, 1000), 1006);
  let t = 0;
  const seen = [];
  for (let k = 0; k < 5; k++) { t = nextTime(t, 1000); seen.push(t); }
  assert.deepEqual(seen, [1000, 1001, 1002, 1003, 1004]);
});

test('tourOrder: ordine dell\'operatore, in coda le opere mancanti; senza, quello casuale', () => {
  const rnd = ['a3', 'a1', 'a4', 'a2'];
  assert.deepEqual(tourOrder(rnd, null), rnd);
  assert.deepEqual(tourOrder(rnd, []), rnd);
  assert.deepEqual(tourOrder(rnd, ['a1', 'a2', 'a3', 'a4']), ['a1', 'a2', 'a3', 'a4']);
  assert.deepEqual(tourOrder(rnd, ['a2', 'zz', 'a1', 'a2']), ['a2', 'a1', 'a3', 'a4']);
  assert.notEqual(tourOrder(rnd, null), rnd);
});
test('moveItem scambia con il vicino e non esce dai bordi', () => {
  const l = ['a', 'b', 'c'];
  assert.deepEqual(moveItem(l, 1, -1), ['b', 'a', 'c']);
  assert.deepEqual(moveItem(l, 1, 1), ['a', 'c', 'b']);
  assert.deepEqual(moveItem(l, 0, -1), l);
  assert.deepEqual(moveItem(l, 2, 1), l);
  assert.deepEqual(l, ['a', 'b', 'c']);
});
