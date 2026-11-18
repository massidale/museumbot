import test from 'node:test';
import assert from 'node:assert/strict';
import {mediaUrls, heardStep, backTo} from '../player.js';

test('heardStep conta l\'ascolto continuo, non i salti', () => {
  assert.equal(heardStep(10, 10.25, false), 0.25);
  assert.equal(heardStep(10, 40, false), 0);        // salto avanti col cursore
  assert.equal(heardStep(40, 30, false), 0);        // indietro di 10 s
  assert.equal(heardStep(10, 10.25, true), 0);      // salto in corso
});
test('backTo: indietro di 10 s, non prima dell\'inizio', () => {
  assert.equal(backTo(25), 15);
  assert.equal(backTo(4), 0);
  assert.equal(backTo(NaN), 0);
});
test('mediaUrls: immagini e audio delle opere, senza doppioni', () => {
  const m = {artworks: [{image: 'img/a.jpg', audio: {en: {flat: {src: 'audio/en/a/flat.mp3'}}}},
                       {image: 'img/a.jpg', audio: {en: {flat: {src: 'audio/en/a/flat.mp3'}}}}]};
  assert.deepEqual(mediaUrls(m).sort(),
    ['audio/en/a/flat.mp3', 'img/a.jpg']);
});

test('prefetch + loadBlob: un solo fetch per lo stesso file', async () => {
  const {prefetch, loadBlob} = await import('../player.js');
  let calls = 0;
  globalThis.fetch = async () => { calls++; return {ok: true, blob: async () => new Blob(['x'], {type: 'audio/mpeg'})}; };
  prefetch(['audio/a.mp3'], null);
  const b = await loadBlob('audio/a.mp3', null);
  assert.equal(b.size, 1);
  await loadBlob('audio/a.mp3', null);   // la voce e' stata tolta: secondo fetch
  assert.equal(calls, 2);
});
test('prefetch fallito: loadBlob ripiega sul fetch', async () => {
  const {prefetch, loadBlob} = await import('../player.js');
  let calls = 0;
  globalThis.fetch = async () => {
    calls++;
    if (calls === 1) throw new Error('rete');
    return {ok: true, blob: async () => new Blob(['y'], {type: 'audio/mpeg'})};
  };
  prefetch(['audio/b.mp3'], null);
  await new Promise(r => setTimeout(r, 0));
  const b = await loadBlob('audio/b.mp3', null);
  assert.equal(b.size, 1);
  assert.equal(calls, 2);
});
