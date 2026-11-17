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
test('mediaUrls comprende le prove del volume', () => {
  const m = {sample: {en: {src: 'audio/en/sample.mp3'}, de: {src: 'audio/de/sample.mp3'}},
             artworks: [{image: 'img/a.jpg', audio: {en: {flat: {src: 'audio/en/a/flat.mp3'}}}}]};
  assert.deepEqual(mediaUrls(m).sort(),
    ['audio/de/sample.mp3', 'audio/en/a/flat.mp3', 'audio/en/sample.mp3', 'img/a.jpg']);
});
