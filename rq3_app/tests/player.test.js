import test from 'node:test';
import assert from 'node:assert/strict';
import {mediaUrls, seekTarget, reached, endedCounts} from '../player.js';

test("seekTarget: indietro si', avanti oltre il punto raggiunto no", () => {
  assert.equal(seekTarget(0, 40), null);           // ricomincia da capo
  assert.equal(seekTarget(20, 40), null);          // torna indietro
  assert.equal(seekTarget(40.4, 40), null);        // entro la tolleranza
  assert.equal(seekTarget(85, 40), 40);            // salto avanti: si torna a 40
  assert.equal(seekTarget(10, 0), 0);
});
test('reached cresce con la riproduzione, non durante un salto', () => {
  assert.equal(reached(10, 10.25, false), 10.25);
  assert.equal(reached(10, 3, false), 10);         // riascolto: il massimo resta
  assert.equal(reached(10, 80, true), 10);         // salto in corso
});
test('endedCounts: la fine vale solo se ci si e\' arrivati ascoltando', () => {
  assert.equal(endedCounts(89.6, 90), true);
  assert.equal(endedCounts(40, 90), false);
  assert.equal(endedCounts(0, NaN), false);
});
test('mediaUrls comprende le prove del volume', () => {
  const m = {sample: {en: {src: 'audio/en/sample.mp3'}, de: {src: 'audio/de/sample.mp3'}},
             artworks: [{image: 'img/a.jpg', audio: {en: {flat: {src: 'audio/en/a/flat.mp3'}}}}]};
  assert.deepEqual(mediaUrls(m).sort(),
    ['audio/de/sample.mp3', 'audio/en/a/flat.mp3', 'audio/en/sample.mp3', 'img/a.jpg']);
});
