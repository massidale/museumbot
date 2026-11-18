import test from 'node:test';
import assert from 'node:assert/strict';
import {isQrMode, nextCount} from '../qr.js';

test('isQrMode: parametro m=qr o modo gia\' salvato', () => {
  assert.equal(isQrMode('?m=qr', null), true);
  assert.equal(isQrMode('', 'qr'), true);
  assert.equal(isQrMode('?m=x', null), false);
  assert.equal(isQrMode('', null), false);
});
test('nextCount: 0 se assente o non numerico', () => {
  assert.equal(nextCount(null), 0);
  assert.equal(nextCount('3'), 3);
  assert.equal(nextCount('abc'), 0);
});
