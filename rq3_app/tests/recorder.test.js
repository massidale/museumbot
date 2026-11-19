import test from 'node:test';
import assert from 'node:assert/strict';
import {pickMime, extFor, clock, MAX_VOICE_SEC} from '../recorder.js';
import {VOICE_TYPE_RE} from '../validate.js';

test('pickMime: mp4 se c\'e\' (Safari), poi webm/opus; vuoto se nessuno', () => {
  assert.equal(pickMime(() => true), 'audio/mp4');
  assert.equal(pickMime(m => m.startsWith('audio/webm')), 'audio/webm;codecs=opus');
  assert.equal(pickMime(() => false), '');
});
test('i formati scelti passano il controllo del server', () => {
  for (const m of ['audio/mp4', 'audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus'])
    assert.ok(VOICE_TYPE_RE.test(m), m);
});
test('extFor dal tipo, senza parametri', () => {
  assert.equal(extFor('audio/mp4'), 'm4a');
  assert.equal(extFor('audio/webm;codecs=opus'), 'webm');
  assert.equal(extFor('audio/ogg'), 'ogg');
  assert.equal(extFor('video/x'), 'bin');
  assert.equal(extFor(undefined), 'bin');
});
test('clock in m:ss', () => {
  assert.equal(clock(0), '0:00'); assert.equal(clock(9.9), '0:09');
  assert.equal(clock(65), '1:05'); assert.equal(clock(MAX_VOICE_SEC), '10:00');
  assert.equal(clock(-3), '0:00');
});
