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
