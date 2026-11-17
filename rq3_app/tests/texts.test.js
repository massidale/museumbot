import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {CATEGORIES} from '../logic.js';

const texts = JSON.parse(readFileSync(new URL('../texts.json', import.meta.url), 'utf8'));
const LANGS = ['en', 'de', 'it'];

// Struttura di un valore: oggetti con le loro chiavi, liste con la loro lunghezza.
function shape(v, path = '') {
  if (Array.isArray(v)) return [`${path}[${v.length}]`, ...v.flatMap((x, i) => shape(x, `${path}[${i}]`))];
  if (v && typeof v === 'object')
    return Object.keys(v).sort().flatMap(k => [`${path}.${k}`, ...shape(v[k], `${path}.${k}`)]);
  return [];
}

// Ogni stringa con il suo percorso.
function strings(v, path = '') {
  if (typeof v === 'string') return [[path, v]];
  if (Array.isArray(v)) return v.flatMap((x, i) => strings(x, `${path}[${i}]`));
  if (v && typeof v === 'object') return Object.entries(v).flatMap(([k, x]) => strings(x, `${path}.${k}`));
  return [[path, v]];
}

test('en, de, it hanno le stesse chiavi e liste della stessa lunghezza', () => {
  assert.deepEqual(Object.keys(texts).sort(), [...LANGS].sort());
  const ref = shape(texts.en);
  for (const l of ['de', 'it']) assert.deepEqual(shape(texts[l]), ref, `lingua ${l}`);
});

test('falk ha le 5 categorie', () => {
  for (const l of LANGS) {
    assert.deepEqual(Object.keys(texts[l].falk).sort(), [...CATEGORIES].sort(), `falk ${l}`);
  }
});

test('nessuna stringa vuota (e niente valori che non siano stringhe)', () => {
  for (const l of LANGS)
    for (const [p, s] of strings(texts[l])) {
      assert.equal(typeof s, 'string', `${l}${p}`);
      assert.ok(s.trim().length > 0, `${l}${p} vuota`);
    }
});

test('i segnaposto {n} e {i} sono gli stessi nelle tre lingue', () => {
  const marks = s => (s.match(/\{[a-z]+\}/g) || []).sort().join(',');
  const ref = new Map(strings(texts.en).map(([p, s]) => [p, marks(s)]));
  for (const l of ['de', 'it'])
    for (const [p, s] of strings(texts[l])) assert.equal(marks(s), ref.get(p), `${l}${p}`);
});
