// Modo QR: l'app aperta dal telefono del visitatore (indirizzo con ?m=qr).

// Vero se l'indirizzo chiede il modo QR o la scheda lo aveva gia' attivato.
export const isQrMode = (search, stored) => new URLSearchParams(search).get('m') === 'qr' || stored === 'qr';

// Quante sessioni ha gia' fatto questo telefono (0 se manca o non e' un numero);
// chi chiama salva count + 1.
export function nextCount(stored) {
  const n = Number(stored);
  return stored != null && stored !== '' && Number.isInteger(n) && n >= 0 ? n : 0;
}
