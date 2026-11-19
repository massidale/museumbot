// Salvataggio in IndexedDB: database 'rq3', store 'sessions' (chiave session_id),
// 'settings' (chiave libera: device, last_export_ids, ...) e 'voice' (registrazione di fine
// giro: {session_id, data: ArrayBuffer, mime, sec, recorded_at}).

const DB_NAME = 'rq3';
const DB_VERSION = 2;

let dbPromise = null;

function openDb() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('sessions')) db.createObjectStore('sessions', {keyPath: 'session_id'});
      if (!db.objectStoreNames.contains('settings')) db.createObjectStore('settings');
      if (!db.objectStoreNames.contains('voice')) db.createObjectStore('voice', {keyPath: 'session_id'});
    };
    req.onsuccess = () => {
      const db = req.result;
      db.onversionchange = () => { db.close(); dbPromise = null; };
      resolve(db);
    };
    req.onerror = () => reject(req.error);
  });
  dbPromise.catch(() => { dbPromise = null; });  // al prossimo accesso si riprova
  return dbPromise;
}

// Una transazione: fn(store) fa le richieste; si risolve col risultato dell'ultima,
// solo quando la transazione e' conclusa (cioe' scritta).
async function run(storeName, mode, fn) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, mode);
    let out;
    const req = fn(tx.objectStore(storeName));
    if (req) req.onsuccess = () => { out = req.result; };
    tx.oncomplete = () => resolve(out);
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error || new Error('transazione annullata'));
  });
}

// Salvataggi di sessioni non riusciti da quando la pagina e' aperta (per l'operatore).
const failures = [];
export const getSaveFailures = () => failures.map(f => ({...f}));

// Un errore resta registrato e si propaga a chi ha chiamato.
export async function putSession(record) {
  try {
    await run('sessions', 'readwrite', s => s.put(record));
  } catch (e) {
    failures.push({session_id: record.session_id, last_step: record.last_step,
                   at: new Date().toISOString(), error: String((e && e.message) || e)});
    throw e;
  }
}

export const getAllSessions = () => run('sessions', 'readonly', s => s.getAll());

export async function deleteSessions(ids) {
  await run('sessions', 'readwrite', s => { for (const id of ids) s.delete(id); });
}

export const getSetting = key => run('settings', 'readonly', s => s.get(key));

export async function setSetting(key, value) {
  await run('settings', 'readwrite', s => s.put(value, key));
}

// Registrazioni di fine giro: una per sessione (una nuova sostituisce la vecchia).
export const putVoice = v => run('voice', 'readwrite', s => s.put(v));
export const getVoice = id => run('voice', 'readonly', s => s.get(id));
export const getVoiceIds = () => run('voice', 'readonly', s => s.getAllKeys());

export async function deleteVoices(ids) {
  await run('voice', 'readwrite', s => { for (const id of ids) s.delete(id); });
}
