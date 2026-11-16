// Salvataggio in IndexedDB: database 'rq3', store 'sessions' (chiave session_id)
// e 'settings' (chiave libera: device, last_export_ids).

const DB_NAME = 'rq3';
const DB_VERSION = 1;

let dbPromise = null;

function openDb() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('sessions')) db.createObjectStore('sessions', {keyPath: 'session_id'});
      if (!db.objectStoreNames.contains('settings')) db.createObjectStore('settings');
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

export async function putSession(record) {
  await run('sessions', 'readwrite', s => s.put(record));
}

export const getAllSessions = () => run('sessions', 'readonly', s => s.getAll());

export async function deleteSessions(ids) {
  await run('sessions', 'readwrite', s => { for (const id of ids) s.delete(id); });
}

export const getSetting = key => run('settings', 'readonly', s => s.get(key));

export async function setSetting(key, value) {
  await run('settings', 'readwrite', s => s.put(value, key));
}
