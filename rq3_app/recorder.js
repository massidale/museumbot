// Registrazione delle risposte a voce di fine giro: microfono -> MediaRecorder -> un solo file.
// Funzioni pure in cima (testate), il resto usa il browser.

export const MAX_VOICE_SEC = 600;   // oltre, la registrazione si ferma da sola

// Primo formato che il browser sa registrare: mp4/AAC su Safari, webm/opus altrove.
const CANDIDATES = ['audio/mp4', 'audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus'];
export const pickMime = isSupported => CANDIDATES.find(m => isSupported(m)) || '';

// Estensione del file dal tipo (senza i parametri).
export function extFor(mime) {
  const base = String(mime || '').split(';')[0].trim();
  return {'audio/mp4': 'm4a', 'audio/webm': 'webm', 'audio/ogg': 'ogg', 'audio/mpeg': 'mp3'}[base] || 'bin';
}

// Secondi -> "m:ss".
export const clock = sec => {
  const s = Math.max(0, Math.floor(sec));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
};

export const canRecord = () =>
  !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder);

// Avvia la registrazione (chiede il microfono). onTick(sec) ogni secondo; onLimit() quando
// si arriva a MAX_VOICE_SEC. Restituisce {stop, cancel}: stop() -> {data, mime, sec}.
export async function startRecording({onTick = () => {}, onLimit = () => {}} = {}) {
  const stream = await navigator.mediaDevices.getUserMedia({audio: true});
  const supported = m => (MediaRecorder.isTypeSupported ? MediaRecorder.isTypeSupported(m) : false);
  const mime = pickMime(supported);
  const rec = new MediaRecorder(stream, mime ? {mimeType: mime} : undefined);
  const chunks = [];
  const t0 = performance.now();
  const elapsed = () => (performance.now() - t0) / 1000;
  rec.ondataavailable = e => { if (e.data && e.data.size) chunks.push(e.data); };
  const release = () => stream.getTracks().forEach(t => t.stop());
  const timer = setInterval(() => {
    onTick(elapsed());
    if (elapsed() >= MAX_VOICE_SEC) onLimit();
  }, 1000);
  rec.start(1000);
  let done = null;
  const stop = () => {
    if (done) return done;
    clearInterval(timer);
    const sec = Math.round(elapsed() * 10) / 10;
    done = new Promise((resolve, reject) => {
      rec.onstop = async () => {
        release();
        const type = rec.mimeType || mime || 'audio/mp4';
        const blob = new Blob(chunks, {type});
        try {
          resolve({data: await blob.arrayBuffer(), mime: type, sec});
        } catch (e) { reject(e); }
      };
      rec.onerror = e => { release(); reject(e.error || e); };
      if (rec.state !== 'inactive') rec.stop(); else rec.onstop();
    });
    return done;
  };
  const cancel = () => {
    clearInterval(timer);
    rec.ondataavailable = null;
    rec.onstop = null;
    try { if (rec.state !== 'inactive') rec.stop(); } catch (e) { /* gia' fermo */ }
    release();
  };
  return {stop, cancel};
}
