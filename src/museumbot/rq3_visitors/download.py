"""Download delle sessioni online di RQ3 dal database: salva un export in data/rq3/raw e le
registrazioni di fine giro in data/rq3/raw/voice (solo quelle che mancano)."""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import requests

from museumbot.common.config import ROOT

URL = "https://museumbot.pages.dev/api/sessions"
VOICE_URL = "https://museumbot.pages.dev/api/voice"
EXT = {"audio/mp4": "m4a", "audio/webm": "webm", "audio/ogg": "ogg", "audio/mpeg": "mp3"}
RAW = ROOT / "data" / "rq3" / "raw"


def out_path(raw_dir: Path, now: datetime) -> Path:
    """Nome del file di export: online_<AAAAMMGG-HHMM>.json."""
    return Path(raw_dir) / f"online_{now:%Y%m%d-%H%M}.json"


def export_token() -> str:
    """Token di lettura da RQ3_EXPORT_TOKEN (ambiente o .env); esce se manca."""
    token = os.getenv("RQ3_EXPORT_TOKEN")
    if not token:
        raise SystemExit(
            "RQ3_EXPORT_TOKEN non configurato: impostalo nell'ambiente o nel file .env")
    return token


def fetch(url: str, token: str, get=requests.get) -> list[dict]:
    """Scarica la lista dei record dal server."""
    resp = get(url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    resp.raise_for_status()
    dati = resp.json()
    if not isinstance(dati, list):
        raise ValueError("la risposta non e' una lista di sessioni")
    return dati


def save(records: list[dict], raw_dir: Path, now: datetime) -> Path:
    """Scrive la lista come l'export dell'app (indent=2)."""
    path = out_path(raw_dir, now)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def voice_name(item: dict) -> str:
    """File della registrazione: <session_id>.<estensione dal tipo, senza parametri>."""
    base = str(item.get("type", "")).split(";")[0].strip()
    return f"{item['session_id']}.{EXT.get(base, 'bin')}"


def fetch_voice(url: str, token: str, voice_dir: Path, get=requests.get) -> int:
    """Scarica le registrazioni non ancora in voice_dir; restituisce quante."""
    auth = {"Authorization": f"Bearer {token}"}
    resp = get(url, headers=auth, timeout=30)
    resp.raise_for_status()
    items = resp.json()
    if not isinstance(items, list):
        raise ValueError("la risposta non e' una lista di registrazioni")
    voice_dir = Path(voice_dir)
    nuove = [it for it in items if not (voice_dir / voice_name(it)).exists()]
    for it in nuove:
        r = get(f"{url}?id={it['session_id']}", headers=auth, timeout=120)
        r.raise_for_status()
        voice_dir.mkdir(parents=True, exist_ok=True)
        (voice_dir / voice_name(it)).write_bytes(r.content)
    return len(nuove)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--voice-url", default=VOICE_URL)
    ap.add_argument("--out-dir", type=Path, default=RAW)
    args = ap.parse_args()
    token = export_token()
    records = fetch(args.url, token)
    path = save(records, args.out_dir, datetime.now())
    print(f"scritto: {path}")
    print(f"sessioni: {len(records)}")
    n = fetch_voice(args.voice_url, token, args.out_dir / "voice")
    print(f"registrazioni nuove: {n} (in {args.out_dir / 'voice'})")


if __name__ == "__main__":
    main()
