"""Download delle sessioni online di RQ3 dal database: salva un export in data/rq3/raw."""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import requests

from museumbot.common.config import ROOT

URL = "https://museumbot.pages.dev/api/sessions"
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out-dir", type=Path, default=RAW)
    args = ap.parse_args()
    records = fetch(args.url, export_token())
    path = save(records, args.out_dir, datetime.now())
    print(f"scritto: {path}")
    print(f"sessioni: {len(records)}")


if __name__ == "__main__":
    main()
