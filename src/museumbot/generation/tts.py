"""Sintesi vocale delle audioguide con ElevenLabs v4 tramite OpenRouter (/audio/speech).

Legge un file di testi generati con il testo inglese (`text`) e quello tedesco (`text_de`) e
scrive un mp3 per testo e lingua in <out>/<lingua>/<opera>/<condizione>.mp3 (la struttura
che legge l'app di RQ3, `rq3_visitors/build_manifest.py`), con la stessa voce per tutti
i testi e tutte e due le lingue (la voce non deve variare fra le condizioni). Riavviabile:
salta gli mp3 gia' presenti. In <out>/manifest.jsonl una riga per file (caratteri, voce,
modello). Prima di tutto il corpus una prova con --limit.

Uso:
    python -m museumbot.generation.tts --in data/wiki/haum_generations.jsonl --limit 1
    python -m museumbot.generation.tts --in data/wiki/haum_generations.jsonl
"""

import argparse
import json
import re
from pathlib import Path

import requests

from museumbot.common.config import ROOT, openrouter_key

URL = "https://openrouter.ai/api/v1/audio/speech"
MODEL = "elevenlabs/eleven-v4"
VOICE = "george"
PRICE = 0.00008  # $ per carattere (listino OpenRouter)
OUT = ROOT / "rq3_app" / "audio"
LANGS = {"en": "text", "de": "text_de"}


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def artwork_slug(r: dict) -> str:
    """Id dell'opera nell'app: il titolo inglese in minuscolo, con i trattini."""
    return slug(r["title"])


def audio_name(r: dict) -> str:
    """<opera>/<condizione>: per i misti la coppia, gia' nella condizione."""
    return f"{artwork_slug(r)}/{r['condition']}"


def jobs(rows: list[dict], out: Path) -> list[tuple[dict, str, Path]]:
    """(riga, lingua, file) per gli mp3 che mancano."""
    todo = []
    for r in rows:
        for lang, field in LANGS.items():
            path = out / lang / f"{audio_name(r)}.mp3"
            if not path.exists():
                if not r.get(field):
                    raise ValueError(f"manca {field} per {audio_name(r)}")
                todo.append((r, lang, path))
    return todo


def speak(text: str, voice: str = VOICE, post=requests.post) -> bytes:
    resp = post(URL, headers={"Authorization": f"Bearer {openrouter_key()}"},
                json={"model": MODEL, "input": text, "voice": voice, "response_format": "mp3"},
                timeout=300)
    resp.raise_for_status()
    return resp.content


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--voice", default=VOICE)
    ap.add_argument("--limit", type=int, help="solo i primi N file (prova)")
    ap.add_argument("--cap", type=float, default=25.0, help="spesa massima stimata in $")
    args = ap.parse_args()
    rows = [json.loads(l) for l in args.inp.open()]
    todo = jobs(rows, args.out)[: args.limit]
    chars = sum(len(r[LANGS[lang]]) for r, lang, _ in todo)
    print(f"{len(todo)} file, {chars} caratteri, ~{chars * PRICE:.2f} $", flush=True)
    if chars * PRICE > args.cap:
        raise SystemExit(f"oltre il tetto di {args.cap} $")
    for r, lang, path in todo:
        text = r[LANGS[lang]]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(speak(text, args.voice))
        with (args.out / "manifest.jsonl").open("a") as fh:
            fh.write(json.dumps({"file": str(path.relative_to(args.out)), "artwork_id":
                                 r["artwork_id"], "condition": r["condition"], "lang": lang,
                                 "chars": len(text), "voice": args.voice, "model": MODEL},
                                ensure_ascii=False) + "\n")
        print(f"  {path.relative_to(args.out)}  {len(text)} car.", flush=True)


if __name__ == "__main__":
    main()
