"""Sintesi vocale delle audioguide tramite OpenRouter (/audio/speech); di default ElevenLabs v4.

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
import subprocess
from pathlib import Path

import requests

from museumbot.common.config import ROOT, openrouter_key

URL = "https://openrouter.ai/api/v1/audio/speech"
MODEL = "elevenlabs/eleven-v4"
VOICE = "george"
# $ per carattere (listino OpenRouter, 18 novembre 2026); modelli non in lista: il piu' caro
PRICES = {"elevenlabs/eleven-v4": 0.00008, "elevenlabs/eleven-v4-turbo": 0.00004,
          "fish-audio/s2.1-pro": 0.000015, "heygen/heygen-voice-1": 0.000015,
          "heygen/heygen-voice-1-turbo": 0.000004, "heygen/heygen-voice-1-turbo:free": 0.0}


def price(model: str) -> float:
    return PRICES.get(model, max(PRICES.values()))
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


def pcm_rate(content_type: str) -> int:
    """Frequenza da un content-type come 'audio/pcm;rate=44100;channels=1'."""
    m = re.search(r"rate=(\d+)", content_type or "")
    return int(m.group(1)) if m else 44100


def pcm_to_mp3(pcm: bytes, rate: int) -> bytes:
    """PCM a 16 bit mono -> mp3 con ffmpeg."""
    cmd = ["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", str(rate), "-ac", "1",
           "-i", "pipe:0", "-b:a", "128k", "-f", "mp3", "pipe:1"]
    return subprocess.run(cmd, input=pcm, capture_output=True, check=True).stdout


def speak(text: str, voice: str = VOICE, model: str = MODEL, post=requests.post) -> bytes:
    """mp3 del testo. HeyGen restituisce solo PCM: lo si converte."""
    pcm = model.startswith("heygen/")
    resp = post(URL, headers={"Authorization": f"Bearer {openrouter_key()}"},
                json={"model": model, "input": text, "voice": voice,
                      "response_format": "pcm" if pcm else "mp3"},
                timeout=300)
    resp.raise_for_status()
    if pcm:
        return pcm_to_mp3(resp.content, pcm_rate(resp.headers.get("content-type")))
    return resp.content


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--voice", default=VOICE)
    ap.add_argument("--limit", type=int, help="solo i primi N file (prova)")
    ap.add_argument("--exclude", nargs="+", default=[], help="opere da saltare (id dell'app)")
    ap.add_argument("--cap", type=float, default=25.0, help="spesa massima stimata in $")
    args = ap.parse_args()
    rows = [r for r in map(json.loads, args.inp.open()) if artwork_slug(r) not in args.exclude]
    todo = jobs(rows, args.out)[: args.limit]
    chars = sum(len(r[LANGS[lang]]) for r, lang, _ in todo)
    cost = chars * price(args.model)
    print(f"{len(todo)} file, {chars} caratteri, ~{cost:.2f} $ ({args.model})", flush=True)
    if cost > args.cap:
        raise SystemExit(f"oltre il tetto di {args.cap} $")
    for r, lang, path in todo:
        text = r[LANGS[lang]]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(speak(text, args.voice, args.model))
        with (args.out / "manifest.jsonl").open("a") as fh:
            fh.write(json.dumps({"file": str(path.relative_to(args.out)), "artwork_id":
                                 r["artwork_id"], "condition": r["condition"], "lang": lang,
                                 "chars": len(text), "voice": args.voice, "model": args.model},
                                ensure_ascii=False) + "\n")
        print(f"  {path.relative_to(args.out)}  {len(text)} car.", flush=True)


if __name__ == "__main__":
    main()
