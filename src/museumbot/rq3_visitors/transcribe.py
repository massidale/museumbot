"""Trascrizione locale delle registrazioni di RQ3 (Whisper sul Mac, niente servizi esterni).

Legge data/rq3/raw/voice/<session_id>[_<suffisso>].<m4a|webm|ogg|mp3> (il suffisso e' n per
la registrazione dell'opera n, o un nome per i file fatti fuori dall'app), usa la lingua della sessione
dall'export online e scrive una riga per file in data/rq3/transcripts.jsonl. I file gia'
trascritti si saltano; quelli di sessioni assenti dall'export (prove cancellate) o in
--exclude non si trascrivono.
"""
import argparse
import json
import subprocess
from pathlib import Path

from museumbot.common.config import ROOT

RAW = ROOT / "data" / "rq3" / "raw"
OUT = ROOT / "data" / "rq3" / "transcripts.jsonl"
MODEL = "openai/whisper-large-v3-turbo"
WHISPER_LANG = {"de": "german", "en": "english"}
AUDIO = {".m4a", ".webm", ".ogg", ".mp3"}


def session_of(path: Path) -> str:
    """session_id dal nome del file: la parte prima del primo '_'."""
    return Path(path).stem.split("_")[0]


def to_do(files: list[Path], langs: dict[str, str], exclude: set[str], done: set[str]) -> list[tuple[Path, str]]:
    """(file, lingua) da trascrivere: sessione nota, non esclusa, file non ancora fatto."""
    out = []
    for f in sorted(files):
        sid = session_of(f)
        if sid in langs and sid not in exclude and f.name not in done:
            out.append((f, langs[sid]))
    return out


def load_langs(export: Path) -> dict[str, str]:
    """session_id -> lingua, dall'export online."""
    return {r["session_id"]: r["lang"] for r in json.loads(Path(export).read_text(encoding="utf-8"))}


def load_done(out: Path) -> set[str]:
    if not Path(out).exists():
        return set()
    return {json.loads(l)["file"] for l in Path(out).read_text(encoding="utf-8").splitlines() if l.strip()}


def decode(path: Path, rate: int = 16000):
    """Audio mono a 16 kHz in float32, letto da ffmpeg dal file (gli m4a dell'iPhone non
    si leggono da una pipe)."""
    import numpy as np
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1",
                          "-ar", str(rate), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def make_asr(model: str):
    """Pipeline di transformers su MPS (GPU del Mac) se c'e'."""
    import torch
    from transformers import pipeline
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    return pipeline("automatic-speech-recognition", model=model, device=device, torch_dtype=dtype)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", type=Path, help="export online (default: il piu' recente)")
    ap.add_argument("--voice-dir", type=Path, default=RAW / "voice")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--exclude", nargs="*", default=[], help="session_id da saltare (prove)")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    export = args.export or sorted(RAW.glob("online_*.json"))[-1]
    files = [f for f in args.voice_dir.iterdir() if f.suffix in AUDIO]
    jobs = to_do(files, load_langs(export), set(args.exclude),
                 load_done(args.out))
    print(f"export: {export.name}; da trascrivere: {len(jobs)}")
    if not jobs:
        return
    asr = make_asr(args.model)
    with open(args.out, "a", encoding="utf-8") as fh:
        for f, lang in jobs:
            r = asr({"raw": decode(f), "sampling_rate": 16000}, return_timestamps=True,
                    generate_kwargs={"language": WHISPER_LANG[lang], "task": "transcribe"})
            row = {"file": f.name, "session_id": session_of(f), "lang": lang,
                   "model": args.model, "text": r["text"].strip()}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"{f.name}: {len(row['text'])} caratteri")


if __name__ == "__main__":
    main()
