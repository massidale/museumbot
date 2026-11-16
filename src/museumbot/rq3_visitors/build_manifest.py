"""Manifesto degli stimoli dell'app RQ3: opere, audio per lingua e condizione, durate."""
import argparse
import json
import subprocess
from itertools import combinations
from pathlib import Path

from museumbot.common.prompts import FALK_CATEGORIES

LANGS = ("en", "de")
EXTS = (".mp3", ".m4a")


def condition_key(name: str) -> str:
    """Chiave canonica: 'flat', una categoria, o 'a+b' con le categorie in ordine Falk."""
    if name == "flat" or name in FALK_CATEGORIES:
        return name
    parti = name.split("+")
    if len(parti) != 2 or parti[0] == parti[1] or any(p not in FALK_CATEGORIES for p in parti):
        raise ValueError(f"condizione ignota: {name}")
    a, b = sorted(parti, key=FALK_CATEGORIES.index)
    return f"{a}+{b}"


def all_conditions() -> list[str]:
    """flat, le 5 pure, le 10 miste: stesso ordine di allConditions() in logic.js."""
    return ["flat", *FALK_CATEGORIES, *(f"{a}+{b}" for a, b in combinations(FALK_CATEGORIES, 2))]


def scan_audio(audio_root: Path, artwork_id: str, lang: str) -> dict[str, Path]:
    """File audio_root/<lang>/<opera>/<condizione>.(mp3|m4a), chiavi canoniche."""
    d = Path(audio_root) / lang / artwork_id
    out = {}
    if d.is_dir():
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in EXTS:
                out[condition_key(p.stem)] = p
    return out


def duration(path: Path) -> float:
    """Durata in secondi, letta con ffprobe."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def build(artworks: list[dict], audio_root: Path, app_root: Path, version: str,
          dur=duration) -> dict:
    """Manifesto; ValueError con tutti gli audio mancanti."""
    mancanti, opere = [], []
    for art in artworks:
        audio = {}
        for lang in LANGS:
            files = scan_audio(audio_root, art["id"], lang)
            audio[lang] = {}
            for c in all_conditions():
                if c not in files:
                    mancanti.append(f"{art['id']} / {lang} / {c}")
                    continue
                src = files[c].resolve().relative_to(Path(app_root).resolve()).as_posix()
                audio[lang][c] = {"src": src, "sec": dur(files[c])}
        opere.append({"id": art["id"], "image": art["image"], "title": art["title"],
                      "artist": art["artist"], "audio": audio})
    if mancanti:
        raise ValueError("audio mancanti (opera / lingua / condizione):\n" + "\n".join(mancanti))
    return {"version": version, "categories": list(FALK_CATEGORIES), "artworks": opere}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artworks", required=True, help="JSON: lista di {id, image, title, artist}")
    ap.add_argument("--app-root", default="rq3_app")
    ap.add_argument("--version", required=True)
    a = ap.parse_args()
    root = Path(a.app_root)
    artworks = json.loads(Path(a.artworks).read_text(encoding="utf-8"))
    m = build(artworks, root / "audio", root, a.version)
    out = root / "manifest.json"
    out.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{out}: {len(m['artworks'])} opere")


if __name__ == "__main__":
    main()
