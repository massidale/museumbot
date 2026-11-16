"""Stimoli finti per provare l'app RQ3: 4 opere, audio parlato + silenzio, immagini segnaposto."""
import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from museumbot.rq3_visitors.build_manifest import LANGS, SAMPLE, all_conditions

VOCI = {"en": "Samantha", "de": "Anna"}
COLORI = ["0x3b5b7a", "0x7a3b4e", "0x4e7a3b", "0x7a6a3b"]
SEC = 90


SAMPLE_SEC = 8
SAMPLE_TEXT = {"en": "This is a sound check.", "de": "Dies ist eine Hörprobe."}


def render_speech(lang: str, testo: str, sec: int, out: Path):
    """say pronuncia il testo, poi silenzio fino a sec secondi (mp3 mono 64k)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        aiff = Path(tmp) / "v.aiff"
        subprocess.run(["say", "-v", VOCI[lang], "-o", str(aiff), testo], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(aiff), "-af",
                        f"apad=whole_dur={sec}", "-t", str(sec), "-ac", "1", "-b:a", "64k",
                        str(out)], check=True)


def render_audio(lang: str, art_id: str, cond: str, out: Path):
    """Opera e condizione pronunciate, poi silenzio fino a 90 s."""
    testo = f"{art_id.replace('_', ' ')}. {cond.replace('+', ' plus ').replace('_', ' ')}."
    render_speech(lang, testo, SEC, out)


def render_sample(lang: str, out: Path):
    """Prova del volume neutra: una frase ripetuta fino a circa 8 s."""
    render_speech(lang, " ".join([SAMPLE_TEXT[lang]] * 3), SAMPLE_SEC, out)


def render_image(i: int, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"color=c={COLORI[i % len(COLORI)]}:s=1200x900", "-frames:v", "1", str(out)],
                   check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--app-root", default="rq3_app")
    ap.add_argument("--data-dir", default="data/rq3")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    root = Path(a.app_root)
    ids = [f"demo_{i}" for i in range(1, 5)]
    artworks = []
    for i, art_id in enumerate(ids):
        render_image(i, root / "img" / f"{art_id}.jpg")
        artworks.append({"id": art_id, "image": f"img/{art_id}.jpg",
                         "title": {"en": f"Demo artwork {i + 1}", "de": f"Demo-Werk {i + 1}"},
                         "artist": f"Demo Artist {i + 1}"})
    jobs = [(lang, art_id, c, root / "audio" / lang / art_id / f"{c}.mp3")
            for art_id in ids for lang in LANGS for c in all_conditions()]
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda j: render_audio(*j), jobs))
    for lang in LANGS:
        render_sample(lang, root / "audio" / lang / f"{SAMPLE}.mp3")
    d = Path(a.data_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "artworks_demo.json").write_text(
        json.dumps(artworks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(jobs)} audio, {len(LANGS)} prove del volume, {len(ids)} immagini, "
          f"{d / 'artworks_demo.json'}")


if __name__ == "__main__":
    main()
