"""Voce Wikipedia intera (testo semplice) per alcune opere del corpus.

Prova "pagina intera" (docs/report-pagina-intera.md): al posto dell'intro troncata di
`fetch_artworks.py` il prompt riceve tutta la voce inglese, senza le sezioni finali di
servizio (note, bibliografia, collegamenti) e senza le sezioni rimaste vuote.

Output: data/wiki/full_pages.jsonl con i campi di data/artworks.jsonl, `source_text` = voce
intera, `revid` della revisione letta e `words`.

Con --lang e --titles legge voci per titolo da un'altra Wikipedia (le opere del museo di
Braunschweig per RQ3, voci tedesche): righe con id = QID, `lang`, `wiki_title`, `revid`,
`source_text` nella lingua della voce. La traduzione inglese si aggiunge a mano
(data/wiki/haum_pages.jsonl).

Uso:
    python -m museumbot.corpus.fetch_full Q568847 Q588695 Q737062
    python -m museumbot.corpus.fetch_full --lang de --titles "Familienbildnis (Rembrandt)" ... \
        --out data/wiki/haum_pages_de.jsonl
"""

import argparse
import json
import re
from pathlib import Path

import requests

from museumbot.common.config import ROOT
from museumbot.corpus.fetch_artworks import UA, WIKI_API

ARTWORKS = ROOT / "data" / "artworks.jsonl"
OUT = ROOT / "data" / "wiki" / "full_pages.jsonl"

# Sezioni di servizio (inglese, tedesco, italiano, spagnolo): da qui in poi la voce non
# descrive piu' l'opera.
TAIL = ("References", "See also", "Notes", "External links", "Further reading", "Sources",
        "Bibliography", "Citations", "Footnotes",
        "Literatur", "Einzelnachweise", "Weblinks", "Anmerkungen", "Quellen", "Siehe auch",
        "Fußnoten", "Belege", "Bibliografie", "Literatur und Weblinks",
        "Note", "Bibliografia", "Voci correlate", "Collegamenti esterni", "Altri progetti",
        "Fonti", "Referencias", "Bibliografía", "Véase también", "Enlaces externos", "Notas")
HEADING = re.compile(r"^(=+)\s*(.*?)\s*=+\s*$")


def strip_page(text: str) -> str:
    """Toglie le sezioni di servizio in coda e i titoli senza testo sotto."""
    lines, out = text.splitlines(), []
    for line in lines:
        m = HEADING.match(line)
        if m and len(m.group(1)) == 2 and m.group(2) in TAIL:
            break
        out.append(line)
    # un titolo e' vuoto se fino al titolo successivo (di livello uguale o superiore) o alla
    # fine non c'e' testo
    keep = []
    for i, line in enumerate(out):
        m = HEADING.match(line)
        if m:
            level, body = len(m.group(1)), False
            for nxt in out[i + 1:]:
                n = HEADING.match(nxt)
                if n and len(n.group(1)) <= level:
                    break
                if not n and nxt.strip():
                    body = True
                    break
            if not body:
                continue
        keep.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(keep)).strip()


def fetch_page(wiki_title: str, lang: str = "en") -> tuple[str, int, str | None]:
    """Testo semplice della voce intera, revid della revisione corrente e QID Wikidata."""
    params = {"action": "query", "format": "json", "prop": "extracts|revisions|pageprops",
              "explaintext": 1, "rvprop": "ids", "ppprop": "wikibase_item", "redirects": 1,
              "titles": wiki_title}
    api = WIKI_API.replace("://en.", f"://{lang}.")
    r = requests.get(api, params=params, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    page = next(iter(r.json()["query"]["pages"].values()))
    qid = page.get("pageprops", {}).get("wikibase_item")
    return page["extract"], page["revisions"][0]["revid"], qid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ids", nargs="*", help="id delle opere in data/artworks.jsonl")
    ap.add_argument("--lang", default="en", help="Wikipedia da cui leggere le voci per titolo")
    ap.add_argument("--titles", nargs="+", help="titoli delle voci (al posto degli id)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if args.titles:
        rows = []
        for t in args.titles:
            raw, revid, qid = fetch_page(t, args.lang)
            text = strip_page(raw)
            rows.append({"id": qid, "qid": qid, "wiki_title": t, "lang": args.lang,
                         "revid": revid, "source_text": text, "words": len(text.split())})
            print(f"{qid} {t}: {rows[-1]['words']} parole (revid {revid})")
    else:
        arts = {a["id"]: a for a in map(json.loads, ARTWORKS.open())}
        rows = []
        for i in args.ids:
            a = arts[i]
            raw, revid, _ = fetch_page(a["wiki_title"])
            text = strip_page(raw)
            rows.append({**a, "source_text": text, "revid": revid, "words": len(text.split())})
            print(f"{i} {a['title']}: {rows[-1]['words']} parole (revid {revid})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
