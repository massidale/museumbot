"""Costruisce il corpus di opere: Wikidata SPARQL -> Wikipedia intro extracts.

Output: data/artworks.jsonl con {id, qid, title, artist, museum, sitelinks, source_text}.

Il `source_text` e' la descrizione "flat" di partenza dell'esperimento: l'intro della voce
Wikipedia inglese, troncata in modo uniforme su confine di frase. La lunghezza della fonte
e' un confondente, quindi va normalizzata prima della generazione.
"""

import argparse
import json
import re
import time

import requests

from museumbot.common.config import ROOT

OUT = ROOT / "data" / "artworks.jsonl"

UA = "museumbot-research/0.1 (https://github.com/, massidale)"
SPARQL = "https://query.wikidata.org/sparql"
WIKI_API = "https://en.wikipedia.org/w/api.php"

# Grandi musei con collezioni ben coperte su Wikipedia, scelti per diversita' geografica
# e di periodo cosi' che il corpus non sia monotematico.
MUSEUMS = {
    "wd:Q188740": "MoMA",
    "wd:Q19675": "Louvre",
    "wd:Q160236": "Uffizi",
    "wd:Q190804": "Rijksmuseum",
    "wd:Q214867": "National Gallery of Art",
    "wd:Q160112": "Museo del Prado",
    "wd:Q23402": "Metropolitan Museum of Art",
    "wd:Q95569": "Kunsthistorisches Museum",
    "wd:Q510324": "National Gallery (London)",
    "wd:Q640447": "Art Institute of Chicago",
    "wd:Q1416890": "Musee d'Orsay",
    "wd:Q1319450": "Museum of Fine Arts, Boston",
    "wd:Q842858": "Tate",
    "wd:Q1895953": "Hermitage",
    "wd:Q1145791": "Museo Reina Sofia",
}

QUERY = """
SELECT ?painting ?paintingLabel ?creatorLabel ?museumLabel ?title ?sl WHERE {{
  VALUES ?museum {{ {museums} }}
  ?painting wdt:P31 wd:Q3305213 ;
            wdt:P276 ?museum ;
            wdt:P170 ?creator ;
            wikibase:sitelinks ?sl .
  FILTER(?sl > {min_sitelinks})
  ?article schema:about ?painting ;
           schema:isPartOf <https://en.wikipedia.org/> ;
           schema:name ?title .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
ORDER BY DESC(?sl)
LIMIT {limit}
"""

MIN_CHARS = 400   # sotto questa soglia l'intro e' troppo povera per differenziare i registri
MAX_CHARS = 1200  # troncamento uniforme, su confine di frase


def run_sparql(min_sitelinks: int, limit: int) -> list[dict]:
    query = QUERY.format(
        museums=" ".join(MUSEUMS), min_sitelinks=min_sitelinks, limit=limit
    )
    r = requests.get(
        SPARQL,
        params={"query": query},
        headers={"Accept": "application/sparql-results+json", "User-Agent": UA},
        timeout=180,
    )
    r.raise_for_status()
    rows = r.json()["results"]["bindings"]

    seen, out = set(), []
    for b in rows:
        qid = b["painting"]["value"].rsplit("/", 1)[-1]
        if qid in seen:
            continue
        seen.add(qid)
        out.append(
            {
                "qid": qid,
                "title": b["paintingLabel"]["value"],
                "wiki_title": b["title"]["value"],
                "artist": b["creatorLabel"]["value"],
                "museum": b["museumLabel"]["value"],
                "sitelinks": int(b["sl"]["value"]),
            }
        )
    return out


def fetch_extracts(wiki_titles: list[str]) -> dict[str, str]:
    """Intro in plain text per un batch di voci (max 20 per richiesta, limite MediaWiki)."""
    extracts: dict[str, str] = {}
    for i in range(0, len(wiki_titles), 20):
        batch = wiki_titles[i : i + 20]
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": 1,
            "exintro": 1,
            "exlimit": "max",
            "redirects": 1,
            "titles": "|".join(batch),
        }
        r = requests.get(WIKI_API, params=params, headers={"User-Agent": UA}, timeout=60)
        r.raise_for_status()
        data = r.json().get("query", {})
        # `redirects`/`normalized` rimappano i titoli richiesti su quelli effettivi
        alias = {}
        for key in ("normalized", "redirects"):
            for m in data.get(key, []):
                alias[m["to"]] = alias.get(m["from"], m["from"])
        for page in data.get("pages", {}).values():
            text = (page.get("extract") or "").strip()
            if not text:
                continue
            name = page.get("title", "")
            extracts[alias.get(name, name)] = text
        print(f"  extracts {min(i + 20, len(wiki_titles))}/{len(wiki_titles)}")
        time.sleep(0.3)
    return extracts


def clean_truncate(text: str) -> str:
    text = re.sub(r"\s*\n\s*", " ", text)        # l'audioguida e' parlata: niente paragrafi
    text = re.sub(r"\s{2,}", " ", text).strip()
    if len(text) <= MAX_CHARS:
        return text
    cut = text[:MAX_CHARS]
    # arretra all'ultimo confine di frase, evitando di spezzare abbreviazioni tipo "c."
    m = list(re.finditer(r"[.!?](?=\s|$)", cut))
    return cut[: m[-1].end()].strip() if m else cut.rsplit(" ", 1)[0] + "."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=100, help="numero di opere da tenere")
    ap.add_argument("--min-sitelinks", type=int, default=15)
    ap.add_argument("--limit", type=int, default=400, help="LIMIT della query SPARQL")
    args = ap.parse_args()

    print(f"SPARQL: {len(MUSEUMS)} musei, sitelinks > {args.min_sitelinks} ...")
    rows = run_sparql(args.min_sitelinks, args.limit)
    print(f"  {len(rows)} dipinti unici")

    print("Wikipedia intro extracts ...")
    extracts = fetch_extracts([r["wiki_title"] for r in rows])

    kept, skipped_missing, skipped_short = [], 0, 0
    for r in rows:
        raw = extracts.get(r["wiki_title"])
        if not raw:
            skipped_missing += 1
            continue
        text = clean_truncate(raw)
        if len(text) < MIN_CHARS:
            skipped_short += 1
            continue
        r["source_text"] = text
        r["id"] = r["qid"]
        kept.append(r)
        if len(kept) >= args.target:
            break

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lens = [len(r["source_text"]) for r in kept]
    print(f"\nscritte {len(kept)} opere in {OUT.relative_to(ROOT)}")
    print(f"  scartate: {skipped_missing} senza intro, {skipped_short} sotto {MIN_CHARS} char")
    print(f"  source_text: min {min(lens)}, mediana {sorted(lens)[len(lens) // 2]}, max {max(lens)}")
    print(f"  artisti distinti: {len({r['artist'] for r in kept})}")


if __name__ == "__main__":
    main()
