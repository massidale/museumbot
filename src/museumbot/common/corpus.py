"""I due corpus di RQ1, letti e puliti sempre allo stesso modo.

    main      testi a prompt singolo dello studio chain (data/main/chain_study.jsonl): DeepInfra
              fp8, ragionamento attivo, stessa sessione. single_a -> variante `full`,
              single_b -> replica `full_rep`, flat -> `flat`; le righe `chain` restano fuori.
    ablation  corpus di settembre (data/ablation/generations.jsonl, routing libero), usato
              solo per l'ablazione: le 8 varianti del blocco piu' la replica `full_rep`.

Ogni testo passa per `clean_guide` a partire dall'originale (`usage.raw` se presente), e il
numero di parole si ricalcola. Le opere con almeno un testo troncato vengono escluse per
intero, cosi' tutte le varianti restano appaiate sulle stesse opere.
"""

import json
from pathlib import Path

from museumbot.common.clean import clean_guide, is_truncated
from museumbot.common.config import ROOT

SOURCES = {
    "main": ROOT / "data" / "main" / "chain_study.jsonl",
    "ablation": ROOT / "data" / "ablation" / "generations.jsonl",
}
CORPORA = tuple(SOURCES)
# Metodo dello studio chain -> variante nel corpus principale.
MAIN_VARIANT = {"single_a": "full", "single_b": "full_rep", "flat": "full"}


def original_text(row: dict) -> str:
    """Testo prima di qualsiasi pulizia: le righe pulite in generazione lo tengono in `raw`."""
    return (row.get("usage") or {}).get("raw") or row["text"]


def to_row(r: dict, variant: str) -> dict:
    text = clean_guide(original_text(r))
    return {
        "artwork_id": r["artwork_id"],
        "title": r["title"],
        "artist": r["artist"],
        "condition": r["condition"],
        "variant": variant,
        "model": r["model"],
        "text": text,
        "words": len(text.split()),
    }


def drop_truncated(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Toglie le opere con almeno un testo troncato; restituisce anche le opere tolte."""
    bad = sorted({r["artwork_id"] for r in rows if is_truncated(r["text"])})
    return [r for r in rows if r["artwork_id"] not in bad], bad


def load_corpus(name: str, path: Path | None = None, verbose: bool = True) -> list[dict]:
    """Righe pulite del corpus `name`, senza le opere con testi troncati."""
    if name not in SOURCES:
        raise ValueError(f"corpus sconosciuto {name!r}; attesi {list(SOURCES)}")
    raw = [json.loads(l) for l in (path or SOURCES[name]).open()]
    if name == "main":
        rows = [to_row(r, MAIN_VARIANT[r["method"]]) for r in raw if r["method"] in MAIN_VARIANT]
    else:
        rows = [to_row(r, r.get("variant", "full")) for r in raw]
    rows, bad = drop_truncated(rows)
    if bad and verbose:
        print(f"[{name}] escluse {len(bad)} opere con testi troncati: {', '.join(bad)}")
    return rows
