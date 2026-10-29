"""I corpus di RQ1, letti e puliti sempre allo stesso modo.

    main      DeepInfra fp8, ragionamento attivo. Dallo studio chain
              (data/main/chain_study.jsonl): single_a -> variante `full`, single_b -> replica
              `full_rep`, flat -> `flat`; le righe `chain` restano fuori. Da
              data/main/replicates.jsonl: altre due repliche di `full`, `full_rep2` e
              `full_rep3` (il flat generato con loro resta nel file, non usato: tutte le
              repliche si misurano rispetto al flat principale).
    ablation  le 7 varianti del blocco diverse da `full` (data/ablation/generations.jsonl,
              DeepInfra fp8, ragionamento attivo), piu' `full` e flat del corpus principale
              come riferimento.
    local     parte 2 (data/local/generations.jsonl): Gemma 4 31B su Kaggle, `full` piu' la
              replica `full_rep`; i testi rimasti in violazione del filtro (`usage.failed`)
              valgono come troncati.

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
    "replicates": ROOT / "data" / "main" / "replicates.jsonl",
    "ablation": ROOT / "data" / "ablation" / "generations.jsonl",
    "local": ROOT / "data" / "local" / "generations.jsonl",
}
CORPORA = ("main", "ablation", "local")
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
    """Righe pulite del corpus `name`, senza le opere con testi troncati o in violazione."""
    if name not in SOURCES:
        raise ValueError(f"corpus sconosciuto {name!r}; attesi {list(SOURCES)}")
    raw = [json.loads(l) for l in (path or SOURCES[name]).open()]
    if name == "main":
        rows = [to_row(r, MAIN_VARIANT[r["method"]]) for r in raw if r["method"] in MAIN_VARIANT]
        rep = [json.loads(l) for l in SOURCES["replicates"].open()] if path is None else []
        rows += [to_row(r, r["variant"]) for r in rep]
        raw += rep
    else:
        rows = [to_row(r, r.get("variant", "full")) for r in raw]
    if name == "ablation" and path is None:
        main = load_corpus("main", verbose=False)
        rows += [r for r in main if r["variant"] == "full"]
    failed = {r["artwork_id"] for r in raw if (r.get("usage") or {}).get("failed")}
    rows, bad = drop_truncated(rows)
    rows = [r for r in rows if r["artwork_id"] not in failed]
    bad = sorted(set(bad) | failed)
    if bad and verbose:
        print(f"[{name}] escluse {len(bad)} opere con testi troncati o in violazione: "
              f"{', '.join(bad)}")
    return rows
