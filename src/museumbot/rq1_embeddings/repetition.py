"""Ripetitivita' lessicale per condizione: quanto si somigliano i testi di una condizione su opere diverse?

Per ogni corpus (`main` = DeepSeek, `local` = Gemma; testi `full` e flat) e condizione, si
prendono tutte le coppie di testi della condizione su opere diverse e si misura la quota di
n-grammi in comune (Jaccard fra gli insiemi). Il contenuto dell'opera cambia da testo a
testo, quindi quello che resta in comune sono le formule del registro.

Scelte fissate prima dei risultati: n-grammi come in `lexicon.ngrams` (unigrammi senza
parole funzionali, n-grammi non fatti solo di parole funzionali), tutti, senza selezione;
un valore per n = 1, 2, 3 e uno sull'unione dei tre; per condizione la media sulle coppie;
riferimento il flat (stesso modello, nessuna categoria), si riporta la differenza.

Scrive results/repetition.json.
"""

import argparse
import json
from itertools import combinations

import numpy as np

from museumbot.common.config import ROOT
from museumbot.common.corpus import load_corpus
from museumbot.common.prompts import CONDITIONS
from museumbot.rq1_embeddings.lexicon import NS, ngrams, tokens

RESULTS = ROOT / "results"
REF = "flat"


def gram_set(text: str, ns=NS) -> set[str]:
    """Insieme degli n-grammi del testo per gli n dati (marcati con n, cosi' non si confondono)."""
    toks = tokens(text)
    return {f"{n}:{g}" for n in ns for g in ngrams(toks, n)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def mean_overlap(texts: list[tuple[str, str]], ns=NS) -> float:
    """Jaccard medio fra le coppie (artwork_id, testo) su opere diverse."""
    sets = [(aid, gram_set(t, ns)) for aid, t in texts]
    vals = [jaccard(a, b) for (ia, a), (ib, b) in combinations(sets, 2) if ia != ib]
    return float(np.mean(vals))


def run() -> dict:
    keys = {str(n): (n,) for n in NS} | {"all": NS}
    out = {"ngrams": list(NS), "reference": REF}
    for corpus in ("main", "local"):
        rows = [r for r in load_corpus(corpus, verbose=False) if r["variant"] == "full"]
        res = {}
        for c in CONDITIONS:
            texts = [(r["artwork_id"], r["text"]) for r in rows if r["condition"] == c]
            res[c] = {"n_texts": len(texts),
                      **{k: round(mean_overlap(texts, ns), 4) for k, ns in keys.items()}}
        for c in CONDITIONS:
            res[c]["delta_vs_flat"] = {k: round(res[c][k] - res[REF][k], 4) for k in keys}
        out[corpus] = res
    return out


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    out = run()
    for corpus in ("main", "local"):
        print(f"\n[{corpus}]  Jaccard medio fra coppie di opere diverse (differenza col flat)")
        print(f"  {'condizione':22} {'n':>4} " + " ".join(f"{k:>16}" for k in ("1", "2", "3", "all")))
        for c in CONDITIONS:
            r = out[corpus][c]
            cells = " ".join(f"{r[k]:.4f} ({r['delta_vs_flat'][k]:+.4f})" for k in ("1", "2", "3", "all"))
            print(f"  {c:22} {r['n_texts']:>4} {cells}")
    path = RESULTS / "repetition.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
