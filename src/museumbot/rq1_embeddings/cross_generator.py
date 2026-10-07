"""Lo shift di categoria punta nella stessa direzione con due generatori diversi?

Passo 2 del profilo continuo: steering vector (riferimento flat) del corpus principale
(DeepSeek V4 Flash) e del corpus `local` (Gemma 4 31B), nello stesso spazio di embedding e
sulle stesse opere, confrontati categoria per categoria:

    cos[c]     = cos(v_main[c], v_local[c])
    rapporto[c] = cos[c] / sqrt(tetto_main[c] * tetto_local[c])

Il tetto e' il coseno fra full e replica dello stesso corpus (`analyze.replicate_ceiling`):
il rapporto e' il coseno rispetto al massimo atteso fra due vettori con la stessa direzione
vera e i due rumori di generazione. Scrive results/cross_generator_<modello>.json.
"""

import argparse
import json

import numpy as np

from museumbot.common.config import ROOT
from museumbot.common.prompts import FALK_CATEGORIES
from museumbot.rq1_embeddings.analyze import load, replicate_ceiling, row_cosines, steering

RESULTS = ROOT / "results"


def cross_cosines(Xa, arts_a, Xb, arts_b) -> tuple[np.ndarray, int]:
    """Coseno per categoria fra gli steering vector di due corpus, sulle opere comuni."""
    common = sorted(set(arts_a) & set(arts_b))
    ia = [arts_a.index(a) for a in common]
    ib = [arts_b.index(a) for a in common]
    return row_cosines(steering(Xa[ia]), steering(Xb[ib])), len(common)


def calibrate(cos, ceil_a, ceil_b) -> np.ndarray:
    return cos / np.sqrt(ceil_a * ceil_b)


def run(model: str, a: str = "main", b: str = "local") -> dict:
    X = {c: load(model, "full", c)[:2] for c in (a, b)}
    R = {c: load(model, "full_rep", c)[:2] for c in (a, b)}
    cos, n = cross_cosines(*X[a], *X[b])
    ceil = {c: replicate_ceiling(*X[c], *R[c]) for c in (a, b)}
    ratio = calibrate(cos, ceil[a], ceil[b])
    fmt = lambda v: {k: round(float(x), 3) for k, x in zip(FALK_CATEGORIES, v)}
    return {"model": model, "corpora": [a, b], "n_artworks": n, "cos": fmt(cos),
            f"ceiling_{a}": fmt(ceil[a]), f"ceiling_{b}": fmt(ceil[b]), "ratio": fmt(ratio)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="qwen")
    args = ap.parse_args()
    out = run(args.model)
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"cross_generator_{args.model}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
