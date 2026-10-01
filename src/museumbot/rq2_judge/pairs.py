"""Coppie da giudicare, costruite in modo deterministico e bilanciato.

Per ogni opera e persona k (j != k):
    A  testo k vs flat     giudice k
    B  testo k vs testo j  giudice k
    C  testo j vs flat     giudice k
    N  testo k vs flat     giudice neutro (baseline)
piu' un controllo di attenzione per opera: il suo flat contro il flat di un'altra opera,
giudice neutro. In ogni coppia `target` e' l'elemento di interesse (k in A, B, N; j in C;
il flat giusto nel controllo).

Le opere sono mescolate con un seed; l'opera in posizione i riceve j = altre(k)[i mod 4],
cosi' su 100 opere ogni j compare 25 volte per ogni k.
"""

import random

from museumbot.common.prompts import FALK_CATEGORIES
from museumbot.rq2_judge.personas import NEUTRAL

SEED = 0
TYPES = ("A", "B", "C", "N")


def artwork_order(ids, seed: int = SEED) -> list[str]:
    order = sorted(set(ids))
    random.Random(seed).shuffle(order)
    return order


def others(k: str) -> list[str]:
    return [c for c in FALK_CATEGORIES if c != k]


def pair(pid, a, typ, persona, target, other) -> dict:
    return {"pair_id": pid, "artwork_id": a, "type": typ, "persona": persona,
            "target": list(target), "other": list(other)}


def build_pairs(ids, seed: int = SEED) -> list[dict]:
    """Tutte le coppie, nell'ordine delle opere mescolate."""
    order = artwork_order(ids, seed)
    out = []
    for i, a in enumerate(order):
        for k in FALK_CATEGORIES:
            j = others(k)[i % 4]
            out += [
                pair(f"{a}|{k}|A", a, "A", k, (a, k), (a, "flat")),
                pair(f"{a}|{k}|B", a, "B", k, (a, k), (a, j)),
                pair(f"{a}|{k}|C", a, "C", k, (a, j), (a, "flat")),
                pair(f"{a}|{k}|N", a, "N", NEUTRAL, (a, k), (a, "flat")),
            ]
        b = order[(i + 1) % len(order)]
        out.append(pair(f"{a}|attention", a, "attention", NEUTRAL, (a, "flat"), (b, "flat")))
    return out


def pilot_artworks(ids, n: int, seed: int = SEED) -> set[str]:
    """Le prime n opere dell'ordine mescolato."""
    return set(artwork_order(ids, seed)[:n])
