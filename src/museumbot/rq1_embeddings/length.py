"""Confondente lunghezza: lo shift di categoria regge se si toglie la lunghezza dei testi?

Il flat e le categorie hanno lunghezze diverse (nel corpus `local` il flat e' molto piu'
corto), quindi una parte di ogni steering vector puo' essere "piu' lungo del flat". Per
ogni corpus (`main`, `local`) e modello di embedding:

  1. probe a 6 classi sulla sola lunghezza (log parole, centrata per opera): quanto
     separa la lunghezza da sola;
  2. probe senza flat (5 classi);
  3. lunghezza tolta dagli embedding: per ogni dimensione si stima il coefficiente della
     lunghezza a parita' di opera e di categoria (effetti fissi di entrambe) e si
     sottrae la componente lineare della lunghezza centrata per opera. Poi probe a 6 classi, test di
     permutazione, coseno fra steering vector corretto e originale e, fra i due corpus,
     rapporto del coseno sui tetti (`cross_generator`), con full e replica corretti.

La correzione e' conservativa: toglie anche il segnale di categoria che e' davvero
lunghezza. Soglie nella spec docs/plans/2026-10-06-profilo-continuo-design.md.
Scrive results/length_<modello>.json.
"""

import argparse
import json

import numpy as np

from museumbot.common.config import ROOT
from museumbot.common.prompts import CONDITIONS, FALK_CATEGORIES
from museumbot.rq1_embeddings.analyze import (
    FALK_IDX, center, load, permutation_test, probe, replicate_ceiling, row_cosines, steering,
)
from museumbot.rq1_embeddings.cross_generator import calibrate, cross_cosines

RESULTS = ROOT / "results"


def length_matrix(meta, arts) -> np.ndarray:
    """(A, C): log del numero di parole, nell'ordine delle opere e di CONDITIONS."""
    w = meta.set_index(["artwork_id", "condition"])["words"]
    return np.log(np.array([[w[(a, c)] for c in CONDITIONS] for a in arts], dtype=float))


def double_center(Z):
    """Toglie le medie per opera e per condizione (effetti fissi di entrambe)."""
    Z = Z - Z.mean(axis=1, keepdims=True)
    return Z - Z.mean(axis=0, keepdims=True)


def residualize(X, L, beta=None):
    """Toglie la componente lineare della lunghezza entro opera. Il coefficiente si stima
    a parita' di opera e di condizione, cosi' non assorbe le differenze fra categorie;
    si sottrae dalla lunghezza centrata per opera. Restituisce (X', beta)."""
    if beta is None:
        Ld = double_center(L)
        beta = np.einsum("ac,acd->d", Ld, double_center(X)) / (Ld**2).sum()
    Lc = L - L.mean(axis=1, keepdims=True)
    return X - Lc[..., None] * beta, beta


def length_probe(L) -> float:
    """Accuracy a 6 classi con la sola lunghezza centrata per opera."""
    Lc = L - L.mean(axis=1, keepdims=True)
    return float(probe(Lc[..., None])[0])


def permutation_p(X, n_perm: int = 1000) -> float:
    Xc = center(X, "mean")
    observed = np.linalg.norm(Xc.mean(axis=0), axis=1).mean()
    return float(permutation_test(Xc, observed, n_perm)[0])


def corpus_report(X, L, Xr) -> dict:
    fmt = lambda v: {c: round(float(x), 3) for c, x in zip(FALK_CATEGORIES, v)}
    Xn, Xrn = X[:, FALK_IDX], Xr[:, FALK_IDX]
    return {
        "words_median": {c: round(float(np.median(np.exp(L[:, i]))), 1)
                         for i, c in enumerate(CONDITIONS)},
        "probe_length_only": round(length_probe(L), 3),
        "probe6": round(float(probe(center(X, "mean"))[0]), 3),
        "probe6_corrected": round(float(probe(center(Xr, "mean"))[0]), 3),
        "probe5_no_flat": round(float(probe(Xn - Xn.mean(axis=1, keepdims=True))[0]), 3),
        "probe5_no_flat_corrected": round(float(probe(Xrn - Xrn.mean(axis=1, keepdims=True))[0]), 3),
        "permutation_p_corrected": round(permutation_p(Xr), 4),
        "cos_steering_corrected_vs_original": fmt(row_cosines(steering(X), steering(Xr))),
        "steering_norm": fmt(np.linalg.norm(steering(X), axis=1)),
        "steering_norm_corrected": fmt(np.linalg.norm(steering(Xr), axis=1))}


def run(model: str) -> dict:
    out, full, rep = {"model": model}, {}, {}
    for corpus in ("main", "local"):
        X, arts, meta = load(model, "full", corpus)
        R, arts_r, meta_r = load(model, "full_rep", corpus)
        L, Lr = length_matrix(meta, arts), length_matrix(meta_r, arts_r)
        Xr, beta = residualize(X, L)
        Rr, _ = residualize(R, Lr, beta)
        out[corpus] = corpus_report(X, L, Xr)
        full[corpus], rep[corpus] = (Xr, arts), (Rr, arts_r)
    cos, n = cross_cosines(*full["main"], *full["local"])
    ceil = {c: replicate_ceiling(*full[c], *rep[c]) for c in ("main", "local")}
    ratio = calibrate(cos, ceil["main"], ceil["local"])
    out["cross_generator_corrected"] = {
        "n_artworks": n, "cos": {c: round(float(x), 3) for c, x in zip(FALK_CATEGORIES, cos)},
        "ratio": {c: round(float(x), 3) for c, x in zip(FALK_CATEGORIES, ratio)}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="qwen")
    args = ap.parse_args()
    out = run(args.model)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"length_{args.model}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
