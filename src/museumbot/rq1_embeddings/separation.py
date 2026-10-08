"""Quanto sono formulaici i testi di una categoria? Analisi preliminari sulla separazione.

Gemma separa le categorie piu' di DeepSeek (report di Gemma, sezione 2). Separazione
maggiore puo' voler dire personalizzazione piu' netta o testi piu' stereotipati: queste
misure lo distinguono, per ogni corpus (`main` = DeepSeek, `local` = Gemma) e condizione.

  - somiglianza fra opere: coseno medio fra gli embedding dei testi della stessa
    condizione su opere diverse. Il flat fa da riferimento: ha solo il contenuto
    dell'opera, quindi l'eccesso di una categoria sul flat e' la parte "di formula";
  - copertura delle formule: per ogni testo, quota dei suoi trigrammi di parole che
    compaiono in almeno MIN_DF dei testi della sua condizione; piu' le formule piu' comuni.

Scrive results/separation_<modello>.json.
"""

import argparse
import json
import re
from collections import Counter

import numpy as np

from museumbot.common.config import ROOT
from museumbot.common.corpus import load_corpus
from museumbot.common.prompts import CONDITIONS
from museumbot.rq1_embeddings.analyze import load

RESULTS = ROOT / "results"
MIN_DF = 0.10
WORD = re.compile(r"[a-z’']+")


def trigrams(text: str) -> set[tuple[str, str, str]]:
    w = WORD.findall(text.lower())
    return {tuple(w[i : i + 3]) for i in range(len(w) - 2)}


def _df(docs) -> tuple[list[set], Counter]:
    grams = [trigrams(d) for d in docs]
    return grams, Counter(g for gs in grams for g in gs)


def formula_coverage(docs: list[str], min_df: float = MIN_DF) -> list[float]:
    grams, df = _df(docs)
    thr = min_df * len(docs)
    return [sum(df[g] >= thr for g in gs) / len(gs) if gs else 0.0 for gs in grams]


def top_formulas(docs: list[str], k: int = 8) -> list[tuple[str, float]]:
    _, df = _df(docs)
    return [(" ".join(g), round(c / len(docs), 3)) for g, c in df.most_common(k)]


def cross_artwork_similarity(E: np.ndarray, arts: list[str]) -> float:
    """Coseno medio fra coppie di testi di opere diverse (embedding gia' normalizzati)."""
    S = E @ E.T
    a = np.array(arts)
    mask = a[:, None] != a[None, :]
    return float(S[mask].mean())


def run(model: str) -> dict:
    out = {"model": model, "min_df": MIN_DF}
    for corpus in ("main", "local"):
        X, arts, _ = load(model, "full", corpus)
        texts = {(r["artwork_id"], r["condition"]): r["text"]
                 for r in load_corpus(corpus, verbose=False) if r["variant"] == "full"}
        res = {}
        for ci, c in enumerate(CONDITIONS):
            E = X[:, ci] / np.linalg.norm(X[:, ci], axis=1, keepdims=True)
            docs = [texts[(a, c)] for a in arts]
            res[c] = {"cross_artwork_similarity": round(cross_artwork_similarity(E, arts), 4),
                      "formula_coverage": round(float(np.mean(formula_coverage(docs))), 4),
                      "top_formulas": top_formulas(docs)}
        base = res["flat"]["cross_artwork_similarity"]
        for c in res:
            res[c]["similarity_above_flat"] = round(res[c]["cross_artwork_similarity"] - base, 4)
        out[corpus] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="qwen")
    args = ap.parse_args()
    out = run(args.model)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"separation_{args.model}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({c: {k: {kk: vv for kk, vv in v.items() if kk != "top_formulas"}
                          for k, v in out[c].items()} for c in ("main", "local")}, indent=1))


if __name__ == "__main__":
    main()
