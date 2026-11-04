"""Miscela di categorie passo per passo, indipendente dal motore di generazione.

A ogni passo ogni esperto (un prompt di categoria con peso w_k > 0) restituisce le prime K
log-probabilita' del prossimo token (per vLLM: `logprobs=K`, distribuzione grezza). Sull'
unione dei token proposti:

    l_k(t) = log p_k(t) / T            se t e' fra le prime K dell'esperto k
           = min_t' log p_k(t') / T    altrimenti (stima per eccesso)
    log p(t) = sum_k w_k l_k(t), rinormalizzata sull'unione (media geometrica pesata)

Si campiona un token con il generatore del testo e lo si aggiunge a tutti gli esperti.
Con un solo esperto e' il campionamento a temperatura T troncato alle prime K. Diagnostica
per token: log-probabilita' per esperto, se il token era fra le prime K di ciascuno, e la
divergenza sum_k w_k KL(q_k || p) degli esperti dalla miscela (zero con un esperto).

Ingresso della minoranza (facoltativo, due esperti): a ogni passo, con probabilita' q il
token lo sceglie da solo l'esperto a peso minore, altrimenti la miscela. q = `minority_q`
fa vincere alla minoranza la sua quota m alla distanza d* fra i due esperti. Con il veto
la minoranza sceglie solo fra i suoi token che sono anche fra le prime K della maggioranza
(rinormalizzati); se non ce n'e' nessuno il passo lo fa la miscela. Spec:
docs/plans/2026-11-04-ingresso-minoranza-design.md.
"""

import math

import numpy as np


def _check(weights) -> None:
    if any(w < 0 for w in weights) or not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
        raise ValueError(f"pesi non validi: {weights}")


def _normalize(x: np.ndarray) -> np.ndarray:
    m = x.max()
    return x - (m + np.log(np.exp(x - m).sum()))


def _tempered(lps: list[dict], toks: list[int], temp: float) -> np.ndarray:
    """(esperti, token): log-probabilita' a temperatura T, con il minimo per i mancanti."""
    out = np.empty((len(lps), len(toks)))
    for k, d in enumerate(lps):
        floor = min(d.values())
        out[k] = [d.get(t, floor) / temp for t in toks]
    return out


def mix_top(lps: list[dict], weights: list[float], temp: float):
    """Token dell'unione, log-probabilita' della miscela, copertura per esperto attivo."""
    if len(lps) != len(weights):
        raise ValueError("un peso per esperto")
    _check(weights)
    active = [(d, w) for d, w in zip(lps, weights) if w > 0]
    toks = sorted({t for d, _ in active for t in d})
    lt = _tempered([d for d, _ in active], toks, temp)
    w = np.array([w for _, w in active])
    mixed = _normalize(w @ lt)
    cov = {t: [t in d for d, _ in active] for t in toks}
    return toks, mixed, cov


def divergence(lps: list[dict], weights: list[float], toks, mixed, temp: float) -> float:
    """sum_k w_k KL(q_k || p) sull'unione, con q_k l'esperto temperato e rinormalizzato."""
    active = [(d, w) for d, w in zip(lps, weights) if w > 0]
    lt = _tempered([d for d, _ in active], toks, temp)
    q = np.array([_normalize(row) for row in lt])
    kl = (np.exp(q) * (q - mixed)).sum(axis=1)
    return max(float(sum(w * k for (_, w), k in zip(active, kl))), 0.0)


def minority_q(alpha: float, dstar: float) -> float:
    """q tale che (1 - q) s + q = m, con s = 1 / (1 + exp((1 - 2m) d*)) la quota della
    minoranza (peso m) nella miscela geometrica a distanza d*; zero a alpha 0, 0.5 e 1."""
    m = min(alpha, 1 - alpha)
    if m <= 0:
        return 0.0
    s = 1 / (1 + math.exp((1 - 2 * m) * dstar))
    return max((m - s) / (1 - s), 0.0)


def minority(weights: list[float]) -> int | None:
    """Indice dell'esperto a peso minore fra quelli attivi; None se non ce n'e' uno solo."""
    active = [(w, k) for k, w in enumerate(weights) if w > 0]
    if len(active) != 2 or active[0][0] == active[1][0]:
        return None
    return min(active)[1]


def vetoed(lps: list[dict], low: int) -> dict:
    """Prime K della minoranza ristrette ai token fra le prime K della maggioranza."""
    high = 1 - low
    return {t: lp for t, lp in lps[low].items() if t in lps[high]}


def sample_token(toks: list[int], mixed: np.ndarray, rng: np.random.Generator) -> int:
    p = np.exp(mixed)
    return int(toks[rng.choice(len(toks), p=p / p.sum())])


def extra_diag(tok: int, solo: bool, toks: list[int], mixed: np.ndarray) -> dict:
    """Diagnostica in piu' con l'ingresso della minoranza: se il token l'ha scelto la sola
    minoranza e la sua log-probabilita' nella miscela geometrica (`pmix`): con `solo`,
    1 - exp(pmix) e' la probabilita' che senza q quel token non sarebbe uscito."""
    return {"solo": solo, "pmix": round(float(mixed[toks.index(tok)]), 4)}


def mix_generate(prompts, weights, step, *, eos: set[int], max_tokens: int, temp: float,
                 seeds: list[int], qs: list[float] | None = None, veto: bool = False
                 ) -> list[tuple[list[int], list[dict]]]:
    """Genera piu' testi insieme. `prompts[i]` sono i prompt (token) degli esperti del testo
    i, `weights[i]` i loro pesi; `step(sequenze)` restituisce per ogni sequenza il dizionario
    token -> log-probabilita' del prossimo token. Gli esperti a peso zero non si chiamano.
    `qs[i]` > 0: ingresso della minoranza con probabilita' q (diagnostica `extra_diag`);
    `veto`: la minoranza sceglie solo fra token che la maggioranza ha fra le sue prime K."""
    texts = []
    for i, (p, w, s) in enumerate(zip(prompts, weights, seeds)):
        _check(w)
        idx = [k for k, x in enumerate(w) if x > 0]
        q = qs[i] if qs else 0.0
        low = minority([w[k] for k in idx]) if q > 0 else None
        if q > 0 and low is None:
            raise ValueError("l'ingresso della minoranza vuole due esperti con pesi diversi")
        texts.append({"prompts": [p[k] for k in idx], "w": [w[k] for k in idx], "q": q,
                      "low": low, "rng": np.random.default_rng(s), "toks": [], "diag": [],
                      "done": False})
    while not all(t["done"] for t in texts):
        live = [t for t in texts if not t["done"]]
        seqs = [pr + t["toks"] for t in live for pr in t["prompts"]]
        res = iter(step(seqs))
        for t in live:
            lps = [next(res) for _ in t["prompts"]]
            toks, mixed, cov = mix_top(lps, t["w"], temp)
            # con q = 0 nessuna estrazione in piu': stesso flusso del generatore di prima
            solo = t["q"] > 0 and t["rng"].random() < t["q"]
            if solo:
                own = vetoed(lps, t["low"]) if veto else lps[t["low"]]
                solo = bool(own)  # niente in comune con la maggioranza: decide la miscela
            if solo:
                stoks, smixed, _ = mix_top([own], [1.0], temp)
                tok = sample_token(stoks, smixed, t["rng"])
            else:
                tok = sample_token(toks, mixed, t["rng"])
            if tok in eos:
                t["done"] = True
                continue
            t["toks"].append(tok)
            t["diag"].append({"logp": [round(d.get(tok, min(d.values())), 4) for d in lps],
                              "covered": cov[tok],
                              "div": round(divergence(lps, t["w"], toks, mixed, temp), 5),
                              **(extra_diag(tok, solo, toks, mixed) if t["q"] > 0 else {})})
            t["done"] = len(t["toks"]) >= max_tokens
    return [(t["toks"], t["diag"]) for t in texts]
