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


def sample_token(toks: list[int], mixed: np.ndarray, rng: np.random.Generator) -> int:
    p = np.exp(mixed)
    return int(toks[rng.choice(len(toks), p=p / p.sum())])


def mix_generate(prompts, weights, step, *, eos: set[int], max_tokens: int, temp: float,
                 seeds: list[int]) -> list[tuple[list[int], list[dict]]]:
    """Genera piu' testi insieme. `prompts[i]` sono i prompt (token) degli esperti del testo
    i, `weights[i]` i loro pesi; `step(sequenze)` restituisce per ogni sequenza il dizionario
    token -> log-probabilita' del prossimo token. Gli esperti a peso zero non si chiamano."""
    texts = []
    for p, w, s in zip(prompts, weights, seeds):
        _check(w)
        idx = [k for k, x in enumerate(w) if x > 0]
        texts.append({"prompts": [p[k] for k in idx], "w": [w[k] for k in idx],
                      "rng": np.random.default_rng(s), "toks": [], "diag": [], "done": False})
    while not all(t["done"] for t in texts):
        live = [t for t in texts if not t["done"]]
        seqs = [pr + t["toks"] for t in live for pr in t["prompts"]]
        res = iter(step(seqs))
        for t in live:
            lps = [next(res) for _ in t["prompts"]]
            toks, mixed, cov = mix_top(lps, t["w"], temp)
            tok = sample_token(toks, mixed, t["rng"])
            if tok in eos:
                t["done"] = True
                continue
            t["toks"].append(tok)
            t["diag"].append({"logp": [round(d.get(tok, min(d.values())), 4) for d in lps],
                              "covered": cov[tok],
                              "div": round(divergence(lps, t["w"], toks, mixed, temp), 5)})
            t["done"] = len(t["toks"]) >= max_tokens
    return [(t["toks"], t["diag"]) for t in texts]
