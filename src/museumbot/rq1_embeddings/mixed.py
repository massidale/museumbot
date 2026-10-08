"""Testi a profilo misto (passo 3 del profilo continuo): stanno davvero fra le due categorie?

Ogni testo misto (opera a, coppia X -> Y, peso alpha) si centra sul flat della sua opera nel
corpus `local`, d = e - e[a, flat], e si confronta con gli steering vector v_X, v_Y del
corpus `local` (tutte le opere):

    posizione = (d - v_X) . (v_Y - v_X) / |v_Y - v_X|^2     0 su X, 1 su Y
    residuo   = |d - proiezione di d sul piano (v_X, v_Y)| / |d|

Verifica del metodo (alpha = 0 e 1, un solo esperto): lo steering vector dei testi passo per
passo contro quello dei testi puri sulle stesse opere, diviso per il tetto full/full_rep
sulle stesse opere. Criteri nella spec docs/plans/2026-10-06-profilo-continuo-design.md.
Curva per coppia: alpha interni dai testi misti, vertici (alpha = 0 e 1) dai testi puri del
corpus `local` su tutte le opere; i vertici passo per passo del pilota servono solo alla
verifica. Scrive results/mixed_<modello>.json e figures/mixed_<modello>_<coppia>.png.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from museumbot.common.config import ROOT
from museumbot.common.prompts import CONDITIONS, LABELS
from museumbot.rq1_embeddings.analyze import FLAT, load, row_cosines, steering

MIXED = ROOT / "data" / "local" / "mixed.jsonl"
EMB = ROOT / "data" / "emb" / "mixed"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"


def position(d, vx, vy):
    seg = vy - vx
    return (np.asarray(d) - vx) @ seg / (seg @ seg)


def residual(d, vx, vy):
    d = np.atleast_2d(d)
    B = np.stack([vx, vy], axis=1)
    coef, *_ = np.linalg.lstsq(B, d.T, rcond=None)
    r = np.linalg.norm(d - (B @ coef).T, axis=1) / np.linalg.norm(d, axis=1)
    return r if len(r) > 1 else float(r[0])


def embed_mixed(alias: str, rows: list[dict]) -> np.ndarray:
    """Embedding dei testi misti, ricalcolati se il file non corrisponde alle righe."""
    from museumbot.rq1_embeddings.embed import MODELS

    EMB.mkdir(parents=True, exist_ok=True)
    path, meta = EMB / f"{alias}.npy", EMB / f"meta_{alias}.csv"
    keys = [f"{r['artwork_id']}|{r['condition']}|{r['alpha']}" for r in rows]
    if path.exists() and meta.exists() and pd.read_csv(meta)["key"].tolist() == keys:
        return np.load(path)
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(MODELS[alias], device="mps")
    emb = m.encode([r["text"] for r in rows], batch_size=8, normalize_embeddings=True,
                   convert_to_numpy=True).astype(np.float32)
    np.save(path, emb)
    pd.DataFrame({"key": keys}).to_csv(meta, index=False)
    return emb


def method_check(Xm, Xp, Xr, idx_p, cond: int) -> float:
    """cos(v passo per passo, v puro) / tetto, sulle stesse opere, per la categoria `cond`."""
    v_step = Xm.mean(axis=0)
    vp = steering(Xp[idx_p])
    vr = steering(Xr[idx_p])
    k = cond - (cond > FLAT)  # steering() esclude il flat
    ceil = row_cosines(vp[k : k + 1], vr[k : k + 1])[0]
    cos = row_cosines(v_step[None], vp[k : k + 1])[0]
    return float(cos / ceil)


def run(alias: str) -> dict:
    rows = [r for r in map(json.loads, MIXED.open()) if not r["usage"]["failed"]]
    E = embed_mixed(alias, rows)
    Xp, arts, meta = load(alias, "full", "local")
    Xr, arts_r, _ = load(alias, "full_rep", "local")
    assert arts == arts_r
    V = steering(Xp)
    ai = {a: i for i, a in enumerate(arts)}
    out = {"model": alias, "pairs": {}}
    words = meta.set_index(["artwork_id", "condition"])["words"]
    for pair in sorted({tuple(r["pair"]) for r in rows}):
        cx, cy = (CONDITIONS.index(c) for c in pair)
        vx, vy = V[cx - (cx > FLAT)], V[cy - (cy > FLAT)]
        sel = [i for i, r in enumerate(rows) if tuple(r["pair"]) == pair]
        a_idx = np.array([ai[rows[i]["artwork_id"]] for i in sel])
        alpha = np.array([rows[i]["alpha"] for i in sel])
        D = E[sel] - Xp[a_idx, FLAT]
        # verifica del metodo: vertici generati passo per passo (pilota)
        checks = {}
        for al, c in ((0.0, cx), (1.0, cy)):
            m = alpha == al
            if m.any():
                checks[f"{al:g}"] = round(method_check(D[m], Xp, Xr, a_idx[m], c), 3)
        # curva: alpha interni dai testi misti, vertici dai testi puri su tutte le opere
        inner = (alpha > 0) & (alpha < 1)
        n = len(arts)
        Dv = np.concatenate([Xp[:, cx] - Xp[:, FLAT], Xp[:, cy] - Xp[:, FLAT]])
        al_all = np.concatenate([np.zeros(n), np.ones(n), alpha[inner]])
        D_all = np.concatenate([Dv, D[inner]])
        w_all = np.concatenate([[words[(a, pair[0])] for a in arts],
                                [words[(a, pair[1])] for a in arts],
                                [rows[i]["words"] for i, k in zip(sel, inner) if k]])
        cov_all = np.concatenate([np.full(2 * n, np.nan),
                                  [rows[i]["usage"]["covered_all"] for i, k in zip(sel, inner) if k]])
        pos, res = position(D_all, vx, vy), residual(D_all, vx, vy)
        res_pure = float(np.median(res[al_all != np.clip(al_all, 1e-9, 1 - 1e-9)]))
        by = {}
        for al in sorted(set(al_all)):
            m = al_all == al
            by[f"{al:g}"] = {
                "n": int(m.sum()), "source": "puri" if al in (0, 1) else "misti",
                "position_median": round(float(np.median(pos[m])), 3),
                "position_iqr": [round(float(q), 3) for q in np.percentile(pos[m], [25, 75])],
                "residual_median": round(float(np.median(res[m])), 3),
                "residual_ratio": round(float(np.median(res[m]) / res_pure), 3),
                "covered_all_mean": None if al in (0, 1) else round(float(np.nanmean(cov_all[m])), 4),
                "words_median": float(np.median(w_all[m]))}
        rho = spearmanr(al_all, pos).statistic
        out["pairs"]["+".join(pair)] = {"by_alpha": by, "residual_pure_median": round(res_pure, 3),
                                        "segment_norm": round(float(np.linalg.norm(vy - vx)), 3),
                                        "spearman_alpha_position": round(float(rho), 3),
                                        "method_check_ratio": checks}
        plot(alias, pair, al_all, pos)
    return out


def plot(alias, pair, alpha, pos) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    rng = np.random.default_rng(0)
    ax.scatter(alpha + rng.uniform(-0.03, 0.03, len(alpha)), pos, s=12, alpha=0.6)
    for al in sorted(set(alpha)):
        ax.hlines(np.median(pos[alpha == al]), al - 0.08, al + 0.08, color="k")
    ax.plot([0, 1], [0, 1], ls=":", color="grey")
    ax.set_xlabel(f"alpha (0 = {LABELS[pair[0]]}, 1 = {LABELS[pair[1]]})")
    ax.set_ylabel("posizione sul segmento v_X -> v_Y")
    ax.set_title(f"Testi misti, {alias}: posizione per alpha (linea: mediana)")
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / f"mixed_{alias}_{'+'.join(pair)}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="qwen")
    args = ap.parse_args()
    out = run(args.model)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"mixed_{args.model}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
