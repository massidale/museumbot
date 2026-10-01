"""Chain vs prompt singolo: la differenza fra i due metodi e' pari al rumore di generazione?

Dati: data/chain_study.jsonl (generation/chain_study.py), tutti dallo stesso provider.

Livello testo, per ogni cella (opera a, categoria c), su embedding L2-normalizzati:

    within  = cos(S_a, S_b)                      rumore: due run del prompt singolo
    between = 1/2 [cos(C, S_a) + cos(C, S_b)]    differenza fra metodi + rumore
    R       = mean(1 - between) / mean(1 - within)

R ~ 1: un testo della chain dista da un testo singolo quanto due testi singoli fra loro.
La cella tiene fissi opera e categoria, quindi la loro varianza si cancella. IC al 95% con
bootstrap sulle opere (le celle della stessa opera non sono indipendenti).

Criterio, fissato prima di vedere i dati: equivalenti se il limite superiore dell'IC di R
e' sotto EQUIV_MARGIN (1.10), cioè se la chain aggiunge al piu' il 10% di dissimilarita'
oltre al rumore.

Livello steering (l'effetto di categoria e' lo stesso?), riferimento flat della sessione:

    tetto  = cos(v_Sa[c], v_Sb[c])
    chain  = 1/2 [cos(v_C[c], v_Sa[c]) + cos(v_C[c], v_Sb[c])]
    rel    = chain / tetto                      ~1 = stesso shift di categoria

Deriva di routing: lo stesso `within`, calcolato fra `full` e `full_rep` del corpus
principale (routing libero, settembre vs ottobre), dice quanto rumore aggiunge cambiare
provider. Tutti i testi, anche quelli del corpus, passano per la stessa `clean_guide`.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from museumbot.common.config import ROOT
from museumbot.common.prompts import CONDITIONS, FALK_CATEGORIES, LABELS
from museumbot.generation.chain_study import OUT as STUDY
from museumbot.generation.chain_study import clean_guide
from museumbot.rq1_embeddings.analyze import FLAT, steering
from museumbot.rq1_embeddings.embed import MODELS

GEN = ROOT / "data" / "generations.jsonl"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
EQUIV_MARGIN = 1.10
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- dati


def collect_rows() -> list[dict]:
    """Testi dello studio + full/full_rep del corpus (per la deriva), tutti puliti."""
    rows = [{"artwork_id": r["artwork_id"], "condition": r["condition"],
             "method": r["method"], "text": r["text"], "words": r["words"]}
            for r in map(json.loads, STUDY.open())]
    for r in map(json.loads, GEN.open()):
        v = r.get("variant", "full")
        if v in ("full", "full_rep") and r["condition"] != "flat":
            t = clean_guide(r["text"])
            rows.append({"artwork_id": r["artwork_id"], "condition": r["condition"],
                         "method": v, "text": t, "words": len(t.split())})
    rows.sort(key=lambda r: (r["method"], r["artwork_id"], r["condition"]))
    return rows


def embed(alias: str, device: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Embedding in cache: si ricalcola solo se i testi sono cambiati."""
    rows = collect_rows()
    meta = pd.DataFrame(rows)
    emb_file = ROOT / "data" / f"emb_{alias}_chain_study.npy"
    meta_file = ROOT / "data" / "meta_chain_study.csv"
    if emb_file.exists() and meta_file.exists():
        cached = pd.read_csv(meta_file)
        E = np.load(emb_file)
        if len(cached) == len(meta) == len(E) and (cached["text"] == meta["text"]).all():
            return meta, E
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(MODELS[alias], device=device)
    E = m.encode(meta["text"].tolist(), batch_size=8, normalize_embeddings=True,
                 show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    np.save(emb_file, E)
    meta.to_csv(meta_file, index=False)
    return meta, E


def index(meta: pd.DataFrame) -> dict[tuple[str, str, str], int]:
    return {(a, c, m): i for i, (a, c, m) in
            enumerate(zip(meta["artwork_id"], meta["condition"], meta["method"]))}


# ------------------------------------------------------------------ livello testo


def pair_cos(E, idx, cells, m1, m2) -> np.ndarray:
    return np.array([E[idx[(a, c, m1)]] @ E[idx[(a, c, m2)]] for a, c in cells])


def complete_cells(idx, methods) -> list[tuple[str, str]]:
    arts = sorted({a for a, _, _ in idx})
    return [(a, c) for a in arts for c in FALK_CATEGORIES
            if all((a, c, m) in idx for m in methods)]


def text_level(E, idx) -> pd.DataFrame:
    cells = complete_cells(idx, ("single_a", "single_b", "chain"))
    within = pair_cos(E, idx, cells, "single_a", "single_b")
    between = (pair_cos(E, idx, cells, "chain", "single_a")
               + pair_cos(E, idx, cells, "chain", "single_b")) / 2
    df = pd.DataFrame(cells, columns=["artwork_id", "condition"])
    df["within"], df["between"] = within, between
    mixed_cells = set(complete_cells(idx, ("full", "full_rep")))
    df["mixed"] = [E[idx[(a, c, "full")]] @ E[idx[(a, c, "full_rep")]]
                   if (a, c) in mixed_cells else np.nan for a, c in cells]
    return df


def ratio(df: pd.DataFrame, num: str, den: str = "within") -> float:
    d = df.dropna(subset=[num, den])
    return float((1 - d[num]).mean() / (1 - d[den]).mean())


def bootstrap(df: pd.DataFrame, stat, n: int = 2000) -> tuple[float, float]:
    """IC al 95% di `stat(df)` ricampionando le opere con reinserimento."""
    groups = {a: g for a, g in df.groupby("artwork_id")}
    arts = list(groups)
    vals = [stat(pd.concat([groups[a] for a in RNG.choice(arts, len(arts))]))
            for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# --------------------------------------------------------------- livello steering


def tensor(E, idx, arts, method) -> np.ndarray:
    """(A, C, D): categorie dal metodo, flat della sessione nella sua posizione."""
    X = np.zeros((len(arts), len(CONDITIONS), E.shape[1]), dtype=np.float32)
    for i, a in enumerate(arts):
        for j, c in enumerate(CONDITIONS):
            X[i, j] = E[idx[(a, "flat", "flat")] if j == FLAT else idx[(a, c, method)]]
    return X


def cos_rows(U, V) -> np.ndarray:
    return np.sum(U * V, axis=1) / (np.linalg.norm(U, axis=1) * np.linalg.norm(V, axis=1))


def steering_stats(Xa, Xb, Xc) -> dict[str, np.ndarray]:
    va, vb, vc = steering(Xa), steering(Xb), steering(Xc)
    ceiling = cos_rows(va, vb)
    chain = (cos_rows(vc, va) + cos_rows(vc, vb)) / 2
    norm_s = (np.linalg.norm(va, axis=1) + np.linalg.norm(vb, axis=1)) / 2
    return {"ceiling": ceiling, "chain": chain, "rel": chain / ceiling,
            "norm_ratio": np.linalg.norm(vc, axis=1) / norm_s}


def steering_level(E, idx, n_boot: int = 1000) -> dict:
    full_cells = complete_cells(idx, ("single_a", "single_b", "chain"))
    arts = [a for a in sorted({a for a, _ in full_cells})
            if (a, "flat", "flat") in idx
            and sum(1 for b, _ in full_cells if b == a) == len(FALK_CATEGORIES)]
    Xs = [tensor(E, idx, arts, m) for m in ("single_a", "single_b", "chain")]
    point = steering_stats(*Xs)
    boot = []
    for _ in range(n_boot):
        s = RNG.choice(len(arts), len(arts))
        boot.append(steering_stats(*(X[s] for X in Xs))["rel"])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    out = {"n_artworks": len(arts)}
    for i, c in enumerate(FALK_CATEGORIES):
        out[LABELS[c]] = {k: float(v[i]) for k, v in point.items()}
        out[LABELS[c]]["rel_ci95"] = [float(lo[i]), float(hi[i])]
    out["rel_mean"] = float(point["rel"].mean())
    return out


# ------------------------------------------------------------------------- figura


def fig_dissimilarity(df: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bins = np.linspace(0, max(1 - df[["within", "between", "mixed"]].min().min(), 0.01), 40)
    series = [("within", "rumore: singolo vs singolo (stesso provider)", "#1F77B4"),
              ("between", "chain vs singolo", "#E8743B"),
              ("mixed", "full vs full_rep (routing libero, set. vs ott.)", "#9E9E9E")]
    for col, lab, color in series:
        d = 1 - df[col].dropna()
        ax.hist(d, bins=bins, alpha=0.45, color=color, label=f"{lab} — media {d.mean():.3f}")
        ax.axvline(d.mean(), color=color, lw=1.5)
    ax.set_xlabel("dissimilarità coseno 1 − cos, per cella (opera × categoria)")
    ax.set_ylabel("celle")
    ax.set_title("Chain vs prompt singolo: differenza fra metodi e rumore di generazione",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen", choices=list(MODELS))
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    meta, E = embed(args.model, args.device)
    idx = index(meta)
    df = text_level(E, idx)

    R = ratio(df, "between")
    R_ci = bootstrap(df, lambda d: ratio(d, "between"))
    D = ratio(df, "mixed")
    D_ci = bootstrap(df.dropna(subset=["mixed"]), lambda d: ratio(d, "mixed"))
    per_cat = {LABELS[c]: ratio(g, "between") for c, g in df.groupby("condition")}
    words = meta.groupby("method")["words"].mean().round(1).to_dict()

    out = {
        "model": args.model,
        "n_cells": len(df),
        "criterion": f"equivalenti se IC95 superiore di R < {EQUIV_MARGIN}",
        "text": {
            "within_mean": float(df["within"].mean()),
            "between_mean": float(df["between"].mean()),
            "R": R, "R_ci95": R_ci,
            "equivalent": R_ci[1] < EQUIV_MARGIN,
            "R_per_category": per_cat,
        },
        "routing_drift": {
            "n_cells": int(df["mixed"].notna().sum()),
            "mixed_mean": float(df["mixed"].mean()),
            "D": D, "D_ci95": D_ci,
        },
        "steering": steering_level(E, idx),
        "words": words,
    }

    print(f"\n[{args.model}] {len(df)} celle")
    print(f"  cos singolo-singolo {out['text']['within_mean']:.4f} | "
          f"chain-singolo {out['text']['between_mean']:.4f}")
    print(f"  R = {R:.3f}  IC95 [{R_ci[0]:.3f}, {R_ci[1]:.3f}]  -> "
          f"{'equivalenti' if out['text']['equivalent'] else 'NON equivalenti'} "
          f"(margine {EQUIV_MARGIN})")
    for k, v in per_cat.items():
        print(f"      {k:24} R {v:.3f}")
    print(f"  deriva di routing: D = {D:.3f}  IC95 [{D_ci[0]:.3f}, {D_ci[1]:.3f}]  "
          f"({out['routing_drift']['n_cells']} celle)")
    st = out["steering"]
    print(f"  steering (rif. flat, {st['n_artworks']} opere): rel medio {st['rel_mean']:.3f}")
    for c in FALK_CATEGORIES:
        s = st[LABELS[c]]
        print(f"      {LABELS[c]:24} tetto {s['ceiling']:.3f}  chain {s['chain']:.3f}  "
              f"rel {s['rel']:.3f} [{s['rel_ci95'][0]:.3f}, {s['rel_ci95'][1]:.3f}]  "
              f"norma {s['norm_ratio']:.2f}x")
    print(f"  parole medie: {words}")

    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    (RESULTS / f"chain_noise_{args.model}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    fig_dissimilarity(df, FIGS / f"chain_noise_{args.model}.png")
    print(f"\n-> results/chain_noise_{args.model}.json, figures/chain_noise_{args.model}.png")


if __name__ == "__main__":
    main()
