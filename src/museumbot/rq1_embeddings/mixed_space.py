"""RQ1 per Gemma sulle categorie combinate: i testi misti nello spazio dei testi puri.

Le categorie sono le quattro che entrano nelle coppie dei testi misti (Explorer,
Facilitator, Professional/Hobbyist, Recharger). Gli embedding dei testi puri si centrano
sulla media delle quattro categorie della stessa opera; i testi misti si centrano con la
stessa media della loro opera, cosi' stanno nello stesso spazio.

  - probe a 4 classi addestrato sui puri, valutato fuori dal fold per opera
    (GroupKFold): per i puri da' l'accuratezza, per i misti le probabilita' delle 4
    classi. Per ogni coppia e alpha: probabilita' media di X, di Y e delle altre due
    (fuga) e quota di testi assegnati a ciascuna classe (la logistica e' poco sicura:
    le probabilita' medie sono morbide anche sui puri);
  - PCA e LDA costruite sui puri, con le traiettorie dei misti (centroidi per alpha).

Scrive results/mixed_space_<modello>.json e figures/mixed_space_<modello>.png.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from museumbot.common.config import ROOT
from museumbot.common.prompts import CONDITIONS, LABELS
from museumbot.rq1_embeddings.analyze import COLORS, load
from museumbot.rq1_embeddings.mixed import MIXED, embed_mixed
from museumbot.rq2_judge.mixed_judge import PAIRS

RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
CATS = [c for c in CONDITIONS if any(c in p for p in PAIRS)]
INNER = (0.25, 0.5, 0.75)
SEED = 0


def subset_center(X):
    """(A, K, D) -> dati centrati sulla media per opera e la media stessa (A, D)."""
    mu = X.mean(axis=1)
    return X - mu[:, None], mu


def fold_probs(Xc, M, mix_arts, n_splits: int = 5):
    """Probabilita' fuori fold: (A, K, K) per i puri, (N, K) per i misti."""
    A, K, D = Xc.shape
    p_pure, p_mix = np.zeros((A, K, K)), np.zeros((len(M), K))
    for tr, te in GroupKFold(n_splits=n_splits).split(np.arange(A), groups=np.arange(A)):
        clf = LogisticRegression(max_iter=3000, C=1.0)
        clf.fit(Xc[tr].reshape(-1, D), np.tile(np.arange(K), len(tr)))
        p_pure[te] = clf.predict_proba(Xc[te].reshape(-1, D)).reshape(len(te), K, K)
        m = np.isin(mix_arts, te)
        if m.any():
            p_mix[m] = clf.predict_proba(M[m])
    return p_pure, p_mix


def mixed_rows(alias):
    rows = [r for r in map(json.loads, MIXED.open()) if not r["usage"]["failed"]]
    E = embed_mixed(alias, rows)
    keep = [i for i, r in enumerate(rows) if r["alpha"] in INNER]
    return [rows[i] for i in keep], E[keep]


def run(alias: str) -> dict:
    X, arts, _ = load(alias, "full", "local")
    idx = [CONDITIONS.index(c) for c in CATS]
    Xc, mu = subset_center(X[:, idx])
    rows, E = mixed_rows(alias)
    ai = {a: i for i, a in enumerate(arts)}
    mix_arts = np.array([ai[r["artwork_id"]] for r in rows])
    M = E - mu[mix_arts]
    p_pure, p_mix = fold_probs(Xc, M, mix_arts)
    acc = float(np.mean(p_pure.argmax(axis=2) == np.arange(len(CATS))))
    out = {"model": alias, "categories": CATS, "probe4_accuracy": round(acc, 3), "pairs": {}}
    for x, y in PAIRS:
        kx, ky = CATS.index(x), CATS.index(y)
        name = f"{x}+{y}"
        by = {"0": p_pure[:, kx], "1": p_pure[:, ky]}
        for al in INNER:
            m = np.array([r["condition"] == name and r["alpha"] == al for r in rows])
            by[f"{al:g}"] = p_mix[m]
        res = {}
        for al in ("0", "0.25", "0.5", "0.75", "1"):
            P = by[al]
            p, hard = P.mean(axis=0), np.bincount(P.argmax(axis=1), minlength=len(CATS)) / len(P)
            res[al] = {"n": len(P), "probs": {c: round(float(v), 3) for c, v in zip(CATS, p)},
                       "assigned": {c: round(float(v), 3) for c, v in zip(CATS, hard)},
                       "p_x": round(float(p[kx]), 3), "p_y": round(float(p[ky]), 3),
                       "p_other": round(float(1 - p[kx] - p[ky]), 3)}
        out["pairs"][name] = res
    out["figure"] = figure(alias, Xc, M, rows, mix_arts, out)
    return out


def trajectories(Z_pure, Z_mix, rows, keep_mix) -> dict:
    """Centroidi per coppia: X puro, alpha interni, Y puro."""
    tr = {}
    for x, y in PAIRS:
        name = f"{x}+{y}"
        pts = [Z_pure[:, CATS.index(x)].mean(axis=0)]
        for al in INNER:
            m = np.array([keep_mix[i] and r["condition"] == name and r["alpha"] == al
                          for i, r in enumerate(rows)])
            pts.append(Z_mix[m].mean(axis=0))
        pts.append(Z_pure[:, CATS.index(y)].mean(axis=0))
        tr[name] = np.array(pts)
    return tr


def figure(alias, Xc, M, rows, mix_arts, out) -> str:
    A, K, D = Xc.shape
    rng = np.random.default_rng(SEED)
    half = rng.permutation(A)
    tr, te = half[: A // 2], half[A // 2 :]
    lda = LinearDiscriminantAnalysis(n_components=2).fit(
        Xc[tr].reshape(-1, D), np.tile(np.arange(K), len(tr)))
    pca = PCA(n_components=2).fit(Xc.reshape(-1, D))
    views = {"LDA (addestrata su meta' delle opere, mostrata sull'altra meta')":
             (lda.transform(Xc[te].reshape(-1, D)).reshape(len(te), K, 2),
              lda.transform(M), np.isin(mix_arts, te)),
             "PCA sui testi puri": (pca.transform(Xc.reshape(-1, D)).reshape(A, K, 2),
                                    pca.transform(M), np.ones(len(M), bool))}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    styles = {"recharger+professional_hobbyist": "-", "facilitator+professional_hobbyist": "--",
              "explorer+facilitator": ":"}
    for ax, (title, (Zp, Zm, keep)) in zip(axes, views.items()):
        for k, c in enumerate(CATS):
            ax.scatter(Zp[:, k, 0], Zp[:, k, 1], s=12, alpha=0.35, c=COLORS[c], label=LABELS[c],
                       edgecolors="none")
        for name, pts in trajectories(Zp, Zm, rows, keep).items():
            x, y = name.split("+")
            ax.plot(pts[:, 0], pts[:, 1], styles[name], color="black", lw=1.6,
                    label=f"{LABELS[x]} → {LABELS[y]}")
            ax.scatter(pts[1:-1, 0], pts[1:-1, 1], s=[30, 55, 80], c="black", zorder=5)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(f"Testi misti nello spazio delle 4 categorie combinate ({alias}); "
                 "punti neri: alpha 0.25, 0.5, 0.75", fontsize=11)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    path = FIGS / f"mixed_space_{alias}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="qwen")
    args = ap.parse_args()
    out = run(args.model)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"mixed_space_{args.model}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
