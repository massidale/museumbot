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

Due metodi (--method): `mix` (miscela dei logit) e `pct` (percentuale nel prompt); il probe
e' sempre addestrato sui puri. Scrive results/mixed_space_<modello>.json e
figures/mixed_space_<modello>.png (mix), results/mixed_space_pct_<modello>.json e
figures/mixed_space_pct_qwen.png (pct, con le traiettorie di entrambi i metodi).
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from museumbot.common.config import ROOT
from museumbot.common.prompts import CONDITIONS, LABELS
from museumbot.rq1_embeddings.analyze import COLORS, load
from museumbot.rq1_embeddings.mixed import SOURCES, embed_mixed, load_rows
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


def mixed_rows(alias, method="mix"):
    rows = load_rows(method)
    E = embed_mixed(alias, rows, SOURCES[method][1])
    keep = [i for i, r in enumerate(rows) if r["alpha"] in INNER]
    return [rows[i] for i in keep], E[keep]


def centered(alias, method, arts, mu):
    """Righe del metodo, loro embedding centrato sulla media della loro opera, indice opera."""
    rows, E = mixed_rows(alias, method)
    ai = {a: i for i, a in enumerate(arts)}
    mix_arts = np.array([ai[r["artwork_id"]] for r in rows])
    return rows, E - mu[mix_arts], mix_arts


def run(alias: str, method: str = "mix") -> dict:
    X, arts, _ = load(alias, "full", "local")
    idx = [CONDITIONS.index(c) for c in CATS]
    Xc, mu = subset_center(X[:, idx])
    rows, M, mix_arts = centered(alias, method, arts, mu)
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
    if alias == "qwen":  # figura citata: solo Qwen3
        sets = {method: (M, rows, mix_arts)}
        if method != "mix":  # il confronto mostra anche la miscela
            r_mix, M_mix, a_mix = centered(alias, "mix", arts, mu)
            sets = {"mix": (M_mix, r_mix, a_mix), **sets}
        figure(alias, Xc, sets, method)
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


def method_trajectories(Z_pure, sets) -> dict:
    """Traiettorie per metodo: {metodo: {coppia: punti}}; gli estremi (puri) sono in comune.

    sets: {metodo: (Z_mix, rows, keep_mix)}.
    """
    return {m: trajectories(Z_pure, Z, rows, keep) for m, (Z, rows, keep) in sets.items()}


def figure(alias, Xc, sets, method="mix") -> str:
    """sets: {metodo: (M, rows, mix_arts)}; con piu' metodi il secondo e' tratteggiato."""
    A, K, D = Xc.shape
    rng = np.random.default_rng(SEED)
    half = rng.permutation(A)
    tr, te = half[: A // 2], half[A // 2 :]
    lda = LinearDiscriminantAnalysis(n_components=2).fit(
        Xc[tr].reshape(-1, D), np.tile(np.arange(K), len(tr)))
    pca = PCA(n_components=2).fit(Xc.reshape(-1, D))
    views = {"LDA": (lda.transform(Xc[te].reshape(-1, D)).reshape(len(te), K, 2),
                     {m: (lda.transform(M), rows, np.isin(arts, te))
                      for m, (M, rows, arts) in sets.items()}),
             "PCA": (pca.transform(Xc.reshape(-1, D)).reshape(A, K, 2),
                     {m: (pca.transform(M), rows, np.ones(len(M), bool))
                      for m, (M, rows, _) in sets.items()})}
    styles = {"mix": ("-", "logit mixing"), "pct": ("--", "prompt percentage")}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    # un colore per coppia, lontano da quelli delle categorie
    pair_colors = {"recharger+professional_hobbyist": "#C62828",
                   "facilitator+professional_hobbyist": "#222222",
                   "explorer+facilitator": "#B8860B"}
    for ax, (title, (Zp, Zsets)) in zip(axes, views.items()):
        for k, c in enumerate(CATS):
            ax.scatter(Zp[:, k, 0], Zp[:, k, 1], s=12, alpha=0.35, c=COLORS[c], label=LABELS[c],
                       edgecolors="none")
        for m, trs in method_trajectories(Zp, Zsets).items():
            for name, pts in trs.items():
                x, y = name.split("+")
                ax.plot(pts[:, 0], pts[:, 1], styles[m][0], color=pair_colors[name], lw=1.6,
                        label=f"{LABELS[x]} → {LABELS[y]}" if m == "mix" or len(sets) == 1
                        else None)
                ax.scatter(pts[1:-1, 0], pts[1:-1, 1], s=[8, 14, 22], c=pair_colors[name],
                           zorder=5)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    if len(sets) > 1:  # stile della linea = metodo
        for m in sets:
            handles.append(Line2D([], [], color="0.4", ls=styles[m][0], lw=1.6))
            labels.append(styles[m][1])
    axes[0].legend(handles, labels, frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    path = FIGS / f"{suffix(alias, method, 'mixed_space')}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def suffix(alias, method, stem="mixed_space") -> str:
    """Nome senza estensione: mixed_space_qwen (mix) o mixed_space_pct_qwen (pct)."""
    return f"{stem}_{alias}" if method == "mix" else f"{stem}_{method}_{alias}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--method", choices=sorted(SOURCES), default="mix")
    args = ap.parse_args()
    out = run(args.model, args.method)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{suffix(args.model, args.method)}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
