"""Analisi dello spazio latente: esiste uno shift direzionale per categoria di Falk?

Il passaggio critico e' il CENTERING PER OPERA. In uno spazio di embedding la varianza
dominante e' *quale opera* si sta descrivendo, non *per chi*: senza centering qualsiasi
proiezione mostra 100 cluster-opera e zero struttura di categoria. Si usano due riferimenti:

    media:  e'[a,c] = e[a,c] - mean_over_c( e[a,.] )    tutte le condizioni trattate allo stesso modo
    flat:   e'[a,c] = e[a,c] - e[a,flat]                 effetto della categoria rispetto al neutro
    v[c]    = mean_over_a( e'[a,c] )                     steering vector della categoria c

Il riferimento flat da' agli steering vector il significato che interessa ("quanto e in che
direzione la categoria sposta il testo rispetto alla descrizione neutra") ed e' identico in
tutte le varianti dell'ablazione. Con la media, invece, i 6 v[c] sommano a zero per
costruzione e i coseni fra categorie sono spinti verso -1/(C-1): un artefatto, non
un'opposizione. La media resta per le procedure per singola opera basate sulle etichette
(probe, varianza), dove sottrarre un solo testo flat aggiungerebbe il suo rumore.

Metriche, in ordine di forza probatoria:
  1. probe lineare cross-validata, raggruppata per opera  -> lo shift e' preciso?     [media]
  2. consistenza split-half della direzione v[c]           -> proprieta' della categoria? [flat]
  3. distanza cross-validata fra condizioni                -> quali categorie sono vicine?
     (indipendente dal riferimento), coseni fra i v[c]       -> direzione degli effetti [flat]

Tetto: se il corpus ha la replica `full_rep` (nel corpus principale e' `single_b`), il
coseno fra gli steering vector delle due run, per categoria, e' il massimo che un confronto
fra coseni puo' raggiungere (sezione `ceiling` del JSON; la usa anche ablation.py).
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold

from museumbot.common.config import ROOT, emb_path, meta_path
from museumbot.common.corpus import CORPORA
from museumbot.common.prompts import CONDITIONS, FALK_CATEGORIES, LABELS

FIGS = ROOT / "figures"
RESULTS = ROOT / "results"

FLAT = CONDITIONS.index("flat")
FALK_IDX = [CONDITIONS.index(c) for c in FALK_CATEGORIES]
COLORS = {
    "explorer": "#E8743B",
    "facilitator": "#19A979",
    "experience_seeker": "#945ECF",
    "professional_hobbyist": "#1F77B4",
    "recharger": "#13A4B4",
    "flat": "#7F7F7F",
}


# --------------------------------------------------------------------------- dati


def load(model: str, variant: str = "full", corpus: str = "main"):
    emb = np.load(emb_path(model, variant, corpus))
    meta = pd.read_csv(meta_path(variant, corpus))
    assert len(meta) == len(emb), f"meta {len(meta)} != emb {len(emb)}"

    arts = sorted(meta["artwork_id"].unique())
    ai = {a: i for i, a in enumerate(arts)}
    ci = {c: i for i, c in enumerate(CONDITIONS)}

    # tensore (opera, condizione, dim); solo le opere complete su tutte le condizioni
    X = np.full((len(arts), len(CONDITIONS), emb.shape[1]), np.nan, dtype=np.float32)
    for row, (a, c) in enumerate(zip(meta["artwork_id"], meta["condition"])):
        X[ai[a], ci[c]] = emb[row]
    ok = ~np.isnan(X).any(axis=(1, 2))
    dropped = int((~ok).sum())
    X = X[ok]
    if dropped:
        print(f"  scartate {dropped} opere incomplete")
    return X, [a for a, k in zip(arts, ok) if k], meta


def center(X, ref="mean"):
    """Centering per opera. `mean`: sottrae la media delle condizioni dell'opera.
    `flat`: sottrae il testo flat dell'opera (la riga flat diventa zero)."""
    if ref == "mean":
        return X - X.mean(axis=1, keepdims=True)
    if ref == "flat":
        return X - X[:, FLAT : FLAT + 1, :]
    raise ValueError(f"riferimento sconosciuto: {ref!r}")


def steering(X):
    """(A, C, D) -> (5, D): steering vector delle categorie rispetto al flat, nell'ordine
    di FALK_CATEGORIES. Il flat e' l'origine e non ha un vettore proprio."""
    return center(X, "flat")[:, FALK_IDX].mean(axis=0)


def flatten(X):
    """(A, C, D) -> matrice (A*C, D), etichette di condizione, gruppi di opera."""
    A, C, D = X.shape
    return X.reshape(A * C, D), np.tile(np.arange(C), A), np.repeat(np.arange(A), C)


# ------------------------------------------------------------------- 1. probe lineare


def probe(Xc, n_splits=5):
    """Accuracy di una logistica sui dati centrati, con GroupKFold per opera."""
    F, y, g = flatten(Xc)
    preds = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=n_splits).split(F, y, g):
        clf = LogisticRegression(max_iter=3000, C=1.0)
        clf.fit(F[tr], y[tr])
        preds[te] = clf.predict(F[te])
    return accuracy_score(y, preds), confusion_matrix(y, preds), preds, y


# --------------------------------------------------- 2. consistenza della direzione


def half_split(A, seed=0):
    """Una divisione casuale (seed fisso) degli indici di opera in due meta' disgiunte."""
    p = np.random.default_rng(seed).permutation(A)
    return p[: A // 2], p[A // 2 :]


def split_half(X):
    """Coseno fra gli steering vector v[c] (riferimento flat) stimati su due meta' disgiunte
    di opere, nell'ordine di FALK_CATEGORIES. Una sola divisione: la riproducibilita' si
    verifica sulla replica (`replicate_ceiling`)."""
    h1, h2 = half_split(len(X))
    return row_cosines(steering(X[h1]), steering(X[h2]))


# ------------------------------------------------ 3. distanza cross-validata e geometria


def crossval_distance(X):
    """Distanza euclidea quadrata cross-validata fra condizioni, (C, C):

        d2[a,b] = (v1[a] - v1[b]) . (v2[a] - v2[b])     v1, v2 da meta' disgiunte di opere

    E' la crossnobis senza normalizzazione per la covarianza del rumore. Imparziale: vale
    ~0 (anche leggermente negativa) se due condizioni non differiscono, mentre la distanza
    semplice e' sempre gonfiata dal rumore. Usa solo differenze fra condizioni, quindi non
    dipende dal riferimento del centering. Stessa divisione di `split_half`.
    """
    h1, h2 = half_split(len(X))
    # media per condizione senza centering: l'effetto-opera si cancella nelle differenze
    v1, v2 = X[h1].mean(axis=0), X[h2].mean(axis=0)
    d1 = v1[:, None] - v1[None]
    d2 = v2[:, None] - v2[None]
    return np.sum(d1 * d2, axis=-1)


def row_cosines(U, V):
    """Coseno riga per riga fra due matrici (K, D)."""
    return np.sum(U * V, axis=1) / (
        np.linalg.norm(U, axis=1) * np.linalg.norm(V, axis=1) + 1e-12)


def replicate_ceiling(X, arts, Xr, arts_r):
    """cos(v[c], v_rep[c]) sulle opere presenti in entrambe le run."""
    common = sorted(set(arts) & set(arts_r))
    i = [arts.index(a) for a in common]
    j = [arts_r.index(a) for a in common]
    return row_cosines(steering(X[i]), steering(Xr[j]))


def cosine_matrix(V):
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return Vn @ Vn.T


# ------------------------------------------------------------------------- figure


def fig_lda(Xc, arts, path):
    """LDA addestrata su meta' delle opere e proiettata sull'altra meta': senza
    questa separazione la figura sovrastima grossolanamente la separabilita'."""
    F, y, g = flatten(Xc)
    A = len(arts)
    tr_a, te_a = map(set, half_split(A))
    tr = np.array([i for i, gi in enumerate(g) if gi in tr_a])
    te = np.array([i for i, gi in enumerate(g) if gi in te_a])

    lda = LinearDiscriminantAnalysis(n_components=2).fit(F[tr], y[tr])
    Z = lda.transform(F[te])

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for ci, c in enumerate(CONDITIONS):
        m = y[te] == ci
        ax.scatter(Z[m, 0], Z[m, 1], s=26, alpha=0.65, c=COLORS[c], label=LABELS[c],
                   edgecolors="none")
        ax.scatter(*Z[m].mean(axis=0), s=280, marker="X", c=COLORS[c],
                   edgecolors="white", linewidths=1.8, zorder=5)
    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")
    ax.set_title("LDA — addestrata su meta' delle opere, proiettata sull'altra meta'\n"
                 "(embedding centrati per opera)", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_pca(Xc, path):
    F, y, _ = flatten(Xc)
    Z = PCA(n_components=2).fit_transform(F)
    mu = {c: Z[y == ci].mean(axis=0) for ci, c in enumerate(CONDITIONS)}
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for ci, c in enumerate(CONDITIONS):
        m = y == ci
        ax.scatter(Z[m, 0], Z[m, 1], s=22, alpha=0.5, c=COLORS[c], label=LABELS[c],
                   edgecolors="none")
        if c != "flat":
            ax.annotate("", xy=mu[c], xytext=mu["flat"],
                        arrowprops=dict(arrowstyle="->", color=COLORS[c], lw=2.2))
    ax.scatter(*mu["flat"], s=140, marker="o", c=COLORS["flat"], edgecolors="white",
               linewidths=1.5, zorder=5)
    ax.axhline(0, color="#ccc", lw=0.8, zorder=0)
    ax.axvline(0, color="#ccc", lw=0.8, zorder=0)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA sui dati centrati per opera\n"
                 "le frecce sono i vettori di steering v[c], dal centroide del flat", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_cosine(M, path):
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    labs = [LABELS[c] for c in FALK_CATEGORIES]
    ax.set_xticks(range(len(labs)), labs, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(labs)), labs, fontsize=9)
    for i in range(len(labs)):
        for j in range(len(labs)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(M[i, j]) > 0.55 else "black")
    ax.set_title("Coseno fra gli steering vector v[c] (riferimento: flat)", fontsize=11)
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_distance(d2, path):
    """Heatmap della distanza quadrata cross-validata (valori x100 per leggibilita')."""
    d2 = d2 * 100
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(d2, cmap="viridis", vmin=0)
    labs = [LABELS[c] for c in CONDITIONS]
    ax.set_xticks(range(len(labs)), labs, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(labs)), labs, fontsize=9)
    hi = d2.max()
    for i in range(len(labs)):
        for j in range(len(labs)):
            if i == j:
                continue
            ax.text(j, i, f"{d2[i, j]:.1f}", ha="center", va="center", fontsize=8,
                    color="black" if d2[i, j] > 0.6 * hi else "white")
    ax.set_title("Distanza quadrata cross-validata fra condizioni (x100)\n"
                 "split-half, indipendente dal riferimento; ~0 = indistinguibili", fontsize=11)
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_confusion(cm, acc, path):
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    labs = [LABELS[c] for c in CONDITIONS]
    ax.set_xticks(range(len(labs)), labs, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(labs)), labs, fontsize=9)
    for i in range(len(labs)):
        for j in range(len(labs)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("predetta")
    ax.set_ylabel("vera")
    ax.set_title(f"Probe lineare — accuracy {acc:.1%} (chance {1 / len(CONDITIONS):.1%})",
                 fontsize=11)
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--corpus", default="main", choices=CORPORA)
    ap.add_argument("--no-ablation", action="store_true")
    args = ap.parse_args()

    FIGS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    tag = args.model + ("" if args.corpus == "main" else f"_{args.corpus}") + (
        "" if args.variant == "full" else f"_{args.variant}")

    X, arts, _ = load(args.model, args.variant, args.corpus)
    A, C, D = X.shape
    print(f"\n{A} opere x {C} condizioni x {D} dim  [{args.model}, corpus {args.corpus}]")

    Xc = center(X, "mean")  # per probe, varianza e figure di proiezione
    Vm = Xc.mean(axis=0)    # (C, D) scostamenti dal baricentro dell'opera
    V = steering(X)         # (5, D) steering vector rispetto al flat

    out = {"model": args.model, "corpus": args.corpus, "variant": args.variant,
           "n_artworks": A, "n_conditions": C, "dim": D}

    # --- varianza: quanto pesa l'opera rispetto alla categoria ---
    var_total = X.reshape(A * C, D).var(axis=0).sum()
    var_art = X.mean(axis=1).var(axis=0).sum()
    var_cond = Vm.var(axis=0).sum()
    out["variance"] = {
        "artwork_share": float(var_art / var_total),
        "condition_share": float(var_cond / var_total),
    }
    print(f"\nvarianza spiegata: opera {var_art / var_total:.1%} | "
          f"categoria {var_cond / var_total:.1%}")
    print("  -> senza centering per opera l'analisi vedrebbe solo le opere")

    # --- 1. probe ---
    acc, cm, preds, y = probe(Xc)
    chance = 1 / C
    out["probe"] = {"accuracy": float(acc), "chance": chance}
    print(f"\n[1] probe lineare (GroupKFold per opera): {acc:.1%}  (chance {chance:.1%})")
    per = {LABELS[CONDITIONS[i]]: float(cm[i, i] / cm[i].sum()) for i in range(C)}
    for k, v in sorted(per.items(), key=lambda t: -t[1]):
        print(f"      {k:24} {v:.1%}")
    out["probe"]["per_class_recall"] = per

    # --- 2. consistenza split-half ---
    sh = split_half(X)
    out["split_half_cosine"] = {LABELS[c]: float(sh[i]) for i, c in enumerate(FALK_CATEGORIES)}
    print("\n[2] consistenza split-half della direzione v[c] (rif. flat):")
    for i, c in enumerate(FALK_CATEGORIES):
        print(f"      {LABELS[c]:24} {sh[i]:.3f}")

    # --- 3. distanza cross-validata e geometria ---
    d2 = crossval_distance(X)
    out["crossval_distance"] = {
        LABELS[a]: {LABELS[b]: float(d2[i, j]) for j, b in enumerate(CONDITIONS)}
        for i, a in enumerate(CONDITIONS)
    }
    pairs = sorted((d2[i, j], CONDITIONS[i], CONDITIONS[j])
                   for i in range(C) for j in range(i + 1, C))
    print("\n[3] distanza quadrata cross-validata — coppie piu' vicine:")
    for v, a, b in pairs[:3]:
        print(f"      {LABELS[a]:24} ~ {LABELS[b]:24} {v:.4f}")
    print("    piu' lontane:")
    for v, a, b in pairs[-2:]:
        print(f"      {LABELS[a]:24} ~ {LABELS[b]:24} {v:.4f}")

    M = cosine_matrix(V)
    out["cosine_matrix"] = {
        LABELS[a]: {LABELS[b]: float(M[i, j]) for j, b in enumerate(FALK_CATEGORIES)}
        for i, a in enumerate(FALK_CATEGORIES)
    }
    K = len(FALK_CATEGORIES)
    cpairs = sorted(
        ((M[i, j], FALK_CATEGORIES[i], FALK_CATEGORIES[j]) for i in range(K) for j in range(i + 1, K)),
        reverse=True,
    )
    print("    coseno fra steering vector (rif. flat) — stessa direzione:")
    for v, a, b in cpairs[:3]:
        print(f"      {LABELS[a]:24} ~ {LABELS[b]:24} {v:+.3f}")
    print("    direzioni piu' opposte:")
    for v, a, b in cpairs[-2:]:
        print(f"      {LABELS[a]:24} ~ {LABELS[b]:24} {v:+.3f}")

    # --- tetto dalla replica ---
    if args.variant == "full" and emb_path(args.model, "full_rep", args.corpus).exists():
        Xr, arts_r, _ = load(args.model, "full_rep", args.corpus)
        ceil = replicate_ceiling(X, arts, Xr, arts_r)
        out["ceiling"] = {LABELS[c]: float(ceil[i]) for i, c in enumerate(FALK_CATEGORIES)}
        print("    tetto (cos fra steering vector di due run dello stesso prompt):")
        print("      " + "  ".join(f"{LABELS[c]} {ceil[i]:.3f}" for i, c in enumerate(FALK_CATEGORIES)))

    # --- figure: solo quelle citate nei report e nella tesi (Qwen3) ---
    if args.model == "qwen":
        fig_pca(Xc, FIGS / f"pca_{tag}.png")
        fig_cosine(M, FIGS / f"cosine_{tag}.png")
        if args.corpus == "main":
            fig_lda(Xc, arts, FIGS / f"lda_{tag}.png")
            fig_distance(d2, FIGS / f"distance_{tag}.png")
            fig_confusion(cm, acc, FIGS / f"confusion_{tag}.png")
        print(f"\nfigure -> {FIGS.relative_to(ROOT)}/*_{tag}.png")

    (RESULTS / f"metrics_{tag}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nmetriche -> results/metrics_{tag}.json")


if __name__ == "__main__":
    main()
