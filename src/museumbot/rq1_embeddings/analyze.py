"""Analisi dello spazio latente: esiste uno shift direzionale per categoria di Falk?

Il passaggio critico e' il CENTERING PER OPERA. In uno spazio di embedding la varianza
dominante e' *quale opera* si sta descrivendo, non *per chi*: senza centering qualsiasi
PCA/t-SNE mostra 100 cluster-opera e zero struttura di categoria.

    e'[a,c] = e[a,c] - mean_over_c( e[a,.] )    rimuove l'identita' dell'opera
    v[c]    = mean_over_a( e'[a,c] )            steering vector della categoria c

Metriche, in ordine di forza probatoria:
  1. probe lineare cross-validata, raggruppata per opera  -> lo shift e' preciso?
  2. consistenza split-half della direzione v[c]           -> e' una proprieta' della categoria?
  3. geometria fra i v[c] (coseni, norme)                  -> quali categorie collassano?
  4. test di permutazione entro opera                      -> p-value
  5. ablazione lessicale                                   -> registro o solo vocabolario?
"""

import argparse
import json
import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold

from museumbot.common.config import ROOT, emb_path, meta_path
from museumbot.common.prompts import CONDITIONS, LABELS

FIGS = ROOT / "figures"
RESULTS = ROOT / "results"

RNG = np.random.default_rng(0)
COLORS = {
    "explorer": "#E8743B",
    "facilitator": "#19A979",
    "experience_seeker": "#945ECF",
    "professional_hobbyist": "#1F77B4",
    "recharger": "#13A4B4",
    "flat": "#7F7F7F",
}


# --------------------------------------------------------------------------- dati


def load(model: str, variant: str = "full"):
    emb = np.load(emb_path(model, variant))
    meta = pd.read_csv(meta_path(variant))
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


def split_half(X, n_rep=200):
    """Coseno fra v[c] stimato su due meta' disgiunte di opere, mediato su n_rep split."""
    A, C, _ = X.shape
    out = np.zeros((n_rep, C))
    for i in range(n_rep):
        p = RNG.permutation(A)
        h1, h2 = p[: A // 2], p[A // 2 :]
        # ogni meta' viene centrata per opera in modo indipendente
        for j, half in enumerate((h1, h2)):
            Xi = X[half] - X[half].mean(axis=1, keepdims=True)
            v = Xi.mean(axis=0)
            if j == 0:
                v1 = v
            else:
                v2 = v
        out[i] = np.sum(v1 * v2, axis=1) / (
            np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-12
        )
    return out.mean(axis=0), out.std(axis=0)


# ------------------------------------------------------------ 4. test di permutazione


def permutation_test(Xc, observed, n_perm=2000):
    """Permuta le etichette di condizione ENTRO ogni opera: preserva l'effetto-opera."""
    A, C, D = Xc.shape
    null = np.zeros(n_perm)
    for i in range(n_perm):
        Xp = np.stack([Xc[a][RNG.permutation(C)] for a in range(A)])
        v = Xp.mean(axis=0)
        null[i] = np.linalg.norm(v, axis=1).mean()
    p = (np.sum(null >= observed) + 1) / (n_perm + 1)
    return p, null


# ------------------------------------------------------------ 5. ablazione lessicale


def top_words(texts_by_cond, k=40):
    """Parole piu' discriminative per categoria, via log-odds con prior informativo."""
    vocab = Counter()
    counts = {}
    for c, texts in texts_by_cond.items():
        w = Counter(re.findall(r"[a-z']{4,}", " ".join(texts).lower()))
        counts[c] = w
        vocab.update(w)
    total = sum(vocab.values())
    out = {}
    for c, w in counts.items():
        n = sum(w.values())
        scores = {}
        for word, cnt in w.items():
            if vocab[word] < 10:
                continue
            p_c = (cnt + 1) / (n + len(vocab))
            p_all = (vocab[word] + 1) / (total + len(vocab))
            scores[word] = np.log(p_c / p_all)
        out[c] = [x for x, _ in sorted(scores.items(), key=lambda t: -t[1])[:k]]
    return out


# ------------------------------------------------------------------------- figure


def fig_lda(Xc, arts, path):
    """LDA addestrata su meta' delle opere e proiettata sull'altra meta': senza
    questa separazione la figura sovrastima grossolanamente la separabilita'."""
    F, y, g = flatten(Xc)
    A = len(arts)
    p = RNG.permutation(A)
    tr_a, te_a = set(p[: A // 2]), set(p[A // 2 :])
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
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for ci, c in enumerate(CONDITIONS):
        m = y == ci
        ax.scatter(Z[m, 0], Z[m, 1], s=22, alpha=0.5, c=COLORS[c], label=LABELS[c],
                   edgecolors="none")
        mu = Z[m].mean(axis=0)
        ax.annotate("", xy=mu, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=COLORS[c], lw=2.2))
    ax.axhline(0, color="#ccc", lw=0.8, zorder=0)
    ax.axvline(0, color="#ccc", lw=0.8, zorder=0)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA sui dati centrati per opera\n"
                 "le frecce sono i vettori di steering v[c]", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_tsne(Xc, path):
    F, y, _ = flatten(Xc)
    Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(F)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for ci, c in enumerate(CONDITIONS):
        m = y == ci
        ax.scatter(Z[m, 0], Z[m, 1], s=22, alpha=0.6, c=COLORS[c], label=LABELS[c],
                   edgecolors="none")
    ax.set_title("t-SNE (figura supplementare)\n"
                 "non conserva direzioni ne' distanze fra cluster: non usarla per lo shift",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_cosine(M, path):
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    labs = [LABELS[c] for c in CONDITIONS]
    ax.set_xticks(range(len(labs)), labs, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(labs)), labs, fontsize=9)
    for i in range(len(labs)):
        for j in range(len(labs)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(M[i, j]) > 0.55 else "black")
    ax.set_title("Coseno fra i vettori di steering v[c]", fontsize=11)
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
    ap.add_argument("--no-ablation", action="store_true")
    args = ap.parse_args()

    FIGS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    tag = args.model if args.variant == "full" else f"{args.model}_{args.variant}"

    X, arts, meta = load(args.model, args.variant)
    A, C, D = X.shape
    print(f"\n{A} opere x {C} condizioni x {D} dim  [{args.model}]")

    Xc = X - X.mean(axis=1, keepdims=True)  # centering per opera
    V = Xc.mean(axis=0)                     # steering vectors

    out = {"model": args.model, "n_artworks": A, "n_conditions": C, "dim": D}

    # --- varianza: quanto pesa l'opera rispetto alla categoria ---
    var_total = X.reshape(A * C, D).var(axis=0).sum()
    var_art = X.mean(axis=1).var(axis=0).sum()
    var_cond = V.var(axis=0).sum()
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
    mu, sd = split_half(X)
    out["split_half_cosine"] = {
        LABELS[c]: {"mean": float(mu[i]), "std": float(sd[i])} for i, c in enumerate(CONDITIONS)
    }
    print("\n[2] consistenza split-half della direzione v[c]:")
    for i, c in enumerate(CONDITIONS):
        print(f"      {LABELS[c]:24} {mu[i]:.3f} +/- {sd[i]:.3f}")

    # --- 3. geometria ---
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    M = Vn @ Vn.T
    out["cosine_matrix"] = {
        LABELS[a]: {LABELS[b]: float(M[i, j]) for j, b in enumerate(CONDITIONS)}
        for i, a in enumerate(CONDITIONS)
    }
    out["steering_norms"] = {
        LABELS[c]: float(np.linalg.norm(V[i])) for i, c in enumerate(CONDITIONS)
    }
    print("\n[3] coppie piu' collineari (categorie che rischiano di collassare):")
    pairs = sorted(
        ((M[i, j], CONDITIONS[i], CONDITIONS[j]) for i in range(C) for j in range(i + 1, C)),
        reverse=True,
    )
    for v, a, b in pairs[:3]:
        print(f"      {LABELS[a]:24} ~ {LABELS[b]:24} {v:+.3f}")
    print("    piu' opposte:")
    for v, a, b in pairs[-2:]:
        print(f"      {LABELS[a]:24} ~ {LABELS[b]:24} {v:+.3f}")

    # --- 4. permutazione ---
    observed = np.linalg.norm(V, axis=1).mean()
    p, null = permutation_test(Xc, observed)
    out["permutation"] = {
        "observed_mean_norm": float(observed),
        "null_mean": float(null.mean()),
        "p_value": float(p),
    }
    print(f"\n[4] permutazione entro opera: ||v|| osservato {observed:.4f} vs "
          f"null {null.mean():.4f}  p = {p:.4g}")

    # --- 5. lunghezza (confondente) ---
    wl = meta.groupby("condition")["words"].agg(["mean", "std"])
    out["words"] = {LABELS[c]: float(wl.loc[c, "mean"]) for c in CONDITIONS if c in wl.index}
    print("\n[5] lunghezza media per condizione (confondente da sorvegliare):")
    for c in CONDITIONS:
        if c in wl.index:
            print(f"      {LABELS[c]:24} {wl.loc[c, 'mean']:.0f} +/- {wl.loc[c, 'std']:.0f}")

    # --- figure ---
    fig_lda(Xc, arts, FIGS / f"lda_{tag}.png")
    fig_pca(Xc, FIGS / f"pca_{tag}.png")
    fig_cosine(M, FIGS / f"cosine_{tag}.png")
    fig_confusion(cm, acc, FIGS / f"confusion_{tag}.png")
    fig_tsne(Xc, FIGS / f"tsne_{tag}.png")
    print(f"\nfigure -> {FIGS.relative_to(ROOT)}/*_{tag}.png")

    # --- parole discriminative ---
    gens = [json.loads(l) for l in (ROOT / "data" / "generations.jsonl").open()]
    gens = [g for g in gens if g.get("variant", "full") == args.variant
            or (g["condition"] == "flat" and g.get("variant", "full") == "full")]
    by_cond = {c: [g["text"] for g in gens if g["condition"] == c] for c in CONDITIONS}
    out["top_words"] = {LABELS[c]: w[:15] for c, w in top_words(by_cond).items()}
    print("\nparole piu' discriminative per categoria:")
    for c in CONDITIONS:
        print(f"      {LABELS[c]:24} {', '.join(out['top_words'][LABELS[c]][:8])}")

    (RESULTS / f"metrics_{tag}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nmetriche -> results/metrics_{tag}.json")


if __name__ == "__main__":
    main()
