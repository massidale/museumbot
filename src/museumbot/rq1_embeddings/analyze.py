"""Analisi dello spazio latente: esiste uno shift direzionale per categoria di Falk?

Il passaggio critico e' il CENTERING PER OPERA. In uno spazio di embedding la varianza
dominante e' *quale opera* si sta descrivendo, non *per chi*: senza centering qualsiasi
PCA/t-SNE mostra 100 cluster-opera e zero struttura di categoria. Si usano due riferimenti:

    media:  e'[a,c] = e[a,c] - mean_over_c( e[a,.] )    tutte le condizioni trattate allo stesso modo
    flat:   e'[a,c] = e[a,c] - e[a,flat]                 effetto della categoria rispetto al neutro
    v[c]    = mean_over_a( e'[a,c] )                     steering vector della categoria c

Il riferimento flat da' agli steering vector il significato che interessa ("quanto e in che
direzione la categoria sposta il testo rispetto alla descrizione neutra") ed e' identico in
tutte le varianti dell'ablazione. Con la media, invece, i 6 v[c] sommano a zero per
costruzione e i coseni fra categorie sono spinti verso -1/(C-1): un artefatto, non
un'opposizione. La media resta per le procedure per singola opera basate sulle etichette
(probe, permutazione, varianza), dove sottrarre un solo testo flat aggiungerebbe il suo rumore.

Metriche, in ordine di forza probatoria:
  1. probe lineare cross-validata, raggruppata per opera  -> lo shift e' preciso?     [media]
  2. consistenza split-half della direzione v[c]           -> proprieta' della categoria? [flat]
  3. distanza cross-validata fra condizioni                -> quali categorie sono vicine?
     (indipendente dal riferimento), coseni e norme dei v[c] -> direzione degli effetti [flat]
  4. test di permutazione entro opera                      -> p-value              [media]
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
from museumbot.common.prompts import CONDITIONS, FALK_CATEGORIES, LABELS

FIGS = ROOT / "figures"
RESULTS = ROOT / "results"

RNG = np.random.default_rng(0)
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


def halves(A, n_rep):
    """n_rep coppie di meta' disgiunte e casuali degli indici di opera."""
    for _ in range(n_rep):
        p = RNG.permutation(A)
        yield p[: A // 2], p[A // 2 :]


def split_half(X, ref="flat", n_rep=200):
    """Coseno fra v[c] stimato su due meta' disgiunte di opere, mediato su n_rep split.

    ref="flat": 5 valori (nell'ordine di FALK_CATEGORIES), affidabilita' degli steering
    vector riportati. ref="mean": 6 valori (tutte le CONDITIONS), include la stabilita' del
    flat; e' piu' alta perche' la media di 6 testi e' un riferimento meno rumoroso di uno solo.
    """
    out = []
    for h1, h2 in halves(len(X), n_rep):
        # ogni meta' viene centrata per opera in modo indipendente
        if ref == "flat":
            v1, v2 = steering(X[h1]), steering(X[h2])
        else:
            v1, v2 = center(X[h1], ref).mean(axis=0), center(X[h2], ref).mean(axis=0)
        out.append(np.sum(v1 * v2, axis=1) / (
            np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-12
        ))
    out = np.array(out)
    return out.mean(axis=0), out.std(axis=0)


# ------------------------------------------------ 3. distanza cross-validata e geometria


def crossval_distance(X, n_rep=200):
    """Distanza euclidea quadrata cross-validata fra condizioni: (C, C) media e std.

        d2[a,b] = (v1[a] - v1[b]) . (v2[a] - v2[b])     v1, v2 da meta' disgiunte di opere

    E' la crossnobis senza normalizzazione per la covarianza del rumore. Imparziale: vale
    ~0 (anche leggermente negativa) se due condizioni non differiscono, mentre la distanza
    semplice e' sempre gonfiata dal rumore. Usa solo differenze fra condizioni, quindi non
    dipende dal riferimento del centering.
    """
    out = []
    for h1, h2 in halves(len(X), n_rep):
        # media per condizione senza centering: l'effetto-opera si cancella nelle differenze
        v1, v2 = X[h1].mean(axis=0), X[h2].mean(axis=0)
        d1 = v1[:, None] - v1[None]
        d2 = v2[:, None] - v2[None]
        out.append(np.sum(d1 * d2, axis=-1))
    out = np.array(out)
    return out.mean(axis=0), out.std(axis=0)


def cosine_matrix(V):
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return Vn @ Vn.T


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
    ap.add_argument("--no-ablation", action="store_true")
    args = ap.parse_args()

    FIGS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    tag = args.model if args.variant == "full" else f"{args.model}_{args.variant}"

    X, arts, meta = load(args.model, args.variant)
    A, C, D = X.shape
    print(f"\n{A} opere x {C} condizioni x {D} dim  [{args.model}]")

    Xc = center(X, "mean")  # per probe, permutazione, varianza e figure di proiezione
    Vm = Xc.mean(axis=0)    # (C, D) scostamenti dal baricentro dell'opera
    V = steering(X)         # (5, D) steering vector rispetto al flat

    out = {"model": args.model, "n_artworks": A, "n_conditions": C, "dim": D}

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
    mu, sd = split_half(X, "flat")
    out["split_half_cosine"] = {
        LABELS[c]: {"mean": float(mu[i]), "std": float(sd[i])} for i, c in enumerate(FALK_CATEGORIES)
    }
    mu_m, sd_m = split_half(X, "mean")
    out["split_half_cosine_mean_ref"] = {
        LABELS[c]: {"mean": float(mu_m[i]), "std": float(sd_m[i])} for i, c in enumerate(CONDITIONS)
    }
    print("\n[2] consistenza split-half della direzione v[c]   (rif. flat | rif. media):")
    sh = dict(zip(FALK_CATEGORIES, zip(mu, sd)))
    for i, c in enumerate(CONDITIONS):
        f = f"{sh[c][0]:.3f} +/- {sh[c][1]:.3f}" if c in sh else "-".center(14)
        print(f"      {LABELS[c]:24} {f}  |  {mu_m[i]:.3f} +/- {sd_m[i]:.3f}")

    # --- 3. distanza cross-validata e geometria ---
    d2, d2_sd = crossval_distance(X)
    out["crossval_distance"] = {
        LABELS[a]: {LABELS[b]: {"mean": float(d2[i, j]), "std": float(d2_sd[i, j])}
                    for j, b in enumerate(CONDITIONS)}
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
    out["steering_norms"] = {
        LABELS[c]: float(np.linalg.norm(V[i])) for i, c in enumerate(FALK_CATEGORIES)
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

    # --- 4. permutazione (sugli scostamenti dalla media: tutte le condizioni scambiabili) ---
    observed = np.linalg.norm(Vm, axis=1).mean()
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
    fig_distance(d2, FIGS / f"distance_{tag}.png")
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
