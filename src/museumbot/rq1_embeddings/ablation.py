"""Ablazione del prompt: quale parte del blocco di categoria produce lo shift latente?

Per ogni variante (vedi prompts.VARIANTS) si ricalcolano le metriche di analyze.py e si
aggiunge la metrica chiave: il coseno fra lo steering vector della variante e quello di
`full`, per categoria. Dice se la parte conserva la *direzione* dello shift, non soltanto
la separabilita'.

Gli steering vector sono presi rispetto al flat (analyze.steering). Il flat non ha varianti
ed e' lo stesso testo in tutte (embed.select_rows), quindi il riferimento e' identico fra
varianti: con il centering sulla media, invece, il riferimento include le 5 categorie della
variante e si sposterebbe con essa, contaminando il confronto con `full`.

Chiude un'analisi fattoriale 2x2x2 sulle tre parti (def, need, style): effetti principali
e interazioni a due vie, calcolati direttamente sulle medie delle 8 celle.

Corpus: `ablation` (common.corpus), cioe' il corpus di settembre a routing libero, pulito e
senza le opere con testi troncati. I numeri non si confrontano direttamente con quelli del
corpus principale, generato su provider fisso.

Calibrazione (`cos_rel`): il coseno con il full diviso per il tetto di rumore della
generazione; ~1 vuol dire indistinguibile dal full, non soltanto "vicino". Il tetto e'
cos(v_full, v_full_rep) del corpus principale (results/metrics_<model>.json, sezione
`ceiling`): due run dello stesso prompt sullo stesso provider e nella stessa sessione. Il
corpus di ablazione ha anche la deriva di provider, quindi il suo tetto vero e' un po' piu'
basso e il rapporto e' conservativo.

Deriva di routing: quanto il routing libero allontana due run dello stesso prompt, sui
singoli testi. D = mean(1 - cos(full, full_rep)) del corpus di ablazione diviso lo stesso
valore del corpus principale (provider fisso), sulle celle (opera, categoria) comuni; IC al
95% con bootstrap sulle opere. Del corpus principale si usano solo i propri coseni: i testi
dei due corpus non vengono mai confrontati fra loro.
"""

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from museumbot.rq1_embeddings.analyze import COLORS, center, load, probe, split_half, steering
from museumbot.common.config import ROOT, emb_path, meta_path
from museumbot.common.prompts import CONDITIONS, FALK_CATEGORIES, LABELS, PARTS, VARIANTS

RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

N_FALK = len(FALK_CATEGORIES)  # le prime 5 di CONDITIONS; flat e' l'ultima
assert CONDITIONS[N_FALK:] == ["flat"], "l'analisi assume flat come ultima condizione"


def cos_with_full(V: np.ndarray, V_full: np.ndarray) -> np.ndarray:
    num = np.sum(V * V_full, axis=1)
    den = np.linalg.norm(V, axis=1) * np.linalg.norm(V_full, axis=1) + 1e-12
    return num / den


def factorial_effects(table: dict[str, float]) -> dict:
    """Effetti principali e interazioni a due vie da una tabella variante -> scalare.

    main[p]        = media(celle con p) - media(celle senza p)
    inter[p×q]     = media(p,q) - media(p,¬q) - media(¬p,q) + media(¬p,¬q), ciascuna
                     mediata sulle 2 celle che restano libere sulla terza parte.
    """
    cells = {VARIANTS[v]: table[v] for v in VARIANTS}  # (def, need, style) -> valore
    idx = {p: i for i, p in enumerate(PARTS)}

    def mean_where(**fixed):
        vals = [val for flags, val in cells.items()
                if all(flags[idx[p]] == on for p, on in fixed.items())]
        return float(np.mean(vals))

    main = {p: mean_where(**{p: True}) - mean_where(**{p: False}) for p in PARTS}
    inter = {}
    for p, q in itertools.combinations(PARTS, 2):
        inter[f"{p}×{q}"] = (
            mean_where(**{p: True, q: True}) - mean_where(**{p: True, q: False})
            - mean_where(**{p: False, q: True}) + mean_where(**{p: False, q: False})
        )
    return {"main": main, "interaction": inter}


CORPUS = "ablation"


def analyse_variant(model: str, variant: str, V_full: np.ndarray | None) -> tuple[dict, np.ndarray]:
    X, _, meta = load(model, variant, CORPUS)
    Xc = center(X, "mean")
    V = steering(X)                                    # (5, D), rispetto al flat
    acc, cm, _, _ = probe(Xc[:, :N_FALK, :])          # 5 classi: flat escluso
    sh_mu, _ = split_half(X, "flat")
    norms = np.linalg.norm(V, axis=1)
    words = meta.groupby("condition")["words"].mean()
    out = {
        "variant": variant,
        "n_artworks": int(X.shape[0]),
        "probe5_accuracy": float(acc),
        "probe5_recall": {LABELS[c]: float(cm[i, i] / cm[i].sum()) for i, c in enumerate(FALK_CATEGORIES)},
        "norm": {LABELS[c]: float(norms[i]) for i, c in enumerate(FALK_CATEGORIES)},
        "norm_mean": float(norms.mean()),
        "split_half": {LABELS[c]: float(sh_mu[i]) for i, c in enumerate(FALK_CATEGORIES)},
        "words": {LABELS[c]: float(words[c]) for c in FALK_CATEGORIES if c in words.index},
    }
    if V_full is not None:
        cos = cos_with_full(V, V_full)
        out["cos_with_full"] = {LABELS[c]: float(cos[i]) for i, c in enumerate(FALK_CATEGORIES)}
        out["cos_with_full_mean"] = float(cos.mean())
    return out, V


def run(model: str) -> dict:
    variants = [v for v in VARIANTS if emb_path(model, v, CORPUS).exists()]
    missing = [v for v in VARIANTS if v not in variants]
    if missing:
        print(f"  varianti senza embedding, saltate: {missing}")
    if "full" not in variants:
        raise SystemExit("serve la variante full per il confronto")

    per_variant = {}
    full, V_full = analyse_variant(model, "full", None)
    full["cos_with_full"] = {LABELS[c]: 1.0 for c in FALK_CATEGORIES}
    full["cos_with_full_mean"] = 1.0
    per_variant["full"] = full
    for v in variants:
        if v == "full":
            continue
        per_variant[v], _ = analyse_variant(model, v, V_full)

    factorial = {}
    if len(per_variant) == len(VARIANTS):
        for metric in ("norm_mean", "cos_with_full_mean", "probe5_accuracy"):
            factorial[metric] = factorial_effects({v: per_variant[v][metric] for v in VARIANTS})

    out = {"model": model, "variants": per_variant, "factorial": factorial}
    ceiling = main_ceiling(model)
    if ceiling:
        out["ceiling"] = ceiling
        calibrate(per_variant, ceiling)
    if all(emb_path(model, v, c).exists() for v in ("full", "full_rep") for c in ("main", CORPUS)):
        out["routing_drift"] = drift_ratio(cell_cosines(model, CORPUS), cell_cosines(model, "main"))
    return out


def cell_cosines(model: str, corpus: str) -> dict[tuple[str, str], float]:
    """cos(full, full_rep) per cella (opera, categoria) di un corpus; il flat e' escluso."""
    def by_cell(variant):
        E = np.load(emb_path(model, variant, corpus))
        m = pd.read_csv(meta_path(variant, corpus))
        return {(a, c): E[i] for i, (a, c) in enumerate(zip(m["artwork_id"], m["condition"]))
                if c != "flat"}
    full, rep = by_cell("full"), by_cell("full_rep")
    return {k: float(full[k] @ rep[k]) for k in full if k in rep}


def drift_ratio(mixed: dict, fixed: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """D = mean(1 - cos routing libero) / mean(1 - cos provider fisso) sulle celle comuni."""
    by_art = defaultdict(lambda: [0.0, 0.0])
    keys = sorted(set(mixed) & set(fixed))
    for a, c in keys:
        by_art[a][0] += 1 - mixed[(a, c)]
        by_art[a][1] += 1 - fixed[(a, c)]
    num, den = np.array(list(by_art.values())).T
    idx = np.random.default_rng(seed).integers(0, len(num), size=(n_boot, len(num)))
    boot = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return {"D": float(num.sum() / den.sum()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "n_cells": len(keys)}


def main_ceiling(model: str) -> dict[str, float] | None:
    """Tetto su provider fisso: cos(v_full, v_full_rep) per categoria nel corpus principale."""
    path = RESULTS / f"metrics_{model}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("ceiling")


def calibrate(per_variant: dict, ceiling: dict[str, float], key: str = "cos_rel") -> None:
    """Aggiunge a ogni variante il coseno con il full diviso per il tetto (`key`, `key_mean`).
    Il full e' escluso: il suo coseno con se stesso vale 1 per costruzione."""
    for v, r in per_variant.items():
        if v == "full":
            continue
        rel = {c: r["cos_with_full"][c] / ceiling[c] for c in ceiling}
        r[key] = rel
        r[f"{key}_mean"] = float(np.mean(list(rel.values())))


def print_summary(res: dict) -> None:
    calibrated = "ceiling" in res
    head = f"\n{'variante':12} {'probe5':>7} {'||v|| media':>12} {'cos·full':>9} {'split-half':>11}"
    print(head + (f" {'cos/tetto':>10}" if calibrated else ""))
    for r in sorted(res["variants"].values(), key=lambda r: -r["cos_with_full_mean"]):
        sh = np.mean(list(r["split_half"].values()))
        line = (f"{r['variant']:12} {r['probe5_accuracy']:7.1%} {r['norm_mean']:12.4f} "
                f"{r['cos_with_full_mean']:9.3f} {sh:11.3f}")
        if calibrated and "cos_rel_mean" in r:
            line += f" {r['cos_rel_mean']:10.3f}"
        print(line)
    if calibrated:
        print("tetto (provider fisso): "
              + "  ".join(f"{c} {v:.3f}" for c, v in res["ceiling"].items()))
    if "routing_drift" in res:
        d = res["routing_drift"]
        print(f"deriva di routing (testi): D = {d['D']:.3f} IC95 [{d['ci95'][0]:.3f}, "
              f"{d['ci95'][1]:.3f}] su {d['n_cells']} celle")
    if res["factorial"]:
        print("\neffetti principali (con parte - senza parte):")
        for metric, eff in res["factorial"].items():
            m = "  ".join(f"{p}: {e:+.3f}" for p, e in eff["main"].items())
            print(f"  {metric:20} {m}")
        print("interazioni a due vie:")
        for metric, eff in res["factorial"].items():
            m = "  ".join(f"{k}: {e:+.3f}" for k, e in eff["interaction"].items())
            print(f"  {metric:20} {m}")


def fig_bars(res: dict, path: Path) -> None:
    """Due pannelli: (a) probe a 5 classi e norma media; (b) coseno con full per categoria."""
    vs = [v for v in VARIANTS if v in res["variants"]]
    R = res["variants"]
    x = np.arange(len(vs))
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    a.bar(x - 0.2, [R[v]["probe5_accuracy"] for v in vs], 0.4, color="#444", label="probe (5 classi)")
    a.axhline(1 / N_FALK, color="#444", ls=":", lw=1)
    a.set_ylabel("accuracy")
    a.set_ylim(0, 1.05)
    a2 = a.twinx()
    a2.bar(x + 0.2, [R[v]["norm_mean"] for v in vs], 0.4, color="#9E9E9E", label="||v|| media")
    a2.set_ylabel("norma media steering")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, frameon=False, loc="lower left", fontsize=9)
    # a2 (twinx) viene disegnata sopra a: si alza lo zorder di a per riportare la
    # legenda in primo piano, rendendo trasparente lo sfondo di a cosi' i colori non cambiano.
    a.set_zorder(a2.get_zorder() + 1)
    a.patch.set_visible(False)
    a.set_title("Ablazione del prompt — separabilita' e ampiezza dello shift", fontsize=11)

    w = 0.8 / N_FALK
    ceil = res.get("ceiling")
    for i, c in enumerate(FALK_CATEGORIES):
        b.bar(x + (i - (N_FALK - 1) / 2) * w, [R[v]["cos_with_full"][LABELS[c]] for v in vs],
              w, color=COLORS[c], label=LABELS[c])
    if ceil:
        b.axhline(np.mean(list(ceil.values())), color="#444", ls="--", lw=1,
                  label="tetto (provider fisso)")
    b.axhline(0, color="#ccc", lw=0.8)
    b.set_ylabel("cos(v_variante, v_full)")
    b.set_ylim(-0.2, 1.05)
    b.set_xticks(x, vs, rotation=30, ha="right")
    b.legend(frameon=False, fontsize=8, ncol=5, loc="lower left")
    b.set_title("Quanto ogni variante conserva la direzione del blocco completo", fontsize=11)
    for ax in (a, b):
        ax.spines[["top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_cosine_heatmap(res: dict, path: Path) -> None:
    """Heatmap variante x categoria del coseno con full, su scala sequenziale ristretta al range osservato."""
    rows = {v: res["variants"][v] for v in VARIANTS if v in res["variants"]}
    vs = list(rows)
    M = np.array([[rows[v]["cos_with_full"][LABELS[c]] for c in FALK_CATEGORIES] for v in vs])
    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    vmin = min(0.4, np.floor(M.min() * 10) / 10)
    im = ax.imshow(M, cmap="YlOrRd", vmin=vmin, vmax=1.0)
    ax.set_xticks(range(N_FALK), [LABELS[c] for c in FALK_CATEGORIES], rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(vs)), vs, fontsize=9)
    for i in range(len(vs)):
        for j in range(N_FALK):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.85 else "black")
    ax.set_title("Coseno fra v_variante[c] e v_full[c]", fontsize=11)
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)

    res = run(args.model)
    print_summary(res)
    out = RESULTS / f"ablation_{args.model}.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nrisultati -> {out.relative_to(ROOT)}")

    fig_bars(res, FIGS / f"ablation_{args.model}.png")
    fig_cosine_heatmap(res, FIGS / f"ablation_cosine_{args.model}.png")
    print(f"figure -> figures/ablation_{args.model}.png, figures/ablation_cosine_{args.model}.png")


if __name__ == "__main__":
    main()
