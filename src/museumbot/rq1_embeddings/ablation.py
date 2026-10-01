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

Calibrazione: ogni variante viene riportata anche come rapporto col tetto di rumore della
generazione; ~1 vuol dire indistinguibile dal full, non soltanto "vicino". Due tetti
racchiudono quello vero (due run del full con la miscela di provider di settembre, non
piu' riproducibile):
  `cos_rel`        tetto = cos(v_full, v_full_rep): routing libero, settembre vs ottobre.
                   Include anche la deriva di provider: tetto basso, rapporto indulgente.
  `cos_rel_strict` tetto = cos(v_single_a, v_single_b) dallo studio chain
                   (results/chain_noise_<model>.json): stesso provider e stessa sessione.
                   Solo rumore di generazione: tetto alto, rapporto severo.
"""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from museumbot.rq1_embeddings.analyze import COLORS, center, load, probe, split_half, steering
from museumbot.common.config import ROOT, emb_path
from museumbot.common.prompts import CONDITIONS, FALK_CATEGORIES, LABELS, PARTS, REPLICATES, VARIANTS

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


def analyse_variant(model: str, variant: str, V_full: np.ndarray | None) -> tuple[dict, np.ndarray]:
    X, _, meta = load(model, variant)
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
    variants = [v for v in VARIANTS if emb_path(model, v).exists()]
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

    replicates = {r: analyse_variant(model, r, V_full)[0]
                  for r in REPLICATES if emb_path(model, r).exists()}
    ceilings = {}
    if "full_rep" in replicates:
        ceilings["lenient"] = replicates["full_rep"]["cos_with_full"]
        calibrate(per_variant, ceilings["lenient"], "cos_rel")
    strict = strict_ceiling(model)
    if strict:
        ceilings["strict"] = strict
        calibrate(per_variant, strict, "cos_rel_strict")

    return {"model": model, "variants": per_variant, "replicates": replicates,
            "ceilings": ceilings, "factorial": factorial}


def strict_ceiling(model: str) -> dict[str, float] | None:
    """Tetto su provider fisso: cos(v_single_a, v_single_b) per categoria, dallo studio chain."""
    path = RESULTS / f"chain_noise_{model}.json"
    if not path.exists():
        return None
    st = json.loads(path.read_text())["steering"]
    return {LABELS[c]: st[LABELS[c]]["ceiling"] for c in FALK_CATEGORIES}


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
    calibrated = "full_rep" in res.get("replicates", {})
    strict = "strict" in res.get("ceilings", {})
    head = f"\n{'variante':12} {'probe5':>7} {'||v|| media':>12} {'cos·full':>9} {'split-half':>11}"
    print(head + (f" {'cos/tetto':>10}" if calibrated else "")
          + (f" {'cos/tetto severo':>17}" if strict else ""))
    rows = sorted(res["variants"].values(), key=lambda r: -r["cos_with_full_mean"])
    for r in [*rows, *res.get("replicates", {}).values()]:
        sh = np.mean(list(r["split_half"].values()))
        line = (f"{r['variant']:12} {r['probe5_accuracy']:7.1%} {r['norm_mean']:12.4f} "
                f"{r['cos_with_full_mean']:9.3f} {sh:11.3f}")
        if calibrated and "cos_rel_mean" in r:
            line += f" {r['cos_rel_mean']:10.3f}"
        if strict and "cos_rel_strict_mean" in r:
            line += f" {r['cos_rel_strict_mean']:17.3f}"
        print(line)
    if calibrated:
        ceil = res["replicates"]["full_rep"]["cos_with_full"]
        print("tetto (cos full_rep·full): " + "  ".join(f"{c} {v:.3f}" for c, v in ceil.items()))
    if strict:
        print("tetto severo (provider fisso): "
              + "  ".join(f"{c} {v:.3f}" for c, v in res["ceilings"]["strict"].items()))
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
    ceil = res.get("replicates", {}).get("full_rep", {}).get("cos_with_full")
    for i, c in enumerate(FALK_CATEGORIES):
        b.bar(x + (i - (N_FALK - 1) / 2) * w, [R[v]["cos_with_full"][LABELS[c]] for v in vs],
              w, color=COLORS[c], label=LABELS[c])
    if ceil:
        b.axhline(np.mean(list(ceil.values())), color="#444", ls="--", lw=1,
                  label="tetto (replica del full)")
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
    rows |= {f"{r} (tetto)": d for r, d in res.get("replicates", {}).items()}
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
    if len(vs) > len(VARIANTS):
        ax.axhline(len(VARIANTS) - 0.5, color="black", lw=1.5)
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
