"""Ablazione del prompt: quale parte del blocco di categoria produce lo shift latente?

Per ogni variante (vedi prompts.VARIANTS) si ricalcolano le metriche di analyze.py e si
aggiunge la metrica chiave: il coseno fra lo steering vector della variante e quello di
`full`, per categoria. Dice se la parte conserva la *direzione* dello shift, non soltanto
la separabilita'.

Gli steering vector sono presi rispetto al flat (analyze.steering). Il flat non ha varianti
ed e' lo stesso testo in tutte (embed.select_rows), quindi il riferimento e' identico fra
varianti: con il centering sulla media, invece, il riferimento include le 5 categorie della
variante e si sposterebbe con essa, contaminando il confronto con `full`.

Corpus: `ablation` (common.corpus): tutte le varianti, la replica `full_rep` e il flat,
generati su un provider fisso (DeepInfra fp8, ragionamento attivo) nella stessa sessione,
puliti e senza le opere con testi troncati.

Calibrazione (`cos_rel`): il coseno con il full diviso per il tetto di rumore della
generazione; ~1 vuol dire indistinguibile dal full, non soltanto "vicino". Il tetto e'
cos(v_full, v_full_rep) dello stesso corpus: due run dello stesso prompt, stesso provider,
stessa sessione, stesso flat.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from museumbot.rq1_embeddings.analyze import (COLORS, center, load, probe, replicate_ceiling,
                                               split_half, steering)
from museumbot.common.config import ROOT, emb_path
from museumbot.common.prompts import CONDITIONS, FALK_CATEGORIES, LABELS, VARIANTS

RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

N_FALK = len(FALK_CATEGORIES)  # le prime 5 di CONDITIONS; flat e' l'ultima
assert CONDITIONS[N_FALK:] == ["flat"], "l'analisi assume flat come ultima condizione"


def cos_with_full(V: np.ndarray, V_full: np.ndarray) -> np.ndarray:
    num = np.sum(V * V_full, axis=1)
    den = np.linalg.norm(V, axis=1) * np.linalg.norm(V_full, axis=1) + 1e-12
    return num / den


CORPUS = "ablation"


def analyse_variant(model: str, variant: str, V_full: np.ndarray | None) -> tuple[dict, np.ndarray]:
    X, _, _ = load(model, variant, CORPUS)
    Xc = center(X, "mean")
    V = steering(X)                                    # (5, D), rispetto al flat
    acc, _, _, _ = probe(Xc[:, :N_FALK, :])          # 5 classi: flat escluso
    sh_mu = split_half(X)
    norms = np.linalg.norm(V, axis=1)
    out = {
        "variant": variant,
        "n_artworks": int(X.shape[0]),
        "probe5_accuracy": float(acc),
        "norm": {LABELS[c]: float(norms[i]) for i, c in enumerate(FALK_CATEGORIES)},
        "norm_mean": float(norms.mean()),
        "split_half": {LABELS[c]: float(sh_mu[i]) for i, c in enumerate(FALK_CATEGORIES)},
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

    out = {"model": model, "variants": per_variant}
    ceiling = own_ceiling(model)
    if ceiling:
        out["ceiling"] = ceiling
        calibrate(per_variant, ceiling)
    return out


def own_ceiling(model: str) -> dict[str, float] | None:
    """Tetto del corpus: cos(v_full, v_full_rep) per categoria, sulle opere comuni."""
    if not emb_path(model, "full_rep", CORPUS).exists():
        return None
    X, arts, _ = load(model, "full", CORPUS)
    Xr, arts_r, _ = load(model, "full_rep", CORPUS)
    ceil = replicate_ceiling(X, arts, Xr, arts_r)
    return {LABELS[c]: float(ceil[i]) for i, c in enumerate(FALK_CATEGORIES)}


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
        print("tetto (replica): "
              + "  ".join(f"{c} {v:.3f}" for c, v in res["ceiling"].items()))

def fig_bars(res: dict, path: Path) -> None:
    """Coseno con full per variante e categoria, con il tetto di rumore."""
    vs = [v for v in VARIANTS if v in res["variants"]]
    R = res["variants"]
    x = np.arange(len(vs))
    fig, b = plt.subplots(figsize=(10, 4.6))

    w = 0.8 / N_FALK
    ceil = res.get("ceiling")
    for i, c in enumerate(FALK_CATEGORIES):
        b.bar(x + (i - (N_FALK - 1) / 2) * w, [R[v]["cos_with_full"][LABELS[c]] for v in vs],
              w, color=COLORS[c], label=LABELS[c])
    if ceil:
        b.axhline(np.mean(list(ceil.values())), color="#444", ls="--", lw=1,
                  label="tetto (replica)")
    b.axhline(0, color="#ccc", lw=0.8)
    b.set_ylabel("cos(v_variante, v_full)")
    b.set_ylim(-0.2, 1.05)
    b.set_xticks(x, vs, rotation=30, ha="right")
    b.legend(frameon=False, fontsize=8, ncol=5, loc="lower left")
    b.set_title("Quanto ogni variante conserva la direzione del blocco completo", fontsize=11)
    b.spines[["top", "right"]].set_visible(False)
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

    if args.model == "qwen":  # la figura citata nel report
        fig_bars(res, FIGS / f"ablation_{args.model}.png")
        print(f"figura -> figures/ablation_{args.model}.png")


if __name__ == "__main__":
    main()
