"""Analisi del giudice sui testi misti (passo 4 del profilo continuo).

Per ogni coppia (X, Y), dai giudizi di data/judgments_mixed.jsonl:

  curve      tasso con cui la persona X preferisce S_x(alpha) a Y puro, e la persona Y
             S_y(alpha) a X puro; Spearman fra alpha e tasso (atteso <= -0.8 per X,
             >= 0.8 per Y) e alpha in cui le due curve si incrociano (atteso 0.4-0.6)
  double     tasso con cui ciascuna persona preferisce S(alpha) al flat; minimo fra le due
             persone per alpha e alpha del suo massimo (atteso interno, non 0 o 1)
  rank       Spearman fra l'ordine del giudice e alpha, per testo (mediana attesa >= 0.7)
  coherence  tasso con cui il testo misto e' scelto come "una voce sola" (atteso >= 0.4)

Controlli come in RQ2: giudizi validi, attenzione, posizione A. Scrive
results/judge_mixed_glm-5.3.json e figures/judge_mixed_glm-5.3.png.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from museumbot.common.config import ROOT
from museumbot.common.prompts import LABELS
from museumbot.rq2_judge.mixed_judge import OUT, PAIRS
from museumbot.rq2_judge.mixed_pairs import ALPHAS

RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
TAG = "glm-5.3"


def crossing(alphas, rx, ry) -> float | None:
    """Primo alpha in cui rx - ry passa da >= 0 a < 0, interpolato linearmente."""
    d = np.asarray(rx, float) - np.asarray(ry, float)
    for i in range(len(d) - 1):
        if d[i] >= 0 > d[i + 1]:
            return float(alphas[i] + (alphas[i + 1] - alphas[i]) * d[i] / (d[i] - d[i + 1]))
    return None


def rank_spearman(ranked_alphas) -> float:
    """Spearman fra la posizione assegnata dal giudice e alpha."""
    return float(spearmanr(np.arange(len(ranked_alphas)), ranked_alphas).statistic)


def rates(rows, typ) -> dict:
    """(persona, alpha) -> (tasso di scelta del target, n), sui giudizi validi."""
    acc = {}
    for r in rows:
        if r["type"] == typ and r["valid"]:
            acc.setdefault((r["persona"], r["alpha"]), []).append(bool(r["chose_target"]))
    return {k: (float(np.mean(v)), len(v)) for k, v in acc.items()}


def _spear(x, y) -> float | None:
    s = spearmanr(x, y).statistic
    return None if np.isnan(s) else round(float(s), 3)


def summarize_pair(rows, pair) -> dict:
    x, y = pair
    rows = [r for r in rows if r.get("pair") == list(pair)]
    cur, dbl = rates(rows, "curve"), rates(rows, "double")
    rx = [cur[(x, a)][0] for a in ALPHAS]
    ry = [cur[(y, a)][0] for a in ALPHAS]
    dx = [dbl[(x, a)][0] for a in ALPHAS]
    dy = [dbl[(y, a)][0] for a in ALPHAS]
    dmin = np.minimum(dx, dy)
    coh = [bool(r["chose_target"]) for r in rows if r["type"] == "coherence" and r["valid"]]
    coh_by = {f"{a:g}": round(float(np.mean([bool(r["chose_target"]) for r in rows
                                             if r["type"] == "coherence" and r["valid"]
                                             and r["alpha"] == a])), 3) for a in (0.25, 0.5, 0.75)}
    rk = [rank_spearman(r["ranked_alphas"]) for r in rows if r["type"] == "rank" and r["valid"]]
    cross = crossing(ALPHAS, rx, ry)
    r3 = lambda v: [round(float(t), 3) for t in v]
    return {
        "curve": {"rate_x": r3(rx), "rate_y": r3(ry), "n_per_point": cur[(x, ALPHAS[0])][1],
                  "spearman_x": _spear(ALPHAS, rx), "spearman_y": _spear(ALPHAS, ry),
                  "crossing": None if cross is None else round(cross, 3)},
        "double": {"rate_x": r3(dx), "rate_y": r3(dy), "min": r3(dmin),
                   "argmax_min": float(ALPHAS[int(np.argmax(dmin))])},
        "rank": {"n": len(rk), "spearman_median": round(float(np.median(rk)), 3),
                 "spearman_mean": round(float(np.mean(rk)), 3),
                 "share_perfect": round(float(np.mean(np.isclose(rk, 1.0))), 3)},
        "coherence": {"n": len(coh), "rate": round(float(np.mean(coh)), 3), "by_alpha": coh_by}}


def criteria(s) -> dict:
    c, d = s["curve"], s["double"]
    return {"curve_spearman_x": c["spearman_x"] is not None and c["spearman_x"] <= -0.8,
            "curve_spearman_y": c["spearman_y"] is not None and c["spearman_y"] >= 0.8,
            "curve_crossing": c["crossing"] is not None and 0.4 <= c["crossing"] <= 0.6,
            "double_interior_max": d["argmax_min"] not in (0.0, 1.0),
            "rank_median": s["rank"]["spearman_median"] >= 0.7,
            "coherence": s["coherence"]["rate"] >= 0.4}


def checks(rows) -> dict:
    ab = [r for r in rows if r["type"] != "rank"]
    att = [r for r in ab if r["type"] == "attention" and r["valid"]]
    valid_ab = [r for r in ab if r["valid"]]
    return {"n": len(rows), "valid_rate": round(float(np.mean([r["valid"] for r in rows])), 4),
            "attention_accuracy": round(float(np.mean([r["chose_target"] for r in att])), 3),
            "position_A_rate": round(float(np.mean([r["choice"] == "A" for r in valid_ab])), 3),
            "cost": round(float(sum(r.get("cost") or 0 for r in rows)), 3)}


def plot(out) -> None:
    fig, axes = plt.subplots(2, len(PAIRS), figsize=(4.2 * len(PAIRS), 7), sharey=True)
    for j, (x, y) in enumerate(PAIRS):
        s = out["pairs"][f"{x}+{y}"]
        ax = axes[0, j]
        ax.plot(ALPHAS, s["curve"]["rate_x"], "o-", label=f"persona {LABELS[x]} (vs {LABELS[y]} puro)")
        ax.plot(ALPHAS, s["curve"]["rate_y"], "s-", label=f"persona {LABELS[y]} (vs {LABELS[x]} puro)")
        ax.set_title(f"{LABELS[x]} → {LABELS[y]}: preferenza")
        ax = axes[1, j]
        ax.plot(ALPHAS, s["double"]["rate_x"], "o-", label=f"persona {LABELS[x]}")
        ax.plot(ALPHAS, s["double"]["rate_y"], "s-", label=f"persona {LABELS[y]}")
        ax.plot(ALPHAS, s["double"]["min"], "k--", label="minimo")
        ax.set_title("contro il flat (doppio visitatore)")
        for a in axes[:, j]:
            a.axhline(0.5, color="grey", lw=0.5)
            a.set_xlabel("alpha")
            a.legend(fontsize=7)
    axes[0, 0].set_ylabel("tasso di preferenza del testo-alpha")
    axes[1, 0].set_ylabel("tasso di preferenza sul flat")
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / f"judge_mixed_{TAG}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    rows = [json.loads(l) for l in OUT.open()]
    out = {"model": TAG, "checks": checks(rows), "pairs": {}}
    for p in PAIRS:
        s = summarize_pair(rows, p)
        s["criteria"] = criteria(s)
        out["pairs"]["+".join(p)] = s
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"judge_mixed_{TAG}.json").write_text(json.dumps(out, indent=2))
    plot(out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
