"""Analisi del giudice sui testi misti (passo 4 del profilo continuo).

Per ogni coppia (X, Y), dai giudizi di data/judgments_mixed.jsonl:

  curve      tasso con cui la persona X preferisce S_x(alpha) a Y puro, e la persona Y
             S_y(alpha) a X puro
  double     tasso con cui ciascuna persona preferisce S(alpha) al flat, e il minimo fra le
             due persone (doppio visitatore)
  coherence  tasso con cui il testo misto e' scelto come "una voce sola" contro un puro

Controlli come in RQ2: giudizi validi, posizione A. Le righe di altri tipi nel file dei
giudizi (ordinamento, attenzione delle prime run) si ignorano. Scrive
results/judge_mixed_glm-5.3.json e figures/judge_mixed_glm-5.3.png.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from museumbot.common.config import ROOT
from museumbot.common.prompts import LABELS
from museumbot.rq2_judge.mixed_judge import OUT, PAIRS
from museumbot.rq2_judge.mixed_pairs import ALPHAS

RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
TAG = "glm-5.3"


def rates(rows, typ) -> dict:
    """(persona, alpha) -> tasso di scelta del target, sui giudizi validi."""
    acc = {}
    for r in rows:
        if r["type"] == typ and r["valid"]:
            acc.setdefault((r["persona"], r["alpha"]), []).append(bool(r["chose_target"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def summarize_pair(rows, pair) -> dict:
    x, y = pair
    rows = [r for r in rows if r.get("pair") == list(pair)]
    cur, dbl = rates(rows, "curve"), rates(rows, "double")
    rx = [cur[(x, a)] for a in ALPHAS]
    ry = [cur[(y, a)] for a in ALPHAS]
    dx = [dbl[(x, a)] for a in ALPHAS]
    dy = [dbl[(y, a)] for a in ALPHAS]
    coh = [bool(r["chose_target"]) for r in rows if r["type"] == "coherence" and r["valid"]]
    r3 = lambda v: [round(float(t), 3) for t in v]
    return {"curve": {"rate_x": r3(rx), "rate_y": r3(ry)},
            "double": {"rate_x": r3(dx), "rate_y": r3(dy), "min": r3(np.minimum(dx, dy))},
            "coherence": round(float(np.mean(coh)), 3)}


def checks(rows) -> dict:
    valid = [r for r in rows if r["valid"]]
    return {"n": len(rows), "valid_rate": round(float(np.mean([r["valid"] for r in rows])), 4),
            "position_A_rate": round(float(np.mean([r["choice"] == "A" for r in valid])), 3)}


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
    rows = [r for r in map(json.loads, OUT.open()) if r["type"] in ("curve", "double", "coherence")]
    out = {"model": TAG, "checks": checks(rows),
           "pairs": {"+".join(p): summarize_pair(rows, p) for p in PAIRS}}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"judge_mixed_{TAG}.json").write_text(json.dumps(out, indent=2))
    plot(out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
