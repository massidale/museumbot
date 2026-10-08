"""RQ2 su Gemma, testi puri e misti insieme: preferenza sul flat in funzione della quota
della propria categoria nel testo.

Per la persona k mescolata con la categoria j (coppie di `mixed_judge.PAIRS`), il tasso con
cui k preferisce un testo al flat, al variare del peso w di k nel testo:

    w = 1     testo puro di k:   RQ2 sui puri (tipo A)      e giudice sui misti al vertice
    w = 0.75, 0.5, 0.25          giudice sui misti (doppio visitatore)
    w = 0     testo puro di j:   RQ2 sui puri (tipo C, j)   e giudice sui misti al vertice

Ai vertici le due stime vengono da giudizi distinti dello stesso confronto: devono
coincidere. Scrive results/judge_combined_glm-5.3.json e figures/judge_combined_glm-5.3.png.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from museumbot.common.config import ROOT
from museumbot.common.prompts import LABELS
from museumbot.rq2_judge.judge import out_path
from museumbot.rq2_judge.mixed_judge import OUT as MIXED_OUT, PAIRS

RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
TAG = "glm-5.3"


def _rate(v) -> tuple[float, int]:
    return (float(np.mean(v)), len(v)) if v else (float("nan"), 0)


def weight_curve(pure_rows, mixed_rows, k: str, j: str) -> dict:
    """{"pure": {w: (tasso, n)}, "mixed": {w: (tasso, n)}} per la persona k, partner j."""
    pure = {1.0: [], 0.0: []}
    for r in pure_rows:
        if not r["valid"] or r["persona"] != k or r["other"][1] != "flat":
            continue
        if r["type"] == "A" and r["target"][1] == k:
            pure[1.0].append(bool(r["chose_target"]))
        elif r["type"] == "C" and r["target"][1] == j:
            pure[0.0].append(bool(r["chose_target"]))
    mixed = {}
    for r in mixed_rows:
        if r["type"] != "double" or not r["valid"] or r["persona"] != k or set(r["pair"]) != {k, j}:
            continue
        w = 1 - r["alpha"] if r["pair"][0] == k else r["alpha"]
        mixed.setdefault(w, []).append(bool(r["chose_target"]))
    return {"pure": {w: _rate(v) for w, v in pure.items()},
            "mixed": {w: _rate(v) for w, v in sorted(mixed.items())}}


def curves(pure_rows, mixed_rows) -> dict:
    out = {}
    for x, y in PAIRS:
        for k, j in ((x, y), (y, x)):
            c = weight_curve(pure_rows, mixed_rows, k, j)
            r3 = lambda d: {f"{w:g}": [round(t, 3), n] for w, (t, n) in d.items()}
            out[f"{k}|{j}"] = {"persona": k, "partner": j, "pure": r3(c["pure"]),
                               "mixed": r3(c["mixed"])}
    return out


def plot(cur) -> None:
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(4.4 * len(PAIRS), 4), sharey=True)
    for ax, (x, y) in zip(axes, PAIRS):
        for k, j, mk in ((x, y, "o"), (y, x, "s")):
            c = cur[f"{k}|{j}"]
            ws = sorted(float(w) for w in c["mixed"])
            ax.plot(ws, [c["mixed"][f"{w:g}"][0] for w in ws], mk + "-",
                    label=f"persona {LABELS[k]} (giudice sui misti)")
            pw = [w for w in (0.0, 1.0) if c["pure"][f"{w:g}"][1]]
            ax.scatter(pw, [c["pure"][f"{w:g}"][0] for w in pw], marker=mk, s=90,
                       facecolors="none", edgecolors="black", zorder=5,
                       label=f"persona {LABELS[k]} (RQ2 sui puri)")
        ax.axhline(0.5, color="grey", lw=0.5)
        ax.set_title(f"{LABELS[x]} – {LABELS[y]}", fontsize=10)
        ax.set_xlabel("quota della categoria della persona nel testo")
        ax.legend(fontsize=7, loc="center right")
    axes[0].set_ylabel("tasso di preferenza sul flat")
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / f"judge_combined_{TAG}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    pure_rows = [json.loads(l) for l in out_path("local").open()]
    mixed_rows = [json.loads(l) for l in MIXED_OUT.open()]
    cur = curves(pure_rows, mixed_rows)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"judge_combined_{TAG}.json").write_text(json.dumps(cur, indent=2))
    plot(cur)
    print(json.dumps(cur, indent=1))


if __name__ == "__main__":
    main()
