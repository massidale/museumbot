"""Analisi dei giudizi di RQ2 (criteri e misure fissati in docs/plans/2026-10-01-rq2-giudice-design.md).

Punteggio di una coppia: media dei due ordini di `chose_target` (1, 0.5 o 0), che
neutralizza il bias di posizione. Le coppie con un ordine non valido sono escluse.

    A      testo k vs flat, persona k           > 0.5: preferisce il personalizzato
    B      testo k vs testo j, persona k        > 0.5: preferenza specifica
    C      testo j vs flat, persona k           descrittiva
    N      testo k vs flat, senza persona       descrittiva

Analisi esplorative (decise dopo aver visto i dati, sezione `exploratory` del JSON):
matrice di preferenza persona x testo (vittoria sul flat: A in diagonale, C fuori),
confusioni in B, consistenza per tipo, posizione per persona, quante motivazioni citano la
propria categoria.
"""

import argparse
import json
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from museumbot.common.config import ROOT
from museumbot.common.prompts import FALK_CATEGORIES, LABELS
from museumbot.rq2_judge.judge import MODEL, OUT, out_path

RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
TYPES = ("A", "B", "C", "N")
THRESHOLDS = {"valid_rate": 0.98}


def tag(corpus: str = "main") -> str:
    """Suffisso di risultati e figure: nessuno per il corpus principale."""
    return MODEL if corpus == "main" else f"{MODEL}_{corpus}"


def load(path=OUT) -> list[dict]:
    """Giudizi delle coppie A, B, C, N; le righe `attention` delle prime run si ignorano."""
    return [r for r in map(json.loads, path.open()) if r["type"] != "attention"]


def category(p: dict) -> str:
    """Categoria k a cui si riferisce la coppia (per N e' il target, non la persona)."""
    return p["target"][1] if p["type"] == "N" else p["persona"]


def pair_scores(rows: list[dict]) -> dict[str, dict]:
    """Punteggio per coppia, solo se entrambi gli ordini sono validi.
    Con piu' righe per lo stesso (coppia, ordine) vale l'ultima valida."""
    by = defaultdict(dict)
    for r in rows:
        if r["valid"]:
            by[r["pair_id"]][r["order"]] = r
    out = {}
    for pid, orders in by.items():
        if len(orders) < 2:
            continue
        r = next(iter(orders.values()))
        ch = [orders[o]["chose_target"] for o in ("target_first", "target_second")]
        out[pid] = {k: r[k] for k in ("artwork_id", "type", "persona", "target", "other")} | {
            "score": float(np.mean(ch)), "consistent": ch[0] == ch[1]}
    return out


def criteria(rows: list[dict]) -> dict:
    valid = [r for r in rows if r["valid"]]
    scores = list(pair_scores(rows).values())
    return {
        "n_judgments": len(rows),
        "valid_rate": len(valid) / len(rows) if rows else float("nan"),
        "order_consistency": float(np.mean([s["consistent"] for s in scores])) if scores else float("nan"),
        "position_A_rate": float(np.mean([r["choice"] == "A" for r in valid])) if valid else float("nan"),
    }


def by_artwork(scores, typ, cat=None) -> dict[str, list[float]]:
    out = defaultdict(list)
    for s in scores.values():
        if s["type"] == typ and (cat is None or category(s) == cat):
            out[s["artwork_id"]].append(s["score"])
    return out


def rate(values) -> dict:
    """Media su tutte le coppie e numero di coppie."""
    n = sum(len(v) for v in values.values())
    if not n:
        return {"mean": None, "n": 0}
    return {"mean": float(sum(sum(v) for v in values.values()) / n), "n": n}


def rates(scores) -> dict:
    out = {}
    for t in TYPES:
        out[t] = {"overall": rate(by_artwork(scores, t))}
        for k in FALK_CATEGORIES:
            out[t][LABELS[k]] = rate(by_artwork(scores, t, k))
    return out


SELF_NAME = {"explorer": r"explorer", "facilitator": r"facilitator",
             "experience_seeker": r"experience seeker", "professional_hobbyist": r"professional|hobbyist",
             "recharger": r"recharger"}


def preference_matrix(scores) -> tuple[np.ndarray, np.ndarray]:
    """Persona k (righe) x categoria del testo (colonne): tasso di vittoria sul flat.
    Diagonale dalle coppie A, fuori diagonale dalle coppie C. Restituisce anche gli n."""
    acc = defaultdict(list)
    for s in scores.values():
        if s["type"] in ("A", "C"):
            acc[(s["persona"], s["target"][1])].append(s["score"])
    K = len(FALK_CATEGORIES)
    M, n = np.full((K, K), np.nan), np.zeros((K, K), int)
    for (k, t), v in acc.items():
        i, j = FALK_CATEGORIES.index(k), FALK_CATEGORIES.index(t)
        M[i, j], n[i, j] = np.mean(v), len(v)
    return M, n


def confusions_b(scores) -> dict:
    """Coppie B in cui il giudice k sceglie almeno una volta il testo di j."""
    acc = defaultdict(list)
    for s in scores.values():
        if s["type"] == "B":
            acc[(s["persona"], s["other"][1])].append(s["score"])
    return {f"{LABELS[k]} -> {LABELS[j]}": {"own_rate": float(np.mean(v)), "n": len(v)}
            for (k, j), v in sorted(acc.items()) if np.mean(v) < 1}


def exploratory(rows, scores) -> dict:
    M, _ = preference_matrix(scores)
    valid = [r for r in rows if r["valid"] and r["type"] in TYPES]
    lab = [LABELS[k] for k in FALK_CATEGORIES]
    off = [float(np.nanmean(np.delete(M[:, j], j))) for j in range(len(lab))]
    return {
        "preference_matrix": {lab[i]: {lab[j]: float(M[i, j]) for j in range(len(lab))}
                              for i in range(len(lab))},
        "accepted_by_others": dict(zip(lab, off)),
        "b_confusions": confusions_b(scores),
        "consistency_by_type": {t: float(np.mean([s["consistent"] for s in scores.values() if s["type"] == t]))
                                for t in TYPES},
        "position_A_by_persona": {p: float(np.mean([r["choice"] == "A" for r in valid if r["persona"] == p]))
                                  for p in sorted({r["persona"] for r in valid})},
        "reason_cites_own_category": {
            LABELS[k]: float(np.mean([bool(re.search(SELF_NAME[k], r["reason"] or "", re.I))
                                      for r in valid if r["persona"] == k]))
            for k in FALK_CATEGORIES},
    }


def fig_matrix(ex: dict, path) -> None:
    lab = [LABELS[k] for k in FALK_CATEGORIES]
    M = np.array([[ex["preference_matrix"][a][b] for b in lab] for a in lab])
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(lab)), lab, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(lab)), lab, fontsize=9)
    for i in range(len(lab)):
        for j in range(len(lab)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if M[i, j] > 0.55 else "black")
    ax.set_xlabel("categoria per cui e' scritto il testo")
    ax.set_ylabel("persona del giudice")
    ax.set_title("Vittoria sul flat (diagonale: coppie A; fuori: coppie C)", fontsize=11)
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_rates(res: dict, path) -> None:
    groups = [*FALK_CATEGORIES, "overall"]
    names = [LABELS[k] for k in FALK_CATEGORIES] + ["Totale"]
    colors = {"A": "#1F77B4", "B": "#E8743B", "C": "#9E9E9E", "N": "#19A979"}
    desc = {"A": "A: k vs flat", "B": "B: k vs j", "C": "C: j vs flat", "N": "N: k vs flat, senza persona"}
    x = np.arange(len(groups))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for i, t in enumerate(TYPES):
        vals = [res["rates"][t]["overall" if g == "overall" else LABELS[g]] for g in groups]
        m = np.array([v["mean"] for v in vals])
        ax.bar(x + (i - 1.5) * w, m, w, color=colors[t], label=desc[t])
    ax.axhline(0.5, color="#444", ls=":", lw=1)
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("tasso di vittoria dell'elemento di interesse")
    ax.set_title(f"RQ2 — giudice {MODEL} con persona", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--criteria-only", action="store_true",
                    help="solo i criteri di utilizzabilita' (dopo il pilota)")
    ap.add_argument("--corpus", default="main", help="corpus dei testi giudicati (main, local)")
    args = ap.parse_args()
    tg = tag(args.corpus)

    rows = load(out_path(args.corpus))
    crit = criteria(rows)
    crit["passed"] = {k: crit[k] >= v for k, v in THRESHOLDS.items()}
    print(f"[{MODEL}] {crit['n_judgments']} giudizi")
    for k in ("valid_rate", "order_consistency", "position_A_rate"):
        soglia = f"  (soglia {THRESHOLDS[k]:.0%}: {'ok' if crit['passed'][k] else 'NON SUPERATA'})" \
            if k in THRESHOLDS else ""
        print(f"  {k:20} {crit[k]:.3f}{soglia}")
    if args.criteria_only:
        return

    scores = pair_scores(rows)
    res = {"model": MODEL, "criteria": crit,
           "rates": rates(scores),
           "exploratory": exploratory(rows, scores)}

    print("\ntassi (media dei due ordini):")
    for t in TYPES:
        o = res["rates"][t]["overall"]
        print(f"  {t:4} {o['mean']:.3f}  n={o['n']}")
        print("       " + "  ".join(f"{LABELS[k]} {res['rates'][t][LABELS[k]]['mean']:.2f}"
                                    for k in FALK_CATEGORIES))

    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    out = RESULTS / f"judge_{tg}.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    ex = res["exploratory"]
    print("\nesplorative — accettazione dei testi altrui da parte delle altre persone:")
    print("  " + "  ".join(f"{k} {v:.2f}" for k, v in ex["accepted_by_others"].items()))
    fig_rates(res, FIGS / f"judge_{tg}.png")
    if args.corpus == "main":  # la matrice e' citata solo per il corpus principale
        fig_matrix(ex, FIGS / f"judge_{tg}_matrix.png")
    print(f"\n-> {out.relative_to(ROOT)}, figures/judge_{tg}*.png")


if __name__ == "__main__":
    main()
