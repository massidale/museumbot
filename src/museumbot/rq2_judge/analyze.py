"""Analisi dei giudizi di RQ2 (criteri e misure fissati in docs/plans/2026-10-01-rq2-giudice-design.md).

Punteggio di una coppia: media dei due ordini di `chose_target` (1, 0.5 o 0), che
neutralizza il bias di posizione. Le coppie con un ordine non valido sono escluse.

    A      testo k vs flat, persona k           > 0.5: preferisce il personalizzato
    B      testo k vs testo j, persona k        > 0.5: preferenza specifica
    C      testo j vs flat, persona k           descrittiva
    A - C  appaiata per (opera, k)              > 0: oltre l'effetto "testo elaborato"
    A - N  appaiata per (opera, k)              effetto della persona

IC al 95% con bootstrap sulle opere. Lunghezza: regressione logistica con IC bootstrap e
sensibilita' sulle coppie con differenza di lunghezza < 10%.

Analisi esplorative (decise dopo aver visto i dati, sezione `exploratory` del JSON):
matrice di preferenza persona x testo (vittoria sul flat: A in diagonale, C fuori),
confusioni in B, consistenza per tipo, posizione per persona, quante motivazioni citano la
propria categoria, e legame con RQ1: correlazione di Spearman fra l'accettazione dei testi
altrui (C) e distanza e coseno fra categorie negli embedding (results/metrics_*.json).
"""

import argparse
import json
import re
from collections import Counter, defaultdict

import matplotlib
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from museumbot.common.config import ROOT
from museumbot.common.prompts import FALK_CATEGORIES, LABELS
from museumbot.rq2_judge.judge import MODEL, OUT, corpus_index, out_path

RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
TYPES = ("A", "B", "C", "N")
THRESHOLDS = {"valid_rate": 0.98, "attention_accuracy": 0.95}
LENGTH_WORDS = re.compile(r"\b(long|longer|lengthy|short|shorter|concise|brief|succinct)\b", re.I)
RNG = np.random.default_rng(0)


def tag(corpus: str = "main") -> str:
    """Suffisso di risultati e figure: nessuno per il corpus principale."""
    return MODEL if corpus == "main" else f"{MODEL}_{corpus}"


def load(path=OUT) -> list[dict]:
    return [json.loads(l) for l in path.open()]


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
    att = [r for r in valid if r["type"] == "attention"]
    scores = [s for s in pair_scores(rows).values() if s["type"] != "attention"]
    return {
        "n_judgments": len(rows),
        "valid_rate": len(valid) / len(rows) if rows else float("nan"),
        "attention_accuracy": float(np.mean([r["chose_target"] for r in att])) if att else float("nan"),
        "order_consistency": float(np.mean([s["consistent"] for s in scores])) if scores else float("nan"),
        "position_A_rate": float(np.mean([r["choice"] == "A" for r in valid])) if valid else float("nan"),
    }


def bootstrap_mean(values: dict[str, list[float]], n: int = 2000) -> tuple[float, float, float]:
    """Media dei valori e IC al 95% ricampionando le opere."""
    arts = list(values)
    sums = np.array([sum(values[a]) for a in arts])
    counts = np.array([len(values[a]) for a in arts])
    point = sums.sum() / counts.sum()
    idx = RNG.integers(0, len(arts), size=(n, len(arts)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(point), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def by_artwork(scores, typ, cat=None) -> dict[str, list[float]]:
    out = defaultdict(list)
    for s in scores.values():
        if s["type"] == typ and (cat is None or category(s) == cat):
            out[s["artwork_id"]].append(s["score"])
    return out


def paired_diff(scores, t1, t2, cat=None) -> dict[str, list[float]]:
    """Differenze t1 - t2 appaiate per (opera, categoria k)."""
    key = lambda s: (s["artwork_id"], category(s))
    s1 = {key(s): s["score"] for s in scores.values() if s["type"] == t1}
    s2 = {key(s): s["score"] for s in scores.values() if s["type"] == t2}
    out = defaultdict(list)
    for (a, k), v in s1.items():
        if (a, k) in s2 and (cat is None or k == cat):
            out[a].append(v - s2[(a, k)])
    return dict(out)


def ci(values) -> dict:
    if not values:
        return {"mean": None, "ci95": None, "n": 0}
    m, lo, hi = bootstrap_mean(values)
    return {"mean": m, "ci95": [lo, hi], "n": sum(len(v) for v in values.values())}


def rates(scores) -> dict:
    out = {}
    for t in TYPES:
        out[t] = {"overall": ci(by_artwork(scores, t))}
        for k in FALK_CATEGORIES:
            out[t][LABELS[k]] = ci(by_artwork(scores, t, k))
    for t1, t2 in (("A", "C"), ("A", "N")):
        name = f"{t1}-{t2}"
        out[name] = {"overall": ci(paired_diff(scores, t1, t2))}
        for k in FALK_CATEGORIES:
            out[name][LABELS[k]] = ci(paired_diff(scores, t1, t2, k))
    return out


def length_analysis(rows, scores, words) -> dict:
    """Effetto della differenza di lunghezza sulla scelta, a parita' di tipo di coppia."""
    data = []
    for r in rows:
        if not r["valid"] or r["type"] not in TYPES:
            continue
        d = words[tuple(r["target"])] - words[tuple(r["other"])]
        data.append((r["artwork_id"], TYPES.index(r["type"]), d, float(r["chose_target"])))
    arts = sorted({a for a, *_ in data})
    sd = np.std([d for _, _, d, _ in data]) or 1.0

    def fit(sample):
        X = np.array([[t == i for i in range(4)] + [d / sd] for _, t, d, _ in sample], float)
        y = np.array([v for *_, v in sample])
        return LogisticRegression(C=np.inf, fit_intercept=False, max_iter=2000).fit(X, y).coef_[0][-1]

    by_art = defaultdict(list)
    for row in data:
        by_art[row[0]].append(row)
    point = fit(data)
    boot = [fit([row for a in RNG.choice(arts, len(arts)) for row in by_art[a]]) for _ in range(200)]
    # sensibilita': coppie con lunghezze entro il 10%
    close = {pid: s for pid, s in scores.items() if s["type"] in TYPES
             and abs(words[tuple(s["target"])] - words[tuple(s["other"])])
             < 0.1 * max(words[tuple(s["target"])], words[tuple(s["other"])])}
    return {
        "logit_per_sd_words": {"coef": float(point), "ci95": [float(np.percentile(boot, 2.5)),
                                                             float(np.percentile(boot, 97.5))],
                               "sd_words": float(sd)},
        "close_length_pairs": {t: ci(by_artwork(close, t)) for t in TYPES},
        "target_longer_rate": {t: float(np.mean([words[tuple(s["target"])] > words[tuple(s["other"])]
                                                 for s in scores.values() if s["type"] == t]))
                               for t in TYPES},
    }


def reasons_summary(rows) -> dict:
    valid = [r for r in rows if r["valid"] and r["type"] in TYPES]
    stop = set("the a an of and to in it is its this that for with as on by i my me more than "
               "while guide audio b both which would be are or but me's".split())
    out = {"mentions_length": {t: float(np.mean([bool(LENGTH_WORDS.search(r["reason"] or ""))
                                                for r in valid if r["type"] == t])) for t in TYPES}}
    top = {}
    for k in FALK_CATEGORIES:
        c = Counter(w for r in valid if r["type"] in "AB" and r["persona"] == k
                    for w in re.findall(r"[a-z']{4,}", (r["reason"] or "").lower()) if w not in stop)
        top[LABELS[k]] = [w for w, _ in c.most_common(12)]
    out["top_words"] = top
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


def rq1_link(M: np.ndarray, corpus: str = "main") -> dict:
    """Spearman fra l'accettazione dei testi altrui (C, fuori diagonale) e la geometria di RQ1."""
    cells = [(i, j) for i in range(len(FALK_CATEGORIES)) for j in range(len(FALK_CATEGORIES)) if i != j]
    c = [M[i, j] for i, j in cells]
    # versione simmetrica: media dei due versi per ciascuna delle 10 coppie
    pairs = [(i, j) for i, j in cells if i < j]
    c_sym = [(M[i, j] + M[j, i]) / 2 for i, j in pairs]
    out = {}
    for model in ("qwen", "bge-m3"):
        path = RESULTS / f"metrics_{model}{'' if corpus == 'main' else '_' + corpus}.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        lab = [LABELS[k] for k in FALK_CATEGORIES]
        dist = lambda i, j: r["crossval_distance"][lab[i]][lab[j]]["mean"]
        cos = lambda i, j: r["cosine_matrix"][lab[i]][lab[j]]
        res = {}
        for name, f in (("distance", dist), ("cosine", cos)):
            rho, p = spearmanr(c, [f(i, j) for i, j in cells])
            rho_s, p_s = spearmanr(c_sym, [f(i, j) for i, j in pairs])
            res[name] = {"rho_20": float(rho), "p_20": float(p), "rho_10": float(rho_s), "p_10": float(p_s)}
        out[model] = res
    return out


def exploratory(rows, scores, corpus: str = "main") -> dict:
    M, n = preference_matrix(scores)
    valid = [r for r in rows if r["valid"] and r["type"] in TYPES]
    lab = [LABELS[k] for k in FALK_CATEGORIES]
    off = [float(np.nanmean(np.delete(M[:, j], j))) for j in range(len(lab))]
    return {
        "preference_matrix": {lab[i]: {lab[j]: float(M[i, j]) for j in range(len(lab))}
                              for i in range(len(lab))},
        "preference_n": {lab[i]: {lab[j]: int(n[i, j]) for j in range(len(lab))} for i in range(len(lab))},
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
        "rq1_link": rq1_link(M, corpus),
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
        lo = m - np.array([v["ci95"][0] for v in vals])
        hi = np.array([v["ci95"][1] for v in vals]) - m
        ax.bar(x + (i - 1.5) * w, m, w, color=colors[t], label=desc[t], yerr=[lo, hi],
               capsize=2, error_kw={"lw": 0.8})
    ax.axhline(0.5, color="#444", ls=":", lw=1)
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("tasso di vittoria dell'elemento di interesse")
    ax.set_title(f"RQ2 — giudice {MODEL} con persona (IC 95% bootstrap sulle opere)", fontsize=11)
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
    t = tag(args.corpus)

    rows = load(out_path(args.corpus))
    crit = criteria(rows)
    crit["passed"] = {k: crit[k] >= v for k, v in THRESHOLDS.items()}
    print(f"[{MODEL}] {crit['n_judgments']} giudizi")
    for k in ("valid_rate", "attention_accuracy", "order_consistency", "position_A_rate"):
        soglia = f"  (soglia {THRESHOLDS[k]:.0%}: {'ok' if crit['passed'][k] else 'NON SUPERATA'})" \
            if k in THRESHOLDS else ""
        print(f"  {k:20} {crit[k]:.3f}{soglia}")
    if args.criteria_only:
        return

    texts, _ = corpus_index(args.corpus)
    words = {k: len(t.split()) for k, t in texts.items()}
    scores = pair_scores(rows)
    res = {"model": MODEL, "criteria": crit,
           "n_pairs": dict(Counter(s["type"] for s in scores.values())),
           "rates": rates(scores),
           "length": length_analysis(rows, scores, words),
           "reasons": reasons_summary(rows),
           "exploratory": exploratory(rows, scores, args.corpus)}

    print("\ntassi (media dei due ordini, IC 95%):")
    for t in (*TYPES, "A-C", "A-N"):
        o = res["rates"][t]["overall"]
        print(f"  {t:4} {o['mean']:.3f} [{o['ci95'][0]:.3f}, {o['ci95'][1]:.3f}]  n={o['n']}")
        print("       " + "  ".join(f"{LABELS[k]} {res['rates'][t][LABELS[k]]['mean']:.2f}"
                                    for k in FALK_CATEGORIES))
    lg = res["length"]["logit_per_sd_words"]
    print(f"\nlunghezza: logit per 1 sd di parole ({lg['sd_words']:.0f}) = {lg['coef']:+.2f} "
          f"[{lg['ci95'][0]:+.2f}, {lg['ci95'][1]:+.2f}]")

    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    out = RESULTS / f"judge_{t}.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    ex = res["exploratory"]
    print("\nesplorative — accettazione dei testi altrui da parte delle altre persone:")
    print("  " + "  ".join(f"{k} {v:.2f}" for k, v in ex["accepted_by_others"].items()))
    for model, r in ex["rq1_link"].items():
        print(f"  legame con RQ1 [{model}]: Spearman con la distanza {r['distance']['rho_20']:+.2f} "
              f"(10 coppie {r['distance']['rho_10']:+.2f}), con il coseno {r['cosine']['rho_20']:+.2f}")
    fig_rates(res, FIGS / f"judge_{t}.png")
    fig_matrix(ex, FIGS / f"judge_{t}_matrix.png")
    print(f"\n-> {out.relative_to(ROOT)}, figures/judge_{t}.png, figures/judge_{t}_matrix.png")


if __name__ == "__main__":
    main()
