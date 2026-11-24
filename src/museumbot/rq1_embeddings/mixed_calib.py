"""Taratura della posizione sul segmento: dove cade una miscela di testo di proporzione nota.

Per ogni opera e coppia X -> Y si incollano l'inizio del testo puro di X e la fine del testo
puro di Y (replica `full_rep` del corpus `local`, tagli a fine frase) in proporzione
(1 - alpha, alpha), alpha da 0.1 a 0.9, in tre composizioni degli stessi pezzi: `XY`, `YX`
(blocchi in un ordine o nell'altro) e `alt` (frasi alternate in modo uniforme). I testi si
collocano sul segmento fra v_X e v_Y come in mixed.py (steering vector da `full`, centratura
sul flat dell'opera). La curva per alpha e' il riferimento con cui leggere quella dei testi
misti: per ogni alpha dei misti, la quota equivalente e' la quota di Y dei testi incollati
che cade nella stessa posizione mediana. Scrive results/mixed_calib_<modello>.json.
"""

import argparse
import json
import re

import numpy as np
from scipy.stats import spearmanr

from museumbot.common.config import ROOT
from museumbot.common.corpus import load_corpus
from museumbot.common.prompts import CONDITIONS
from museumbot.generation.mix_diag import ABBREV
from museumbot.rq1_embeddings.analyze import FLAT, load, steering
from museumbot.rq1_embeddings.mixed import embed_mixed, position
from museumbot.rq2_judge.mixed_judge import PAIRS

EMB = ROOT / "data" / "emb" / "mixed_calib"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
ALPHAS = tuple(round(0.1 * i, 1) for i in range(1, 10))
ORDERS = ("XY", "YX", "alt")
# fine frase: punteggiatura (con eventuali virgolette o parentesi) seguita da spazio
END = re.compile(r"[.!?…][\"”’)\]]*(?=\s)")


def split_sentences(text: str) -> list[str]:
    """Frasi del testo; non si taglia dopo un'iniziale, un'abbreviazione o dentro "81.5"."""
    out, start = [], 0
    for m in END.finditer(text):
        piece = text[start : m.end()]
        if ABBREV.search(piece):
            continue
        out.append(piece.strip())
        start = m.end()
    out.append(text[start:].strip())
    return [s for s in out if s]


def _closest(cum: np.ndarray, target: float) -> int:
    """Numero di frasi (almeno 1, al massimo tutte meno una) con lunghezza piu' vicina."""
    k = np.arange(1, len(cum))
    return int(k[np.argmin(np.abs(cum[k - 1] - target))]) if len(k) else 1


def compose(sx: list[str], sy: list[str], alpha: float) -> tuple[list[str], list[str]]:
    """Prefisso di X con ~(1 - alpha) L e suffisso di Y con ~alpha L caratteri, L la media
    delle due lunghezze. Ogni pezzo ha almeno una frase."""
    L = (len(" ".join(sx)) + len(" ".join(sy))) / 2
    cx = np.cumsum([len(s) + 1 for s in sx])
    cy = np.cumsum([len(s) + 1 for s in reversed(sy)])
    kx, ky = _closest(cx, (1 - alpha) * L), _closest(cy, alpha * L)
    return sx[:kx], sy[len(sy) - ky :]


def interleave(px: list[str], py: list[str]) -> list[str]:
    """Frasi dei due pezzi alternate in modo uniforme: ogni frase al posto (i + 0.5)/n del
    suo pezzo, poi in ordine di posto (a pari posto prima X)."""
    slots = [((i + 0.5) / len(px), 0, s) for i, s in enumerate(px)]
    slots += [((j + 0.5) / len(py), 1, s) for j, s in enumerate(py)]
    return [s for *_, s in sorted(slots)]


def glue(px: list[str], py: list[str], order: str) -> tuple[str, float]:
    """Testo incollato nella composizione data e quota effettiva di Y in caratteri."""
    tx, ty = " ".join(px), " ".join(py)
    text = {"XY": f"{tx} {ty}", "YX": f"{ty} {tx}"}.get(order) or " ".join(interleave(px, py))
    return text, len(ty) / (len(tx) + len(ty))


def build_rows(texts: dict, arts: list[str]) -> list[dict]:
    """Righe dei testi incollati; `texts[(opera, categoria)]` e' il testo puro."""
    rows = []
    for x, y in PAIRS:
        for a in arts:
            sx, sy = split_sentences(texts[a, x]), split_sentences(texts[a, y])
            for al in ALPHAS:
                px, py = compose(sx, sy, al)
                for order in ORDERS:
                    text, eff = glue(px, py, order)
                    rows.append({"artwork_id": a, "pair": [x, y], "order": order,
                                 "condition": f"{x}+{y}:{order}", "alpha": al,
                                 "alpha_eff": eff, "text": text})
    return rows


def summarize(pos: np.ndarray, alpha: np.ndarray, eff: np.ndarray) -> dict:
    by = {}
    for al in sorted(set(alpha.tolist())):
        m = alpha == al
        by[f"{al:g}"] = {"n": int(m.sum()),
                         "position_median": round(float(np.median(pos[m])), 3),
                         "position_iqr": [round(float(q), 3) for q in np.percentile(pos[m], [25, 75])],
                         "alpha_eff_mean": round(float(eff[m].mean()), 3),
                         # margine dei tagli a fine frase: quota effettiva dal 5 al 95 percentile
                         "alpha_eff_p5_p95": [round(float(q), 3) for q in np.percentile(eff[m], [5, 95])]}
    return {"by_alpha": by,
            "spearman_alpha_eff_position": round(float(spearmanr(eff, pos).statistic), 3)}


def equivalent_share(p: float, by: dict) -> float | None:
    """Quota di Y dei testi incollati con posizione mediana `p`, per interpolazione sulla
    curva (posizione mediana, quota effettiva media); la curva si rende monotona con il
    massimo cumulato e nei tratti piatti vale la quota piu' bassa. None fuori
    dall'intervallo coperto."""
    pts = sorted((v["alpha_eff_mean"], v["position_median"]) for v in by.values())
    q = np.array([a for a, _ in pts])
    pos = np.maximum.accumulate([b for _, b in pts])
    keep = np.r_[True, np.diff(pos) > 0]
    q, pos = q[keep], pos[keep]
    if not pos[0] <= p <= pos[-1]:
        return None
    return round(float(np.interp(p, pos, q)), 3)


def with_vertices(by: dict, vert: dict) -> dict:
    """Curva della taratura con i testi puri interi come quota 0 e 1: con almeno una frase
    per pezzo la quota effettiva non arriva agli estremi."""
    ends = {k: {"alpha_eff_mean": float(k), "position_median": v} for k, v in vert.items()}
    return {**by, **ends}


def equivalent_shares(out: dict, mixed: dict) -> dict:
    """Quota equivalente per coppia, composizione e alpha interno dei misti."""
    res = {}
    for name, pair in mixed["pairs"].items():
        inner = {a: v["position_median"] for a, v in pair["by_alpha"].items() if a not in ("0", "1")}
        cal = out["pairs"][name]
        res[name] = {o: {a: equivalent_share(p, with_vertices(cal[o]["by_alpha"],
                                                              cal["vertices_full_rep"]))
                         for a, p in inner.items()} for o in ORDERS}
    return res


def run(alias: str) -> dict:
    Xp, arts, _ = load(alias, "full", "local")
    Xr, arts_r, _ = load(alias, "full_rep", "local")
    assert arts == arts_r
    texts = {(r["artwork_id"], r["condition"]): r["text"]
             for r in load_corpus("local", verbose=False) if r["variant"] == "full_rep"}
    rows = build_rows(texts, arts)
    E = embed_mixed(alias, rows, EMB)
    V = steering(Xp)
    ai = {a: i for i, a in enumerate(arts)}
    out = {"model": alias, "pairs": {}}
    for x, y in PAIRS:
        cx, cy = CONDITIONS.index(x), CONDITIONS.index(y)
        vx, vy = V[cx - (cx > FLAT)], V[cy - (cy > FLAT)]
        # vertici: testi puri della replica, interi
        vert = {f"{al:g}": round(float(np.median(position(Xr[:, c] - Xp[:, FLAT], vx, vy))), 3)
                for al, c in ((0.0, cx), (1.0, cy))}
        res = {"vertices_full_rep": vert}
        for order in ORDERS:
            sel = [i for i, r in enumerate(rows) if r["pair"] == [x, y] and r["order"] == order]
            a_idx = np.array([ai[rows[i]["artwork_id"]] for i in sel])
            pos = position(E[sel] - Xp[a_idx, FLAT], vx, vy)
            res[order] = summarize(pos, np.array([rows[i]["alpha"] for i in sel]),
                                   np.array([rows[i]["alpha_eff"] for i in sel]))
        out["pairs"][f"{x}+{y}"] = res
    mixed = RESULTS / f"mixed_{alias}.json"
    if mixed.exists():
        out["equivalent_share_mix"] = equivalent_shares(out, json.loads(mixed.read_text()))
    return out


STYLE = {"alt": ("#2a78d6", "-", "alternate"), "XY": ("#eb6834", "--", "X block, then Y"),
         "YX": ("#1baf7a", ":", "Y block, then X")}


def plot() -> None:
    """Curve della taratura (posizione mediana contro quota effettiva, vertici compresi) e
    misti alla loro alpha: una riga per modello di embedding, una colonna per coppia."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from museumbot.common.prompts import LABELS

    models = [m for m in ("qwen", "bge-m3")
              if (RESULTS / f"mixed_calib_{m}.json").exists() and (RESULTS / f"mixed_{m}.json").exists()]
    if not models:
        return
    fig, axes = plt.subplots(len(models), len(PAIRS), figsize=(3.6 * len(PAIRS), 3.3 * len(models)),
                             sharex=True, sharey=True, squeeze=False)
    for row, m in zip(axes, models):
        cal, mix = (json.loads((RESULTS / f"{n}_{m}.json").read_text()) for n in ("mixed_calib", "mixed"))
        for ax, (x, y) in zip(row, PAIRS):
            c = cal["pairs"][f"{x}+{y}"]
            for o in ("alt", "XY", "YX"):
                color, ls, label = STYLE[o]
                by = with_vertices(c[o]["by_alpha"], c["vertices_full_rep"])
                pts = sorted((v["alpha_eff_mean"], v["position_median"]) for v in by.values())
                ax.plot(*zip(*pts), ls=ls, color=color, lw=2 if o == "alt" else 1.4, label=label)
            # margine dei tagli a fine frase, sulla curva alt
            for v in c["alt"]["by_alpha"].values():
                lo, hi = v["alpha_eff_p5_p95"]
                ax.hlines(v["position_median"], lo, hi, color=STYLE["alt"][0], lw=1, alpha=0.5)
            bm = mix["pairs"][f"{x}+{y}"]["by_alpha"]
            al = sorted(bm, key=float)
            ax.plot([float(a) for a in al], [bm[a]["position_median"] for a in al], "o-",
                    color="#222222", ms=6, lw=1, label="mixed texts (at their α)")
            ax.plot([0, 1], [0, 1], ls=":", color="grey", lw=0.8)
            ax.set_title(f"{LABELS[x]} → {LABELS[y]}", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
        row[0].set_ylabel(f"position on the segment\n({'Qwen3' if m == 'qwen' else 'BGE-M3'})")
    for ax in axes[-1]:
        ax.set_xlabel("share of Y (glued: effective; mixed: α)")
    axes[0][0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "mixed_calib.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="qwen")
    args = ap.parse_args()
    out = run(args.model)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"mixed_calib_{args.model}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    plot()


if __name__ == "__main__":
    main()
