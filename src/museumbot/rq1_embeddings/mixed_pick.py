"""Opere per il pilota della miscela con ingresso della minoranza: vicine alla media dei misti.

Per coppia e alpha (0.25, 0.75) ogni testo misto si centra sul flat della sua opera,
d = e - e[flat]; distanza coseno di d dalla media dei d della coppia. Si fa la media dei
ranghi sui due alpha e sui due modelli di embedding; per coppia le N opere col rango medio
piu' basso. La prova a un'opera usa l'opera col rango medio piu' basso sulle tre coppie.
Scrive data/local/mixq_pilot.json. Spec: docs/plans/2026-11-04-ingresso-minoranza-design.md.
"""

import argparse
import json

import numpy as np
from scipy.stats import rankdata

from museumbot.common.config import ROOT
from museumbot.rq1_embeddings.analyze import FLAT, load
from museumbot.rq1_embeddings.mixed import EMB, embed_mixed, load_rows

OUT = ROOT / "data" / "local" / "mixq_pilot.json"
ALPHAS = (0.25, 0.75)
MODELS = ("qwen", "bge-m3")


def mean_distance_ranks(D: np.ndarray) -> np.ndarray:
    """Rango (1 = piu' vicina) della distanza coseno di ogni riga dalla media delle righe."""
    mu = D.mean(axis=0)
    cos = D @ mu / (np.linalg.norm(D, axis=1) * np.linalg.norm(mu))
    return rankdata(1 - cos)


def pick(ranks: dict[str, list[np.ndarray]], arts: list[str], n: int) -> dict:
    """ranks: coppia -> ranghi (uno per alpha e modello) sulle opere `arts`."""
    mean = {p: np.mean(r, axis=0) for p, r in ranks.items()}
    out = {"pairs": {p: [arts[i] for i in np.argsort(m, kind="stable")[:n]]
                     for p, m in mean.items()}}
    overall = np.mean(list(mean.values()), axis=0)
    out["smoke"] = arts[int(np.argsort(overall, kind="stable")[0])]
    return out


def collect() -> tuple[dict, list[str]]:
    rows = load_rows("mix")
    ranks, arts = {}, None
    for alias in MODELS:
        E = embed_mixed(alias, rows, EMB)
        Xp, art_ids, _ = load(alias, "full", "local")
        ai = {a: i for i, a in enumerate(art_ids)}
        for pair in sorted({r["condition"] for r in rows}):
            for al in ALPHAS:
                sel = [i for i, r in enumerate(rows) if r["condition"] == pair and r["alpha"] == al]
                ids = sorted(rows[i]["artwork_id"] for i in sel)
                if arts is None:
                    arts = ids
                assert ids == arts, "ogni coppia e alpha deve avere le stesse opere"
                sel.sort(key=lambda i: rows[i]["artwork_id"])
                D = E[sel] - Xp[[ai[a] for a in arts], FLAT]
                ranks.setdefault(pair, []).append(mean_distance_ranks(D))
    return ranks, arts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-n", type=int, default=10, help="opere per coppia")
    args = ap.parse_args()
    ranks, arts = collect()
    out = {"n": args.n, "alphas": list(ALPHAS), "models": list(MODELS), **pick(ranks, arts, args.n)}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
