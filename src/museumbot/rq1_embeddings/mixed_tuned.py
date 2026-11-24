"""Miscela classica a alpha tarata (0.4 e 0.6): i misti cadono piu' vicino alla quota voluta?

I testi di data/local/mixed_46.jsonl (alpha 0.4 e 0.6, 100 opere per coppia) si confrontano con
i misti a 0.25 e 0.75 di data/local/mixed.jsonl sulle stesse opere. Ogni testo si colloca sul
segmento come in mixed.py e si legge sulla curva `alt` della taratura (mixed_calib.py, vertici
compresi): obiettivo 0.25 per alpha 0.25 e 0.4, 0.75 per alpha 0.75 e 0.6. Misure:

- quota equivalente della posizione mediana, per coppia e alpha; vince l'alpha piu' vicina
  all'obiettivo, una quota fuori dalla curva perde;
- testi con quota equivalente entro 0.15 dall'obiettivo;
- testi con almeno un token improbabile per entrambi gli esperti (log p < log 0.01).

Scrive results/mixed_tuned_<modello>.json; richiede results/mixed_calib_<modello>.json.
"""

import argparse
import json
import math

import numpy as np

from museumbot.common.config import ROOT
from museumbot.common.prompts import CONDITIONS
from museumbot.rq1_embeddings.analyze import FLAT, load, steering
from museumbot.rq1_embeddings.mixed import EMB as EMB_MIX
from museumbot.rq1_embeddings.mixed import embed_mixed, load_rows, position
from museumbot.rq1_embeddings.mixed_calib import equivalent_share, with_vertices
from museumbot.rq2_judge.mixed_judge import PAIRS

TUNED = ROOT / "data" / "local" / "mixed_46.jsonl"
EMB = ROOT / "data" / "emb" / "mixed_46"
RESULTS = ROOT / "results"
TARGET = {0.25: 0.25, 0.4: 0.25, 0.5: 0.5, 0.6: 0.75, 0.75: 0.75}
# coppie del confronto: alpha tarata -> alpha dei misti con lo stesso obiettivo
VERSUS = {0.4: 0.25, 0.6: 0.75}
NEAR = 0.15
LOW = math.log(0.01)


def distance(share: float | None, target: float) -> float:
    """Distanza dall'obiettivo; fuori dalla curva e' piu' lontana di qualunque quota."""
    return math.inf if share is None else abs(share - target)


def unlikely_tokens(diag: list[dict], low: float = LOW) -> int:
    """Token con log p sotto `low` per tutti e due gli esperti."""
    return sum(all(lp < low for lp in d["logp"]) for d in diag)


def summarize(shares: list, pos: np.ndarray, broken: list[int], target: float, curve: dict) -> dict:
    p = float(np.median(pos))
    s = equivalent_share(p, curve)
    return {"n": len(shares), "target": target, "position_median": round(p, 3),
            "share_of_median": s,
            "near_target": sum(distance(x, target) <= NEAR for x in shares),
            "off_curve": sum(x is None for x in shares),
            "with_unlikely_token": sum(b > 0 for b in broken)}


def winners(by: dict) -> dict:
    """Per ogni alpha tarata: e' piu' vicina all'obiettivo dell'alpha dei misti?"""
    out = {}
    for tuned, base in VERSUS.items():
        t, b = by[f"{tuned:g}"], by[f"{base:g}"]
        out[f"{tuned:g}_vs_{base:g}"] = (distance(t["share_of_median"], t["target"])
                                         < distance(b["share_of_median"], b["target"]))
    return out


def run(alias: str) -> dict:
    mix = load_rows("mix")
    tuned = [r for r in map(json.loads, TUNED.open()) if not r["usage"]["failed"]]
    E = np.concatenate([embed_mixed(alias, mix, EMB_MIX), embed_mixed(alias, tuned, EMB)])
    rows = mix + tuned
    Xp, arts, _ = load(alias, "full", "local")
    V = steering(Xp)
    ai = {a: i for i, a in enumerate(arts)}
    cal = json.loads((RESULTS / f"mixed_calib_{alias}.json").read_text())["pairs"]
    out = {"model": alias, "near": NEAR, "low_logp": round(LOW, 3), "pairs": {}}
    for x, y in PAIRS:
        name = f"{x}+{y}"
        cx, cy = CONDITIONS.index(x), CONDITIONS.index(y)
        vx, vy = V[cx - (cx > FLAT)], V[cy - (cy > FLAT)]
        curve = with_vertices(cal[name]["alt"]["by_alpha"], cal[name]["vertices_full_rep"])
        by = {}
        for al, target in TARGET.items():
            sel = [i for i, r in enumerate(rows) if r["condition"] == name and r["alpha"] == al]
            a_idx = np.array([ai[rows[i]["artwork_id"]] for i in sel])
            pos = position(E[sel] - Xp[a_idx, FLAT], vx, vy)
            shares = [equivalent_share(float(p), curve) for p in pos]
            broken = [unlikely_tokens(rows[i]["usage"]["diag"]) for i in sel]
            by[f"{al:g}"] = summarize(shares, pos, broken, target, curve)
        out["pairs"][name] = {"by_alpha": by, "tuned_closer": winners(by)}
    out["tuned_closer_count"] = {k: sum(p["tuned_closer"][k] for p in out["pairs"].values())
                                 for k in next(iter(out["pairs"].values()))["tuned_closer"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="qwen")
    args = ap.parse_args()
    out = run(args.model)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"mixed_tuned_{args.model}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
