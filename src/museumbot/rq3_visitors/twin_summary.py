"""Confronto fra gemello LLM (data/rq3/twin.jsonl) e visitatori, sulle stesse prove.

Tiene solo le prove con tutte e 4 le chiamate valide (2 modi x 2 ordini). Per ogni prova e
modo, voti del gemello = media dei due ordini. Descrive gemello con persona, gemello senza
persona e visitatori come summary.py (voti, diff, esiti, intervallo bootstrap sui visitatori),
poi il segno prova per prova contro quello del visitatore, l'effetto della persona e
quello della posizione. Temi dei commenti del gemello (data/rq3/temi_gemello.json, ordine del
visitatore) contati per commento, accanto a quelli dei visitatori (data/rq3/temi.json) contati
per visitatore. Scrive results/rq3_twin.json.
"""
import json
from collections import Counter, defaultdict

import numpy as np

from museumbot.common.config import ROOT
from museumbot.rq3_visitors.summary import by, cluster_ci, describe, load_trials, themes
from museumbot.rq3_visitors.twin import MODES, ORDERS, OUT as TWIN, RQ3

OUT = ROOT / "results" / "rq3_twin.json"


def valid_calls(rows: list[dict]) -> dict:
    """Ultima risposta valida per (sessione, opera, modo, ordine)."""
    return {(r["session_id"], r["artwork_id"], r["mode"], r["order"]): r
            for r in rows if r["valid"]}


def twin_trials(calls: dict, trials: list[dict]) -> dict[str, list[dict]]:
    """Per modo, prove del gemello (voti medi sui due ordini) nel formato di trials.csv;
    solo le prove complete, nell'ordine di `trials`."""
    out = {m: [] for m in MODES}
    for t in trials:
        k = (t["session_id"], t["artwork_id"])
        if not all(k + (m, o) in calls for m in MODES for o in ORDERS):
            continue
        for m in MODES:
            c = [calls[k + (m, o)] for o in ORDERS]
            p = float(np.mean([x["rating_personal"] for x in c]))
            f = float(np.mean([x["rating_comparison"] for x in c]))
            out[m].append(t | {"rating_personal": p, "rating_comparison": f, "diff": p - f,
                               "orders": [x["rating_personal"] - x["rating_comparison"] for x in c]})
    return out


def sign(x: float) -> str:
    return "personal" if x > 0 else "flat" if x < 0 else "pari"


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Kappa di Cohen fra due codifiche della stessa lista di prove."""
    n = len(pairs)
    if not n:
        return None
    a, b = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    po = sum(x == y for x, y in pairs) / n
    pe = sum(a[k] * b[k] for k in a) / n ** 2
    return round((po - pe) / (1 - pe), 2) if pe < 1 else None


def agreement(twin: list[dict], human: list[dict]) -> dict:
    """Segno della diff del gemello contro quello del visitatore, prova per prova, con il kappa
    di Cohen sui tre esiti e senza le prove pari per uno dei due."""
    seq = [(sign(h["diff"]), sign(t["diff"])) for t, h in zip(twin, human)]
    pairs = Counter(seq)
    same = sum(n for (a, b), n in pairs.items() if a == b)
    no_ties = [x for x in seq if "pari" not in x]
    return {"stesso_segno": same, "prove": len(twin), "kappa": kappa(seq),
            "senza_pari": {"prove": len(no_ties), "stesso_segno": sum(a == b for a, b in no_ties),
                           "kappa": kappa(no_ties)},
            "tabella": {f"visitatore_{a}": {f"gemello_{b}": pairs[(a, b)]
                                            for b in ("personal", "pari", "flat")}
                        for a in ("personal", "pari", "flat")}}


def order_effect(twin: list[dict]) -> dict:
    """Guida 1 meno Guida 2 nelle singole chiamate, e prove con lo stesso segno nei due ordini."""
    g12, groups, same = [], [], 0
    for t in twin:
        pg = int(t["personal_guide"])
        d_vis, d_rev = t["orders"]  # diff personal - museo, ordine del visitatore e inverso
        g12 += [d_vis if pg == 1 else -d_vis, -d_rev if pg == 1 else d_rev]
        groups += [t["session_id"]] * 2
        same += sign(d_vis) == sign(d_rev)
    ties = sum(d == 0 for t in twin for d in t["orders"])
    return {"guida1_meno_guida2": round(float(np.mean(g12)), 2), "ic95": cluster_ci(g12, groups),
            "stesso_segno_nei_due_ordini": same, "chiamate_pari": ties,
            "chiamate": 2 * len(twin)}


def persona_effect(twin: dict) -> dict:
    """diff con persona meno diff senza persona, per prova."""
    d = [p["diff"] - n["diff"] for p, n in zip(twin["persona"], twin["neutral"])]
    return {"diff": round(float(np.mean(d)), 2),
            "ic95": cluster_ci(d, [t["session_id"] for t in twin["persona"]])}


def twin_themes(statements: list[dict]) -> dict:
    """Per modo, temi del gemello contati per commento (una prova = un commento)."""
    return {m: {k: {x: v[x] for x in ("visitatori", "per_guida")} | {"commenti": v["visitatori"]}
                for k, v in themes([s | {"session_id": f"{s['session_id']}/{s['artwork_id']}"}
                                    for s in statements if s["mode"] == m]).items()}
            for m in MODES}


def group(r: dict) -> str:
    return "misti" if "+" in r["personal"] else "puri"


def summarize(calls: dict, trials: list[dict], human_st=(), twin_st=()) -> dict:
    twin = twin_trials(calls, trials)
    keys = {(t["session_id"], t["artwork_id"]) for t in twin["persona"]}
    human = [t for t in trials if (t["session_id"], t["artwork_id"]) in keys]
    sets = {"visitatori": human, **{f"gemello_{m}": twin[m] for m in MODES}}
    out = {"prove_complete": len(human), "prove_totali": len(trials)}
    for name, rows in sets.items():
        out[name] = {"tutto": describe(rows, ci=True),
                     "puri_misti": by(rows, group), "per_opera": by(rows, lambda r: r["artwork_id"])}
    for m in MODES:
        out[f"accordo_{m}"] = agreement(twin[m], human)
        out[f"ordine_{m}"] = order_effect(twin[m])
    out["persona_meno_neutral"] = persona_effect(twin)
    if twin_st:
        out["temi"] = {"visitatori": {k: {x: v[x] for x in ("visitatori", "per_guida")}
                                      for k, v in themes(list(human_st)).items()},
                       **{f"gemello_{m}": t for m, t in twin_themes(list(twin_st)).items()}}
    return out


def main():
    trials = load_trials(RQ3 / "clean" / "trials.csv")
    calls = valid_calls([json.loads(l) for l in TWIN.open()])
    human_st = json.loads((RQ3 / "temi.json").read_text())["affermazioni"]
    twin_st = json.loads((RQ3 / "temi_gemello.json").read_text())["affermazioni"]
    res = summarize(calls, trials, human_st, twin_st)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("prove_complete", "persona_meno_neutral")}))
    for name in ("visitatori", "gemello_persona", "gemello_neutral"):
        print(name, res[name]["tutto"])


if __name__ == "__main__":
    main()
