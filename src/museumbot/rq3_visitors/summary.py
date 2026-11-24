"""Analisi esplorativa di RQ3 sul dataset pulito: numeri, temi dei commenti, fonti.

Legge data/rq3/clean/trials.csv, la codifica tematica dei commenti (data/rq3/temi.json), la
codifica frase per frase delle voci Wikipedia e dei testi del museo
(data/wiki/haum_codifica.jsonl) e i testi generati (domande per condizione).
Scrive results/rq3_summary.json; il testo e' in docs/analisi/rq3-esplorativa.md.

diff = voto della personalizzata - voto del flat. Intervalli al 95% con bootstrap sui
visitatori (le opere di uno stesso visitatore restano insieme).
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from museumbot.common.config import ROOT

RQ3 = ROOT / "data" / "rq3"
WIKI = ROOT / "data" / "wiki"
OUT = ROOT / "results" / "rq3_summary.json"
CATS = ["explorer", "facilitator", "experience_seeker", "professional_hobbyist", "recharger"]
CODES = {"V": "si_vede", "S": "storia", "I": "interpretazione", "A": "artista_contesto",
         "P": "collocazione_fortuna"}
SOGLIA_MASSA = 4  # visitatori diversi perche' un tema sia "di massa"


def load_trials(path: Path) -> list[dict]:
    """Righe di trials.csv con i voti come numeri."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("rating_personal", "rating_comparison", "diff", "position", "personal_guide"):
            r[k] = int(r[k])
    return rows


def esiti(diffs) -> dict:
    """Vittorie della personalizzata, pari e vittorie del flat."""
    d = np.asarray(diffs)
    return {"personal": int((d > 0).sum()), "pari": int((d == 0).sum()), "flat": int((d < 0).sum())}


def cluster_ci(values, groups, reps=2000, seed=0) -> list[float]:
    """Intervallo al 95% della media con bootstrap sui gruppi (visitatori)."""
    v, g = np.asarray(values, float), np.asarray(groups)
    ids = np.unique(g)
    sums = np.array([v[g == i].sum() for i in ids])
    counts = np.array([(g == i).sum() for i in ids])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ids), size=(reps, len(ids)))
    means = sums[pick].sum(1) / counts[pick].sum(1)
    return [round(float(x), 2) for x in np.percentile(means, [2.5, 97.5])]


def describe(rows: list[dict], ci: bool = False) -> dict:
    """Media dei voti, diff, esiti e numerosita' di un gruppo di prove."""
    d = [r["diff"] for r in rows]
    out = {"prove": len(rows), "visitatori": len({r["session_id"] for r in rows}),
           "voto_personal": round(float(np.mean([r["rating_personal"] for r in rows])), 2),
           "voto_flat": round(float(np.mean([r["rating_comparison"] for r in rows])), 2),
           "diff": round(float(np.mean(d)), 2), "esiti": esiti(d)}
    if ci:
        out["ic95"] = cluster_ci(d, [r["session_id"] for r in rows])
    return out


def by(rows: list[dict], key) -> dict:
    """describe per valore di key (funzione della riga)."""
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    return {k: describe(v) for k, v in sorted(groups.items())}


def adjusted(rows: list[dict]) -> list[float]:
    """diff meno la diff media dell'opera."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["artwork_id"]].append(r["diff"])
    mean = {k: np.mean(v) for k, v in groups.items()}
    return [r["diff"] - mean[r["artwork_id"]] for r in rows]


def per_category(rows: list[dict]) -> dict:
    """Per categoria: diff sulle prove che la contengono, sui puri, sui misti e pesata
    (peso 1 alla pura, 0.5 alla mista)."""
    out = {}
    for c in CATS:
        mine = [r for r in rows if c in r["personal"].split("+")]
        if not mine:
            continue
        pure = [r["diff"] for r in mine if "+" not in r["personal"]]
        mixed = [r["diff"] for r in mine if "+" in r["personal"]]
        w = [1.0 if "+" not in r["personal"] else 0.5 for r in mine]
        out[c] = {"prove": len(mine), "visitatori": len({r["session_id"] for r in mine}),
                  "diff": round(float(np.mean([r["diff"] for r in mine])), 2),
                  "diff_pesata": round(float(np.average([r["diff"] for r in mine], weights=w)), 2),
                  "puri": {"prove": len(pure), "diff": round(float(np.mean(pure)), 2) if pure else None},
                  "misti": {"prove": len(mixed), "diff": round(float(np.mean(mixed)), 2) if mixed else None}}
    return out


def guide_order(rows: list[dict]) -> dict:
    """Voto della Guida 1 meno quello della Guida 2, e diff corretta per opera secondo la
    posizione della personalizzata."""
    g12 = [(r["rating_personal"] - r["rating_comparison"]) * (1 if r["personal_guide"] == 1 else -1)
           for r in rows]
    adj = adjusted(rows)
    pos = {g: round(float(np.mean([a for a, r in zip(adj, rows) if r["personal_guide"] == g])), 2)
           for g in (1, 2)}
    return {"guida1_meno_guida2": round(float(np.mean(g12)), 2),
            "ic95": cluster_ci(g12, [r["session_id"] for r in rows]),
            "esiti_guida1": esiti(g12), "diff_corretta_personal_in_guida": pos}


def first_artwork(rows: list[dict]) -> dict:
    """diff della prima opera del giro e delle altre, grezza e corretta per opera."""
    first = {}
    for r in rows:
        first[r["session_id"]] = min(first.get(r["session_id"], 99), r["position"])
    is_first = [r["position"] == first[r["session_id"]] for r in rows]
    adj = adjusted(rows)
    out = {}
    for name, flag in (("prima", True), ("altre", False)):
        sel = [i for i, f in enumerate(is_first) if f == flag]
        out[name] = {"prove": len(sel), "diff": round(float(np.mean([rows[i]["diff"] for i in sel])), 2),
                     "diff_corretta": round(float(np.mean([adj[i] for i in sel])), 2)}
    return out


def themes(statements: list[dict], soglia: int = SOGLIA_MASSA) -> dict:
    """Per tema: visitatori diversi che lo citano e, per guida e segno, quanti visitatori."""
    out = defaultdict(lambda: {"visitatori": set(), "per_guida": defaultdict(set),
                               "per_opera": defaultdict(set)})
    for s in statements:
        t = out[s["tema"]]
        t["visitatori"].add(s["session_id"])
        t["per_opera"][s.get("artwork_id") or "generale"].add(s["session_id"])
        if s.get("guida") and s.get("segno"):
            t["per_guida"][f"{s['guida']}{'+' if s['segno'] > 0 else '-'}"].add(s["session_id"])
    return {k: {"visitatori": len(v["visitatori"]), "di_massa": len(v["visitatori"]) >= soglia,
                "per_guida": {g: len(x) for g, x in sorted(v["per_guida"].items())},
                "per_opera": {o: len(x) for o, x in sorted(v["per_opera"].items())}}
            for k, v in sorted(out.items(), key=lambda kv: -len(kv[1]["visitatori"]))}


def themes_by_group(statements: list[dict], sessions: list[dict]) -> dict:
    """Per categoria scelta (e per puri / misti): visitatori, quanti hanno commentato e, per
    tema, quanti visitatori lo citano. Un visitatore con due categorie conta in tutte e due."""
    groups = {}
    for s in sessions:
        cats = s["categories"].split("+")
        groups[s["session_id"]] = cats + ["misti" if len(cats) > 1 else "puri"]
    out = defaultdict(lambda: {"visitatori": set(), "con_commenti": set(), "temi": defaultdict(set)})
    for sid, gs in groups.items():
        for g in gs:
            out[g]["visitatori"].add(sid)
    for st in statements:
        for g in groups.get(st["session_id"], []):
            out[g]["con_commenti"].add(st["session_id"])
            out[g]["temi"][st["tema"]].add(st["session_id"])
    return {g: {"visitatori": len(v["visitatori"]), "con_commenti": len(v["con_commenti"]),
                "temi": {t: len(x) for t, x in sorted(v["temi"].items(), key=lambda kv: -len(kv[1]))}}
            for g, v in sorted(out.items())}


def theme_votes(statements: list[dict], rows: list[dict]) -> dict:
    """Per tema, le preferenze di chi lo cita: diff ed esiti su tutte le sue opere, e per le
    affermazioni su un'opera con guida e segno se il voto va nello stesso verso del commento
    (personal+ o flat- chiedono diff > 0)."""
    trials = {(r["session_id"], r["artwork_id"]): r for r in rows}
    out = {}
    for tema in {s["tema"] for s in statements}:
        mine = [s for s in statements if s["tema"] == tema]
        who = {s["session_id"] for s in mine}
        d = [r["diff"] for r in rows if r["session_id"] in who]
        verso = {"stesso": 0, "pari": 0, "opposto": 0}
        for s in mine:
            r = trials.get((s["session_id"], s.get("artwork_id")))
            if r is None or not s.get("guida") or not s.get("segno"):
                continue
            want = s["segno"] * (1 if s["guida"] == "personal" else -1)
            got = np.sign(r["diff"]) * want
            verso["stesso" if got > 0 else "pari" if got == 0 else "opposto"] += 1
        out[tema] = {"prove": len(d), "diff": round(float(np.mean(d)), 2) if d else None,
                     "esiti": esiti(d), "voto_e_commento": verso}
    return out


def fit(sessions: list[dict]) -> dict:
    """Quanti visitatori danno almeno 5 (su 7) ad almeno una delle categorie scelte."""
    best = [max(int(x.split(":")[1]) for x in s["category_fit"].split()) for s in sessions]
    return {"sessioni": len(best), "almeno_5": sum(b >= 5 for b in best)}


def coding_shares(rows: list[dict]) -> dict:
    """Quota di parole per tipo di contenuto, per fonte e opera."""
    acc = defaultdict(lambda: defaultdict(float))
    for r in rows:
        for c, w in zip(r["codes"], r["words"]):
            acc[(r["source"], r["artwork_id"])][c] += w
    out = defaultdict(dict)
    for (src, art), w in acc.items():
        tot = sum(w.values())
        out[src][art] = {"parole": round(tot), "si_vede_parole": round(w.get("V", 0)),
                         **{CODES[c]: round(100 * w.get(c, 0) / tot) for c in CODES}}
    return dict(out)


def questions(gens: list[dict]) -> dict:
    """Domande medie per testo: categoria pura, misto che la contiene, flat (5 opere di RQ3)."""
    acc = defaultdict(list)
    for g in gens:
        cond = g["condition"]
        n = g["text"].count("?")
        if "+" in cond:
            for c in cond.split("+"):
                acc[f"{c} (misto)"].append(n)
        else:
            acc[cond].append(n)
    return {k: round(float(np.mean(v)), 1) for k, v in sorted(acc.items())}


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def summarize(rows, sessions, statements, coding, gens, artworks) -> dict:
    pure = [r for r in rows if "+" not in r["personal"]]
    mixed = [r for r in rows if "+" in r["personal"]]
    gens = [g for g in gens if g["artwork_id"] in artworks]
    return {"totale": describe(rows, ci=True), "categorie_scelte": fit(sessions),
            "puri": describe(pure, ci=True), "misti": describe(mixed, ci=True),
            "per_opera": by(rows, lambda r: r["artwork_id"]),
            "per_categoria": per_category(rows),
            "per_condizione": by(rows, lambda r: r["personal"]),
            "ordine_guide": guide_order(rows),
            "prima_opera": first_artwork(rows),
            "temi": {k: {**v, "preferenze": theme_votes(statements, rows)[k]}
                     for k, v in themes(statements).items()},
            "temi_per_gruppo": themes_by_group(statements, sessions),
            "soglia_di_massa": SOGLIA_MASSA,
            "fonti": coding_shares(coding),
            "domande_per_testo": questions(gens)}


def _artwork_id(title: str, ids: set[str]) -> str:
    slug = title.lower().replace(" ", "-")
    return next((i for i in ids if i == slug or slug.endswith(i) or i.endswith(slug)), slug)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", type=Path, default=RQ3 / "clean" / "trials.csv")
    ap.add_argument("--temi", type=Path, default=RQ3 / "temi.json")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    rows = load_trials(args.trials)
    artworks = {r["artwork_id"] for r in rows}
    gens = []
    for f in ("haum_generations.jsonl", "haum2_generations.jsonl"):
        for g in load_jsonl(WIKI / f):
            gens.append({**g, "artwork_id": _artwork_id(g["title"], artworks)})
    statements = json.loads(args.temi.read_text())["affermazioni"]
    with open(args.trials.parent / "sessions.csv", newline="") as f:
        sessions = list(csv.DictReader(f))
    res = summarize(rows, sessions, statements, load_jsonl(WIKI / "haum_codifica.jsonl"), gens, artworks)
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(f"prove {res['totale']['prove']}, visitatori {res['totale']['visitatori']}: scritto {args.out}")


if __name__ == "__main__":
    main()
