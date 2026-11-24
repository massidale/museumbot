"""Dataset pulito di RQ3: una riga per opera valida e una per sessione, con i commenti scritti
e le trascrizioni delle registrazioni, se ci sono.

Legge tutti gli export online (data/rq3/raw/online_*.json, uniti con collect.merge), le
esclusioni scritte a mano (data/rq3/exclusions.json), le trascrizioni
(data/rq3/transcripts.jsonl), i file registrati fuori dall'app (raw/voice/extra.json) e le
annotazioni scritte dall'operatore (data/rq3/operator.json).
Scrive in data/rq3/clean/ trials.csv, sessions.csv ed excluded.csv (cosa e' stato tolto e
perche').

Regole: fuori le sessioni in exclusions.json (prove); fuori le opere saltate col tasto
dell'app, quelle senza i due voti e quelle con una guida a 0 s di ascolto (saltata prima che ci
fosse il tasto, o una guida votata senza ascoltarla); fuori le sessioni senza opere valide.

Registrazioni: <session_id>_<n> e' il commento sull'opera n del giro (app 0.6.0), <session_id>
le risposte finali. extra.json puo' spostare un file su un'altra sessione (dispositivi
scambiati) o su un'opera (artwork_id); senza artwork_id va con le risposte finali (registrazioni
fatte fuori dall'app, note dell'operatore). I file in exclusions.json "voice" sono inutilizzabili:
restano solo come nota. Una sessione senza opere valide ma con un commento finale resta in
sessions.csv (trials_valid 0).

Operatore: operator.json ha "notes" (pareri riferiti dai visitatori che non volevano essere
registrati, scritti in italiano: vanno in oral_text_it o final_text_it) e "ratings" (voti di
un'opera saltata per sbaglio, riportati dall'operatore). Queste note e i file di extra.json con
"operator": true portano la nota OPERATOR.
"""
import argparse
import csv
import json
from pathlib import Path

from museumbot.common.config import ROOT
from museumbot.rq3_visitors.collect import load_file, merge, validate

RQ3 = ROOT / "data" / "rq3"
RAW = RQ3 / "raw"
OUT = RQ3 / "clean"

TRIAL_COLS = ["session_id", "day", "device", "lang", "categories", "n_categories", "position",
              "artwork_id", "personal", "comparison", "personal_guide", "rating_personal",
              "rating_comparison", "diff", "heard_personal", "heard_comparison",
              "plays_personal", "plays_comparison", "comment", "oral_file", "oral_text",
              "oral_text_it", "oral_note", "note"]
SESSION_COLS = ["session_id", "day", "started_at", "device", "lang", "categories",
                "category_fit", "trials_valid", "trials_total", "completed", "voice",
                "final_file", "final_text", "final_text_it", "final_note"]
EXCLUDED_COLS = ["session_id", "artwork_id", "reason"]
OPERATOR = "annotazione dell'operatore, tradotta dall'inglese: da non citare letteralmente"


def trial_problem(t: dict) -> str | None:
    """Perche' l'opera non entra nel dataset, o None se e' valida."""
    if t.get("skipped"):
        return "opera saltata col tasto"
    r = t.get("ratings") or [None, None]
    if r[0] is None or r[1] is None:
        return "senza voti"
    h = [(x or 0) for x in (t.get("heard_sec") or [0, 0])]
    if h[0] <= 0 and h[1] <= 0:
        return "opera saltata (0 s di ascolto)"
    if h[0] <= 0 or h[1] <= 0:
        return "una guida non ascoltata (0 s)"
    return None


def _cats(rec: dict) -> str:
    return "+".join(rec.get("categories") or [])


def trial_row(rec: dict, t: dict, oral: dict | None = None) -> dict:
    """Riga di un'opera: voti e ascolto riportati a personalizzata / confronto."""
    p = 0 if t["first"] == "personal" else 1
    c = 1 - p
    oral = oral or {}
    return {
        "session_id": rec["session_id"], "day": rec["started_at"][:10],
        "device": rec["device"], "lang": rec["lang"], "categories": _cats(rec),
        "n_categories": len(rec.get("categories") or []), "position": t["index"] + 1,
        "artwork_id": t["artwork_id"], "personal": t["personal"], "comparison": t["comparison"],
        "personal_guide": p + 1, "rating_personal": t["ratings"][p],
        "rating_comparison": t["ratings"][c], "diff": t["ratings"][p] - t["ratings"][c],
        "heard_personal": t["heard_sec"][p], "heard_comparison": t["heard_sec"][c],
        "plays_personal": t["plays"][p], "plays_comparison": t["plays"][c],
        "comment": (t.get("why") or "").strip(),
        "oral_file": oral.get("file", ""), "oral_text": oral.get("text", ""),
        "oral_text_it": oral.get("text_it", ""), "oral_note": oral.get("note", ""),
        "note": t.get("note", ""),
    }


def voice_status(rec: dict) -> str:
    """Com'e' andata la registrazione finale nell'app (le altre sono in final_note)."""
    v = rec.get("voice") or {}
    if isinstance(v.get("sec"), (int, float)):
        return "registrata nell'app"
    if v.get("no_mic"):
        return "microfono non disponibile"
    if v.get("skipped"):
        return "saltata"
    return "assente"


def session_row(rec: dict, n_valid: int, final: dict | None) -> dict:
    final = final or {}
    fit = rec.get("category_fit") or {}
    return {
        "session_id": rec["session_id"], "day": rec["started_at"][:10],
        "started_at": rec["started_at"], "device": rec["device"], "lang": rec["lang"],
        "categories": _cats(rec), "category_fit": " ".join(f"{k}:{v}" for k, v in fit.items()),
        "trials_valid": n_valid, "trials_total": len(rec.get("trials") or []),
        "completed": bool(rec.get("completed")), "voice": voice_status(rec),
        "final_file": final.get("file", ""), "final_text": final.get("text", ""),
        "final_text_it": final.get("text_it", ""), "final_note": final.get("note", ""),
    }


def slot_of(file: str, extra: dict[str, dict]) -> tuple[str, str | int | None]:
    """(session_id, posto) di una registrazione: posto = n (opera n del giro) o artwork_id per un
    commento su un'opera, None per le risposte finali."""
    e = extra.get(file)
    if e:
        return e["session_id"], e.get("artwork_id")
    sid, _, suf = Path(file).stem.partition("_")
    return sid, (int(suf) if suf.isdigit() else None)


def combine(parts: list[dict]) -> dict:
    """Piu' registrazioni nello stesso posto: file con '; ', testi separati da una riga vuota."""
    join = lambda k, sep: sep.join(p[k] for p in parts if p.get(k))
    return {"file": join("file", "; "), "text": join("text", "\n\n"),
            "text_it": join("text_it", "\n\n"), "note": join("note", "; ")}


def _notes(*xs: str) -> str:
    return "; ".join(x for x in xs if x)


def gather(transcripts: dict[str, dict], extra: dict[str, dict], unusable: dict[str, str],
           notes: list[dict] = ()) -> dict[tuple, list[dict]]:
    """Registrazioni e note scritte dell'operatore per posto. Un file inutilizzabile porta solo
    la nota."""
    slots: dict[tuple, list[dict]] = {}
    for f in sorted(set(transcripts) | set(unusable)):
        if f in unusable:
            part = {"note": f"{f} inutilizzabile: {unusable[f]}"}
        else:
            r, e = transcripts[f], extra.get(f, {})
            note = _notes(OPERATOR if e.get("operator") else "", e.get("source", ""), r.get("note", ""))
            part = {"file": f, "text": r.get("text", ""), "text_it": r.get("text_it", ""), "note": note}
        slots.setdefault(slot_of(f, extra), []).append(part)
    for n in notes:
        part = {"text_it": n["text_it"], "note": _notes(OPERATOR, n.get("note", ""))}
        slots.setdefault((n["session_id"], n.get("artwork_id")), []).append(part)
    return slots


def fix_ratings(t: dict, fix: dict | None) -> dict:
    """Opera con i voti riportati dall'operatore (saltata per sbaglio col tasto)."""
    if not fix:
        return t
    return {**t, "ratings": list(fix["ratings"]), "skipped": False,
            "note": _notes("voti riportati dall'operatore", fix.get("note", ""))}


def build(records: list[dict], exclusions: dict[str, str], transcripts: dict[str, dict],
          extra: dict[str, dict], unusable: dict[str, str] | None = None,
          operator: dict | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    """(opere, sessioni, escluse). transcripts, extra e unusable sono indicizzati per nome del
    file; operator e' il contenuto di operator.json."""
    operator = operator or {}
    slots = gather(transcripts, extra, unusable or {}, operator.get("notes", []))
    fixes = {(f["session_id"], f["artwork_id"]): f for f in operator.get("ratings", [])}
    trials, sessions, excluded = [], [], []
    for rec in records:
        sid = rec["session_id"]
        if sid in exclusions:
            excluded.append({"session_id": sid, "artwork_id": "", "reason": exclusions[sid]})
            continue
        valid = []
        for t in rec.get("trials") or []:
            t = fix_ratings(t, fixes.get((sid, t["artwork_id"])))
            why = trial_problem(t)
            if why:
                excluded.append({"session_id": sid, "artwork_id": t["artwork_id"], "reason": why})
            else:
                said = slots.get((sid, t["index"] + 1), []) + slots.get((sid, t["artwork_id"]), [])
                valid.append(trial_row(rec, t, combine(said) if said else None))
        final = combine(slots[(sid, None)]) if (sid, None) in slots else None
        if not valid:
            if final and (final["text"] or final["text_it"]):
                excluded.append({"session_id": sid, "artwork_id": "",
                                 "reason": "nessuna opera valida (resta il commento finale)"})
                sessions.append(session_row(rec, 0, final))
            else:
                excluded.append({"session_id": sid, "artwork_id": "", "reason": "nessuna opera valida"})
            continue
        sessions.append(session_row(rec, len(valid), final))
        trials += valid
    return trials, sessions, excluded


def load_records(files: list[Path]) -> list[dict]:
    """Sessioni valide di tutti gli export, una per session_id."""
    out = []
    for f in files:
        righe, _ = load_file(f)
        out += [r for r in righe if not validate(r)]
    return merge(out)


def load_jsonl(path: Path) -> dict[str, dict]:
    """Trascrizioni per nome del file."""
    if not Path(path).exists():
        return {}
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["file"]: r for r in rows}


def load_json(path: Path, default):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default


def write_csv(path: Path, rows: list[dict], cols: list[str]):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exports", nargs="*", type=Path, help="default: raw/online_*.json")
    ap.add_argument("--exclusions", type=Path, default=RQ3 / "exclusions.json")
    ap.add_argument("--transcripts", type=Path, default=RQ3 / "transcripts.jsonl")
    ap.add_argument("--extra", type=Path, default=RAW / "voice" / "extra.json")
    ap.add_argument("--operator", type=Path, default=RQ3 / "operator.json")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    files = args.exports or sorted(RAW.glob("online_*.json"))
    excl = load_json(args.exclusions, {})
    extra = {e["file"]: e for e in load_json(args.extra, [])}
    trials, sessions, excluded = build(load_records(files), excl.get("sessions", {}),
                                       load_jsonl(args.transcripts), extra, excl.get("voice", {}),
                                       load_json(args.operator, {}))
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "trials.csv", trials, TRIAL_COLS)
    write_csv(args.out / "sessions.csv", sessions, SESSION_COLS)
    write_csv(args.out / "excluded.csv", excluded, EXCLUDED_COLS)
    print(f"export: {len(files)}; sessioni: {len(sessions)}; opere: {len(trials)}; "
          f"escluse: {len(excluded)}; scritto in {args.out}")


if __name__ == "__main__":
    main()
