"""Raccolta degli export dell'iPad: unisce i CSV in un solo responses.csv senza doppioni.

Ogni export ha una riga per opera, coi dati della sessione ripetuti (colonne in COLUMNS, le
stesse di CSV_COLUMNS in rq3_app/logic.js); una sessione senza opere ha una riga con le
colonne dell'opera vuote. Le righe si rimettono insieme in un record per sessione, si
controllano, si tiene la versione piu' avanti di ogni sessione e si riscrive un CSV.
"""
import argparse
import csv
import json
from pathlib import Path

from museumbot.common.config import ROOT

COLUMNS = (
    "session_id", "device", "app_version", "manifest_version", "seed", "lang",
    "started_at", "updated_at", "finished_at", "completed", "last_step", "aborted_by_operator",
    "chosen", "description_order", "category_main", "category_secondary", "player_errors",
    "trial_index", "artwork_id", "personal", "comparison", "first", "audio_1", "audio_2",
    "plays_1", "plays_2", "heard_sec_1", "heard_sec_2", "reached_end_1", "reached_end_2",
    "choice", "choice_position", "strength", "why", "shown_at", "answered_at",
)
REQUIRED = ("session_id", "device", "app_version", "manifest_version", "seed", "lang",
            "started_at", "updated_at", "completed", "last_step", "trials")
CHOICES = ("personal", "comparison")
OUT = ROOT / "data" / "rq3" / "responses.csv"


# ---- dalle celle ai valori ----

def _opt(s: str) -> str | None:
    return s if s != "" else None


def _bool(s: str) -> bool:
    if s not in ("true", "false"):
        raise ValueError(f"booleano {s!r}")
    return s == "true"


def _int(s: str) -> int | None:
    return int(s) if s != "" else None


def _float(s: str) -> float | None:
    return float(s) if s != "" else None


def _optbool(s: str) -> bool | None:
    return _bool(s) if s != "" else None


def _list(s: str) -> list[str]:
    return s.split("|") if s != "" else []


def to_record(rows: list[dict]) -> dict:
    """Le righe di una sessione (stesso session_id) -> record come quello dell'app."""
    r0 = rows[0]
    rec = {
        "session_id": r0["session_id"], "device": r0["device"], "app_version": r0["app_version"],
        "manifest_version": r0["manifest_version"], "seed": int(r0["seed"]), "lang": r0["lang"],
        "started_at": r0["started_at"], "updated_at": r0["updated_at"],
        "finished_at": _opt(r0["finished_at"]), "completed": _bool(r0["completed"]),
        "last_step": r0["last_step"], "aborted_by_operator": _bool(r0["aborted_by_operator"]),
        "chosen": _list(r0["chosen"]), "description_order": _list(r0["description_order"]),
        "categories": ({"main": r0["category_main"], "secondary": _opt(r0["category_secondary"])}
                       if r0["category_main"] else None),
        "player_errors": json.loads(r0["player_errors"]) if r0["player_errors"] else [],
        "trials": [],
    }
    for r in rows:
        if r["trial_index"] == "":
            continue
        rec["trials"].append({
            "index": int(r["trial_index"]), "artwork_id": r["artwork_id"],
            "personal": r["personal"], "comparison": r["comparison"], "first": r["first"],
            "audio": [r["audio_1"], r["audio_2"]],
            # vuoti nelle sessioni registrate prima che l'app li contasse
            "plays": [_int(r["plays_1"]), _int(r["plays_2"])],
            "heard_sec": [_float(r["heard_sec_1"]), _float(r["heard_sec_2"])],
            "reached_end": [_optbool(r["reached_end_1"]), _optbool(r["reached_end_2"])],
            "choice": _opt(r["choice"]), "choice_position": _int(r["choice_position"]),
            "strength": _int(r["strength"]), "why": r["why"],
            "shown_at": _opt(r["shown_at"]), "answered_at": _opt(r["answered_at"]),
        })
    return rec


# ---- dai valori alle celle (per responses.csv) ----

def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return "|".join(v)
    return str(v)


def to_rows(rec: dict) -> list[dict]:
    """Un record -> righe del CSV (una per opera; una sola se non ci sono opere)."""
    cats = rec.get("categories") or {}
    base = {
        **{k: rec.get(k) for k in ("session_id", "device", "app_version", "manifest_version",
                                   "seed", "lang", "started_at", "updated_at", "finished_at",
                                   "completed", "last_step", "chosen", "description_order")},
        "aborted_by_operator": bool(rec.get("aborted_by_operator")),
        "category_main": cats.get("main"), "category_secondary": cats.get("secondary"),
        "player_errors": json.dumps(rec["player_errors"], ensure_ascii=False)
        if rec.get("player_errors") else None,
    }
    if not rec.get("trials"):
        return [{c: _cell(base.get(c)) for c in COLUMNS}]
    out = []
    for t in rec["trials"]:
        row = {**base, "trial_index": t["index"],
               **{k: t.get(k) for k in ("artwork_id", "personal", "comparison", "first", "choice",
                                        "choice_position", "strength", "why", "shown_at",
                                        "answered_at")}}
        for name in ("audio", "plays", "heard_sec", "reached_end"):
            row[f"{name}_1"], row[f"{name}_2"] = t[name]
        out.append({c: _cell(row.get(c)) for c in COLUMNS})
    return out


def write_csv(path: Path, records: list[dict]) -> None:
    """CSV con BOM, come l'export dell'app (Excel e Numbers lo aprono come UTF-8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\r\n")
        w.writeheader()
        for rec in records:
            w.writerows(to_rows(rec))


# ---- controllo e unione ----

def validate(rec: dict) -> list[str]:
    """Problemi del record: campi mancanti, scelte non ammesse."""
    errori = [f"manca {k}" for k in REQUIRED if k not in rec]
    trials = rec.get("trials")
    if not isinstance(trials, list):
        return errori + (["trials non e' una lista"] if "trials" in rec else [])
    for i, t in enumerate(trials):
        if not isinstance(t, dict):
            errori.append(f"prova {i}: non e' un oggetto")
            continue
        if t.get("choice") not in (None, *CHOICES):
            errori.append(f"prova {i}: choice {t.get('choice')!r}")
        if not isinstance(t.get("why", ""), str):
            errori.append(f"prova {i}: why {t.get('why')!r}")
    return errori


def answered(rec: dict) -> int:
    """Numero di prove con risposta."""
    return sum(1 for t in rec.get("trials", []) if t.get("choice") is not None)


def _rank(rec: dict) -> tuple:
    return (bool(rec.get("completed")), answered(rec), rec.get("updated_at") or "")


def merge(records: list[dict]) -> list[dict]:
    """Un record per session_id: completo, poi piu' risposte, poi piu' recente."""
    best: dict[str, dict] = {}
    for r in records:
        sid = r["session_id"]
        if sid not in best or _rank(r) > _rank(best[sid]):
            best[sid] = r
    return sorted(best.values(), key=lambda r: r.get("started_at") or "")


def load_file(path: Path) -> tuple[list[dict], list[str]]:
    """Record delle sessioni di un export; colonne mancanti o celle illeggibili -> errori."""
    path = Path(path)
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        mancanti = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if mancanti:
            return [], [f"{path.name}: colonne mancanti {', '.join(mancanti)}"]
        gruppi: dict[str, list[dict]] = {}
        for row in reader:
            gruppi.setdefault(row["session_id"], []).append(row)
    records, errori = [], []
    for sid, rows in gruppi.items():
        try:
            records.append(to_record(rows))
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            errori.append(f"{path.name}: sessione {sid}: {e}")
    return records, errori


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    validi, invalidi = [], 0
    for f in args.files:
        records, errori = load_file(f)
        for e in errori:
            print("saltato:", e)
        invalidi += len(errori)
        for r in records:
            p = validate(r)
            if p:
                print(f"saltato: {f.name} sessione {r.get('session_id')}: {'; '.join(p)}")
                invalidi += 1
            else:
                validi.append(r)
    out = merge(validi)
    write_csv(args.out, out)
    print(f"sessioni: {len(out)}")
    print(f"complete: {sum(1 for r in out if r['completed'])}")
    print(f"prove con risposta: {sum(answered(r) for r in out)}")
    print(f"sessioni non valide: {invalidi}")
    print(f"scritto: {args.out}")


if __name__ == "__main__":
    main()
