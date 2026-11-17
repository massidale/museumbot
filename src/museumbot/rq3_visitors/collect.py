"""Raccolta degli export dell'iPad: unisce i file JSON in un solo responses.jsonl senza doppioni."""
import argparse
import json
from pathlib import Path

from museumbot.common.config import ROOT

REQUIRED = ("session_id", "device", "app_version", "manifest_version", "seed", "lang",
            "started_at", "updated_at", "completed", "last_step", "trials")
CHOICES = ("personal", "comparison")
OUT = ROOT / "data" / "rq3" / "responses.jsonl"


def validate(rec: dict) -> list[str]:
    """Problemi del record: campi mancanti, scelte o motivi non ammessi."""
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
    """Sessioni di un export: lista JSON (l'export dell'app) o una riga JSON per sessione."""
    path = Path(path)
    testo = path.read_text(encoding="utf-8")
    if testo.lstrip().startswith("["):
        try:
            dati = json.loads(testo)
        except json.JSONDecodeError as e:
            return [], [f"{path.name}: JSON non valido ({e.msg}, riga {e.lineno})"]
        righe = [r for r in dati if isinstance(r, dict)]
        return righe, [f"{path.name}: elemento {n} non e' un oggetto"
                       for n, r in enumerate(dati, 1) if not isinstance(r, dict)]
    righe, errori = [], []
    for n, line in enumerate(testo.splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            errori.append(f"{path.name}:{n}: JSON non valido ({e.msg})")
            continue
        if not isinstance(obj, dict):
            errori.append(f"{path.name}:{n}: non e' un oggetto")
            continue
        righe.append(obj)
    return righe, errori


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    validi, invalidi = [], 0
    for f in args.files:
        righe, errori = load_file(f)
        for e in errori:
            print("saltato:", e)
        invalidi += len(errori)
        for n, r in enumerate(righe, 1):
            p = validate(r)
            if p:
                print(f"saltato: {f.name} record {n} ({r.get('session_id')}): {'; '.join(p)}")
                invalidi += 1
            else:
                validi.append(r)
    out = merge(validi)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print(f"sessioni: {len(out)}")
    print(f"complete: {sum(1 for r in out if r['completed'])}")
    print(f"prove con risposta: {sum(answered(r) for r in out)}")
    print(f"record non validi: {invalidi}")
    print(f"scritto: {args.out}")


if __name__ == "__main__":
    main()
