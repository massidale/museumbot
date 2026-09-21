"""Genera le audioguide: 6 condizioni (5 categorie di Falk + flat) per ogni opera.

Scrive data/generations.jsonl in append-only, con chiave (artwork_id, condition). Il run
e' riavviabile: cio' che e' gia' presente viene saltato, quindi un'interruzione non costa
una seconda generazione.

Guardie di spesa: un tetto in dollari controllato dopo ogni chiamata, e il costo cumulato
stampato durante il run.
"""

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from config import openrouter_key
from prompts import CONDITIONS, VARIANTS, build_messages

ROOT = Path(__file__).resolve().parent.parent
ARTWORKS = ROOT / "data" / "artworks.jsonl"
OUT = ROOT / "data" / "generations.jsonl"
API = "https://openrouter.ai/api/v1/chat/completions"

# Il paper vieta i riferimenti espliciti alla categoria: se compaiono, il testo rivelerebbe
# la sua condizione in chiaro e inquinerebbe l'analisi latente. Si rigenera.
BANNED = re.compile(
    r"\bfor those\b|\bfor who\b|\bas a (?:visitor|listener|professional|curious)\b"
    r"|\bexperience seeker\b|\brecharger\b|\bfacilitator\b|\bfalk\b",
    re.I,
)


class Budget:
    """Costo cumulato con tetto: solleva appena la spesa supera il limite."""

    def __init__(self, cap: float):
        self.cap, self.spent, self.n = cap, 0.0, 0
        self._lock = threading.Lock()

    def add(self, cost: float) -> None:
        with self._lock:
            self.spent += cost
            self.n += 1
            if self.spent > self.cap:
                raise RuntimeError(f"tetto di spesa superato: ${self.spent:.4f} > ${self.cap}")

    def check(self) -> None:
        with self._lock:
            if self.spent > self.cap:
                raise RuntimeError(f"tetto di spesa superato: ${self.spent:.4f}")


def generate_one(
    session: requests.Session, model: str, msgs: list[dict], budget: Budget, temp: float
) -> tuple[str, dict]:
    """Genera un testo, rigenerando se viola il vincolo di stile o torna vuoto."""
    for attempt in range(4):
        budget.check()
        r = session.post(
            API,
            json={
                "model": model,
                "messages": msgs,
                "temperature": temp,
                "max_tokens": 2000,
                "usage": {"include": True},
            },
            timeout=300,
        )
        if r.status_code in (429, 500, 502, 503, 529):
            continue
        r.raise_for_status()
        d = r.json()
        if "error" in d:
            raise RuntimeError(d["error"])
        usage = d.get("usage") or {}
        budget.add(float(usage.get("cost") or 0.0))
        # Il prompt richiede gia' soltanto il testo parlato. Evitiamo euristiche su
        # preamboli o heading: potrebbero modificare un incipit perfettamente valido.
        text = (d["choices"][0]["message"].get("content") or "").strip()
        if text and not BANNED.search(text):
            return text, {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "cost": usage.get("cost"),
                "attempts": attempt + 1,
            }
    raise RuntimeError("nessun testo valido dopo 4 tentativi")


def read_done(path: Path) -> set[tuple[str, str, str, str]]:
    """Chiavi gia' generate. Le righe storiche senza `variant` valgono come `full`."""
    done = set()
    if path.exists():
        for line in path.open():
            r = json.loads(line)
            done.add((r["artwork_id"], r["condition"], r["model"], r.get("variant", "full")))
    return done


# Nota: "flat forced to full" dello spec e' implementato come "flat saltato per le
# varianti non-full" (equivalente se esistono le righe flat di full; su un dataset nuovo
# lanciare prima --variant full).
def plan_jobs(artworks, variants, done, model) -> list[tuple[dict, str, str]]:
    """Terne (opera, condizione, variante) da generare. `flat` esiste solo in `full`."""
    return [
        (a, c, v)
        for v in variants
        for a in artworks
        for c in CONDITIONS
        if not (c == "flat" and v != "full")
        and (a["id"], c, model, v) not in done
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, help="usa solo le prime N opere (smoke test)")
    ap.add_argument("--cap", type=float, default=1.00, help="tetto di spesa in dollari")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--variant", default="full", choices=list(VARIANTS),
                    help="variante del blocco di categoria (ablazione)")
    ap.add_argument("--all-variants", action="store_true",
                    help="genera tutte le varianti diverse da full in un solo run")
    args = ap.parse_args()

    artworks = [json.loads(l) for l in ARTWORKS.open()]
    if args.limit:
        artworks = artworks[: args.limit]

    variants = [v for v in VARIANTS if v != "full"] if args.all_variants else [args.variant]
    done = read_done(OUT)
    todo = plan_jobs(artworks, variants, done, args.model)
    print(f"[{args.model}] {len(artworks)} opere, varianti {variants}")
    print(f"  da generare {len(todo)}")
    if not todo:
        return

    budget = Budget(args.cap)
    key = openrouter_key()
    lock = threading.Lock()
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {key}"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fh = OUT.open("a")

    def work(job):
        art, cond, variant = job
        msgs = build_messages(cond, art["title"], art["artist"], art["source_text"], variant)
        text, usage = generate_one(session, args.model, msgs, budget, args.temperature)
        row = {
            "artwork_id": art["id"],
            "title": art["title"],
            "artist": art["artist"],
            "condition": cond,
            "variant": variant,
            "model": args.model,
            "temperature": args.temperature,
            "text": text,
            "words": len(text.split()),
            "usage": usage,
        }
        with lock:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if budget.n % 25 == 0 or budget.n == len(todo):
                print(f"  {budget.n}/{len(todo)}  ${budget.spent:.4f}", flush=True)
        return row

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(work, todo))
    finally:
        fh.close()

    print(f"\ngenerati {len(rows)} testi — costo ${budget.spent:.4f}")
    w = sorted(r["words"] for r in rows)
    print(f"  parole: min {w[0]}, mediana {w[len(w) // 2]}, max {w[-1]}")
    retried = sum(1 for r in rows if r["usage"]["attempts"] > 1)
    print(f"  rigenerazioni per violazione/errore: {retried}")


if __name__ == "__main__":
    main()
