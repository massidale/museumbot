"""Chain del paper vs prompt singolo: la differenza fra i due metodi e' solo rumore?

Disegno. Per ogni (opera, categoria) si generano tre testi nella stessa sessione:
  single_a, single_b   prompt singolo (variante full), due run indipendenti
  chain                chain del paper (turni 1-2 condivisi + Prompt 3), con la stessa fonte
piu' un testo flat per opera, che serve da riferimento per gli steering vector.

Il rumore della generazione e' cos(single_a, single_b); la differenza fra metodi e'
cos(chain, single_*). Se le due sono uguali, il metodo non lascia traccia oltre al rumore.
Assunzione: la chain ha lo stesso rumore del singolo. Se fosse piu' rumorosa, l'eccesso
verrebbe contato come differenza fra metodi: un risultato "uguale al rumore" e' quindi
conservativo, uno "maggiore del rumore" va verificato con una seconda run della chain.

Tutto viene generato su un provider fisso (default deepinfra/fp8), con ragionamento al
default del modello, e i job sono mescolati fra metodi: nessun metodo viene generato in un
momento diverso dagli altri. Il file e' separato dal corpus principale, che e' stato
generato con routing libero; il confronto fra i due rumori misura quanto pesa quel routing.

Ragionamento: attivato esplicitamente. Il default dipende dal provider (Venice ragiona,
DeepInfra no), e il corpus principale e' stato generato in prevalenza con ragionamento.

Pulizia: la chain apre con una frase di presentazione ("Here is a 250-word audio guide ...
for the Recharger"), separata da `---`, usa markdown e spesso scrive in formato copione:
intestazioni ("Audio Guide Script (approx. 250 words):", "Audio Guide: <titolo>") e
didascalie di regia ("(Soft, inviting tone)", "(Fade out)"). Si valuta il testo parlato:
`common.clean.clean_guide` toglie tutto questo ed e' applicata a TUTTI i metodi, prima del filtro sui
riferimenti espliciti. Il testo originale resta in `usage.raw` quando la pulizia lo
modifica; l'analisi riapplica `clean_guide` al testo salvato, quindi le regole aggiunte dopo
la generazione valgono anche per i testi gia' generati.

Riavviabile: le righe gia' presenti vengono saltate; il prefisso della chain (turni 1-2) e'
salvato alla prima esecuzione e riusato, come nel paper.
"""

import argparse
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor

import requests

from museumbot.common.clean import clean_guide
from museumbot.common.config import ROOT, openrouter_key
from museumbot.common.prompts import FALK_CATEGORIES, USER_TEMPLATE, build_messages
from museumbot.generation.compare_chain import (
    CHAIN_CATEGORY_NAME,
    CHAIN_P1,
    CHAIN_P2,
    CHAIN_P3,
)
from museumbot.generation.generate import ARTWORKS, Budget, generate_one, provider_routing

OUT = ROOT / "data" / "main" / "chain_study.jsonl"
PREFIX = ROOT / "data" / "main" / "chain_study_prefix.json"
METHODS = ("single_a", "single_b", "chain")


def chain_messages(prefix: list[dict], art: dict, category: str) -> list[dict]:
    """Prefisso condiviso + Prompt 3 del paper, con la stessa fonte del prompt singolo."""
    p3 = CHAIN_P3.format(
        title=art["title"], artist=art["artist"], category=CHAIN_CATEGORY_NAME[category]
    )
    src = USER_TEMPLATE.format(
        title=art["title"], artist=art["artist"], source_text=art["source_text"]
    )
    return prefix + [{"role": "user", "content": f"{p3}\n\n{src}"}]


def messages_for(method: str, art: dict, condition: str, prefix: list[dict]) -> list[dict]:
    if method == "chain":
        return chain_messages(prefix, art, condition)
    return build_messages(condition, art["title"], art["artist"], art["source_text"], "full")


def plan(artworks: list[dict], done: set[tuple[str, str, str]], seed: int = 0) -> list[tuple]:
    """Job (opera, condizione, metodo) mancanti, mescolati fra metodi e opere."""
    jobs = [(a, "flat", "flat") for a in artworks]
    jobs += [(a, c, m) for a in artworks for c in FALK_CATEGORIES for m in METHODS]
    jobs = [j for j in jobs if (j[0]["id"], j[1], j[2]) not in done]
    random.Random(seed).shuffle(jobs)
    return jobs


def read_done() -> set[tuple[str, str, str]]:
    if not OUT.exists():
        return set()
    return {(r["artwork_id"], r["condition"], r["method"])
            for r in map(json.loads, OUT.open())}


def get_prefix(session, model, budget, temp, routing) -> list[dict]:
    """Turni 1-2 della chain: generati una volta, poi letti da file."""
    if PREFIX.exists():
        return json.loads(PREFIX.read_text())["messages"]
    m = [{"role": "user", "content": CHAIN_P1}]
    r1, u1 = generate_one(session, model, m, budget, temp, routing, banned=None, reasoning=True)
    m += [{"role": "assistant", "content": r1}, {"role": "user", "content": CHAIN_P2}]
    r2, u2 = generate_one(session, model, m, budget, temp, routing, banned=None, reasoning=True)
    m += [{"role": "assistant", "content": r2}]
    PREFIX.write_text(json.dumps({"model": model, "usage": [u1, u2], "messages": m},
                                 ensure_ascii=False, indent=2))
    print(f"  prefisso chain: P1 {len(r1.split())} parole, P2 {len(r2.split())} parole")
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash")
    ap.add_argument("--provider", default="deepinfra/fp8")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, help="usa solo le prime N opere (smoke test)")
    ap.add_argument("--cap", type=float, default=0.50, help="tetto di spesa in dollari")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    artworks = [json.loads(l) for l in ARTWORKS.open()]
    if args.limit:
        artworks = artworks[: args.limit]
    todo = plan(artworks, read_done())
    print(f"[{args.model} @ {args.provider}] {len(artworks)} opere, da generare {len(todo)}")
    if not todo:
        return

    budget = Budget(args.cap)
    routing = provider_routing(args.provider)
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {openrouter_key()}"})
    prefix = get_prefix(session, args.model, budget, args.temperature, routing)

    lock = threading.Lock()
    fh = OUT.open("a")

    failed = []

    def work(job):
        art, cond, method = job
        msgs = messages_for(method, art, cond, prefix)
        try:
            text, usage = generate_one(session, args.model, msgs, budget, args.temperature,
                                       routing, reasoning=True, clean=clean_guide)
        except RuntimeError as e:
            if "tetto di spesa" in str(e):
                raise
            with lock:
                failed.append((art["id"], cond, method, str(e)))
                print(f"  saltato {art['id']} {cond} {method}: {e}", flush=True)
            return None
        row = {
            "artwork_id": art["id"],
            "title": art["title"],
            "artist": art["artist"],
            "condition": cond,
            "method": method,
            "model": args.model,
            "provider_requested": args.provider,
            "temperature": args.temperature,
            "text": text,
            "words": len(text.split()),
            "usage": usage,
        }
        with lock:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if budget.n % 50 == 0:
                print(f"  {budget.n} chiamate  ${budget.spent:.4f}", flush=True)
        return row

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(work, todo))
    finally:
        fh.close()

    rows = [r for r in rows if r]
    providers = {r["usage"]["provider"] for r in rows}
    print(f"\ngenerati {len(rows)} testi — costo ${budget.spent:.4f} — provider {providers}")
    if failed:
        print(f"  {len(failed)} celle saltate: rilanciare per riprovarle")


if __name__ == "__main__":
    main()
