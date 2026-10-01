"""Giudizi a coppie con GLM-5.3 su Ollama Cloud.

Ogni coppia (pairs.py) e' giudicata in entrambi gli ordini: `target_first` mette
l'elemento di interesse come "Audio guide A", `target_second` come "B". Scrive
data/judgments.jsonl in append-only con chiave (pair_id, order): il run e' riavviabile.

Ragionamento: `reasoning_effort: "low"`. Con `none` (o `think: false`) Ollama non spegne il
ragionamento di GLM-5.3 ma lo riversa nella risposta; `low` da' JSON pulito con un
ragionamento minimo, che viene salvato.

Spesa: Ollama non restituisce il costo, che si calcola dai token con il listino del modello.
"""

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from museumbot.common.config import ROOT, ollama_key
from museumbot.common.corpus import load_corpus
from museumbot.rq2_judge.pairs import build_pairs, pilot_artworks
from museumbot.rq2_judge.personas import persona_prompt

API = "https://ollama.com/v1/chat/completions"
OUT = ROOT / "data" / "judgments.jsonl"
MODEL = "glm-5.3"
PRICE = {"glm-5.3": (1.40, 4.40)}  # dollari per milione di token (input, output)
ORDERS = ("target_first", "target_second")

QUESTION = (
    "Which of the two would you rather listen to? Answer only with a JSON object: "
    '{"reason": "<one or two sentences>", "choice": "A" or "B"}'
)


class TokenBudget:
    """Spesa calcolata dai token, con tetto: solleva appena supera il limite."""

    def __init__(self, cap: float, price: tuple[float, float]):
        self.cap, self.price, self.spent, self.n = cap, price, 0.0, 0
        self._lock = threading.Lock()

    def add(self, prompt_tokens: int, completion_tokens: int) -> float:
        cost = (prompt_tokens * self.price[0] + completion_tokens * self.price[1]) / 1e6
        with self._lock:
            self.spent += cost
            self.n += 1
            if self.spent > self.cap:
                raise RuntimeError(f"tetto di spesa superato: ${self.spent:.4f} > ${self.cap}")
        return cost

    def check(self) -> None:
        if self.spent > self.cap:
            raise RuntimeError(f"tetto di spesa superato: ${self.spent:.4f}")


def parse_choice(content: str) -> tuple[str, str] | None:
    """(scelta, motivazione) dall'ultimo oggetto JSON valido della risposta, o None."""
    for blob in reversed(re.findall(r"\{[^{}]*\}", content or "")):
        try:
            d = json.loads(blob)
        except json.JSONDecodeError:
            continue
        choice = str(d.get("choice", "")).strip().upper()
        if choice in ("A", "B"):
            return choice, str(d.get("reason", "")).strip()
    return None


def build_messages(p: dict, order: str, texts: dict, titles: dict) -> list[dict]:
    """Messaggi per una coppia in un ordine. `texts` e' indicizzato da (opera, condizione)."""
    first, second = (p["target"], p["other"]) if order == "target_first" else (p["other"], p["target"])
    title, artist = titles[p["artwork_id"]]
    user = (
        f"You are standing in front of {title} by {artist}. "
        "The museum offers two audio guides for it.\n\n"
        f"Audio guide A:\n{texts[tuple(first)]}\n\n"
        f"Audio guide B:\n{texts[tuple(second)]}\n\n"
        f"{QUESTION}"
    )
    return [{"role": "system", "content": persona_prompt(p["persona"])},
            {"role": "user", "content": user}]


def judge_one(session, msgs: list[dict], budget: TokenBudget, model: str = MODEL) -> dict:
    """Una chiamata con ritentativi: errori HTTP transitori e risposte senza scelta valida."""
    body = {"model": model, "messages": msgs, "temperature": 0,
            "reasoning_effort": "low", "max_tokens": 1000}
    out = {"valid": False}
    for attempt in range(4):
        budget.check()
        try:
            r = session.post(API, json=body, timeout=180)
        except requests.RequestException as e:
            out["error"] = str(e)
            time.sleep(2 ** attempt)
            continue
        if r.status_code in (429, 500, 502, 503, 504, 529):
            out["error"] = f"http {r.status_code}"
            time.sleep(2 ** attempt * 2)
            continue
        r.raise_for_status()
        d = r.json()
        usage = d.get("usage") or {}
        cost = budget.add(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        m = d["choices"][0]["message"]
        content = m.get("content") or ""
        parsed = parse_choice(content)
        out = {
            "valid": parsed is not None,
            "choice": parsed[0] if parsed else None,
            "reason": parsed[1] if parsed else None,
            "content": content,
            "reasoning_chars": len(m.get("reasoning") or m.get("reasoning_content") or ""),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cost": cost,
            "attempts": attempt + 1,
        }
        if parsed:
            return out
    return out


def read_done() -> set[tuple[str, str]]:
    if not OUT.exists():
        return set()
    return {(r["pair_id"], r["order"]) for r in map(json.loads, OUT.open()) if r["valid"]}


def corpus_index() -> tuple[dict, dict]:
    """Testi puliti del corpus principale (variante full e flat) e titoli per opera."""
    rows = [r for r in load_corpus("main", verbose=False) if r["variant"] == "full"]
    texts = {(r["artwork_id"], r["condition"]): r["text"] for r in rows}
    titles = {r["artwork_id"]: (r["title"], r["artist"]) for r in rows}
    return texts, titles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int,
                    help="pilota: coppie A/B/C/N solo sulle prime N opere (attenzione su tutte)")
    ap.add_argument("--cap", type=float, default=15.0, help="tetto di spesa in dollari")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    texts, titles = corpus_index()
    pairs = build_pairs(titles)
    if args.limit:
        keep = pilot_artworks(titles, args.limit)
        pairs = [p for p in pairs if p["type"] == "attention" or p["artwork_id"] in keep]
    done = read_done()
    jobs = [(p, o) for p in pairs for o in ORDERS if (p["pair_id"], o) not in done]
    print(f"[{MODEL}] {len(pairs)} coppie, giudizi da fare {len(jobs)}")
    if not jobs:
        return

    budget = TokenBudget(args.cap, PRICE[MODEL])
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {ollama_key()}"})
    lock = threading.Lock()
    fh = OUT.open("a")

    def work(job):
        p, order = job
        res = judge_one(session, build_messages(p, order, texts, titles), budget)
        chose_target = None
        if res["valid"]:
            chose_target = (res["choice"] == "A") == (order == "target_first")
        row = p | {"order": order, "model": MODEL, "chose_target": chose_target} | res
        with lock:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if budget.n % 100 == 0:
                print(f"  {budget.n} chiamate  ${budget.spent:.3f}", flush=True)
        return row

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(work, jobs))
    finally:
        fh.close()
    bad = sum(not r["valid"] for r in rows)
    print(f"\n{len(rows)} giudizi, non validi {bad} — spesa ${budget.spent:.3f}")


if __name__ == "__main__":
    main()
