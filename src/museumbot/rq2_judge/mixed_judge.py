"""Giudice GLM-5.3 sui testi misti (passo 4 del profilo continuo).

Confronti da `mixed_pairs.build_mixed_pairs`, testi del corpus `local` (puri e replica) e
di data/local/mixed.jsonl (misti validi). Stesse persone, modello e parametri di RQ2
(`judge.py`); un solo ordine A/B per confronto. Domande:

    curve, double, attention   la stessa di RQ2 (quale ascolteresti)
    coherence                  quale suona come una voce sola, coerente dall'inizio alla fine
    rank                       ordinare 5 testi da "piu' per X" a "piu' per Y" (definizioni di
                               Falk delle due categorie nel messaggio)

Scrive data/judgments_mixed.jsonl in append-only con chiave pair_id: riavviabile.
"""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import requests

from museumbot.common.config import ROOT, ollama_key
from museumbot.common.corpus import load_corpus
from museumbot.common.prompts import CATEGORY_PARTS
from museumbot.rq2_judge.judge import MODEL, PRICE, QUESTION, TokenBudget, judge_one
from museumbot.rq2_judge.mixed_pairs import build_mixed_pairs, parse_order
from museumbot.rq2_judge.pairs import pilot_artworks
from museumbot.rq2_judge.personas import NEUTRAL, persona_prompt

OUT = ROOT / "data" / "judgments_mixed.jsonl"
MIXED = ROOT / "data" / "local" / "mixed.jsonl"
PAIRS = [("recharger", "professional_hobbyist"), ("facilitator", "professional_hobbyist"),
         ("explorer", "facilitator")]

COHERENCE_Q = (
    "Which of the two reads as if written in a single, consistent voice from beginning to "
    "end? Answer only with a JSON object: "
    '{"reason": "<one or two sentences>", "choice": "A" or "B"}'
)


def text_index(local_rows, mixed_rows) -> tuple[dict, dict]:
    """Testi per (opera, etichetta) e titoli per opera; i misti falliti restano fuori."""
    texts, titles = {}, {}
    for r in local_rows:
        lab = r["condition"] + ("|rep" if r["variant"] == "full_rep" else "")
        texts[(r["artwork_id"], lab)] = r["text"]
        titles[r["artwork_id"]] = (r["title"], r["artist"])
    for r in mixed_rows:
        if not r["usage"].get("failed"):
            texts[(r["artwork_id"], f"{r['condition']}|{r['alpha']:g}")] = r["text"]
    return texts, titles


def build_ab(p: dict, order: str, texts: dict, titles: dict, question: str) -> list[dict]:
    first, second = (p["target"], p["other"]) if order == "target_first" else (p["other"], p["target"])
    title, artist = titles[p["artwork_id"]]
    user = (f"You are standing in front of {title} by {artist}. "
            "The museum offers two audio guides for it.\n\n"
            f"Audio guide A:\n{texts[tuple(first)]}\n\n"
            f"Audio guide B:\n{texts[tuple(second)]}\n\n{question}")
    return [{"role": "system", "content": persona_prompt(p["persona"])},
            {"role": "user", "content": user}]


def describe(cat: str) -> str:
    part = CATEGORY_PARTS[cat]
    return f"{part['name'].split(' ', 1)[1]}: {part['def']}"


def build_rank(p: dict, texts: dict, titles: dict) -> list[dict]:
    title, artist = titles[p["artwork_id"]]
    x, y = p["pair"]
    guides = "\n\n".join(f"Audio guide {k}:\n{texts[tuple(t)]}"
                         for k, t in enumerate(p["texts"], 1))
    user = (f"You are standing in front of {title} by {artist}. The museum offers five audio "
            "guides for it.\n\n"
            f"Visitor type X is {describe(x)}\nVisitor type Y is {describe(y)}\n\n"
            f"{guides}\n\n"
            "Order the five guides from the one best suited to visitor type X to the one best "
            "suited to visitor type Y. Answer only with a JSON object: "
            '{"reason": "<one or two sentences>", "order": [the five guide numbers]}')
    return [{"role": "system", "content": persona_prompt(NEUTRAL)},
            {"role": "user", "content": user}]


def judge_item(session, p: dict, texts, titles, budget) -> dict:
    if p["type"] == "rank":
        res = judge_one(session, build_rank(p, texts, titles), budget,
                        parse=lambda c: parse_order(c, len(p["texts"])))
        ranked = [p["alphas"][k - 1] for k in res["choice"]] if res["valid"] else None
        return p | {"model": MODEL, "ranked_alphas": ranked} | res
    q = COHERENCE_Q if p["type"] == "coherence" else QUESTION
    res = judge_one(session, build_ab(p, p["order"], texts, titles, q), budget)
    chose = (res["choice"] == "A") == (p["order"] == "target_first") if res["valid"] else None
    return p | {"model": MODEL, "chose_target": chose} | res


def read_done() -> set[str]:
    if not OUT.exists():
        return set()
    return {r["pair_id"] for r in map(json.loads, OUT.open()) if r["valid"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, help="pilota: solo le prime N opere (ordine di RQ2)")
    ap.add_argument("--cap", type=float, default=15.0, help="tetto di spesa in dollari")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    local = [r for r in load_corpus("local", verbose=False) if r["variant"] in ("full", "full_rep")]
    texts, titles = text_index(local, map(json.loads, MIXED.open()))
    arts = sorted(titles)
    if args.limit:
        keep = pilot_artworks(arts, args.limit)
        arts = [a for a in arts if a in keep]
    items = build_mixed_pairs(arts, PAIRS)
    need = lambda p: [tuple(t) for t in p["texts"]] if p["type"] == "rank" else [
        tuple(p["target"]), tuple(p["other"])]
    missing = [p["pair_id"] for p in items if any(k not in texts for k in need(p))]
    items = [p for p in items if p["pair_id"] not in set(missing)]
    done = read_done()
    jobs = [p for p in items if p["pair_id"] not in done]
    print(f"[{MODEL}] {len(arts)} opere, {len(items)} confronti ({len(missing)} senza testi), "
          f"da fare {len(jobs)}", flush=True)
    if not jobs:
        return

    budget = TokenBudget(args.cap, PRICE[MODEL])
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {ollama_key()}"})
    lock = threading.Lock()
    with OUT.open("a") as fh:
        def work(p):
            row = judge_item(session, p, texts, titles, budget)
            with lock:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                if budget.n % 100 == 0:
                    print(f"  {budget.n} chiamate  ${budget.spent:.3f}", flush=True)
            return row

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(work, jobs))
    bad = sum(not r["valid"] for r in rows)
    print(f"\n{len(rows)} giudizi, non validi {bad} — spesa ${budget.spent:.3f}")


if __name__ == "__main__":
    main()
