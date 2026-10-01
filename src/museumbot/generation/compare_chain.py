"""Analisi preliminare: prompt chain del paper vs prompt singolo collassato.

Domanda: con un modello 2026 (DeepSeek V4) il collasso della chain in un unico prompt
produce ancora audioguide equivalenti? Se si', la chain e' un artefatto dei modelli del
2024 e si puo' procedere con il prompt singolo.

Controllo di equita': la chain del paper non passa alcuna fonte testuale, si affida alla
conoscenza parametrica del modello. Qui la stessa fonte Wikipedia viene passata a entrambi
i metodi, cosi' l'unica variabile che cambia e' chain vs collasso (e non la presenza della
fonte, che altrimenti spiegherebbe da sola ogni differenza).

I turni 1-2 della chain non dipendono dall'opera: vengono eseguiti una volta sola e il
prefisso di conversazione viene riusato per ogni (opera, categoria), esattamente come nel
paper, dove i Prompt 3-7 seguono un'unica coppia di turni introduttivi.
"""

import argparse
import json

import requests

from museumbot.common.config import ROOT, openrouter_key
from museumbot.common.prompts import USER_TEMPLATE, build_messages

OUT = ROOT / "results" / "chain_vs_single.json"
API = "https://openrouter.ai/api/v1/chat/completions"

# Prompt 1, 2 e 3 riportati verbatim dal paper (Sez. 3.2 / Appendice A).
CHAIN_P1 = "Do you know the five museum visitor categories by John H. Falk?"
CHAIN_P2 = (
    "Can you recognize the needs for each of these categories during the description "
    "of an artwork?"
)
CHAIN_P3 = (
    "Now create a 250 words audio guide for the painting {title} by {artist} that fits "
    "the needs of the {category}. Remember to include most important information and "
    "lesser-known facts using a proper style for the given visitor category. Don't say "
    'something like "For those" "For who" "As" or explicitly refer to the target '
    "category of that description."
)

# Nomi come li userebbe il paper nel Prompt 3.
CHAIN_CATEGORY_NAME = {
    "explorer": "Explorer",
    "facilitator": "Facilitator",
    "experience_seeker": "Experience Seeker",
    "professional_hobbyist": "Professional/Hobbyist",
    "recharger": "Recharger",
}

class Client:
    def __init__(self, model: str, key: str):
        self.model, self.cost = model, 0.0
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {key}"})

    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        # V4 Pro e' un reasoning model: i token di ragionamento consumano `max_tokens`,
        # quindi serve margine oltre alle ~350 parole di audioguida.
        last = None
        for attempt in range(3):
            r = self.s.post(
                API,
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 3000,
                    "usage": {"include": True},
                },
                timeout=600,
            )
            r.raise_for_status()
            d = r.json()
            if "error" in d:
                raise RuntimeError(d["error"])
            self.cost += float((d.get("usage") or {}).get("cost") or 0.0)
            ch = d["choices"][0]
            text = (ch["message"].get("content") or "").strip()
            if text:
                return text
            last = ch.get("finish_reason")
            print(f"    [retry {attempt + 1}/3: content vuoto, finish_reason={last}]")
        raise RuntimeError(f"nessun contenuto dopo 3 tentativi (finish_reason={last})")


def check(text: str) -> dict:
    return {"words": len(text.split())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek/deepseek-v4-pro")
    ap.add_argument("--artworks", type=int, default=3)
    ap.add_argument(
        "--categories",
        nargs="+",
        default=["explorer", "professional_hobbyist", "recharger"],
        help="sottoinsieme di categorie per l'analisi preliminare",
    )
    args = ap.parse_args()

    rows = [json.loads(l) for l in (ROOT / "data" / "artworks.jsonl").open()]
    # opere distanziate nel corpus, per non testare tre quadri dello stesso autore
    picks = [rows[0], rows[len(rows) // 3], rows[2 * len(rows) // 3]][: args.artworks]

    cl = Client(args.model, openrouter_key())

    # --- prefisso della chain, calcolato una volta e riusato ---
    print(f"[{args.model}] chain: turni 1-2 (condivisi) ...")
    m = [{"role": "user", "content": CHAIN_P1}]
    r1 = cl.chat(m)
    m += [{"role": "assistant", "content": r1}, {"role": "user", "content": CHAIN_P2}]
    r2 = cl.chat(m)
    prefix = m + [{"role": "assistant", "content": r2}]
    print(f"  risposta P1: {len(r1.split())} parole | risposta P2: {len(r2.split())} parole")

    results = []
    for art in picks:
        for cat in args.categories:
            print(f"  {art['title'][:34]:36} {cat}")

            # metodo A: chain del paper (+ stessa fonte, per equita')
            p3 = CHAIN_P3.format(
                title=art["title"], artist=art["artist"], category=CHAIN_CATEGORY_NAME[cat]
            )
            src = USER_TEMPLATE.format(
                title=art["title"], artist=art["artist"], source_text=art["source_text"]
            )
            chain_text = cl.chat(prefix + [{"role": "user", "content": f"{p3}\n\n{src}"}])

            # metodo B: prompt singolo collassato
            single_text = cl.chat(
                build_messages(cat, art["title"], art["artist"], art["source_text"])
            )

            results.append(
                {
                    "artwork": art["title"],
                    "artist": art["artist"],
                    "category": cat,
                    "chain": chain_text,
                    "single": single_text,
                    "chain_check": check(chain_text),
                    "single_check": check(single_text),
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {"model": args.model, "chain_p1": r1, "chain_p2": r2, "results": results},
            ensure_ascii=False,
            indent=2,
        )
    )

    print(f"\n{'opera':30} {'categoria':22} {'chain':>12} {'single':>12}")
    for r in results:
        c, s = r["chain_check"], r["single_check"]
        print(
            f"{r['artwork'][:29]:30} {r['category']:22} "
            f"{c['words']:9}w    {s['words']:9}w"
        )
    print(f"\ncosto: ${cl.cost:.4f}  ->  {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
