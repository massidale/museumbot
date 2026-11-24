"""Gemello LLM di RQ3: GLM-5.3 rifa le prove dei visitatori con la loro persona.

Per ogni prova del dataset pulito (data/rq3/clean/trials.csv) il modello riceve, nella lingua
della sessione, le descrizioni scelte dal visitatore nell'app (nell'ordine di scelta, con il
voto di aderenza 1-7) e i due testi dell'opera (testo di Gemma e testo del museo accorciato),
chiamati Guida 1 e Guida 2. Risponde con la domanda a voce dell'app e un voto 1-7 per guida.

Due ordini per prova: `visitor` (quello che ha avuto il visitatore) e `reversed`. Due modi:
`persona` e `neutral` (senza descrizioni, controllo). Una chiamata per opera, senza memoria.
Scrive data/rq3/twin.jsonl in append-only con chiave (session_id, artwork_id, mode, order).
Disegno in docs/decisioni.md (2026-11-23).
"""
import argparse
import csv
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import requests

from museumbot.common.config import ROOT, ollama_key
from museumbot.rq2_judge.judge import MODEL, PRICE, TokenBudget, judge_one

RQ3 = ROOT / "data" / "rq3"
OUT = RQ3 / "twin.jsonl"
GENERATIONS = [ROOT / "data" / "wiki" / f for f in ("haum_generations.jsonl", "haum2_generations.jsonl")]
MODES = ("persona", "neutral")
ORDERS = ("visitor", "reversed")

# frasi del prompt che non sono nell'app; il resto viene da rq3_app/texts.json
PHRASES = {
    "en": {
        "opening": "You are visiting an art museum.",
        "chose": "When you started the audio guide, you were asked which description fits you. "
                 "You chose:",
        "fit": "fits you: {v}",
        "scale": "(1 = {low}, 7 = {high})",
        "stand": "You are standing in front of {title} by {artist}. You listen to two audio "
                 "guides for this painting.",
        "rate": "Then rate each guide (1 = {low}, 7 = {high}; the two ratings can be equal):",
        "json": 'Answer only with a JSON object. In "comment", answer the question in a few '
                'sentences, as you would say it out loud: {"comment": "...", "guide1": <1-7>, '
                '"guide2": <1-7>}',
    },
    "de": {
        "opening": "Sie besuchen ein Kunstmuseum.",
        "chose": "Zu Beginn des Audioguides wurden Sie gefragt, welche Beschreibung zu Ihnen "
                 "passt. Sie haben gewählt:",
        "fit": "passt zu Ihnen: {v}",
        "scale": "(1 = {low}, 7 = {high})",
        "stand": "Sie stehen vor dem Gemälde {title} von {artist}. Sie hören zwei Audioguides "
                 "zu diesem Gemälde.",
        "rate": "Bewerten Sie dann jeden Audioguide (1 = {low}, 7 = {high}; beide Bewertungen "
                "dürfen gleich sein):",
        "json": 'Antworten Sie nur mit einem JSON-Objekt. Beantworten Sie in "comment" die '
                'Frage in wenigen Sätzen, so wie Sie es laut sagen würden: {"comment": "...", '
                '"guide1": <1-7>, "guide2": <1-7>}',
    },
}


def parse_fit(s: str) -> dict[str, int]:
    """'explorer:6 recharger:3' -> {'explorer': 6, 'recharger': 3}."""
    return {k: int(v) for k, v in (x.split(":") for x in s.split())}


def persona_prompt(lang: str, categories: list[str], fit: dict, app: dict) -> str:
    """Prompt di sistema: apertura e descrizioni scelte, con l'aderenza."""
    p, t = PHRASES[lang], app[lang]
    lines = [p["opening"], "", p["chose"]]
    for c in categories:
        lines.append(f'- "{t["falk"][c]}" ({p["fit"].format(v=fit[c])})')
    lines.append(p["scale"].format(low=t["fit_low"], high=t["fit_high"]))
    return "\n".join(lines)


def neutral_prompt(lang: str) -> str:
    return PHRASES[lang]["opening"]


def user_prompt(lang: str, title: str, artist: str, guide1: str, guide2: str, app: dict) -> str:
    """Messaggio con le due guide e le domande dell'app."""
    p, t = PHRASES[lang], app[lang]
    g1, g2 = (t["guide_label"].format(n=n) for n in (1, 2))
    return "\n\n".join([
        p["stand"].format(title=title, artist=artist),
        f"{g1}:\n{guide1}",
        f"{g2}:\n{guide2}",
        t["trial_voice_q"],
        "\n".join([p["rate"].format(low=t["rating_low"], high=t["rating_high"]),
                   *(f"- {t['q_rating'].format(guide=g)}" for g in (g1, g2))]),
        p["json"],
    ])


def guides(trial: dict, order: str) -> tuple[str, str, int]:
    """Quale testo e' Guida 1 e Guida 2 in un ordine, e la posizione della personalizzata."""
    pg = int(trial["personal_guide"])
    if order == "reversed":
        pg = 3 - pg
    pair = ("personal", "comparison") if pg == 1 else ("comparison", "personal")
    return pair[0], pair[1], pg


def parse_ratings(content: str) -> dict | None:
    """Commento e voti dall'ultimo oggetto JSON con guide1 e guide2 interi 1-7, o None."""
    for blob in reversed(re.findall(r"\{[^{}]*\}", content or "")):
        try:
            d = json.loads(blob)
        except json.JSONDecodeError:
            continue
        try:
            g1, g2 = int(d["guide1"]), int(d["guide2"])
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= g1 <= 7 and 1 <= g2 <= 7:
            return {"comment": str(d.get("comment", "")).strip(), "guide1": g1, "guide2": g2}
    return None


parse_ratings.empty = {"comment": None, "guide1": None, "guide2": None}


def personal_rating(row: dict) -> tuple[int, int]:
    """(voto della personalizzata, voto del museo) da una riga valida."""
    g = (row["guide1"], row["guide2"])
    pg = row["personal_guide"]
    return g[pg - 1], g[2 - pg]


def load_texts() -> tuple[dict, dict, dict]:
    """Testi generati per (opera, condizione, lingua), testi del museo per (opera, lingua),
    titolo e artista per opera."""
    arts = json.loads((RQ3 / "artworks_haum.json").read_text())
    slug = {a["title"]["en"]: a["id"] for a in arts}
    titles = {a["id"]: (a["title"], a["artist"]) for a in arts}
    gen = {}
    for path in GENERATIONS:
        for r in map(json.loads, path.open()):
            if r["title"] in slug:
                for lang, key in (("en", "text"), ("de", "text_de")):
                    gen[(slug[r["title"]], r["condition"], lang)] = r[key]
    museum = {}
    for r in map(json.loads, (RQ3 / "museum_texts.jsonl").open()):
        museum[(r["id"], "en")] = r["text_short"]
        museum[(r["id"], "de")] = r["text_short_de"]
    return gen, museum, titles


def read_csv(path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_jobs(trials, sessions, app, gen, museum, titles) -> list[dict]:
    """Una richiesta per prova, modo e ordine, con i messaggi gia' pronti."""
    sess = {s["session_id"]: s for s in sessions}
    jobs = []
    for t in trials:
        s, lang, art = sess[t["session_id"]], t["lang"], t["artwork_id"]
        text = {"personal": gen[(art, t["personal"], lang)], "comparison": museum[(art, lang)]}
        title, artist = titles[art]
        system = {"persona": persona_prompt(lang, s["categories"].split("+"),
                                            parse_fit(s["category_fit"]), app),
                  "neutral": neutral_prompt(lang)}
        for mode in MODES:
            for order in ORDERS:
                first, second, pg = guides(t, order)
                user = user_prompt(lang, title[lang], artist, text[first], text[second], app)
                jobs.append({
                    "session_id": t["session_id"], "artwork_id": art, "lang": lang,
                    "personal": t["personal"], "mode": mode, "order": order,
                    "personal_guide": pg,
                    "messages": [{"role": "system", "content": system[mode]},
                                 {"role": "user", "content": user}],
                })
    return jobs


def key(r: dict) -> tuple:
    return r["session_id"], r["artwork_id"], r["mode"], r["order"]


def read_done(path=OUT) -> set[tuple]:
    if not path.exists():
        return set()
    return {key(r) for r in map(json.loads, path.open()) if r["valid"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="pilota: solo le prime N prove")
    ap.add_argument("--cap", type=float, default=5.0, help="tetto di spesa in dollari")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    trials = read_csv(RQ3 / "clean" / "trials.csv")[: args.limit]
    sessions = read_csv(RQ3 / "clean" / "sessions.csv")
    app = json.loads((ROOT / "rq3_app" / "texts.json").read_text())
    jobs = build_jobs(trials, sessions, app, *load_texts())
    done = read_done()
    jobs = [j for j in jobs if key(j) not in done]
    print(f"[{MODEL}] {len(trials)} prove, chiamate da fare {len(jobs)}")
    if not jobs:
        return

    budget = TokenBudget(args.cap, PRICE[MODEL])
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {ollama_key()}"})
    lock = threading.Lock()
    fh = OUT.open("a")

    def work(job):
        res = judge_one(session, job["messages"], budget, parse=parse_ratings)
        row = {k: v for k, v in job.items() if k != "messages"} | {"model": MODEL} | res
        if res["valid"]:
            row["rating_personal"], row["rating_comparison"] = personal_rating(row)
        with lock:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
        return row

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(work, jobs))
    finally:
        fh.close()
    bad = sum(not r["valid"] for r in rows)
    print(f"{len(rows)} risposte, non valide {bad} — spesa ${budget.spent:.3f}")


if __name__ == "__main__":
    main()
