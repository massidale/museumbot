"""Lessico distintivo per condizione: quali parole e formule usa una condizione piu' delle altre?

Per ogni corpus (`main` = DeepSeek, `local` = Gemma; testi `full` e flat) e condizione:

  - log-odds ratio con prior di Dirichlet informativo (Monroe, Colaresi e Quinn 2008),
    condizione contro il resto, separato per unigrammi, bigrammi e trigrammi; il prior e'
    il conteggio del termine su tutto il corpus. Si ordina per z-score: premia i termini
    frequenti nella condizione e rari nel resto, senza far dominare quelli rari;
  - quota di testi: frazione dei testi della condizione che contengono il termine
    (formula del registro se alta, tic di pochi testi se bassa);
  - eco del prompt: una parola piena del termine ricorre nel blocco di categoria del
    system prompt, a meno della flessione (stesso prefisso di almeno STEM lettere, o parola
    identica se piu' corta: "discuss" ~ "discussion"); non definito per il flat, che non ha
    blocco.

Scelte fissate prima dei risultati: K = 15 termini per condizione; un termine entra solo
se compare in almeno MIN_DOC_SHARE = 5% dei testi della condizione; unigrammi senza parole
funzionali (lista inglese di scikit-learn), n-grammi fatti solo di parole funzionali
esclusi; un termine che condivide una parola piena (stesso criterio dell'eco) con uno
gia' scelto si salta, cosi' ogni termine porta una parola nuova.

Scrive results/lexicon.json.
"""

import argparse
import json
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from museumbot.common.config import ROOT
from museumbot.common.corpus import load_corpus
from museumbot.common.prompts import CATEGORY_PARTS, CONDITIONS, build_block

RESULTS = ROOT / "results"
K = 15
MIN_DOC_SHARE = 0.05
NS = (1, 2, 3)
STEM = 5
WORD = re.compile(r"[a-z']+")


def tokens(text: str) -> list[str]:
    return WORD.findall(text.lower().replace("’", "'"))


def ngrams(toks: list[str], n: int) -> list[str]:
    """n-grammi come stringhe; unigrammi senza parole funzionali, n-grammi non tutti funzionali."""
    out = []
    for i in range(len(toks) - n + 1):
        g = toks[i : i + n]
        if all(w in ENGLISH_STOP_WORDS for w in g):
            continue
        out.append(" ".join(g))
    return out


def log_odds_z(y_c: Counter, y_r: Counter, prior: Counter) -> dict[str, float]:
    """z-score del log-odds ratio condizione vs resto con prior di Dirichlet (Monroe et al.)."""
    n_c, n_r, a0 = sum(y_c.values()), sum(y_r.values()), sum(prior.values())
    z = {}
    for w, a in prior.items():
        yc, yr = y_c.get(w, 0), y_r.get(w, 0)
        d = (np.log((yc + a) / (n_c + a0 - yc - a))
             - np.log((yr + a) / (n_r + a0 - yr - a)))
        z[w] = float(d / np.sqrt(1 / (yc + a) + 1 / (yr + a)))
    return z


def content(words) -> list[str]:
    return [w for w in words if w not in ENGLISH_STOP_WORDS]


def same_word(a: str, b: str) -> bool:
    """Stessa parola a meno della flessione: prefisso comune di STEM lettere."""
    if min(len(a), len(b)) < STEM:
        return a == b
    return a[:STEM] == b[:STEM]


def overlaps(a: str, b: str) -> bool:
    """I due termini condividono una parola piena."""
    return any(same_word(x, y) for x in content(a.split()) for y in content(b.split()))


def echo(term: str, block: str | None) -> bool | None:
    if block is None:
        return None
    return overlaps(term, " ".join(tokens(block)))


def distinctive(texts_by_cond: dict[str, list[str]], k: int = K,
                min_doc_share: float = MIN_DOC_SHARE) -> dict[str, list[dict]]:
    """Per condizione, i k termini con z piu' alto fra unigrammi, bigrammi e trigrammi."""
    scored = {c: [] for c in texts_by_cond}
    for n in NS:
        grams = {c: [ngrams(tokens(t), n) for t in ts] for c, ts in texts_by_cond.items()}
        counts = {c: Counter(g for gs in grams[c] for g in gs) for c in grams}
        prior = sum(counts.values(), Counter())
        for c in grams:
            rest = sum((counts[o] for o in grams if o != c), Counter())
            z = log_odds_z(counts[c], rest, prior)
            df = Counter(g for gs in grams[c] for g in set(gs))
            for w, cnt in counts[c].items():
                share = df[w] / len(grams[c])
                if share >= min_doc_share and z[w] > 0:
                    scored[c].append({"term": w, "n": n, "z": round(z[w], 2),
                                      "count": cnt, "doc_share": round(share, 3)})
    out = {}
    for c, items in scored.items():
        block = build_block(c) if c in CATEGORY_PARTS else None
        chosen = []
        for it in sorted(items, key=lambda t: -t["z"]):
            if any(overlaps(it["term"], s["term"]) for s in chosen):
                continue
            chosen.append({**it, "echo": echo(it["term"], block)})
            if len(chosen) == k:
                break
        out[c] = chosen
    return out


def echo_share(items: list[dict]) -> float | None:
    flags = [it["echo"] for it in items if it["echo"] is not None]
    return round(sum(flags) / len(flags), 3) if flags else None


def run() -> dict:
    out = {"k": K, "min_doc_share": MIN_DOC_SHARE}
    for corpus in ("main", "local"):
        rows = [r for r in load_corpus(corpus, verbose=False) if r["variant"] == "full"]
        by_cond = {c: [r["text"] for r in rows if r["condition"] == c] for c in CONDITIONS}
        terms = distinctive(by_cond)
        out[corpus] = {c: {"echo_share": echo_share(terms[c]), "terms": terms[c]}
                       for c in CONDITIONS}
    return out


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    out = run()
    for corpus in ("main", "local"):
        print(f"\n[{corpus}]")
        for c in CONDITIONS:
            r = out[corpus][c]
            terms = ", ".join(t["term"] + ("*" if t["echo"] else "") for t in r["terms"][:8])
            print(f"  {c:22} eco {r['echo_share']}  {terms}")
    path = RESULTS / "lexicon.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
