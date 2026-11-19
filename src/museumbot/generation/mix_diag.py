"""Come si alternano i due esperti dentro un testo misto? Analisi della diagnostica per token.

Per ogni token generato `vllm_mix` salva la log-probabilita' che ciascun esperto dava al
token scelto. Il margine

    m(t) = log p_X(t) - log p_Y(t)

e' positivo se la parola era piu' "di X", negativo se "di Y". I token si raggruppano in
frasi (tokenizer del generatore); ogni frase ha la somma dei margini dei suoi token
(positiva: frase piu' "di X"). Per ogni testo a alpha = 0.5, poi in media sui testi:

  - numero di frasi;
  - somma massima e somma minima fra le frasi: quanto una frase arriva a pendere verso X o
    verso Y; somma media con segno: verso quale esperto pende il testo (0 = equilibrio);
  - cambi di segno fra frasi consecutive, contro l'atteso con le stesse frasi in ordine
    casuale (2pq/n, esatto): meno cambi del caso = blocchi di frasi dello stesso esperto.

Con `--source scores` i margini vengono dalla rilettura dei testi con i due prompt puri
(`vllm_score`, data/local/scores_<metodo>.jsonl): stessa quantita', calcolata sul testo gia'
scritto invece che durante la generazione, e senza troncamento alle prime 100. E' l'unica
fonte per i testi a percentuale (`--method pct`), e per il confronto si usa per entrambi.

Limiti: solo il token scelto, non le distribuzioni intere; nei misti, nei token assenti dalle
prime 100 di un esperto il valore e' il minimo della sua lista. Scrive
results/mix_diag.json (diagnostica dei misti) e results/mix_diag_<metodo>_scores.json.
"""

import argparse
import json
import re

import numpy as np

from museumbot.common.config import ROOT
from museumbot.generation.vllm_gen import MODEL
from museumbot.generation.vllm_score import scores_path

MIXED = ROOT / "data" / "local" / "mixed.jsonl"
RESULTS = ROOT / "results"
END = re.compile(r"[.!?…][\"”’)\]]*\s*$")
# punto che non chiude una frase: iniziale ("W.", "D.C.") o abbreviazione comune
ABBREV = re.compile(r"(?:\b[A-Z]|\b(?:St|Dr|Mr|Mrs|Ms|Jr|Sr|c|ca|vol|no))\.\s*$")


def split_sentences(ids, margins, decode, min_tokens: int = 4) -> list[np.ndarray]:
    """Margini raggruppati per frase. Si chiude dopo la punteggiatura di fine frase solo se
    il token seguente comincia con uno spazio o a capo (non "81.5") e il punto non segue
    un'iniziale o un'abbreviazione."""
    out, cur = [], []
    for k, i in enumerate(ids):
        cur.append(k)
        nxt = decode([ids[k + 1]]) if k + 1 < len(ids) else " "
        text = decode([ids[j] for j in cur])
        if (len(cur) >= min_tokens and END.search(text) and nxt[:1].isspace()
                and not ABBREV.search(text)):
            out.append(np.asarray(margins)[cur]); cur = []
    if cur:
        out.append(np.asarray(margins)[cur])
    return out


def sign_changes(v) -> int:
    s = np.sign(np.asarray(v)[np.asarray(v) != 0])
    return int((s[1:] != s[:-1]).sum())


def expected_sign_changes(v) -> float:
    """Cambi di segno attesi con i valori in ordine casuale: 2pq/n (p positivi, q negativi)."""
    v = np.asarray(v)
    p, q = int((v > 0).sum()), int((v < 0).sum())
    return 2 * p * q / (p + q) if p + q else 0.0


def sentence_stats(sents: list[np.ndarray]) -> dict:
    """Statistiche di un testo sulle somme dei margini per frase."""
    v = np.array([x.sum() for x in sents])
    return {"n_sentences": len(v), "max": float(v.max()), "min": float(v.min()),
            "mean": float(v.mean()),
            "changes": sign_changes(v), "expected": expected_sign_changes(v)}


def mix_margins(rows, alpha, encode):
    """(coppia, token, margini) dei misti a `alpha`, dalla diagnostica della generazione."""
    for r in rows:
        if r["alpha"] != alpha or r["usage"]["failed"]:
            continue
        ids = encode(r["usage"]["raw"])
        d = r["usage"]["diag"]
        if len(ids) != len(d):
            continue  # ritokenizzazione diversa dalla generazione
        yield r["condition"], ids, np.array([t["logp"][0] - t["logp"][1] for t in d])


def scored_margins(rows, alpha):
    """(coppia, token, margini) dalle riletture di `vllm_score`."""
    for r in rows:
        if r["alpha"] == alpha:
            yield r["condition"], r["ids"], np.array(r["logp"][0]) - np.array(r["logp"][1])


def pair_stats(items, decode) -> dict:
    """Medie sui testi delle statistiche per frase, per coppia."""
    by = {}
    for pair, ids, m in items:
        by.setdefault(pair, []).append(sentence_stats(split_sentences(ids, m, decode)))
    out = {}
    for pair in sorted(by):
        stats = by[pair]
        mean = lambda k: float(np.mean([st[k] for st in stats]))
        out[pair] = {
            "n": len(stats),
            "sentences_mean": round(mean("n_sentences"), 2),
            "sentence_sum_max_mean": round(mean("max"), 2),
            "sentence_sum_min_mean": round(mean("min"), 2),
            "sentence_sum_mean": round(mean("mean"), 2),
            "sign_changes_observed": round(mean("changes"), 2),
            "sign_changes_expected": round(mean("expected"), 2)}
    return out


def result_name(method: str, source: str) -> str:
    return "mix_diag" if source == "diag" else f"mix_diag_{method}_scores"


def run(alpha: float = 0.5, method: str = "mix", source: str = "diag") -> dict:
    from transformers import AutoTokenizer

    if source == "diag" and method != "mix":
        raise ValueError("la diagnostica per token c'e' solo nei misti")
    tok = AutoTokenizer.from_pretrained(MODEL)
    decode = lambda ids: tok.decode(ids)
    if source == "diag":
        rows = [json.loads(l) for l in MIXED.open()]
        items = mix_margins(rows, alpha, lambda raw: tok.encode(raw, add_special_tokens=False))
    else:
        items = scored_margins(map(json.loads, scores_path(method).open()), alpha)
    return {"alpha": alpha, "method": method, "source": source,
            "pairs": pair_stats(items, decode)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--method", choices=["mix", "pct"], default="mix")
    ap.add_argument("--source", choices=["diag", "scores"], default="diag")
    args = ap.parse_args()
    out = run(args.alpha, args.method, args.source)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{result_name(args.method, args.source)}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
