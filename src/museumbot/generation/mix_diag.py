"""Come si alternano i due esperti dentro un testo misto? Analisi della diagnostica per token.

Per ogni token generato `vllm_mix` salva la log-probabilita' che ciascun esperto dava al
token scelto. Il margine

    m(t) = log p_X(t) - log p_Y(t)

e' positivo se la parola era piu' "di X", negativo se "di Y". I token si raggruppano in
frasi (tokenizer del generatore) e si misura, sui testi a alpha = 0.5:

  - cambi di segno del margine medio fra frasi consecutive, contro l'atteso con le stesse
    frasi in ordine casuale (2pq/n, esatto): meno cambi del caso = blocchi di frasi dello
    stesso esperto;
  - margine medio, mediato per decimi del testo (ogni parte del testo pesa uguale): quale
    esperto prevale nel complesso.

Limiti: solo il token scelto, non le distribuzioni intere; nei token assenti dalle prime
100 di un esperto il valore e' il minimo della sua lista. Scrive results/mix_diag.json.
"""

import argparse
import json
import re

import numpy as np

from museumbot.common.config import ROOT
from museumbot.generation.vllm_gen import MODEL

MIXED = ROOT / "data" / "local" / "mixed.jsonl"
RESULTS = ROOT / "results"
END = re.compile(r"[.!?…][\"”’)\]]*\s*$")


def split_sentences(ids, margins, decode, min_tokens: int = 4) -> list[np.ndarray]:
    """Margini raggruppati per frase: si chiude dopo la punteggiatura di fine frase."""
    out, cur = [], []
    for k, i in enumerate(ids):
        cur.append(k)
        if len(cur) >= min_tokens and END.search(decode([ids[j] for j in cur])):
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


def decile_means(texts: list[np.ndarray], bins: int = 10) -> np.ndarray:
    """(testi, bins): margine medio di ogni testo per intervallo di posizione relativa."""
    out = []
    for m in texts:
        pos = np.minimum((np.arange(len(m)) * bins) // len(m), bins - 1)
        out.append([m[pos == b].mean() for b in range(bins)])
    return np.array(out)


def mean_margin(texts: list[np.ndarray], bins: int = 10) -> float:
    """Margine medio sui testi, mediato per intervallo di posizione relativa."""
    return float(decile_means(texts, bins).mean())


def run(alpha: float = 0.5) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    decode = lambda ids: tok.decode(ids)
    rows = [json.loads(l) for l in MIXED.open()]
    out = {"alpha": alpha, "pairs": {}}
    for pair in sorted({r["condition"] for r in rows}):
        obs, null, texts = [], [], []
        for r in rows:
            if r["condition"] != pair or r["alpha"] != alpha or r["usage"]["failed"]:
                continue
            ids = tok.encode(r["usage"]["raw"], add_special_tokens=False)
            d = r["usage"]["diag"]
            if len(ids) != len(d):
                continue  # ritokenizzazione diversa dalla generazione
            m = np.array([t["logp"][0] - t["logp"][1] for t in d])
            sents = split_sentences(ids, m, decode)
            sm = np.array([s.mean() for s in sents])
            obs.append(sign_changes(sm))
            null.append(expected_sign_changes(sm))
            texts.append(m)
        out["pairs"][pair] = {
            "n": len(obs),
            "sign_changes_observed": round(float(np.mean(obs)), 2),
            "sign_changes_expected": round(float(np.mean(null)), 2),
            "margin_mean": round(mean_margin(texts), 3)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()
    out = run(args.alpha)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "mix_diag.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
