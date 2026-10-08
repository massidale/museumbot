"""Come si alternano i due esperti dentro un testo misto? Analisi della diagnostica per token.

Per ogni token generato `vllm_mix` salva la log-probabilita' che ciascun esperto dava al
token scelto. Il margine

    m(t) = log p_X(t) - log p_Y(t)

e' positivo se la parola era piu' "di X", negativo se "di Y". I token si raggruppano in
frasi (tokenizer del generatore) e si misura, sui testi a alpha = 0.5:

  - cambi di segno del margine medio fra frasi consecutive, contro le stesse frasi in
    ordine casuale: meno cambi del caso = blocchi di frasi dello stesso esperto;
  - "appartenenza" delle frasi: dispersione dei margini medi per frase divisa per quella
    di tagli in punti casuali con le stesse lunghezze (1 = i confini di frase non contano);
  - profilo del margine per decimi del testo: tendenze di posizione (apertura, chiusura).

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


def positional_profile(texts: list[np.ndarray], bins: int = 10) -> np.ndarray:
    """Margine medio per intervallo di posizione relativa, mediato sui testi."""
    acc = np.zeros(bins)
    for m in texts:
        pos = np.minimum((np.arange(len(m)) * bins) // len(m), bins - 1)
        acc += np.array([m[pos == b].mean() for b in range(bins)])
    return acc / len(texts)


def ownership_ratio(sents: list[np.ndarray], rng, n: int = 100) -> float:
    """std dei margini medi per frase / std con tagli casuali delle stesse lunghezze."""
    allm = np.concatenate(sents)
    lens = [len(s) for s in sents]
    real = np.std([s.mean() for s in sents])
    fake = []
    for _ in range(n):
        rolled = np.roll(allm, rng.integers(len(allm)))
        cuts = np.cumsum(rng.permutation(lens))[:-1]
        fake.append(np.std([c.mean() for c in np.split(rolled, cuts)]))
    return float(real / np.mean(fake))


def run(alpha: float = 0.5, seed: int = 0) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    decode = lambda ids: tok.decode(ids)
    rng = np.random.default_rng(seed)
    rows = [json.loads(l) for l in MIXED.open()]
    out = {"alpha": alpha, "pairs": {}}
    for pair in sorted({r["condition"] for r in rows}):
        obs, null, own, texts, skipped = [], [], [], [], 0
        for r in rows:
            if r["condition"] != pair or r["alpha"] != alpha or r["usage"]["failed"]:
                continue
            ids = tok.encode(r["usage"]["raw"], add_special_tokens=False)
            d = r["usage"]["diag"]
            if len(ids) != len(d):
                skipped += 1  # ritokenizzazione diversa dalla generazione
                continue
            m = np.array([t["logp"][0] - t["logp"][1] for t in d])
            sents = split_sentences(ids, m, decode)
            sm = np.array([s.mean() for s in sents])
            obs.append(sign_changes(sm))
            null.append(np.mean([sign_changes(rng.permutation(sm)) for _ in range(200)]))
            own.append(ownership_ratio(sents, rng))
            texts.append(m)
        prof = positional_profile(texts)
        out["pairs"][pair] = {
            "n": len(obs), "skipped": skipped,
            "sign_changes_observed": round(float(np.mean(obs)), 2),
            "sign_changes_shuffled": round(float(np.mean(null)), 2),
            "share_below_chance": round(float(np.mean(np.array(obs) < np.array(null))), 3),
            "ownership_ratio_median": round(float(np.median(own)), 3),
            "ownership_share_above_1": round(float(np.mean(np.array(own) > 1)), 3),
            "profile_by_decile": [round(float(v), 3) for v in prof],
            "mean_margin": round(float(np.mean([m.mean() for m in texts])), 3)}
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
