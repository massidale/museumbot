"""Generazione con un modello aperto in locale (MLX), con miscela di categorie.

Parte 2 del progetto (docs/plans/2026-10-06-profilo-continuo-design.md). Il ciclo di
decoding e' scritto a mano perche' serve l'accesso alle distribuzioni sul prossimo token:

    per ogni esperto k (un system prompt di categoria) con peso w_k > 0:
        log p_k(t) = log_softmax(logit_k / T)
    log p(t) = sum_k w_k * log p_k(t), rinormalizzata (media geometrica pesata)
    si campiona UN token e lo si passa a tutti gli esperti

Con pesi one-hot il ciclo coincide con la generazione normale a temperatura T: i testi di
categoria puri si generano con lo stesso codice dei misti.

Prompt: `prompts.local_messages`, cioe' i prompt della parte 1 byte per byte piu'
`prompts.LOCAL_SUFFIX` in coda al system prompt, uguale per tutte le condizioni. Pulizia e
filtro dei riferimenti espliciti come in `generate.py`; scrive
data/local/generations.jsonl, riavviabile.

Uso:
    python -m museumbot.generation.local gen --limit 5            # pilota
    python -m museumbot.generation.local gen --variant full_rep
    python -m museumbot.generation.local report
"""

import argparse
import hashlib
import json
import math
import re
import statistics
import time
from pathlib import Path

import mlx.core as mx

from museumbot.common.clean import clean_guide, is_truncated
from museumbot.common.config import ROOT
from museumbot.common.prompts import REPLICATES, local_messages, prompt_variant
from museumbot.generation.generate import ARTWORKS, BANNED, plan_jobs, read_done

MODEL = "mlx-community/gemma-4-12B-it-4bit"
OUT = ROOT / "data" / "local" / "generations.jsonl"
MARKDOWN = re.compile(r"\*\*|__|^\s*#|^\s*[-*] ", re.M)
ATTEMPTS = 4


def log_softmax(x):
    return x - mx.logsumexp(x)


def mix_logprobs(logprobs: list, weights: list[float]):
    """Media geometrica pesata di distribuzioni (in log), rinormalizzata."""
    if len(logprobs) != len(weights):
        raise ValueError("un peso per esperto")
    if any(w < 0 for w in weights) or not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
        raise ValueError(f"pesi non validi: {weights}")
    mixed = sum(w * lp for w, lp in zip(weights, logprobs))
    return mixed - mx.logsumexp(mixed)


def decode(model, prompts: list[list[int]], weights: list[float], *, temp: float, seed: int,
           max_tokens: int, eos: set[int], make_cache) -> tuple[list[int], list[dict]]:
    """Genera un testo dalla miscela degli esperti. Restituisce i token e, per token, la
    log-probabilita' che ogni esperto attivo gli dava e la divergenza degli esperti dalla
    miscela (sum_k w_k KL(p_k || p), zero con un solo esperto)."""
    active = [k for k, w in enumerate(weights) if w > 0]
    w = [weights[k] for k in active]
    caches = [make_cache() for _ in active]
    logits = [model(mx.array([prompts[k]]), cache=c)[0, -1] for k, c in zip(active, caches)]
    key = mx.random.key(seed)
    out, diag = [], []
    for _ in range(max_tokens):
        lps = [log_softmax(l / temp) for l in logits]
        mixed = mix_logprobs(lps, w)
        key, sub = mx.random.split(key)
        tok = int(mx.random.categorical(mixed, key=sub).item())
        div = sum(wk * mx.sum(mx.exp(lp) * (lp - mixed)) for wk, lp in zip(w, lps))
        if tok in eos:
            break
        out.append(tok)
        diag.append({"logp": [float(lp[tok].item()) for lp in lps], "div": max(float(div.item()), 0.0)})
        logits = [model(mx.array([[tok]]), cache=c)[0, -1] for c in caches]
    return out, diag


def row_seed(artwork_id: str, condition: str, variant: str, attempt: int = 0) -> int:
    """Seed deterministico per testo: la replica ha un seed diverso dal full."""
    h = hashlib.sha256(f"{artwork_id}|{condition}|{variant}|{attempt}".encode()).hexdigest()
    return int(h[:8], 16)


def check_pilot(rows: list[dict]) -> dict:
    """Criteri del passo 1 (spec): troncamenti <= 1/30, violazioni <= 2/30 (Markdown nel
    grezzo o rigenerazione per riferimento esplicito), mediana 200-300 parole, nessun
    testo sotto 120."""
    n = len(rows)
    words = [r["words"] for r in rows]
    trunc = sum(is_truncated(r["text"]) for r in rows)
    viol = sum(bool(MARKDOWN.search(r["usage"].get("raw") or r["text"]))
               or r["usage"].get("attempts", 1) > 1 for r in rows)
    med = statistics.median(words)
    rep = {"n": n, "truncated": trunc, "violations": viol, "median_words": med,
           "min_words": min(words), "max_words": max(words)}
    rep["passed"] = (trunc <= n / 30 and viol <= 2 * n / 30 and 200 <= med <= 300
                     and min(words) >= 120)
    return rep


# ------------------------------------------------------------------ modello vero (MLX)


def chat_ids(tokenizer, msgs: list[dict]) -> list[int]:
    """Prompt tokenizzato con il chat template, ragionamento disattivato."""
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                         enable_thinking=False)


def generate_text(model, tokenizer, msgs_per_expert, weights, *, temp, seed, max_tokens):
    from mlx_lm.models.cache import make_prompt_cache

    prompts = [chat_ids(tokenizer, m) for m in msgs_per_expert]
    t0 = time.time()
    toks, diag = decode(model, prompts, weights, temp=temp, seed=seed, max_tokens=max_tokens,
                        eos=set(tokenizer.eos_token_ids),
                        make_cache=lambda: make_prompt_cache(model))
    secs = time.time() - t0
    raw = tokenizer.decode(toks).strip()
    usage = {"prompt_tokens": [len(p) for p in prompts], "completion_tokens": len(toks),
             "seconds": round(secs, 1), "tok_per_s": round(len(toks) / secs, 2) if secs else None}
    return raw, usage, diag


def generate_one(model, tokenizer, art, cond, variant, *, temp, max_tokens) -> dict:
    """Un testo di categoria (pesi one-hot), rigenerato se viola il filtro o e' vuoto."""
    msgs = local_messages(cond, art["title"], art["artist"], art["source_text"],
                          prompt_variant(variant))
    for attempt in range(ATTEMPTS):
        seed = row_seed(art["id"], cond, variant, attempt)
        raw, usage, _ = generate_text(model, tokenizer, [msgs], [1.0], temp=temp, seed=seed,
                                      max_tokens=max_tokens)
        text = clean_guide(raw)
        if text and not BANNED.search(text):
            break
    usage |= {"raw": raw, "attempts": attempt + 1}
    return {"artwork_id": art["id"], "title": art["title"], "artist": art["artist"],
            "condition": cond, "variant": variant, "weights": {cond: 1.0}, "model": MODEL,
            "temperature": temp, "seed": seed, "text": text, "words": len(text.split()),
            "usage": usage}


def gen(args) -> None:
    from mlx_lm import load

    artworks = [json.loads(l) for l in ARTWORKS.open()]
    if args.limit:
        artworks = artworks[: args.limit]
    todo = plan_jobs(artworks, [args.variant], read_done(OUT), MODEL)
    print(f"[{MODEL}] {len(artworks)} opere, variante {args.variant}: da generare {len(todo)}")
    if not todo:
        return
    model, tokenizer = load(MODEL)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as fh:
        for i, (art, cond, variant) in enumerate(todo, 1):
            row = generate_one(model, tokenizer, art, cond, variant, temp=args.temperature,
                               max_tokens=args.max_tokens)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            u = row["usage"]
            print(f"  {i}/{len(todo)} {art['title'][:30]:30s} {cond:22s} {row['words']:4d} parole "
                  f"{u['seconds']:6.1f}s {u['tok_per_s']} tok/s", flush=True)


def report(path: Path = OUT) -> dict:
    rows = [json.loads(l) for l in path.open()]
    rep = check_pilot(rows)
    for k, v in rep.items():
        print(f"{k:14s} {v}")
    secs = [r["usage"]["seconds"] for r in rows]
    print(f"{'secondi/testo':14s} mediana {statistics.median(secs):.1f}")
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="genera i testi di categoria (pesi one-hot)")
    g.add_argument("--limit", type=int, help="usa solo le prime N opere (pilota)")
    g.add_argument("--variant", default="full", choices=["full", *REPLICATES])
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--max-tokens", type=int, default=800)
    sub.add_parser("report", help="criteri del pilota sulle righe generate")
    args = ap.parse_args()
    gen(args) if args.cmd == "gen" else report()


if __name__ == "__main__":
    main()
