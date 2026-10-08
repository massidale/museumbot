"""Testi a profilo misto fra due categorie con vLLM (Kaggle): passo 3 del profilo continuo.

Per ogni (opera, coppia (X, Y), alpha) i due esperti sono i prompt di X e di Y
(`prompts.local_messages`), con pesi (1 - alpha, alpha). La miscela e' passo per passo da
fuori (`mixing.mix_generate`): a ogni passo una chiamata a vLLM da un token per esperto con
le prime TOP log-probabilita' grezze, cache del prefisso attiva. Pulizia, filtro e
rigenerazione come i testi puri (`vllm_gen.generate_rows`). Ogni riga salva la diagnostica
per token (log-probabilita' per esperto, copertura, divergenza).

Uso (su Kaggle, dopo il clone del repo):
    python -m museumbot.generation.vllm_mix --pair recharger professional_hobbyist \\
        --alphas 0 0.5 1 --limit 20 --out /kaggle/working/mixed.jsonl
"""

import argparse
import json
import time
from pathlib import Path

from museumbot.common.clean import clean_guide
from museumbot.common.config import ROOT
from museumbot.common.prompts import FALK_CATEGORIES, local_messages
from museumbot.generation.generate import ARTWORKS
from museumbot.generation.mixing import mix_generate
from museumbot.generation.rows import ATTEMPTS, row_seed
from museumbot.generation.vllm_gen import (
    LLM_ARGS, MAX_TOKENS, MODEL, TEMPERATURE, generate_rows, replace_rows,
)

OUT = ROOT / "data" / "local" / "mixed.jsonl"
TOP = 100      # log-probabilita' per esperto e per passo
# Testi generati insieme. Con il 31B su 2x T4 la cache KV tiene ~6 600 token: 3 testi misti
# (6 sequenze da ~950 token) ci stanno, e a ogni passo vLLM riusa il prefisso in cache e
# calcola solo il token nuovo. Con gruppi piu' grandi ricalcola i prompt a ogni passo.
GROUP = 3


def pair_name(pair) -> str:
    return "+".join(pair)


def mix_seed(artwork_id: str, pair, alpha: float, attempt: int) -> int:
    return row_seed(artwork_id, pair_name(pair), f"a{alpha:g}", attempt)


def make_mix_row(art, pair, alpha, raw, attempt, extra=None, failed=False) -> dict:
    text = clean_guide(raw)
    extra = dict(extra or {})
    diag = extra.pop("diag", [])
    n = max(len(diag), 1)
    return {"artwork_id": art["id"], "title": art["title"], "artist": art["artist"],
            "condition": pair_name(pair), "variant": "mix", "pair": list(pair),
            "alpha": alpha, "weights": {pair[0]: round(1 - alpha, 6), pair[1]: alpha},
            "model": MODEL, "temperature": TEMPERATURE,
            "seed": mix_seed(art["id"], pair, alpha, attempt),
            "text": text, "words": len(text.split()),
            "usage": {"raw": raw, "attempts": attempt + 1, "failed": failed,
                      "covered_all": round(sum(all(d["covered"]) for d in diag) / n, 4),
                      "div_mean": round(sum(d["div"] for d in diag) / n, 5),
                      **extra, "diag": diag}}


def mix_key(r: dict) -> tuple:
    return r["artwork_id"], r["condition"], r["alpha"]


def failed_mix_jobs(rows: list[dict], arts: dict) -> list[tuple]:
    """(opera, coppia, alpha) dei testi rimasti in violazione del filtro."""
    return [(arts[r["artwork_id"]], tuple(r["pair"]), r["alpha"])
            for r in rows if r["usage"].get("failed")]


def plan_mix_jobs(artworks, pairs, alphas, done) -> list[tuple]:
    return [(a, tuple(p), al) for p in pairs for al in alphas for a in artworks
            if (a["id"], pair_name(p), al) not in done]


def read_done(path: Path) -> set:
    if not path.exists():
        return set()
    return {mix_key(r) for r in map(json.loads, path.open())}


def vllm_mixer(llm, group: int = GROUP):
    """Backend vero: (opera, coppia, alpha, tentativo) -> (testo grezzo, extra)."""
    from transformers import GenerationConfig
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    tok = llm.get_tokenizer()
    eos = GenerationConfig.from_pretrained(MODEL).eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) | {tok.eos_token_id}
    params = SamplingParams(max_tokens=1, logprobs=TOP, temperature=1.0, seed=0)

    def ids(cond, art):
        msgs = local_messages(cond, art["title"], art["artist"], art["source_text"])
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        return tok.encode(text, add_special_tokens=False)

    def step(seqs):
        outs = llm.generate([TokensPrompt(prompt_token_ids=s) for s in seqs], params,
                            use_tqdm=False)
        return [{t: lp.logprob for t, lp in o.outputs[0].logprobs[0].items()} for o in outs]

    def gen(requests):
        out = []
        for i in range(0, len(requests), group):
            batch = requests[i : i + group]
            t0 = time.time()
            res = mix_generate([[ids(c, a) for c in p] for a, p, al, k in batch],
                               [[1 - al, al] for a, p, al, k in batch], step, eos=eos,
                               max_tokens=MAX_TOKENS, temp=TEMPERATURE,
                               seeds=[mix_seed(a["id"], p, al, k) for a, p, al, k in batch])
            secs = round(time.time() - t0, 1)
            out += [(tok.decode(toks, skip_special_tokens=True).strip(),
                     {"completion_tokens": len(toks), "group_seconds": secs, "diag": diag})
                    for toks, diag in res]
        return out

    return gen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pair", nargs=2, action="append", choices=FALK_CATEGORIES)
    ap.add_argument("--alphas", nargs="+", type=float)
    ap.add_argument("--retry-failed", type=int, metavar="FINO_A",
                    help="rigenera i testi con `failed` in --out, tentativi da ATTEMPTS a "
                         "FINO_A escluso, e riscrive il file")
    ap.add_argument("--limit", type=int, help="usa solo le prime N opere (pilota)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--group", type=int, default=GROUP, help="testi misti generati insieme")
    args = ap.parse_args()

    from vllm import LLM

    if args.retry_failed:
        rows = [json.loads(l) for l in args.out.open()]
        jobs = failed_mix_jobs(rows, {a["id"]: a for a in map(json.loads, ARTWORKS.open())})
        print(f"[{MODEL}] da rigenerare {len(jobs)} testi misti, tentativi "
              f"{ATTEMPTS}..{args.retry_failed - 1}", flush=True)
        gen = vllm_mixer(LLM(MODEL, **LLM_ARGS, max_logprobs=TOP, logprobs_mode="raw_logprobs"),
                         args.group)
        new = generate_rows(jobs, gen, start=ATTEMPTS, stop=args.retry_failed, make=make_mix_row)
        for r in new:
            print(f"  {r['title'][:30]:30s} {r['condition']} {r['alpha']} "
                  f"tentativi {r['usage']['attempts']} falliti {r['usage']['failed']}", flush=True)
        args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                    for r in replace_rows(rows, new, key=mix_key)))
        return
    artworks = [json.loads(l) for l in ARTWORKS.open()]
    if args.limit:
        artworks = artworks[: args.limit]
    todo = plan_mix_jobs(artworks, args.pair, args.alphas, read_done(args.out))
    print(f"[{MODEL}] coppie {args.pair}, alpha {args.alphas}: da generare {len(todo)}",
          flush=True)
    if not todo:
        return
    gen = vllm_mixer(LLM(MODEL, **LLM_ARGS, max_logprobs=TOP, logprobs_mode="raw_logprobs"),
                     args.group)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    chunk = 4 * args.group  # righe scritte su disco a ogni blocco
    for i in range(0, len(todo), chunk):
        rows = generate_rows(todo[i : i + chunk], gen, make=make_mix_row)
        with args.out.open("a") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {i + len(rows)}/{len(todo)}  {time.time() - t0:.0f}s  "
              f"falliti {sum(r['usage']['failed'] for r in rows)}  "
              f"copertura {sum(r['usage']['covered_all'] for r in rows) / len(rows):.3f}",
              flush=True)


if __name__ == "__main__":
    main()
