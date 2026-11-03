"""Testi misti a percentuale nel prompt con vLLM (Kaggle): alternativa alla miscela per token.

Per ogni (opera, coppia (X, Y), alpha) un solo prompt con il blocco a percentuale
(`prompts.pct_messages`): ascoltatore (1 - alpha) X e alpha Y. Generazione normale, come i
testi puri (`vllm_gen.vllm_backend`): un seed per testo, ragionamento disattivato. Pulizia,
filtro e rigenerazione come i testi puri (`vllm_gen.generate_rows`). Le righe hanno lo stesso
formato dei misti (`vllm_mix`), con `variant: "pct"` e senza diagnostica per token.

Uso (su Kaggle, dopo il clone del repo):
    python -m museumbot.generation.vllm_pct --pair recharger professional_hobbyist \\
        --alphas 0.25 0.5 0.75 --limit 2 --out /kaggle/working/prompted.jsonl
"""

import argparse
import json
import time
from pathlib import Path

from museumbot.common.clean import clean_guide
from museumbot.common.config import ROOT
from museumbot.common.prompts import FALK_CATEGORIES, pct_messages
from museumbot.generation.generate import ARTWORKS
from museumbot.generation.rows import ATTEMPTS, row_seed
from museumbot.generation.vllm_gen import (
    CHUNK, MAX_TOKENS, MODEL, TEMPERATURE, generate_rows, load_llm, replace_rows,
)
from museumbot.generation.vllm_mix import (
    failed_mix_jobs, mix_key, pair_name, plan_mix_jobs, read_done,
)

OUT = ROOT / "data" / "local" / "prompted.jsonl"


def pct_seed(artwork_id: str, pair, alpha: float, attempt: int) -> int:
    return row_seed(artwork_id, pair_name(pair), f"p{alpha:g}", attempt)


def make_pct_row(art, pair, alpha, raw, attempt, extra=None, failed=False) -> dict:
    text = clean_guide(raw)
    return {"artwork_id": art["id"], "title": art["title"], "artist": art["artist"],
            "condition": pair_name(pair), "variant": "pct", "pair": list(pair),
            "alpha": alpha, "weights": {pair[0]: round(1 - alpha, 6), pair[1]: alpha},
            "model": MODEL, "temperature": TEMPERATURE,
            "seed": pct_seed(art["id"], pair, alpha, attempt),
            "text": text, "words": len(text.split()),
            "usage": {"raw": raw, "attempts": attempt + 1, "failed": failed, **(extra or {})}}


def vllm_pct_backend(llm):
    """Backend vero: (opera, coppia, alpha, tentativo) -> (testo grezzo, extra)."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    tok = llm.get_tokenizer()

    def gen(requests):
        prompts, params = [], []
        for a, p, al, k in requests:
            msgs = pct_messages(p[0], p[1], al, a["title"], a["artist"], a["source_text"])
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            prompts.append(TokensPrompt(prompt_token_ids=tok.encode(text, add_special_tokens=False)))
            params.append(SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                                         seed=pct_seed(a["id"], p, al, k)))
        t0 = time.time()
        outs = llm.generate(prompts, params, use_tqdm=False)
        secs = round(time.time() - t0, 1)
        return [(o.outputs[0].text.strip(),
                 {"prompt_tokens": len(o.prompt_token_ids),
                  "completion_tokens": len(o.outputs[0].token_ids),
                  "finish_reason": o.outputs[0].finish_reason, "batch_seconds": secs})
                for o in outs]

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
    args = ap.parse_args()

    if args.retry_failed:
        rows = [json.loads(l) for l in args.out.open()]
        jobs = failed_mix_jobs(rows, {a["id"]: a for a in map(json.loads, ARTWORKS.open())})
        print(f"[{MODEL}] da rigenerare {len(jobs)} testi a percentuale, tentativi "
              f"{ATTEMPTS}..{args.retry_failed - 1}", flush=True)
        new = (generate_rows(jobs, vllm_pct_backend(load_llm()), start=ATTEMPTS,
                             stop=args.retry_failed, make=make_pct_row) if jobs else [])
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
    gen = vllm_pct_backend(load_llm())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i in range(0, len(todo), CHUNK):
        rows = generate_rows(todo[i : i + CHUNK], gen, make=make_pct_row)
        with args.out.open("a") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {i + len(rows)}/{len(todo)}  {time.time() - t0:.0f}s  "
              f"falliti {sum(r['usage']['failed'] for r in rows)}", flush=True)


if __name__ == "__main__":
    main()
