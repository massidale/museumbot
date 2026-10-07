"""Testi di categoria puri con vLLM (Kaggle, 2x T4): passo 2 del profilo continuo.

Stessi prompt di `local.py` (`prompts.local_messages`), stessa pulizia e stesso filtro dei
riferimenti espliciti di `generate.py`: chi viola il filtro o torna vuoto si rigenera con
un nuovo seed, fino a ATTEMPTS tentativi; dopo l'ultimo la riga resta con `failed`.
Un seed per testo (`rows.row_seed`), quindi i testi del batch non condividono il rumore.
Scrive in append, riavviabile come `generate.py`. vLLM si importa solo nel backend vero.

Uso (su Kaggle, dopo il clone del repo):
    python -m museumbot.generation.vllm_gen --variant full --out /kaggle/working/generations.jsonl
    python -m museumbot.generation.vllm_gen --variant full_rep --out ...
"""

import argparse
import json
import time
from pathlib import Path

from museumbot.common.clean import clean_guide
from museumbot.common.config import ROOT
from museumbot.common.prompts import REPLICATES, local_messages, prompt_variant
from museumbot.generation.generate import ARTWORKS, BANNED, plan_jobs, read_done
from museumbot.generation.rows import ATTEMPTS, row_seed

MODEL = "google/gemma-4-31B-it-qat-w4a16-ct"
OUT = ROOT / "data" / "local" / "generations.jsonl"
TEMPERATURE = 0.7
MAX_TOKENS = 800
CHUNK = 120  # testi per chiamata a vLLM: ogni blocco si scrive su disco


def needs_retry(raw: str) -> bool:
    text = clean_guide(raw)
    return not text or bool(BANNED.search(text))


def make_row(art, cond, variant, raw, attempt, extra=None, failed=False) -> dict:
    text = clean_guide(raw)
    return {"artwork_id": art["id"], "title": art["title"], "artist": art["artist"],
            "condition": cond, "variant": variant, "weights": {cond: 1.0}, "model": MODEL,
            "temperature": TEMPERATURE, "seed": row_seed(art["id"], cond, variant, attempt),
            "text": text, "words": len(text.split()),
            "usage": {"raw": raw, "attempts": attempt + 1, "failed": failed, **(extra or {})}}


def generate_rows(jobs, gen) -> list[dict]:
    """`gen(richieste)` riceve (opera, condizione, variante, tentativo) e restituisce
    (testo grezzo, extra) per ciascuna. Rigenera solo i testi da rifare."""
    rows, pending = [], [(a, c, v, 0) for a, c, v in jobs]
    while pending:
        retry = []
        for (a, c, v, k), (raw, extra) in zip(pending, gen(pending)):
            last = k == ATTEMPTS - 1
            if needs_retry(raw) and not last:
                retry.append((a, c, v, k + 1))
            else:
                rows.append(make_row(a, c, v, raw, k, extra, failed=needs_retry(raw)))
        pending = retry
    return rows


def vllm_backend(llm):
    """Backend vero: prompt con il chat template, ragionamento disattivato, un seed per
    richiesta."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    tok = llm.get_tokenizer()

    def gen(requests):
        prompts, params = [], []
        for a, c, v, k in requests:
            msgs = local_messages(c, a["title"], a["artist"], a["source_text"], prompt_variant(v))
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            prompts.append(TokensPrompt(prompt_token_ids=tok.encode(text, add_special_tokens=False)))
            params.append(SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                                         seed=row_seed(a["id"], c, v, k)))
        t0 = time.time()
        outs = llm.generate(prompts, params, use_tqdm=False)
        secs = round(time.time() - t0, 1)
        return [(o.outputs[0].text.strip(),
                 {"prompt_tokens": len(o.prompt_token_ids),
                  "completion_tokens": len(o.outputs[0].token_ids),
                  "finish_reason": o.outputs[0].finish_reason, "batch_seconds": secs})
                for o in outs]

    return gen


# 2x T4: niente bfloat16 (float16), modello diviso sulle due GPU. Restano ~4 GB per GPU dopo
# i pesi: niente encoder multimodale, niente CUDA graph, al piu' 32 sequenze insieme.
LLM_ARGS = dict(dtype="float16", tensor_parallel_size=2, max_model_len=4096,
                gpu_memory_utilization=0.92, enable_prefix_caching=True, enforce_eager=True,
                max_num_seqs=32, limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})


def load_llm():
    from vllm import LLM

    return LLM(MODEL, **LLM_ARGS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", default="full", choices=["full", *REPLICATES])
    ap.add_argument("--limit", type=int, help="usa solo le prime N opere (pilota)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    artworks = [json.loads(l) for l in ARTWORKS.open()]
    if args.limit:
        artworks = artworks[: args.limit]
    todo = plan_jobs(artworks, [args.variant], read_done(args.out), MODEL)
    print(f"[{MODEL}] {len(artworks)} opere, variante {args.variant}: da generare {len(todo)}",
          flush=True)
    if not todo:
        return
    gen = vllm_backend(load_llm())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i in range(0, len(todo), CHUNK):
        rows = generate_rows(todo[i : i + CHUNK], gen)
        with args.out.open("a") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        done = i + len(rows)
        print(f"  {done}/{len(todo)}  {time.time() - t0:.0f}s  "
              f"falliti {sum(r['usage']['failed'] for r in rows)}", flush=True)


if __name__ == "__main__":
    main()
