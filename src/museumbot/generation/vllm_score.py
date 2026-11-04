"""Rilettura dei testi misti con i due prompt puri (vLLM, Kaggle): margine per token dei testi a
percentuale, che non hanno la diagnostica della miscela.

Per ogni testo (opera, coppia (X, Y), alpha) il testo grezzo generato si accoda al prompt di X
e a quello di Y (`prompts.local_messages`) e vLLM restituisce la log-probabilita' di ogni
token del testo dato il prefisso (`prompt_logprobs`, distribuzione grezza): e' la stessa
quantita' che `vllm_mix` salva durante la miscela, senza il troncamento alle prime 100.
Con `--method mix` si rileggono i misti, per confrontare con la loro diagnostica.

Uso (su Kaggle, dopo il clone del repo):
    python -m museumbot.generation.vllm_score --method pct --alphas 0.5 \\
        --out /kaggle/working/scores_pct.jsonl
"""

import argparse
import json
import time
from pathlib import Path

from museumbot.common.config import ROOT
from museumbot.common.prompts import local_messages
from museumbot.generation.vllm_gen import LLM_ARGS, MODEL
from museumbot.generation.vllm_mix import OUT as MIXED
from museumbot.generation.vllm_pct import OUT as PCT

SOURCES = {"mix": MIXED, "pct": PCT}
SCORES = ROOT / "data" / "local"
CHUNK = 20  # testi per chiamata (due sequenze ciascuno)
# Le log-probabilita' del prompt passano per i logit di tutto il vocabolario (262k token)
# a ogni posizione: prefill a blocchi piccoli e meno cache KV, senza cache del prefisso.
SCORE_ARGS = {**LLM_ARGS, "enable_prefix_caching": False, "gpu_memory_utilization": 0.85,
              "max_num_batched_tokens": 512, "max_num_seqs": 4,
              "logprobs_mode": "raw_logprobs"}


def scores_path(method: str) -> Path:
    return SCORES / f"scores_{method}.jsonl"


def score_key(r: dict) -> tuple:
    return r["artwork_id"], r["condition"], r["alpha"], r["variant"]


def text_logprobs(prompt_logprobs, ids, start: int) -> list[float]:
    """Log-probabilita' dei token del testo, che nella sequenza partono da `start`."""
    return [prompt_logprobs[start + k][t].logprob for k, t in enumerate(ids)]


def score_rows(rows, prompt, encode, score) -> list[dict]:
    """`prompt(categoria, riga)` da' i token del prompt, `encode(grezzo)` quelli del testo,
    `score([(prompt, testo)])` le log-probabilita' dei token del testo. Una lista per
    esperto (X, Y) in `logp`, allineata a `ids`."""
    jobs, ids = [], []
    for r in rows:
        t = encode(r["usage"]["raw"])
        ids.append(t)
        jobs += [(prompt(c, r), t) for c in r["pair"]]
    lps = score(jobs)
    return [{"artwork_id": r["artwork_id"], "condition": r["condition"], "alpha": r["alpha"],
             "variant": r["variant"], "ids": t, "logp": [lps[2 * i], lps[2 * i + 1]]}
            for i, (r, t) in enumerate(zip(rows, ids))]


def vllm_scorer(llm):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    tok = llm.get_tokenizer()
    arts = {a["id"]: a for a in map(json.loads, (ROOT / "data" / "artworks.jsonl").open())}
    params = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=1.0, seed=0)

    def prompt(cond, r):
        a = arts[r["artwork_id"]]
        msgs = local_messages(cond, a["title"], a["artist"], a["source_text"])
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        return tok.encode(text, add_special_tokens=False)

    def encode(raw):
        return tok.encode(raw, add_special_tokens=False)

    def score(jobs):
        outs = llm.generate([TokensPrompt(prompt_token_ids=p + t) for p, t in jobs], params,
                            use_tqdm=False)
        return [text_logprobs(o.prompt_logprobs, t, len(p)) for o, (p, t) in zip(outs, jobs)]

    return prompt, encode, score


def select(rows, alphas, limit=None) -> list[dict]:
    arts = sorted({r["artwork_id"] for r in rows}, key=[r["artwork_id"] for r in rows].index)
    keep = set(arts[:limit]) if limit else set(arts)
    return [r for r in rows if r["alpha"] in alphas and r["artwork_id"] in keep
            and not r["usage"]["failed"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--method", choices=list(SOURCES), default="pct")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.5])
    ap.add_argument("--limit", type=int, help="usa solo le prime N opere (pilota)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out = args.out or scores_path(args.method)
    done = {score_key(r) for r in map(json.loads, out.open())} if out.exists() else set()
    rows = [r for r in select([json.loads(l) for l in SOURCES[args.method].open()],
                              args.alphas, args.limit) if score_key(r) not in done]
    print(f"[{MODEL}] {args.method}, alpha {args.alphas}: da rileggere {len(rows)}", flush=True)
    if not rows:
        return
    from vllm import LLM

    prompt, encode, score = vllm_scorer(LLM(MODEL, **SCORE_ARGS))
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i in range(0, len(rows), CHUNK):
        new = score_rows(rows[i : i + CHUNK], prompt, encode, score)
        with out.open("a") as fh:
            for r in new:
                fh.write(json.dumps(r) + "\n")
        print(f"  {i + len(new)}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
