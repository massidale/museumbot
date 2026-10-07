"""Prova di fattibilita' su Kaggle (2x T4): vLLM con Gemma 4 31B QAT a 4 bit.

Passi, ognuno indipendente e registrato in /kaggle/working/probe.json:
  env        GPU, driver, versioni
  install    vLLM e repo museumbot (branch restructure-src)
  big        Gemma 4 31B QAT, tensor parallel 2: 6 testi di un'opera, throughput in batch,
             log-probabilita' passo per passo con la cache del prefisso
"""

import json
import os
import subprocess
import sys
import time
import traceback

OUT = "/kaggle/working/probe.json"
REPO = "/tmp/museumbot"
os.environ.setdefault("HF_HOME", "/tmp/hf")
LOG = {}


def save():
    with open(OUT, "w") as f:
        json.dump(LOG, f, indent=2, ensure_ascii=False)


def sh(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr)[-4000:]


def step(name):
    def wrap(fn):
        t0 = time.time()
        print(f"\n===== {name}", flush=True)
        try:
            LOG[name] = {"ok": True, **(fn() or {})}
        except Exception as e:
            LOG[name] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                         "trace": traceback.format_exc()[-3000:]}
        LOG[name]["seconds"] = round(time.time() - t0, 1)
        print(json.dumps(LOG[name], indent=2, ensure_ascii=False)[:3000], flush=True)
        save()
        return LOG[name]["ok"]
    return wrap


@step("env")
def env():
    return {"nvidia_smi": sh("nvidia-smi"), "python": sys.version,
            "pip": sh("pip list 2>/dev/null | grep -i -E '^(torch|transformers|vllm) '"),
            "disk": sh("df -h /tmp /kaggle/working")}


@step("install")
def install():
    out = sh("pip uninstall -y -q torchaudio; pip install -q -U vllm python-dotenv 2>&1 | tail -5")
    clone = sh(f"git clone -q -b restructure-src https://github.com/massidale/museumbot {REPO}")
    sys.path.insert(0, f"{REPO}/src")
    import torch
    import vllm
    return {"pip": out, "clone": clone, "vllm": vllm.__version__, "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpus": torch.cuda.device_count()}


def quality(text: str) -> dict:
    from museumbot.common.clean import clean_guide, is_truncated
    from museumbot.generation.generate import BANNED

    clean = clean_guide(text)
    words = clean.split()
    return {"words": len(words), "truncated": is_truncated(clean),
            "banned": bool(BANNED.search(clean)),
            "markdown": any(m in text for m in ("**", "\n#", "\n- ")),
            "distinct_ratio": round(len(set(words)) / max(len(words), 1), 2)}


def prompts_for(tok, arts, conds):
    from museumbot.common.prompts import local_messages
    from vllm.inputs import TokensPrompt

    out = []
    for a in arts:
        for c in conds:
            msgs = local_messages(c, a["title"], a["artist"], a["source_text"])
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            out.append(TokensPrompt(prompt_token_ids=tok.encode(text, add_special_tokens=False)))
    return out


def artworks():
    return [json.loads(l) for l in open(f"{REPO}/data/artworks.jsonl")]


def run_big():
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    t0 = time.time()
    sys.path.insert(0, f"{REPO}/src")
    from museumbot.generation.vllm_gen import LLM_ARGS, MODEL

    llm = LLM(MODEL, **LLM_ARGS, max_logprobs=100)
    load_s = round(time.time() - t0, 1)
    tok = llm.get_tokenizer()
    conds = ["explorer", "facilitator", "experience_seeker", "professional_hobbyist",
             "recharger", "flat"]
    arts = artworks()

    # 6 testi di un'opera
    p = prompts_for(tok, arts[:1], conds)
    t0 = time.time()
    outs = llm.generate(p, SamplingParams(temperature=0.7, max_tokens=800))
    dt = time.time() - t0
    texts = {c: {"text": o.outputs[0].text, **quality(o.outputs[0].text)}
             for c, o in zip(conds, outs)}
    n6 = sum(len(o.outputs[0].token_ids) for o in outs)

    # throughput in batch: 32 opere, Explorer
    pb = prompts_for(tok, arts[:32], ["explorer"])
    t0 = time.time()
    ob = llm.generate(pb, SamplingParams(temperature=0.7, max_tokens=800))
    dtb = time.time() - t0
    nb = sum(len(o.outputs[0].token_ids) for o in ob)
    qb = [quality(o.outputs[0].text) for o in ob]

    # passo per passo: 8 opere x 2 esperti, 30 passi da un token con 100 log-probabilita'
    ids = [list(x["prompt_token_ids"]) for x in
           prompts_for(tok, arts[:8], ["explorer", "experience_seeker"])]
    sp = SamplingParams(temperature=1.0, max_tokens=1, logprobs=100)
    steps, nlp = [], None
    for _ in range(30):
        t1 = time.time()
        res = llm.generate([TokensPrompt(prompt_token_ids=x) for x in ids], sp, use_tqdm=False)
        steps.append(time.time() - t1)
        nlp = len(res[0].outputs[0].logprobs[0])
        for i in range(0, len(ids), 2):  # stesso token ai due esperti (qui: quello del primo)
            t = res[i].outputs[0].token_ids[0]
            ids[i].append(t)
            ids[i + 1].append(t)
    return {"load_seconds": load_s,
            "six_texts": texts, "six_tok_s": round(n6 / dt, 1),
            "batch32_tok_s": round(nb / dtb, 1), "batch32_seconds": round(dtb, 1),
            "batch32_words": sorted(q["words"] for q in qb),
            "batch32_truncated": sum(q["truncated"] for q in qb),
            "batch32_markdown": sum(q["markdown"] for q in qb),
            "batch32_banned": sum(q["banned"] for q in qb),
            "batch32_low_distinct": sum(q["distinct_ratio"] < 0.3 for q in qb),
            "step_logprobs_returned": nlp,
            "step_seconds_first": round(steps[0], 2),
            "step_seconds_median": round(sorted(steps)[len(steps) // 2], 3)}


if LOG["install"]["ok"]:
    step("big")(run_big)

save()
print("\nFINE", {k: v["ok"] for k, v in LOG.items()})
