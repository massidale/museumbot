"""Prova "pagina intera" con vLLM (Kaggle, 2x T4): fonte = voce Wikipedia intera, prompt con
l'indicazione degli aspetti su cui concentrarsi per categoria (`prompts.wiki_messages`).

Per ogni opera di --pages: un testo puro per categoria di Falk (5), con --flat anche il flat,
e un testo misto a alpha = 0.5 per ogni coppia di categorie (10). Puri come `vllm_gen`, misti come
`vllm_mix` (miscela classica, nessun ingresso ne' debito della minoranza); pulizia, filtro e
rigenerazione come gli altri corpus. Tutte le righe hanno variant "wiki"; i misti hanno
`pair` e `alpha`. Report: docs/report-pagina-intera.md.

Uso (su Kaggle):
    python -m museumbot.generation.vllm_wiki --out /kaggle/working/wiki.jsonl
    python -m museumbot.generation.vllm_wiki --pages data/wiki/haum_pages.jsonl --flat \
        --group 3 --out /kaggle/working/haum.jsonl     # opere del museo di Braunschweig (RQ3)
"""

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

from museumbot.common.config import ROOT
from museumbot.common.prompts import CONDITIONS, FALK_CATEGORIES, wiki_messages
from museumbot.generation.vllm_gen import LLM_ARGS, MODEL, generate_rows, make_row, vllm_backend
from museumbot.generation.vllm_mix import TOP, make_mix_row, vllm_mixer

PAGES = ROOT / "data" / "wiki" / "full_pages.jsonl"
OUT = ROOT / "data" / "wiki" / "generations.jsonl"
VARIANT = "wiki"
ALPHA = 0.5
# La voce piu' lunga (Olympia) e' ~3 350 token: prompt ~3 900, piu' 800 di testo.
MAX_MODEL_LEN = 5120
# Un testo misto alla volta: voce condivisa piu' due code da ~650 token stanno nei ~6 600
# token di cache KV; con due testi insieme si arriva al limite. Con voci corte (~800 token)
# ne stanno 3 (--group 3).
GROUP = 1


def pairs() -> list[tuple[str, str]]:
    return list(combinations(FALK_CATEGORIES, 2))


def pure_jobs(pages, flat: bool = False) -> list[tuple]:
    conds = CONDITIONS if flat else FALK_CATEGORIES
    return [(a, c, VARIANT) for a in pages for c in conds]


def mix_jobs(pages) -> list[tuple]:
    """Opera per opera, cosi' la voce resta nella cache fra un testo e l'altro."""
    return [(a, p, ALPHA) for a in pages for p in pairs()]


def wiki_mix_row(*args, **kw) -> dict:
    return {**make_mix_row(*args, **kw), "variant": VARIANT}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pages", type=Path, default=PAGES)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--flat", action="store_true", help="genera anche il flat per ogni opera")
    ap.add_argument("--group", type=int, default=GROUP, help="testi misti generati insieme")
    args = ap.parse_args()

    from vllm import LLM

    pages = [json.loads(l) for l in args.pages.open()]
    llm = LLM(MODEL, **{**LLM_ARGS, "max_model_len": MAX_MODEL_LEN}, max_logprobs=TOP,
              logprobs_mode="raw_logprobs")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = generate_rows(pure_jobs(pages, args.flat), vllm_backend(llm, messages=wiki_messages))
    with args.out.open("w") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"puri {len(rows)}  {time.time() - t0:.0f}s  "
          f"falliti {sum(r['usage']['failed'] for r in rows)}", flush=True)
    gen = vllm_mixer(llm, args.group, messages=wiki_messages)
    for a in pages:
        rows = generate_rows(mix_jobs([a]), gen, make=wiki_mix_row)
        with args.out.open("a") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        print(f"misti {a['title']}: {len(rows)}  {time.time() - t0:.0f}s  "
              f"falliti {sum(r['usage']['failed'] for r in rows)}  copertura "
              f"{sum(r['usage']['covered_all'] for r in rows) / len(rows):.3f}", flush=True)


if __name__ == "__main__":
    main()
