"""Kaggle (2x T4): rilettura dei testi a alpha = 0.5 con i due prompt puri (`vllm_score`), per
il margine per token dei testi a percentuale. Con LIMIT (pilota) rilegge anche i misti delle
stesse opere, da confrontare con la loro diagnostica. Output: /kaggle/working/scores_*.jsonl."""

import os
import subprocess
import sys

LIMIT = "3"  # opere del pilota; "" = tutte, solo testi a percentuale
REPO = "/tmp/museumbot"
os.environ.setdefault("HF_HOME", "/tmp/hf")


def run(cmd: str) -> None:
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


run("pip uninstall -y -q torchaudio; pip install -q vllm==0.31.0 python-dotenv")
run(f"git clone -q -b restructure-src https://github.com/massidale/museumbot {REPO}")
run(f"git -C {REPO} log --oneline -1")
run(f"{sys.executable} {REPO}/kaggle/patch_vllm_turing.py")
limit = f"--limit {LIMIT} " if LIMIT else ""
for method in (("mix", "pct") if LIMIT else ("pct",)):
    run(f"PYTHONPATH={REPO}/src {sys.executable} -m museumbot.generation.vllm_score "
        f"--method {method} --alphas 0.5 {limit}--out /kaggle/working/scores_{method}.jsonl")
