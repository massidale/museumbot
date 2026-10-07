"""Kaggle (2x T4): testi a profilo misto (passo 3 del profilo continuo), miscela passo per
passo con vLLM. Stessa configurazione di kaggle/gen. Output: /kaggle/working/mixed.jsonl.
Pilota: coppia Recharger-Professional/Hobbyist, alpha 0 / 0.5 / 1, prime 20 opere."""

import os
import subprocess
import sys

REPO = "/tmp/museumbot"
os.environ.setdefault("HF_HOME", "/tmp/hf")


def run(cmd: str) -> None:
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


run("pip uninstall -y -q torchaudio; pip install -q vllm==0.31.0 python-dotenv")
run(f"git clone -q -b restructure-src https://github.com/massidale/museumbot {REPO}")
run(f"git -C {REPO} log --oneline -1")
run(f"{sys.executable} {REPO}/kaggle/patch_vllm_turing.py")
run(f"PYTHONPATH={REPO}/src {sys.executable} -m museumbot.generation.vllm_mix "
    f"--pair recharger professional_hobbyist --alphas 0 0.5 1 --limit 20 "
    f"--out /kaggle/working/mixed.jsonl")
