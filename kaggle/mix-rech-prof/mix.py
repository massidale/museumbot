"""Kaggle (2x T4): giro completo dei testi misti per la coppia recharger -> professional_hobbyist, alpha 0.25 /
0.5 / 0.75 sulle 100 opere (passo 3 del profilo continuo). Parte da data/local/mixed.jsonl
del repo e salta i testi gia' presenti. Output: /kaggle/working/mixed.jsonl."""

import os
import shutil
import subprocess
import sys

REPO = "/tmp/museumbot"
OUT = "/kaggle/working/mixed.jsonl"
os.environ.setdefault("HF_HOME", "/tmp/hf")


def run(cmd: str) -> None:
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


run("pip uninstall -y -q torchaudio; pip install -q vllm==0.31.0 python-dotenv")
run(f"git clone -q -b restructure-src https://github.com/massidale/museumbot {REPO}")
run(f"git -C {REPO} log --oneline -1")
run(f"{sys.executable} {REPO}/kaggle/patch_vllm_turing.py")
if os.path.exists(f"{REPO}/data/local/mixed.jsonl"):
    shutil.copy(f"{REPO}/data/local/mixed.jsonl", OUT)
run(f"PYTHONPATH={REPO}/src {sys.executable} -m museumbot.generation.vllm_mix "
    f"--pair recharger professional_hobbyist --alphas 0.25 0.5 0.75 --group 6 --out {OUT}")
