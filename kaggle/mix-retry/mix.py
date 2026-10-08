"""Kaggle (2x T4): rigenera i testi misti rimasti in violazione del filtro, tentativi 5-12,
stessa configurazione delle sessioni dei testi misti. Parte da data/local/mixed.jsonl del
repo. Output: /kaggle/working/mixed.jsonl (il file intero, aggiornato)."""

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
shutil.copy(f"{REPO}/data/local/mixed.jsonl", OUT)
run(f"PYTHONPATH={REPO}/src {sys.executable} -m museumbot.generation.vllm_mix "
    f"--retry-failed 12 --group 6 --out {OUT}")
