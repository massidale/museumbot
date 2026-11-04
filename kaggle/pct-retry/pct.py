"""Kaggle (2x T4): rigenera i testi a percentuale rimasti in violazione del filtro, dal tentativo
dopo l'ultimo fatto fino al 200esimo, stessa configurazione della sessione dei testi a percentuale;
ogni riga registra le violazioni dei tentativi scartati (`usage.violations`). Parte da
data/local/prompted.jsonl del repo. Output: /kaggle/working/prompted.jsonl (il file intero, aggiornato)."""

import os
import shutil
import subprocess
import sys

REPO = "/tmp/museumbot"
OUT = "/kaggle/working/prompted.jsonl"
os.environ.setdefault("HF_HOME", "/tmp/hf")


def run(cmd: str) -> None:
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


run("pip uninstall -y -q torchaudio; pip install -q vllm==0.31.0 python-dotenv")
run(f"git clone -q -b restructure-src https://github.com/massidale/museumbot {REPO}")
run(f"git -C {REPO} log --oneline -1")
run(f"{sys.executable} {REPO}/kaggle/patch_vllm_turing.py")
shutil.copy(f"{REPO}/data/local/prompted.jsonl", OUT)
run(f"PYTHONPATH={REPO}/src {sys.executable} -m museumbot.generation.vllm_pct "
    f"--retry-failed 200 --out {OUT}")
