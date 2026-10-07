"""Kaggle (2x T4): rigenera i testi del corpus `local` rimasti in violazione del filtro dopo
ATTEMPTS tentativi, con i tentativi successivi fino a 12. Stessa configurazione di
kaggle/gen. Output: /kaggle/working/generations.jsonl (il corpus intero, aggiornato)."""

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
run(f"PYTHONPATH={REPO}/src {sys.executable} -m museumbot.generation.vllm_gen --retry-failed 12 "
    f"--in {REPO}/data/local/generations.jsonl --out /kaggle/working/generations.jsonl")
