"""Kaggle (2x T4): testi misti a percentuale nel prompt per le tre coppie (recharger ->
professional_hobbyist, facilitator -> professional_hobbyist, explorer -> facilitator), alpha
0.25 / 0.5 / 0.75. LIMIT = prime N opere (pilota); stringa vuota per il giro completo sulle
100 opere. Parte da data/local/prompted.jsonl del repo e salta i testi gia' presenti.
Output: /kaggle/working/prompted.jsonl."""

import os
import shutil
import subprocess
import sys

LIMIT = ""  # opere del pilota ("2"); "" = tutte
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
if os.path.exists(f"{REPO}/data/local/prompted.jsonl"):
    shutil.copy(f"{REPO}/data/local/prompted.jsonl", OUT)
limit = f"--limit {LIMIT} " if LIMIT else ""
run(f"PYTHONPATH={REPO}/src {sys.executable} -m museumbot.generation.vllm_pct "
    f"--pair recharger professional_hobbyist --pair facilitator professional_hobbyist "
    f"--pair explorer facilitator --alphas 0.25 0.5 0.75 {limit}--out {OUT}")
