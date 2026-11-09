"""Kaggle (2x T4): giro completo della miscela classica a alpha 0.4, tre coppie, 100 opere
(seed come i misti). Parte da data/local/mixed_46.jsonl del repo (pilota a 10 opere per
coppia) e salta i testi gia' presenti. Spec: docs/plans/2026-11-09-miscela-alpha-tarata-corpus.md.
Output: /kaggle/working/mixed_46.jsonl."""

import os
import shutil
import subprocess
import sys

REPO = "/tmp/museumbot"
OUT = "/kaggle/working/mixed_46.jsonl"
os.environ.setdefault("HF_HOME", "/tmp/hf")
PAIRS = ("recharger professional_hobbyist", "facilitator professional_hobbyist",
         "explorer facilitator")


def run(cmd: str) -> None:
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


run("pip uninstall -y -q torchaudio; pip install -q vllm==0.31.0 python-dotenv")
run(f"git clone -q -b restructure-src https://github.com/massidale/museumbot {REPO}")
run(f"git -C {REPO} log --oneline -1")
run(f"{sys.executable} {REPO}/kaggle/patch_vllm_turing.py")
shutil.copy(f"{REPO}/data/local/mixed_46.jsonl", OUT)
pairs = " ".join(f"--pair {p}" for p in PAIRS)
run(f"PYTHONPATH={REPO}/src {sys.executable} -m museumbot.generation.vllm_mix {pairs} "
    f"--alphas 0.4 --group 6 --out {OUT}")
