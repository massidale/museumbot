"""Kaggle (2x T4): testi di categoria puri per il corpus `local`, full e replica nella stessa
sessione. Output: /kaggle/working/generations.jsonl (riavviabile: salta cio' che c'e')."""

import os
import subprocess
import sys

REPO = "/tmp/museumbot"
OUT = "/kaggle/working/generations.jsonl"
os.environ.setdefault("HF_HOME", "/tmp/hf")


def run(cmd: str) -> None:
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


run("pip uninstall -y -q torchaudio; pip install -q vllm==0.31.0 python-dotenv")
run(f"git clone -q -b restructure-src https://github.com/massidale/museumbot {REPO}")
run(f"git -C {REPO} log --oneline -1")
run(f"{sys.executable} {REPO}/kaggle/patch_vllm_turing.py")
env = f"PYTHONPATH={REPO}/src"
limit = os.environ.get("LIMIT", "")
for variant in ("full", "full_rep"):
    run(f"{env} {sys.executable} -m museumbot.generation.vllm_gen --variant {variant} "
        f"--out {OUT} {'--limit ' + limit if limit else ''}")
run(f"wc -l {OUT}")
