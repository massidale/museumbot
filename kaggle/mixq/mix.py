"""Kaggle (2x T4): pilota della miscela con ingresso della minoranza (d* = 4), tre coppie,
alpha 0.25 e 0.75, stessi seed dei misti. SMOKE = True: una sola opera (`smoke` di
data/local/mixq_pilot.json) per le tre coppie, 6 testi; False: le 10 opere per coppia.
Spec: docs/plans/2026-11-04-ingresso-minoranza-design.md. Output: /kaggle/working/mixed_q.jsonl."""

import json
import os
import subprocess
import sys

SMOKE = True
REPO = "/tmp/museumbot"
OUT = "/kaggle/working/mixed_q.jsonl"
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
ids = f"{REPO}/data/local/mixq_pilot.json"
if SMOKE:
    pilot = json.load(open(ids))
    ids = "/kaggle/working/mixq_smoke.json"
    json.dump({"pairs": {p: [pilot["smoke"]] for p in pilot["pairs"]}}, open(ids, "w"))
pairs = " ".join(f"--pair {p}" for p in PAIRS)
run(f"PYTHONPATH={REPO}/src {sys.executable} -m museumbot.generation.vllm_mix {pairs} "
    f"--alphas 0.25 0.75 --dstar 4 --ids {ids} --group 6 --out {OUT}")
