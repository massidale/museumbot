"""Patch di vLLM per le T4 (Turing, 64 KB di shared memory per blocco).

L'attenzione Triton di vLLM usa blocchi KV da 32 token e il pipelining di default: con le
teste da 512 di Gemma 4 servono 96 KB e il kernel non parte (OutOfResources). Sotto la
capacita' 8.0 si scende a blocchi da 16 e un solo stadio. Sulle GPU piu' recenti non cambia
nulla. Si ferma con errore se il punto da modificare non esiste (versione di vLLM diversa).
"""

import pathlib

import vllm.v1.attention.ops.triton_unified_attention as tua

ANCHOR = "    # USE_TD requires BLOCK_SIZE % TILE_SIZE == 0"
PATCH = """    # [museumbot] Turing (T4): 64 KB di shared memory, blocchi piu' piccoli e un solo stadio
    if not current_platform.has_device_capability(80):
        TILE_SIZE_PREFILL = min(TILE_SIZE_PREFILL, 16)
        TILE_SIZE_DECODE = min(TILE_SIZE_DECODE, 16)
        launch_num_stages = 1

"""

path = pathlib.Path(tua.__file__)
src = path.read_text()
if "[museumbot]" in src:
    print("patch gia' applicata:", path)
else:
    assert src.count(ANCHOR) == 1, f"punto di patch non trovato in {path}"
    assert "launch_num_stages" in src and "current_platform" in src
    path.write_text(src.replace(ANCHOR, PATCH + ANCHOR))
    print("patch applicata:", path)
