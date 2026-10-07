"""Parti delle righe generate comuni ai backend locali (MLX sul Mac, vLLM su Kaggle):
seed per testo e criteri del pilota (passo 1 della spec del profilo continuo)."""

import hashlib
import re
import statistics

from museumbot.common.clean import is_truncated

MARKDOWN = re.compile(r"\*\*|__|^\s*#|^\s*[-*] ", re.M)
ATTEMPTS = 4


def row_seed(artwork_id: str, condition: str, variant: str, attempt: int = 0) -> int:
    """Seed deterministico per testo: la replica ha un seed diverso dal full."""
    h = hashlib.sha256(f"{artwork_id}|{condition}|{variant}|{attempt}".encode()).hexdigest()
    return int(h[:8], 16)


def check_pilot(rows: list[dict]) -> dict:
    """Criteri del passo 1 (spec): troncamenti <= 1/30, violazioni <= 2/30 (Markdown nel
    grezzo o rigenerazione per riferimento esplicito), mediana 200-300 parole, nessun
    testo sotto 120."""
    n = len(rows)
    words = [r["words"] for r in rows]
    trunc = sum(is_truncated(r["text"]) for r in rows)
    viol = sum(bool(MARKDOWN.search(r["usage"].get("raw") or r["text"]))
               or r["usage"].get("attempts", 1) > 1 for r in rows)
    med = statistics.median(words)
    rep = {"n": n, "truncated": trunc, "violations": viol, "median_words": med,
           "min_words": min(words), "max_words": max(words)}
    rep["passed"] = (trunc <= n / 30 and viol <= 2 * n / 30 and 200 <= med <= 300
                     and min(words) >= 120)
    return rep
