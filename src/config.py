"""Configurazione condivisa per gli script del progetto."""

import os
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent

# Le variabili gia' presenti nell'ambiente hanno precedenza sul file locale.
load_dotenv(ROOT / ".env", override=False)


def openrouter_key() -> str:
    """Restituisce la chiave OpenRouter configurata nell'ambiente o in `.env`."""
    key = os.getenv("OPENROUTER_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_KEY non configurata: impostala nell'ambiente o nel file .env"
        )
    return key


def emb_path(alias: str, variant: str = "full") -> Path:
    """File degli embedding per (modello, variante). `full` mantiene il nome storico."""
    suffix = "" if variant == "full" else f"_{variant}"
    return ROOT / "data" / f"emb_{alias}{suffix}.npy"


def meta_path(variant: str = "full") -> Path:
    """Metadati allineati riga per riga con `emb_path`. `full` mantiene il nome storico."""
    suffix = "" if variant == "full" else f"_{variant}"
    return ROOT / "data" / f"meta{suffix}.csv"
