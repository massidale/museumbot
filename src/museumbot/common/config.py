"""Configurazione condivisa per gli script del progetto."""

import os
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[3]  # src/museumbot/common/config.py -> root

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


def emb_path(alias: str, variant: str = "full", corpus: str = "main") -> Path:
    """File degli embedding per (modello, variante) di un corpus (vedi common.corpus)."""
    return ROOT / "data" / "emb" / corpus / f"{alias}_{variant}.npy"


def meta_path(variant: str = "full", corpus: str = "main") -> Path:
    """Metadati allineati riga per riga con `emb_path`, uguali per tutti i modelli."""
    return ROOT / "data" / "emb" / corpus / f"meta_{variant}.csv"


def ollama_key() -> str:
    """Restituisce la chiave di Ollama Cloud configurata nell'ambiente o in `.env`."""
    key = os.getenv("OLLAMA_CLOUD_KEY")
    if not key:
        raise RuntimeError(
            "OLLAMA_CLOUD_KEY non configurata: impostala nell'ambiente o nel file .env"
        )
    return key
