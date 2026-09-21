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
