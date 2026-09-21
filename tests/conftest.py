"""Mette src/ sul path: gli script importano i moduli fratelli per nome semplice."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
