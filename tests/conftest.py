"""Mette src/ sul path, cosi' `import museumbot` funziona anche senza installazione editable."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
