import json
from pathlib import Path

import pytest

from prompts import CONDITIONS, build_system

SNAP = json.loads((Path(__file__).parent / "snapshots" / "system_full.json").read_text())


@pytest.mark.parametrize("cond", CONDITIONS)
def test_full_matches_snapshot(cond):
    """La variante full deve riprodurre byte per byte il prompt originale."""
    assert build_system(cond) == SNAP[cond]
