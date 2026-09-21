import json
from pathlib import Path

import pytest

from prompts import CONDITIONS, build_system

SNAP = json.loads((Path(__file__).parent / "snapshots" / "system_full.json").read_text())


@pytest.mark.parametrize("cond", CONDITIONS)
def test_full_matches_snapshot(cond):
    """La variante full deve riprodurre byte per byte il prompt originale."""
    assert build_system(cond) == SNAP[cond]


from prompts import CATEGORY_PARTS, FALK_CATEGORIES, VARIANTS, build_block


def test_variants_table():
    assert list(VARIANTS) == [
        "full", "no_def", "no_need", "no_style",
        "def_only", "need_only", "style_only", "name_only",
    ]
    assert VARIANTS["full"] == (True, True, True)
    assert VARIANTS["name_only"] == (False, False, False)
    assert VARIANTS["no_style"] == (True, True, False)
    assert VARIANTS["need_only"] == (False, True, False)


@pytest.mark.parametrize("cat", FALK_CATEGORIES)
def test_full_block_equals_legacy_block(cat):
    from prompts import CATEGORY_BLOCKS
    assert build_block(cat, "full") == CATEGORY_BLOCKS[cat]


@pytest.mark.parametrize("cat", FALK_CATEGORIES)
def test_name_only_has_no_parts(cat):
    p = CATEGORY_PARTS[cat]
    block = build_block(cat, "name_only")
    assert block == f"Your listener is {p['name']}."
    for part in ("def", "need", "style"):
        assert p[part] not in block


@pytest.mark.parametrize("cat", FALK_CATEGORIES)
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_parts_present_iff_active_and_in_order(cat, variant):
    p = CATEGORY_PARTS[cat]
    flags = dict(zip(("def", "need", "style"), VARIANTS[variant]))
    block = build_block(cat, variant)
    positions = []
    for part in ("def", "need", "style"):
        if flags[part]:
            assert p[part] in block
            positions.append(block.index(p[part]))
        else:
            assert p[part] not in block
    assert positions == sorted(positions)
    assert block.startswith(f"Your listener is {p['name']}")


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_flat_ignores_variant(variant):
    assert build_system("flat", variant) == build_system("flat", "full")


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        build_block("explorer", "bogus")
