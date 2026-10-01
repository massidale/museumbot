from collections import Counter

from museumbot.common.prompts import CATEGORY_PARTS, FALK_CATEGORIES
from museumbot.rq2_judge.pairs import build_pairs, pilot_artworks
from museumbot.rq2_judge.personas import NEUTRAL, persona_prompt

IDS = [f"Q{i}" for i in range(100)]


def test_counts_per_artwork():
    pairs = build_pairs(IDS)
    assert len(pairs) == 100 * (5 * 4 + 1)
    assert Counter(p["type"] for p in pairs) == {"A": 500, "B": 500, "C": 500, "N": 500,
                                                 "attention": 100}


def test_j_is_balanced_and_shared_by_b_and_c():
    pairs = build_pairs(IDS)
    b = {(p["artwork_id"], p["persona"]): p["other"][1] for p in pairs if p["type"] == "B"}
    c = {(p["artwork_id"], p["persona"]): p["target"][1] for p in pairs if p["type"] == "C"}
    assert b == c
    counts = Counter((k, j) for (_, k), j in b.items())
    assert all(j != k for k, j in counts)
    assert set(counts.values()) == {25} and len(counts) == 20


def test_targets_and_personas():
    for p in build_pairs(IDS):
        a = p["artwork_id"]
        if p["type"] in ("A", "B", "N"):
            assert p["target"] == [a, p["persona"] if p["type"] != "N" else p["target"][1]]
        if p["type"] in ("A", "C", "N"):
            assert p["other"] == [a, "flat"]
        if p["type"] == "N":
            assert p["persona"] == NEUTRAL
        if p["type"] == "attention":
            assert p["target"] == [a, "flat"] and p["other"][0] != a


def test_order_is_deterministic():
    assert build_pairs(IDS) == build_pairs(list(reversed(IDS)))
    assert len(pilot_artworks(IDS, 20)) == 20


def test_persona_has_definition_but_not_style():
    for k in FALK_CATEGORIES:
        s = persona_prompt(k)
        assert CATEGORY_PARTS[k]["def"] in s and CATEGORY_PARTS[k]["name"] in s
        assert CATEGORY_PARTS[k]["style"] not in s and CATEGORY_PARTS[k]["need"] not in s
    assert persona_prompt(NEUTRAL) == "You are visiting an art museum."
