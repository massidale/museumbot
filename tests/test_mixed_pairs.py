from collections import Counter

from museumbot.rq2_judge.mixed_pairs import (
    ALPHAS, NEUTRAL, build_mixed_pairs, parse_order, series,
)

PAIR = ("recharger", "professional_hobbyist")
ARTS = [f"Q{i}" for i in range(10)]


def test_series_uses_pure_vertices_and_replica_against_same_category():
    assert series(PAIR, 0.0, side="x") == "recharger"
    assert series(PAIR, 1.0, side="x") == "professional_hobbyist|rep"
    assert series(PAIR, 0.0, side="y") == "recharger|rep"
    assert series(PAIR, 1.0, side="y") == "professional_hobbyist"
    assert series(PAIR, 0.5, side="x") == "recharger+professional_hobbyist|0.5"
    assert series(PAIR, 0.0) == "recharger" and series(PAIR, 1.0) == "professional_hobbyist"


def test_curve_pairs_compare_with_the_other_pure_text():
    ps = [p for p in build_mixed_pairs(ARTS, [PAIR]) if p["type"] == "curve"]
    assert len(ps) == len(ARTS) * len(ALPHAS) * 2
    px = [p for p in ps if p["persona"] == "recharger"]
    assert all(p["other"][1] == "professional_hobbyist" for p in px)
    py = [p for p in ps if p["persona"] == "professional_hobbyist"]
    assert all(p["other"][1] == "recharger" for p in py)


def test_double_visitor_against_flat_for_both_personas():
    ps = [p for p in build_mixed_pairs(ARTS, [PAIR]) if p["type"] == "double"]
    assert len(ps) == len(ARTS) * len(ALPHAS) * 2
    assert {p["other"][1] for p in ps} == {"flat"}
    assert {p["persona"] for p in ps} == set(PAIR)


def test_coherence_compares_inner_alphas_with_both_pures_neutral():
    ps = [p for p in build_mixed_pairs(ARTS, [PAIR]) if p["type"] == "coherence"]
    assert len(ps) == len(ARTS) * 3 * 2
    assert {p["persona"] for p in ps} == {NEUTRAL}
    assert {p["other"][1] for p in ps} == set(PAIR)


def test_rank_items_shuffle_five_texts_reproducibly():
    a = [p for p in build_mixed_pairs(ARTS, [PAIR]) if p["type"] == "rank"]
    b = [p for p in build_mixed_pairs(ARTS, [PAIR]) if p["type"] == "rank"]
    assert len(a) == len(ARTS) and a == b
    assert sorted(a[0]["alphas"]) == list(ALPHAS) and len(a[0]["texts"]) == 5
    assert len({tuple(p["alphas"]) for p in a}) > 1  # ordini diversi fra opere


def test_orders_are_balanced_and_ids_unique():
    ps = [p for p in build_mixed_pairs(ARTS, [PAIR]) if p["type"] != "rank"]
    assert len({p["pair_id"] for p in ps}) == len(ps)
    c = Counter(p["order"] for p in ps)
    assert abs(c["target_first"] - c["target_second"]) <= 0.15 * len(ps)


def test_attention_once_per_artwork_neutral():
    ps = [p for p in build_mixed_pairs(ARTS, [PAIR, ("explorer", "facilitator")])
          if p["type"] == "attention"]
    assert len(ps) == len(ARTS) and {p["persona"] for p in ps} == {NEUTRAL}
    assert all(p["target"][0] == p["artwork_id"] != p["other"][0] for p in ps)


def test_parse_order_accepts_permutations_only():
    assert parse_order('ok {"reason": "r", "order": [3, 1, 5, 2, 4]}', 5) == ([3, 1, 5, 2, 4], "r")
    assert parse_order('{"order": [1, 1, 2, 3, 4]}', 5) is None
    assert parse_order('{"order": [1, 2, 3]}', 5) is None
    assert parse_order("nessun json", 5) is None
