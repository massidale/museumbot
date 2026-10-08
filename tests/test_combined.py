import pytest

from museumbot.rq2_judge.combined import weight_curve


def pure(typ, persona, target_cond, chose, valid=True):
    return {"type": typ, "persona": persona, "target": ["Q1", target_cond],
            "other": ["Q1", "flat"], "chose_target": chose, "valid": valid}


def mixed(persona, pair, alpha, chose):
    return {"type": "double", "persona": persona, "pair": list(pair), "alpha": alpha,
            "chose_target": chose, "valid": True}


def test_weight_curve_joins_pure_and_mixed_by_share_of_own_category():
    P = [pure("A", "x", "x", True), pure("A", "x", "x", True), pure("A", "x", "x", False),
         pure("C", "x", "y", False), pure("C", "x", "z", True),  # z: altro partner, escluso
         pure("A", "y", "y", True, valid=False)]
    M = [mixed("x", ("x", "y"), 0.0, True), mixed("x", ("x", "y"), 0.25, True),
         mixed("x", ("x", "y"), 0.5, False), mixed("x", ("x", "y"), 1.0, False),
         mixed("y", ("x", "y"), 0.25, True)]  # altra persona, esclusa
    c = weight_curve(P, M, "x", "y")
    assert c["pure"][1.0] == (pytest.approx(2 / 3), 3)
    assert c["pure"][0.0] == (0.0, 1)
    assert c["mixed"][1.0] == (1.0, 1)   # alpha 0: testo tutto di x
    assert c["mixed"][0.75] == (1.0, 1)  # alpha 0.25
    assert c["mixed"][0.5] == (0.0, 1)
    assert c["mixed"][0.0] == (0.0, 1)


def test_weight_curve_flips_alpha_when_persona_is_second_of_pair():
    M = [mixed("y", ("x", "y"), 0.25, True), mixed("y", ("x", "y"), 1.0, False)]
    c = weight_curve([], M, "y", "x")
    assert c["mixed"][0.25] == (1.0, 1) and c["mixed"][1.0] == (0.0, 1)
