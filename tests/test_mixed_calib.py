import numpy as np
import pytest

from museumbot.rq1_embeddings.mixed_calib import (
    compose, equivalent_share, glue, interleave, split_sentences, summarize,
)


def test_split_sentences_keeps_abbreviations_and_decimals():
    text = "St. Jerome sits here. It is 81.5 cm wide! Painted by J. Smith? Yes."
    assert split_sentences(text) == ["St. Jerome sits here.", "It is 81.5 cm wide!",
                                     "Painted by J. Smith?", "Yes."]


def test_split_sentences_keeps_closing_quote_and_unfinished_tail():
    assert split_sentences('He said "look." Then\nwe go') == ['He said "look."', "Then\nwe go"]


S = [f"Sentence number {i} is here." for i in range(10)]


@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75])
def test_compose_share_is_close_to_alpha(alpha):
    px, py = compose(S, S, alpha)
    assert px == S[: len(px)] and py == S[len(S) - len(py):]
    _, eff = glue(px, py, "XY")
    assert eff == pytest.approx(alpha, abs=0.06)


def test_compose_keeps_at_least_one_sentence_per_piece():
    px, py = compose(["Only one."], ["A.", "B."], 0.9)
    assert len(px) >= 1 and len(py) >= 1


def test_glue_orders_use_the_same_pieces():
    xy, e1 = glue(["X one.", "X two."], ["Y one."], "XY")
    yx, e2 = glue(["X one.", "X two."], ["Y one."], "YX")
    assert xy == "X one. X two. Y one." and yx == "Y one. X one. X two."
    assert e1 == e2


def test_summarize_medians_by_alpha():
    alpha = np.repeat([0.25, 0.5, 0.75], 3)
    out = summarize(alpha + 0.05, alpha, alpha)
    assert out["by_alpha"]["0.5"]["position_median"] == pytest.approx(0.55)
    assert out["by_alpha"]["0.25"]["n"] == 3 and out["spearman_alpha_eff_position"] > 0.9


def test_interleave_spreads_the_shorter_piece():
    assert interleave(["x1", "x2", "x3", "x4"], ["y1"]) == ["x1", "x2", "y1", "x3", "x4"]
    assert interleave(["x1", "x2"], ["y1", "y2"]) == ["x1", "y1", "x2", "y2"]


def test_glue_alt_keeps_content_and_share():
    px, py = ["X one.", "X two."], ["Y one."]
    alt, e = glue(px, py, "alt")
    assert alt == "X one. Y one. X two."
    assert sorted(alt.split()) == sorted(glue(px, py, "XY")[0].split())
    assert e == glue(px, py, "XY")[1]


def test_equivalent_share_interpolates_and_flags_out_of_range():
    by = {"0.25": {"alpha_eff_mean": 0.25, "position_median": 0.4},
          "0.5": {"alpha_eff_mean": 0.5, "position_median": 0.6},
          "0.75": {"alpha_eff_mean": 0.75, "position_median": 0.55}}
    assert equivalent_share(0.5, by) == pytest.approx(0.375)
    assert equivalent_share(0.6, by) == pytest.approx(0.5)
    assert equivalent_share(0.1, by) is None and equivalent_share(0.9, by) is None
