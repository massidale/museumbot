import numpy as np
import pytest

from museumbot.generation.mix_diag import (
    sentence_stats, sign_changes, split_sentences,
)

WORDS = {1: "Look", 2: " at", 3: " it", 4: ".", 5: " Now", 6: " rest", 7: " here", 8: "!"}


def decode(ids):
    return "".join(WORDS[i] for i in ids)


def test_split_sentences_cuts_after_closing_punctuation():
    ids = [1, 2, 3, 4, 5, 6, 7, 8]
    m = np.arange(8.0)
    out = split_sentences(ids, m, decode, min_tokens=2)
    assert [s.tolist() for s in out] == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_split_sentences_keeps_trailing_tokens():
    out = split_sentences([1, 2, 4, 5, 6], np.ones(5), decode, min_tokens=2)
    assert [len(s) for s in out] == [3, 2]


def test_sign_changes_ignores_zeros():
    assert sign_changes(np.array([1.0, 2.0, -1.0, 0.0, -3.0, 4.0])) == 2


def test_sentence_stats_on_sentence_sums():
    # frasi: somme +3, -1, +2
    st = sentence_stats([np.array([1.0, 2.0]), np.array([-1.0]), np.array([0.5, 1.5])])
    assert st == pytest.approx({"n_sentences": 3, "max": 3.0, "min": -1.0, "mean": 4 / 3,
                                "changes": 2, "expected": 2 * 2 * 1 / 3})


def test_expected_sign_changes_matches_all_orderings():
    from itertools import permutations

    from museumbot.generation.mix_diag import expected_sign_changes, sign_changes

    v = np.array([0.5, -1.0, 2.0, -0.3, 0.0, 1.0])
    brute = np.mean([sign_changes(p) for p in permutations(v)])
    assert expected_sign_changes(v) == pytest.approx(brute)
    assert expected_sign_changes(np.array([1.0, 2.0])) == 0.0


WORDS2 = {1: "It", 2: " measures", 3: " 81", 4: ".", 5: "5", 6: " cm", 7: " Then", 8: " St",
          9: " Peter", 10: " W", 11: " Mellon", 12: " gave", 13: " it", 14: "!"}


def decode2(ids):
    return "".join(WORDS2[i] for i in ids)


def test_split_sentences_keeps_decimals_together():
    ids = [1, 2, 3, 4, 5, 6, 4, 7, 13, 14]  # "It measures 81.5 cm. Then it!"
    out = split_sentences(ids, np.arange(10.0), decode2, min_tokens=2)
    assert [len(s) for s in out] == [7, 3]


def test_split_sentences_skips_abbreviations_and_initials():
    ids = [7, 8, 4, 9, 2, 13, 4, 10, 4, 11, 12, 13, 14]  # "Then St. Peter measures it. W. Mellon gave it!"
    out = split_sentences(ids, np.arange(13.0), decode2, min_tokens=2)
    assert [len(s) for s in out] == [7, 6]


def test_scored_margins_are_expert_difference_at_alpha():
    from museumbot.generation.mix_diag import scored_margins

    rows = [{"condition": "a+b", "alpha": 0.5, "ids": [1, 4], "logp": [[-1.0, -2.0], [-3.0, -1.0]]},
            {"condition": "a+b", "alpha": 0.25, "ids": [1], "logp": [[0.0], [0.0]]}]
    out = list(scored_margins(rows, 0.5))
    assert len(out) == 1 and out[0][0] == "a+b" and out[0][1] == [1, 4]
    assert out[0][2].tolist() == [2.0, -1.0]


def test_pair_stats_averages_texts_per_pair():
    from museumbot.generation.mix_diag import pair_stats

    items = [("a+b", [1, 2, 3, 4, 5, 6, 7, 8], np.array([1.0] * 4 + [-1.0] * 4)),
             ("a+b", [1, 2, 3, 4, 5, 6, 7, 8], np.array([2.0] * 8))]
    out = pair_stats(items, decode)["a+b"]
    assert out["n"] == 2 and out["sentences_mean"] == 2.0
    assert out["sentence_sum_max_mean"] == 6.0 and out["sign_changes_observed"] == 0.5


def test_result_name_and_diag_only_for_mix():
    from museumbot.generation.mix_diag import result_name, run

    assert result_name("mix", "diag") == "mix_diag"
    assert result_name("pct", "scores") == "mix_diag_pct_scores"
    with pytest.raises(ValueError):
        run(method="pct", source="diag")
