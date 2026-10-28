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
    assert st == pytest.approx({"n_sentences": 3, "max": 3.0, "min": -1.0, "abs_mean": 2.0,
                                "changes": 2, "expected": 2 * 2 * 1 / 3})


def test_expected_sign_changes_matches_all_orderings():
    from itertools import permutations

    from museumbot.generation.mix_diag import expected_sign_changes, sign_changes

    v = np.array([0.5, -1.0, 2.0, -0.3, 0.0, 1.0])
    brute = np.mean([sign_changes(p) for p in permutations(v)])
    assert expected_sign_changes(v) == pytest.approx(brute)
    assert expected_sign_changes(np.array([1.0, 2.0])) == 0.0
