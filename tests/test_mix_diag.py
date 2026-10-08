import numpy as np

from museumbot.generation.mix_diag import (
    ownership_ratio, positional_profile, sign_changes, split_sentences,
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


def test_positional_profile_averages_by_relative_position():
    prof = positional_profile([np.array([1.0, 1.0, 3.0, 3.0])], bins=2)
    assert prof.tolist() == [1.0, 3.0]


def test_ownership_ratio_high_for_blocks_and_near_one_for_noise():
    rng = np.random.default_rng(0)
    blocks = [np.full(10, s) + rng.normal(0, 0.1, 10) for s in (1, -1, 1, -1, 1, -1)]
    noise = [rng.normal(0, 1, 10) for _ in range(6)]
    assert ownership_ratio(blocks, np.random.default_rng(1)) > 1.5
    assert 0.7 < ownership_ratio(noise, np.random.default_rng(1)) < 1.3
