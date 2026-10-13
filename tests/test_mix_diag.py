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


def test_decile_means_one_row_per_text():
    from museumbot.generation.mix_diag import decile_means

    M = decile_means([np.array([1.0, 1.0, 3.0, 3.0]), np.array([0.0, 2.0])], bins=2)
    assert M.tolist() == [[1.0, 3.0], [0.0, 2.0]]


def test_profile_ci_brackets_the_mean_and_collapses_without_variation():
    from museumbot.generation.mix_diag import profile_ci

    same = [np.array([1.0, 1.0, -1.0, -1.0])] * 20
    mean, lo, hi = profile_ci(same, bins=2, n_boot=200, rng=np.random.default_rng(0))
    assert mean.tolist() == [1.0, -1.0] and lo.tolist() == [1.0, -1.0] and hi.tolist() == [1.0, -1.0]
    rng = np.random.default_rng(1)
    noisy = [rng.normal(0, 1, 50) for _ in range(40)]
    mean, lo, hi = profile_ci(noisy, bins=5, n_boot=500, rng=np.random.default_rng(2))
    assert np.all(lo <= mean) and np.all(mean <= hi) and np.all(hi - lo > 0)
