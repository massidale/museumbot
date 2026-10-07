import math

import numpy as np
import pytest

from museumbot.generation.mixing import mix_generate, mix_top, sample_token


def lp(d):
    """Log-probabilita' normalizzate da probabilita'."""
    return {t: math.log(p) for t, p in d.items()}


def test_mix_top_one_hot_is_tempered_expert():
    a = lp({1: 0.5, 2: 0.3, 3: 0.2})
    toks, mixed, _ = mix_top([a, lp({4: 1.0})], [1.0, 0.0], temp=0.5)
    p = dict(zip(toks, np.exp(mixed)))
    want = np.array([0.5, 0.3, 0.2]) ** 2
    want /= want.sum()
    assert set(p) == {1, 2, 3}  # l'esperto a peso zero non entra
    assert [p[1], p[2], p[3]] == pytest.approx(want)


def test_mix_top_geometric_mean_over_union_with_floor():
    a = lp({1: 0.6, 2: 0.4})
    b = lp({2: 0.7, 3: 0.3})
    toks, mixed, cov = mix_top([a, b], [0.5, 0.5], temp=1.0)
    p = dict(zip(toks, np.exp(mixed)))
    # token assente da un esperto: prende il minimo della sua lista (stima per eccesso)
    raw = {1: 0.5 * math.log(0.6) + 0.5 * math.log(0.3),
           2: 0.5 * math.log(0.4) + 0.5 * math.log(0.7),
           3: 0.5 * math.log(0.4) + 0.5 * math.log(0.3)}
    z = math.log(sum(math.exp(v) for v in raw.values()))
    assert p[2] == pytest.approx(math.exp(raw[2] - z))
    assert math.isclose(sum(p.values()), 1.0)
    assert cov == {1: [True, False], 2: [True, True], 3: [False, True]}


def test_mix_top_rejects_bad_weights():
    with pytest.raises(ValueError):
        mix_top([lp({1: 1.0})], [0.7], temp=1.0)


def test_sample_token_reproducible_and_follows_mass():
    toks, mixed = [7, 8], np.log(np.array([0.999, 0.001]))
    r1 = [sample_token(toks, mixed, np.random.default_rng(3)) for _ in range(5)]
    r2 = [sample_token(toks, mixed, np.random.default_rng(3)) for _ in range(5)]
    assert r1 == r2 and r1[0] == 7


EOS = 0


def fake_step(sequences):
    """Esperto = primo token del prompt: 10 vuole salire, 20 vuole scendere; dopo 6 token
    generati entrambi chiudono."""
    out = []
    for seq in sequences:
        expert, gen = seq[0], len(seq) - 2  # prompt di 2 token
        last = seq[-1] if gen else 50
        nxt = last + 1 if expert == 10 else last - 1
        out.append(lp({EOS: 0.98, nxt: 0.02}) if gen >= 6 else lp({nxt: 0.98, 99: 0.02}))
    return out


def test_mix_generate_one_hot_follows_single_expert_and_stops():
    texts = mix_generate([[[10, 1], [20, 1]]], [[1.0, 0.0]], fake_step, eos={EOS},
                         max_tokens=50, temp=0.7, seeds=[0])
    toks, diag = texts[0]
    assert toks[:3] == [51, 52, 53] and EOS not in toks and len(toks) == 6
    assert len(diag) == len(toks)


def test_mix_generate_runs_texts_in_parallel_and_respects_max_tokens():
    texts = mix_generate([[[10, 1]], [[20, 1]]], [[1.0], [1.0]], fake_step, eos={EOS},
                         max_tokens=3, temp=0.7, seeds=[0, 1])
    assert texts[0][0] == [51, 52, 53] and texts[1][0] == [49, 48, 47]


def test_mix_generate_feeds_same_token_to_both_experts():
    seen = []

    def spy(seqs):
        seen.append([list(s) for s in seqs])
        return fake_step(seqs)

    mix_generate([[[10, 1], [20, 1]]], [[0.5, 0.5]], spy, eos={EOS}, max_tokens=3,
                 temp=0.7, seeds=[2])
    for call in seen[1:]:
        assert call[0][2:] == call[1][2:]  # stesso testo generato dopo i due prompt


def test_mix_generate_diag_records_coverage_and_divergence():
    (toks, diag), = mix_generate([[[10, 1], [20, 1]]], [[0.5, 0.5]], fake_step, eos={EOS},
                                 max_tokens=4, temp=0.7, seeds=[5])
    assert all(set(d) == {"logp", "covered", "div"} for d in diag)
    assert all(len(d["logp"]) == 2 and d["div"] >= 0 for d in diag)
