import math

import numpy as np
import pytest

from museumbot.generation.mixing import (
    extra_diag, minority, minority_q, mix_generate, mix_top, sample_token, vetoed,
)


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


def test_minority_q_zero_at_ends_and_middle_and_symmetric():
    assert minority_q(0.0, 4) == minority_q(1.0, 4) == minority_q(0.5, 4) == 0.0
    assert minority_q(0.25, 4) == pytest.approx(minority_q(0.75, 4))
    q = minority_q(0.25, 4)
    s = 1 / (1 + math.exp(0.5 * 4))
    assert (1 - q) * s + q == pytest.approx(0.25)  # la minoranza vince m a distanza d*
    assert q == pytest.approx(0.148, abs=1e-3)


def test_minority_index():
    assert minority([0.75, 0.25]) == 1 and minority([0.25, 0.75]) == 0
    assert minority([0.5, 0.5]) is None and minority([1.0, 0.0]) is None


def test_mix_generate_q_zero_unchanged():
    a = mix_generate([[[10, 1], [20, 1]]], [[0.7, 0.3]], fake_step, eos={EOS}, max_tokens=6,
                     temp=0.7, seeds=[4])
    b = mix_generate([[[10, 1], [20, 1]]], [[0.7, 0.3]], fake_step, eos={EOS}, max_tokens=6,
                     temp=0.7, seeds=[4], qs=[0.0])
    assert a == b and "solo" not in a[0][1][0]


def test_mix_generate_q_one_lets_minority_decide():
    # q = 1: decide sempre la minoranza (esperto 20, peso 0.1), che vuole scendere
    (toks, diag), = mix_generate([[[10, 1], [20, 1]]], [[0.9, 0.1]], fake_step, eos={EOS},
                                 max_tokens=3, temp=0.7, seeds=[0], qs=[1.0])
    assert toks == [49, 48, 47] and all(d["solo"] for d in diag)
    # la maggioranza (esperto 10) vuole salire: nella miscela il token della minoranza e' raro
    assert all(math.exp(d["pmix"]) < 0.05 for d in diag)


def test_extra_diag_reads_mix_logprob_of_token():
    d = extra_diag(8, True, [7, 8], np.log(np.array([0.9, 0.1])))
    assert d == {"solo": True, "pmix": round(math.log(0.1), 4)}


def test_mix_generate_q_needs_two_unequal_experts():
    with pytest.raises(ValueError):
        mix_generate([[[10, 1], [20, 1]]], [[0.5, 0.5]], fake_step, eos={EOS}, max_tokens=3,
                     temp=0.7, seeds=[0], qs=[0.2])


def test_vetoed_keeps_only_tokens_the_majority_lists():
    low, high = lp({1: 0.6, 2: 0.3, 3: 0.1}), lp({2: 0.5, 3: 0.4, 4: 0.1})
    assert set(vetoed([high, low], 1)) == {2, 3}
    assert vetoed([high, low], 1)[2] == low[2]  # valori della minoranza


def veto_step(sequences):
    """Maggioranza (10) e minoranza (20) hanno in comune solo il token 7; poi chiudono."""
    out = []
    for seq in sequences:
        if len(seq) > 2:
            out.append(lp({EOS: 1.0}))
        elif seq[0] == 10:
            out.append(lp({5: 0.9, 7: 0.1}))
        else:
            out.append(lp({6: 0.95, 7: 0.05}))
    return out


def test_mix_generate_veto_restricts_minority_to_shared_tokens():
    # q = 1: decide sempre la minoranza; senza veto sceglie quasi sempre 6, con il veto solo 7
    free = [mix_generate([[[10, 1], [20, 1]]], [[0.9, 0.1]], veto_step, eos={EOS},
                         max_tokens=1, temp=1.0, seeds=[s], qs=[1.0])[0][0][0] for s in range(20)]
    veto = [mix_generate([[[10, 1], [20, 1]]], [[0.9, 0.1]], veto_step, eos={EOS},
                         max_tokens=1, temp=1.0, seeds=[s], qs=[1.0], veto=True)[0][0][0]
            for s in range(20)]
    assert 6 in free and set(veto) == {7}


def test_mix_generate_veto_falls_back_to_mix_without_shared_tokens():
    def disjoint(sequences):
        return [lp({EOS: 1.0}) if len(s) > 2 else lp({5: 1.0} if s[0] == 10 else {6: 1.0})
                for s in sequences]
    (toks, diag), = mix_generate([[[10, 1], [20, 1]]], [[0.9, 0.1]], disjoint, eos={EOS},
                                 max_tokens=1, temp=1.0, seeds=[0], qs=[1.0], veto=True)
    assert not diag[0]["solo"]  # passo fatto dalla miscela


def test_debt_weight_grows_with_debt_and_stops_at_half():
    from museumbot.generation.mixing import debt_weight
    assert debt_weight(0.25, 0.25, 0.0) == 0.25 and debt_weight(0.25, 0.25, -1.0) == 0.25
    assert debt_weight(0.25, 0.25, 0.5) == pytest.approx(0.375)
    assert debt_weight(0.25, 0.25, 3.0) == 0.5


def test_debt_update_counts_only_contested_steps():
    from museumbot.generation.mixing import debt_update
    assert debt_update(0.4, 2.9, 0.25) == 0.4 and debt_update(0.4, -2.9, 0.25) == 0.4
    assert debt_update(0.0, -4.0, 0.25) == pytest.approx(0.25)   # vince la maggioranza
    assert debt_update(0.0, 4.0, 0.25) == pytest.approx(-0.75)   # vince la minoranza


def test_mix_generate_debt_zero_or_without_minority_unchanged():
    base = dict(eos={EOS}, max_tokens=6, temp=0.7, seeds=[4])
    a = mix_generate([[[10, 1], [20, 1]]], [[0.7, 0.3]], fake_step, **base)
    assert a == mix_generate([[[10, 1], [20, 1]]], [[0.7, 0.3]], fake_step, **base, debts=[0.0])
    b = mix_generate([[[10, 1], [20, 1]]], [[0.5, 0.5]], fake_step, **base)
    assert b == mix_generate([[[10, 1], [20, 1]]], [[0.5, 0.5]], fake_step, **base, debts=[0.25])


def test_mix_generate_debt_raises_minority_weight_up_to_half():
    from museumbot.generation.mixing import debt_update, debt_weight
    # gli esperti vogliono token diversi a ogni passo: ogni passo e' conteso (|margine| 3.9)
    (toks, diag), = mix_generate([[[10, 1], [20, 1]]], [[0.9, 0.1]], fake_step, eos={EOS},
                                 max_tokens=5, temp=0.7, seeds=[1], debts=[4.0])
    assert diag[0]["wlow"] == pytest.approx(0.1) and diag[0]["debt"] == 0
    debt = 0.0
    for d in diag:  # la diagnostica ripercorre il debito passo per passo
        assert d["debt"] == pytest.approx(debt, abs=1e-4)
        assert d["wlow"] == pytest.approx(debt_weight(0.1, 4.0, debt), abs=1e-4)
        debt = debt_update(debt, d["logp"][1] - d["logp"][0], 0.1)
    assert max(d["wlow"] for d in diag) == 0.5


def test_mix_generate_entry_and_debt_are_alternative():
    with pytest.raises(ValueError):
        mix_generate([[[10, 1], [20, 1]]], [[0.9, 0.1]], fake_step, eos={EOS}, max_tokens=3,
                     temp=0.7, seeds=[0], qs=[0.2], debts=[0.25])
