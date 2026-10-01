import numpy as np
import pytest

from museumbot.rq1_embeddings.ablation import cos_with_full, factorial_effects, steering


def synthetic(rng, A=10, C=6, D=8, scale=1.0):
    """Ogni opera ha un offset proprio; ogni condizione uno shift fisso scalato."""
    art = rng.normal(size=(A, 1, D))
    shifts = rng.normal(size=(1, C, D)) * scale
    return art + shifts + rng.normal(size=(A, C, D)) * 0.01, shifts[0]


def test_steering_recovers_shifts_up_to_condition_mean():
    rng = np.random.default_rng(0)
    X, shifts = synthetic(rng)
    V = steering(X)
    expected = shifts - shifts.mean(axis=0, keepdims=True)
    assert np.allclose(V, expected, atol=0.02)


def test_cos_with_full_is_one_for_identical():
    rng = np.random.default_rng(1)
    X, _ = synthetic(rng)
    V = steering(X)
    assert np.allclose(cos_with_full(V, V), 1.0)


def test_cos_with_full_drops_when_shift_removed():
    rng = np.random.default_rng(2)
    X_full, _ = synthetic(rng, scale=1.0)
    X_half, _ = synthetic(np.random.default_rng(2), scale=0.0)  # stessa opera, nessuno shift
    c = cos_with_full(steering(X_half), steering(X_full))
    assert np.all(np.abs(c) < 0.6)


def test_factorial_main_effects_signs():
    # metrica = 1 se style presente, 0 altrimenti -> effetto di style = 1, altri = 0
    table = {"full": 1, "no_def": 1, "no_need": 1, "no_style": 0,
             "def_only": 0, "need_only": 0, "style_only": 1, "name_only": 0}
    eff = factorial_effects(table)
    assert eff["main"]["style"] == pytest.approx(1.0)
    assert eff["main"]["def"] == pytest.approx(0.0)
    assert eff["main"]["need"] == pytest.approx(0.0)
    assert all(abs(v) < 1e-9 for v in eff["interaction"].values())


def test_factorial_interaction():
    # metrica = 1 solo se def E need presenti -> interazione def×need positiva
    table = {"full": 1, "no_def": 0, "no_need": 0, "no_style": 1,
             "def_only": 0, "need_only": 0, "style_only": 0, "name_only": 0}
    eff = factorial_effects(table)
    # (d,n)=1 ; (d,¬n)=0 ; (¬d,n)=0 ; (¬d,¬n)=0  ->  1 - 0 - 0 + 0
    assert eff["interaction"]["def×need"] == pytest.approx(1.0)
    assert eff["main"]["def"] == pytest.approx(0.5)
