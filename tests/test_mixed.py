import numpy as np
import pytest

from museumbot.rq1_embeddings.mixed import position, residual

VX = np.array([1.0, 0.0, 0.0])
VY = np.array([0.0, 1.0, 0.0])


def test_position_is_zero_at_x_one_at_y_and_half_in_between():
    assert position(VX, VX, VY) == pytest.approx(0.0)
    assert position(VY, VX, VY) == pytest.approx(1.0)
    assert position(0.5 * VX + 0.5 * VY, VX, VY) == pytest.approx(0.5)


def test_position_ignores_components_off_the_segment():
    d = 0.25 * VX + 0.75 * VY + np.array([0.0, 0.0, 3.0])
    assert position(d, VX, VY) == pytest.approx(0.75)


def test_residual_zero_in_plane_and_relative_off_plane():
    assert residual(0.3 * VX + 2.0 * VY, VX, VY) == pytest.approx(0.0, abs=1e-12)
    d = np.array([3.0, 0.0, 4.0])
    assert residual(d, VX, VY) == pytest.approx(4.0 / 5.0)


def test_functions_work_row_wise_on_matrices():
    D = np.stack([VX, VY, 0.5 * (VX + VY)])
    assert position(D, VX, VY) == pytest.approx([0.0, 1.0, 0.5])
    assert residual(D, VX, VY) == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
