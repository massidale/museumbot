import json

import numpy as np
import pytest

from museumbot.rq1_embeddings import mixed
from museumbot.rq1_embeddings.mixed import load_rows, plot_compare, position, residual, result_name

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


def test_result_name():
    assert result_name("qwen", "mix") == "mixed_qwen"
    assert result_name("bge-m3", "pct") == "mixed_pct_bge-m3"


def test_load_rows_drops_failed(tmp_path, monkeypatch):
    f = tmp_path / "rows.jsonl"
    ok = {"artwork_id": "a", "condition": "x+y", "alpha": 0.5, "text": "t", "usage": {"failed": False}}
    bad = {**ok, "artwork_id": "b", "usage": {"failed": True}}
    f.write_text("\n".join(json.dumps(r) for r in (bad, ok)) + "\n")
    monkeypatch.setitem(mixed.SOURCES, "mix", (f, tmp_path))
    assert [r["artwork_id"] for r in load_rows("mix")] == ["a"]


def test_plot_compare_skips_when_a_json_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(mixed, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(mixed, "FIGS", tmp_path / "figures")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "mixed_qwen.json").write_text(json.dumps({"pairs": {}}))
    plot_compare("qwen")  # manca mixed_pct_qwen.json
    assert not (tmp_path / "figures").exists()
