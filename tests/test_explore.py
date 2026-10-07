import json

import numpy as np
import pandas as pd
import pytest

from museumbot.common.config import ROOT
from museumbot.common.prompts import CONDITIONS, FALK_CATEGORIES
from museumbot.explore import (
    build,
    format_df,
    prompt_rows,
    query,
    score_rows,
    text_rows,
)

C = len(CONDITIONS)
FLAT = CONDITIONS.index("flat")


def exact(rng, A=6, D=8):
    """Offset per opera + shift fisso per condizione, senza rumore: ogni testo centrato
    sul flat coincide con lo steering vector della sua categoria."""
    shifts = rng.normal(size=(C, D))
    X = rng.normal(size=(A, 1, D)) * 5 + shifts[None]
    return X, shifts


def test_score_rows_cos_own_is_one_without_noise():
    X, shifts = exact(np.random.default_rng(0))
    rows = score_rows(X, [f"Q{i}" for i in range(len(X))], "qwen")
    assert all(r["cos_own"] == pytest.approx(1.0) for r in rows)
    assert all(r["model"] == "qwen" for r in rows)


def test_score_rows_cos_with_other_category_matches_steering_geometry():
    X, shifts = exact(np.random.default_rng(1))
    rows = score_rows(X, [f"Q{i}" for i in range(len(X))], "qwen")
    v = shifts - shifts[FLAT]
    ie, ir = CONDITIONS.index("explorer"), CONDITIONS.index("recharger")
    want = v[ie] @ v[ir] / (np.linalg.norm(v[ie]) * np.linalg.norm(v[ir]))
    r = next(r for r in rows if r["artwork_id"] == "Q0" and r["condition"] == "explorer")
    assert r["cos_recharger"] == pytest.approx(want)
    assert r["norm"] == pytest.approx(np.linalg.norm(v[ie]))


def test_score_rows_skips_flat_and_covers_every_category():
    X, _ = exact(np.random.default_rng(2), A=4)
    rows = score_rows(X, ["a", "b", "c", "d"], "bge-m3")
    assert len(rows) == 4 * len(FALK_CATEGORIES)
    assert {r["condition"] for r in rows} == set(FALK_CATEGORIES)


def test_prompt_rows_cover_variants_and_replicate_reuses_full():
    rows = {(r["condition"], r["variant"]): r["system_prompt"] for r in prompt_rows()}
    assert "Your listener is an Explorer: a curiosity-driven" in rows[("explorer", "full")]
    assert rows[("explorer", "full_rep")] == rows[("explorer", "full")]
    assert rows[("explorer", "name_only")].count("Your listener is an Explorer.") == 1
    assert "Your listener" not in rows[("flat", "full")]


def study_row(a, c, method, text="Look at the sky. It is wide."):
    return {"artwork_id": a, "title": "T", "artist": "A", "condition": c, "method": method,
            "model": "m", "text": text, "words": len(text.split()), "usage": {}}


def test_text_rows_add_corpus_and_unique_id(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        study_row("Q1", "explorer", "single_a"), study_row("Q1", "explorer", "single_b"),
        study_row("Q1", "flat", "flat")]) + "\n")
    rows = text_rows("main", p)
    assert all(r["corpus"] == "main" for r in rows)
    ids = [r["text_id"] for r in rows]
    assert len(set(ids)) == 3
    assert "main|full_rep|Q1|explorer" in ids


def test_format_df_prints_long_text_in_full():
    long = "word " * 300
    df = pd.DataFrame([{"condition": "explorer", "cos_own": 0.123456, "text": long}])
    out = format_df(df)
    assert long.strip() in out
    assert "0.123" in out and "explorer" in out


@pytest.mark.skipif(not (ROOT / "data" / "emb" / "main").exists(), reason="dati assenti")
def test_build_then_query_joins_texts_and_scores(tmp_path):
    db = tmp_path / "explore.duckdb"
    build(db, corpora=("main",), models=("qwen",))
    df = query(db, """
        SELECT t.condition, s.cos_own, t.text, p.system_prompt
        FROM scores s JOIN texts t USING (text_id)
        JOIN prompts p ON p.condition = t.condition AND p.variant = t.variant
        WHERE t.condition = 'recharger' AND t.variant = 'full'
        ORDER BY s.cos_own DESC LIMIT 3""")
    assert len(df) == 3 and df["cos_own"].is_monotonic_decreasing
    assert df["system_prompt"].str.contains("Recharger").all()
    n = query(db, "SELECT count(*) AS n FROM judgments")["n"][0]
    assert n > 0
