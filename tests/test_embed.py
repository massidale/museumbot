import pytest

from museumbot.rq1_embeddings.embed import select_rows


def row(a, c, v, model="m"):
    return {"artwork_id": a, "condition": c, "variant": v, "model": model,
            "title": "t", "artist": "x", "words": 1, "text": "hi"}


def test_select_rows_full_treats_missing_variant_as_full():
    rows = [{"artwork_id": "Q1", "condition": "flat", "model": "m", "title": "t",
             "artist": "x", "words": 1, "text": "hi"}, row("Q1", "explorer", "full")]
    out = select_rows(rows, "m", "full")
    assert [(r["artwork_id"], r["condition"]) for r in out] == [("Q1", "explorer"), ("Q1", "flat")]


def test_select_rows_variant_borrows_flat_from_full():
    rows = [row("Q1", "flat", "full"), row("Q1", "explorer", "full"),
            row("Q1", "explorer", "no_style")]
    out = select_rows(rows, "m", "no_style")
    assert {(r["condition"], r["variant"]) for r in out} == {("flat", "full"), ("explorer", "no_style")}


def test_select_rows_rejects_duplicates():
    rows = [row("Q1", "explorer", "full"), row("Q1", "explorer", "full")]
    with pytest.raises(SystemExit):
        select_rows(rows, "m", "full")


def test_select_rows_rejects_other_model():
    assert select_rows([row("Q1", "explorer", "full", model="other")], "m", "full") == []
