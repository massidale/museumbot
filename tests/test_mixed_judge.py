from museumbot.rq2_judge.mixed_judge import (
    COHERENCE_Q, build_ab, build_rank, text_index,
)
from museumbot.rq2_judge.mixed_pairs import build_mixed_pairs
from museumbot.rq2_judge.judge import QUESTION

PAIR = ("recharger", "professional_hobbyist")


def local_rows():
    rows = []
    for a in ("Q1", "Q2"):
        for c in ("recharger", "professional_hobbyist", "flat"):
            rows.append({"artwork_id": a, "condition": c, "variant": "full",
                         "title": f"T{a}", "artist": "A", "text": f"{a} {c} full"})
            if c != "flat":
                rows.append({"artwork_id": a, "condition": c, "variant": "full_rep",
                             "title": f"T{a}", "artist": "A", "text": f"{a} {c} rep"})
    return rows


def mixed_rows():
    return [{"artwork_id": a, "condition": "recharger+professional_hobbyist", "alpha": al,
             "text": f"{a} mix {al}", "usage": {"failed": al == 0.75 and a == "Q2"}}
            for a in ("Q1", "Q2") for al in (0.25, 0.5, 0.75)]


def test_text_index_labels_pure_replica_mixed_and_skips_failed():
    texts, titles = text_index(local_rows(), mixed_rows())
    assert texts[("Q1", "recharger")] == "Q1 recharger full"
    assert texts[("Q1", "recharger|rep")] == "Q1 recharger rep"
    assert texts[("Q1", "recharger+professional_hobbyist|0.5")] == "Q1 mix 0.5"
    assert ("Q2", "recharger+professional_hobbyist|0.75") not in texts
    assert titles["Q2"] == ("TQ2", "A")


def test_build_ab_places_target_by_order_and_uses_question():
    texts, titles = text_index(local_rows(), mixed_rows())
    p = next(q for q in build_mixed_pairs(["Q1", "Q2"], [PAIR])
             if q["type"] == "coherence")
    first = build_ab(p, "target_first", texts, titles, COHERENCE_Q)[1]["content"]
    second = build_ab(p, "target_second", texts, titles, COHERENCE_Q)[1]["content"]
    tgt = texts[tuple(p["target"])]
    assert first.index(tgt) < first.index("Audio guide B")
    assert second.index(tgt) > second.index("Audio guide B")
    assert COHERENCE_Q in first and QUESTION not in first


def test_build_rank_lists_five_numbered_texts_and_both_definitions():
    texts, titles = text_index(local_rows(), mixed_rows())
    p = next(q for q in build_mixed_pairs(["Q1"], [PAIR]) if q["type"] == "rank")
    msg = build_rank(p, texts, titles)[1]["content"]
    for k in range(1, 6):
        assert f"Audio guide {k}:" in msg
    assert "Recharger" in msg and "Professional or Hobbyist" in msg
    assert msg.index(texts[tuple(p["texts"][0])]) < msg.index("Audio guide 2:")
