import pytest

from museumbot.rq2_judge.judge import TokenBudget, build_messages, parse_choice


def test_parse_clean_json():
    assert parse_choice('{"reason": "More vivid.", "choice": "B"}') == ("B", "More vivid.")


def test_parse_json_after_leaked_reasoning():
    content = 'The user wants JSON {"reason": "<x>", "choice": "A"|"B"}. So:\n{"reason": "Calm.", "choice": "a"}'
    assert parse_choice(content) == ("A", "Calm.")


def test_parse_fenced_json():
    assert parse_choice('```json\n{"choice": "A", "reason": "ok"}\n```') == ("A", "ok")


def test_parse_without_valid_choice():
    assert parse_choice("I would pick the first one.") is None
    assert parse_choice('{"reason": "both", "choice": "both"}') is None


def test_budget_from_tokens():
    b = TokenBudget(cap=10.0, price=(1.40, 4.40))
    assert b.add(1_000_000, 1_000_000) == pytest.approx(5.80)
    b2 = TokenBudget(cap=0.01, price=(1.40, 4.40))
    assert b2.add(1000, 100) == pytest.approx((1000 * 1.40 + 100 * 4.40) / 1e6)
    with pytest.raises(RuntimeError):
        b2.add(1_000_000, 0)


def test_messages_put_target_in_requested_position():
    p = {"artwork_id": "Q1", "persona": "recharger", "target": ["Q1", "recharger"],
         "other": ["Q1", "flat"]}
    texts = {("Q1", "recharger"): "CALM TEXT", ("Q1", "flat"): "FLAT TEXT"}
    titles = {"Q1": ("The Starry Night", "Vincent van Gogh")}
    first = build_messages(p, "target_first", texts, titles)[1]["content"]
    second = build_messages(p, "target_second", texts, titles)[1]["content"]
    assert first.index("CALM TEXT") < first.index("FLAT TEXT")
    assert second.index("FLAT TEXT") < second.index("CALM TEXT")
    assert "The Starry Night by Vincent van Gogh" in first
    assert "Recharger" in build_messages(p, "target_first", texts, titles)[0]["content"]
