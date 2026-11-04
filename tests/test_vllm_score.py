from types import SimpleNamespace

import pytest

from museumbot.generation.vllm_score import score_key, score_rows, text_logprobs

ROW = {"artwork_id": "Q1", "condition": "explorer+facilitator", "pair": ["explorer", "facilitator"],
       "alpha": 0.5, "variant": "pct", "usage": {"raw": "Look.", "failed": False}}


def lp(x):
    return SimpleNamespace(logprob=x)


def test_text_logprobs_reads_actual_token_after_prompt():
    # prompt di 2 token, testo [7, 8]: posizioni 2 e 3
    plp = [None, {5: lp(-9.0)}, {7: lp(-0.5), 1: lp(-0.1)}, {8: lp(-2.0)}]
    assert text_logprobs(plp, [7, 8], start=2) == [-0.5, -2.0]


def test_text_logprobs_fails_without_actual_token():
    with pytest.raises(KeyError):
        text_logprobs([None, {1: lp(-1.0)}], [7], start=1)


def test_score_rows_one_prompt_per_expert_and_same_text():
    calls = []

    def score(jobs):
        calls.extend(jobs)
        return [[-1.0 * (k + 1)] * len(t) for k, (_, t) in enumerate(jobs)]

    prompt = lambda cond, row: [100 if cond == "explorer" else 200]
    out = score_rows([ROW], prompt, lambda raw: [7, 8], score)
    assert [p for p, _ in calls] == [[100], [200]] and all(t == [7, 8] for _, t in calls)
    r = out[0]
    assert r["ids"] == [7, 8] and r["logp"] == [[-1.0, -1.0], [-2.0, -2.0]]
    assert score_key(r) == ("Q1", "explorer+facilitator", 0.5, "pct")
