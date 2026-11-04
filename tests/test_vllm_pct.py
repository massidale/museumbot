from museumbot.generation.rows import row_seed
from museumbot.generation.vllm_gen import generate_rows
from museumbot.generation.vllm_mix import mix_key, mix_seed
from museumbot.generation.vllm_pct import make_pct_row, pct_seed

ART = {"id": "Q1", "title": "T", "artist": "A", "source_text": "S"}
PAIR = ("recharger", "professional_hobbyist")
OK = "Breathe slowly. Look at the brushwork."


def test_pct_seed_differs_from_mix_seed():
    assert pct_seed("Q1", PAIR, 0.5, 0) == row_seed("Q1", "recharger+professional_hobbyist",
                                                    "p0.5", 0)
    assert pct_seed("Q1", PAIR, 0.5, 0) != mix_seed("Q1", PAIR, 0.5, 0)


def test_make_pct_row_fields():
    r = make_pct_row(ART, PAIR, 0.25, OK, 0, {"completion_tokens": 7})
    assert r["variant"] == "pct" and r["alpha"] == 0.25 and r["pair"] == list(PAIR)
    assert r["weights"] == {"recharger": 0.75, "professional_hobbyist": 0.25}
    assert r["condition"] == "recharger+professional_hobbyist"
    assert r["seed"] == pct_seed("Q1", PAIR, 0.25, 0)
    assert r["usage"]["completion_tokens"] == 7 and r["text"] == OK
    assert "diag" not in r["usage"] and mix_key(r) == ("Q1", "recharger+professional_hobbyist", 0.25)


def test_generate_rows_retries_pct_with_new_seed():
    answers = {0: "Ideal for those who rest.", 1: OK}
    seen = []

    def gen(reqs):
        seen.append([(a["id"], p, al, k) for a, p, al, k in reqs])
        return [(answers[k], {}) for a, p, al, k in reqs]

    (r,) = generate_rows([(ART, PAIR, 0.5)], gen, make=make_pct_row)
    assert seen == [[("Q1", PAIR, 0.5, 0)], [("Q1", PAIR, 0.5, 1)]]
    assert r["usage"]["attempts"] == 2 and not r["usage"]["failed"]
    assert r["seed"] == pct_seed("Q1", PAIR, 0.5, 1) and r["text"] == OK


def test_known_violations_rebuilds_last_attempt_for_old_rows():
    from museumbot.generation.vllm_pct import known_violations

    old = {"usage": {"attempts": 12, "raw": "For those who like it."}}
    assert known_violations(old) == [{"attempt": 12, "matches": ["For those"]}]
    new = {"usage": {"attempts": 3, "raw": OK, "violations": [{"attempt": 1, "matches": ["x"]}]}}
    assert known_violations(new) == [{"attempt": 1, "matches": ["x"]}]
