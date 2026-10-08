from museumbot.generation.rows import row_seed
from museumbot.generation.vllm_gen import generate_rows
from museumbot.generation.vllm_mix import make_mix_row, mix_seed, plan_mix_jobs

ART = {"id": "Q1", "title": "T", "artist": "A", "source_text": "S"}
PAIR = ("recharger", "professional_hobbyist")
OK = "Breathe slowly. Look at the brushwork."
DIAG = [{"logp": [-0.1, -2.0], "covered": [True, True], "div": 0.2},
        {"logp": [-0.3, -0.4], "covered": [True, False], "div": 0.0}]


def test_mix_seed_depends_on_pair_alpha_and_attempt():
    s = mix_seed("Q1", PAIR, 0.5, 0)
    assert s == row_seed("Q1", "recharger+professional_hobbyist", "a0.5", 0)
    assert s != mix_seed("Q1", PAIR, 0.25, 0) != mix_seed("Q1", PAIR, 0.5, 1)


def test_make_mix_row_records_weights_and_diagnostics():
    r = make_mix_row(ART, PAIR, 0.25, OK, 0, {"diag": DIAG, "completion_tokens": 2})
    assert r["weights"] == {"recharger": 0.75, "professional_hobbyist": 0.25}
    assert r["pair"] == list(PAIR) and r["alpha"] == 0.25 and r["variant"] == "mix"
    assert r["condition"] == "recharger+professional_hobbyist"
    assert r["seed"] == mix_seed("Q1", PAIR, 0.25, 0)
    u = r["usage"]
    assert u["covered_all"] == 0.5 and u["div_mean"] == 0.1 and u["diag"] == DIAG
    assert r["text"] == OK and not u["failed"]


def test_plan_mix_jobs_skips_done():
    done = {("Q1", "recharger+professional_hobbyist", 0.5)}
    jobs = plan_mix_jobs([ART, {**ART, "id": "Q2"}], [PAIR], [0.0, 0.5], done)
    assert ("Q1", 0.5) not in [(a["id"], al) for a, _, al in jobs]
    assert len(jobs) == 3


def test_generate_rows_builds_mix_rows_and_retries_with_new_seed():
    answers = {0: "Ideal for those who rest.", 1: OK}

    def gen(reqs):
        return [(answers[k], {"diag": DIAG}) for a, p, al, k in reqs]

    (r,) = generate_rows([(ART, PAIR, 0.5)], gen, make=make_mix_row)
    assert r["usage"]["attempts"] == 2 and r["seed"] == mix_seed("Q1", PAIR, 0.5, 1)


def test_mix_key_distinguishes_alphas():
    from museumbot.generation.vllm_gen import replace_rows
    from museumbot.generation.vllm_mix import mix_key

    rows = [make_mix_row(ART, PAIR, al, OK, 0) for al in (0.25, 0.5, 0.75)]
    new = make_mix_row(ART, PAIR, 0.5, OK + " Again.", 5)
    out = replace_rows(rows, [new], key=mix_key)
    assert [r["alpha"] for r in out] == [0.25, 0.5, 0.75]
    assert out[1] is new and out[0] is rows[0] and out[2] is rows[2]


def test_failed_mix_jobs_lists_only_failed_rows():
    from museumbot.generation.vllm_mix import failed_mix_jobs

    rows = [make_mix_row(ART, PAIR, 0.25, OK, 0),
            make_mix_row(ART, PAIR, 0.75, "for those", 3, failed=True)]
    assert failed_mix_jobs(rows, {"Q1": ART}) == [(ART, PAIR, 0.75)]
