import json

from museumbot.generation.generate import plan_jobs, read_done


def test_read_done_defaults_missing_variant_to_full(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(
        json.dumps({"artwork_id": "Q1", "condition": "explorer", "model": "m"}) + "\n"
        + json.dumps({"artwork_id": "Q1", "condition": "explorer", "model": "m",
                      "variant": "no_style"}) + "\n"
    )
    assert read_done(p) == {("Q1", "explorer", "m", "full"), ("Q1", "explorer", "m", "no_style")}


def test_plan_jobs_skips_done_and_flat_for_non_full():
    arts = [{"id": "Q1"}, {"id": "Q2"}]
    done = {("Q1", "explorer", "m", "no_style")}
    jobs = plan_jobs(arts, ["no_style"], done, "m")
    keys = {(a["id"], c, v) for a, c, v in jobs}
    assert ("Q1", "explorer", "no_style") not in keys
    assert ("Q2", "explorer", "no_style") in keys
    assert not any(c == "flat" for _, c, _ in jobs)
    assert len(jobs) == 2 * 5 - 1


def test_plan_jobs_full_includes_flat():
    arts = [{"id": "Q1"}]
    jobs = plan_jobs(arts, ["full"], set(), "m")
    assert ("Q1", "flat", "full") in {(a["id"], c, v) for a, c, v in jobs}
    assert len(jobs) == 6


def test_plan_jobs_replicate_skips_flat():
    jobs = plan_jobs([{"id": "Q1"}], ["full_rep"], {("Q1", "explorer", "m", "full")}, "m")
    assert {(c, v) for _, c, v in jobs} == {
        (c, "full_rep") for c in ("explorer", "facilitator", "experience_seeker",
                                  "professional_hobbyist", "recharger")
    }


def test_provider_routing():
    from museumbot.generation.generate import provider_routing

    assert provider_routing(None) is None
    assert provider_routing("deepinfra") == {"only": ["deepinfra"], "allow_fallbacks": False}
    assert provider_routing("deepinfra/fp8") == {
        "only": ["deepinfra"], "allow_fallbacks": False, "quantizations": ["fp8"]}


def test_chain_study_plan_covers_cells_and_skips_done():
    from museumbot.generation.chain_study import METHODS, plan

    arts = [{"id": "Q1"}, {"id": "Q2"}]
    jobs = plan(arts, {("Q1", "flat", "flat"), ("Q1", "explorer", "chain")})
    keys = {(a["id"], c, m) for a, c, m in jobs}
    assert len(jobs) == 2 * (1 + 5 * len(METHODS)) - 2
    assert ("Q1", "explorer", "chain") not in keys and ("Q2", "flat", "flat") in keys
    assert jobs == plan(arts, {("Q1", "flat", "flat"), ("Q1", "explorer", "chain")})  # seed fisso


def test_chain_messages_extend_prefix_with_p3_and_source():
    from museumbot.generation.chain_study import messages_for

    art = {"title": "T", "artist": "A", "source_text": "SRC"}
    prefix = [{"role": "user", "content": "p1"}, {"role": "assistant", "content": "r1"}]
    m = messages_for("chain", art, "recharger", prefix)
    assert m[:2] == prefix and m[-1]["role"] == "user"
    assert "Recharger" in m[-1]["content"] and "SRC" in m[-1]["content"]
    assert messages_for("single_a", art, "recharger", prefix)[0]["role"] == "system"


def test_clean_guide_strips_preamble_markdown_and_word_count():
    from museumbot.generation.chain_study import clean_guide

    raw = ("Here is a 250-word audio guide for *The Starry Night*, written for the **Explorer**.\n\n"
           "---\n\nPause here. Look at *The Starry Night* and its **swirling** sky.\n\n"
           "Van Gogh painted it in 1889.\n\n---\n\n*(Word count: 250)*")
    assert clean_guide(raw) == ("Pause here. Look at The Starry Night and its swirling sky.\n\n"
                                "Van Gogh painted it in 1889.")


def test_clean_guide_leaves_plain_text_untouched():
    from museumbot.generation.chain_study import clean_guide

    plain = "Stand back a little.\n\nThe painting measures 73 by 92 cm, 3*4 grid aside."
    assert clean_guide(plain) == plain
