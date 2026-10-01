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
