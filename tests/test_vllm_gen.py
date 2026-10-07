from museumbot.generation.rows import ATTEMPTS, row_seed
from museumbot.generation.vllm_gen import MODEL, generate_rows, make_row, needs_retry

ART = {"id": "Q1", "title": "T", "artist": "A", "source_text": "S"}
OK = "Look at the sky. It is wide and calm."


def test_needs_retry_on_empty_or_explicit_reference():
    assert needs_retry("")
    assert needs_retry("This is perfect for those who love art.")
    assert not needs_retry(OK)


def test_make_row_cleans_and_records_generation():
    raw = "Here is the audio guide:\n\n" + OK
    r = make_row(ART, "explorer", "full", raw, attempt=0, extra={"completion_tokens": 9})
    assert r["text"] == OK and r["words"] == len(OK.split())
    assert r["weights"] == {"explorer": 1.0} and r["model"] == MODEL
    assert r["seed"] == row_seed("Q1", "explorer", "full", 0)
    assert r["usage"]["raw"] == raw and r["usage"]["attempts"] == 1
    assert r["usage"]["completion_tokens"] == 9 and not r["usage"]["failed"]


def fake_backend(answers):
    """Restituisce, per (opera, condizione), la risposta del tentativo richiesto."""
    calls = []

    def gen(requests):
        calls.append(requests)
        return [(answers[(a["id"], c)][min(k, len(answers[(a["id"], c)]) - 1)], {})
                for a, c, v, k in requests]

    return gen, calls


def test_generate_rows_retries_only_failures_with_new_seed():
    jobs = [(ART, "explorer", "full"), (ART, "recharger", "full")]
    gen, calls = fake_backend({("Q1", "explorer"): [OK],
                               ("Q1", "recharger"): ["Ideal for those who rest.", OK]})
    rows = generate_rows(jobs, gen)
    by = {r["condition"]: r for r in rows}
    assert by["explorer"]["usage"]["attempts"] == 1
    assert by["recharger"]["usage"]["attempts"] == 2
    assert by["recharger"]["seed"] == row_seed("Q1", "recharger", "full", 1)
    assert len(calls) == 2 and len(calls[1]) == 1


def test_generate_rows_keeps_failed_row_after_last_attempt():
    gen, calls = fake_backend({("Q1", "explorer"): ["for those"] * ATTEMPTS})
    (r,) = generate_rows([(ART, "explorer", "full")], gen)
    assert r["usage"]["failed"] and r["usage"]["attempts"] == ATTEMPTS
    assert len(calls) == ATTEMPTS
