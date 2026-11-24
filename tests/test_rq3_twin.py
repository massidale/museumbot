from museumbot.rq3_visitors.twin import (build_jobs, guides, parse_fit, parse_ratings,
                                         persona_prompt, personal_rating)

APP = {lang: {
    "falk": {"explorer": f"E-{lang}", "recharger": f"R-{lang}"},
    "fit_low": "low", "fit_high": "high", "rating_low": "rl", "rating_high": "rh",
    "guide_label": "Guide {n}", "q_rating": "Liked {guide}?", "trial_voice_q": f"Q-{lang}",
} for lang in ("en", "de")}


def test_parse_fit():
    assert parse_fit("explorer:6 recharger:3") == {"explorer": 6, "recharger": 3}


def test_persona_ordine_di_scelta_e_aderenza():
    p = persona_prompt("en", ["recharger", "explorer"], {"explorer": 6, "recharger": 3}, APP)
    assert p.index("R-en") < p.index("E-en")
    assert 'R-en" (fits you: 3)' in p and 'E-en" (fits you: 6)' in p
    assert "(1 = low, 7 = high)" in p
    assert "recharger" not in p  # niente nome della categoria


def test_guides_ordini():
    t = {"personal_guide": "2"}
    assert guides(t, "visitor") == ("comparison", "personal", 2)
    assert guides(t, "reversed") == ("personal", "comparison", 1)


def test_parse_ratings():
    c = 'ok {"x": 1} {"comment": "Guide 2.", "guide1": 3, "guide2": "6"}'
    assert parse_ratings(c) == {"comment": "Guide 2.", "guide1": 3, "guide2": 6}
    assert parse_ratings('{"comment": "", "guide1": 0, "guide2": 5}') is None
    assert parse_ratings("niente") is None


def test_personal_rating():
    assert personal_rating({"guide1": 3, "guide2": 6, "personal_guide": 2}) == (6, 3)
    assert personal_rating({"guide1": 3, "guide2": 6, "personal_guide": 1}) == (3, 6)


def test_build_jobs():
    trials = [{"session_id": "s1", "artwork_id": "a", "lang": "de", "personal": "explorer",
               "personal_guide": "1"}]
    sessions = [{"session_id": "s1", "categories": "explorer", "category_fit": "explorer:5"}]
    gen = {("a", "explorer", "de"): "GEMMA"}
    museum = {("a", "de"): "MUSEUM"}
    titles = {"a": ({"en": "T", "de": "T-de"}, "Artist")}
    jobs = build_jobs(trials, sessions, APP, gen, museum, titles)
    assert len(jobs) == 4
    by = {(j["mode"], j["order"]): j for j in jobs}
    user = by[("persona", "visitor")]["messages"][1]["content"]
    assert "T-de" in user and user.index("GEMMA") < user.index("MUSEUM") and "Q-de" in user
    assert "Liked Guide 1?" in user and "Liked Guide 2?" in user
    rev = by[("persona", "reversed")]
    assert rev["personal_guide"] == 2
    assert rev["messages"][1]["content"].index("MUSEUM") < rev["messages"][1]["content"].index("GEMMA")
    assert "E-de" in rev["messages"][0]["content"]
    assert "E-de" not in by[("neutral", "visitor")]["messages"][0]["content"]
