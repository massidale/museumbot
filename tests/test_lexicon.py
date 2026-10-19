from collections import Counter

from museumbot.rq1_embeddings.lexicon import distinctive, echo, log_odds_z, ngrams, overlaps


def test_ngrams_drop_stopword_only_grams():
    toks = "take a moment in the room".split()
    assert ngrams(toks, 1) == ["moment", "room"]
    assert "in the" not in ngrams(toks, 2)
    assert "take a moment" in ngrams(toks, 3)


def test_log_odds_rewards_frequent_in_condition_rare_in_rest():
    y_c = Counter({"breathe": 40, "paint": 50, "rare": 1})
    y_r = Counter({"breathe": 2, "paint": 250, "rare": 0})
    z = log_odds_z(y_c, y_r, y_c + y_r)
    assert z["breathe"] > z["rare"] > 0
    assert z["paint"] < 0


def test_overlaps_on_shared_content_words():
    assert overlaps("you are standing", "standing before one")
    assert overlaps("discuss", "encourage discussion")
    assert not overlaps("art", "heart")
    assert not overlaps("in the room", "in the garden")  # solo parole funzionali in comune


def test_echo_matches_content_words_up_to_inflection():
    block = "Your listener is a Facilitator: encourage discussion among companions."
    assert echo("your companions", block) is True
    assert echo("discuss", block) is True
    assert echo("your group", block) is False
    assert echo("discuss", None) is None


def test_distinctive_applies_doc_share_and_dedup():
    a = ["take a moment to breathe slowly"] * 10
    b = ["the painting shows a river"] * 10
    out = distinctive({"recharger": a, "flat": b}, k=5, min_doc_share=0.05)
    terms = [t["term"] for t in out["recharger"]]
    assert terms and all(not overlaps(x, y) for i, x in enumerate(terms) for y in terms[i + 1:])
    assert all(t["doc_share"] == 1.0 for t in out["recharger"])
    assert out["flat"][0]["echo"] is None
