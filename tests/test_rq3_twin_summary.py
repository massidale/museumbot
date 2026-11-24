from museumbot.rq3_visitors.twin_summary import (agreement, kappa, order_effect, twin_themes,
                                                 twin_trials, valid_calls)


def call(mode, order, p, f, valid=True, art="a"):
    return {"session_id": "s", "artwork_id": art, "mode": mode, "order": order, "valid": valid,
            "rating_personal": p, "rating_comparison": f}


TRIAL = {"session_id": "s", "artwork_id": "a", "personal": "explorer", "personal_guide": 1,
         "rating_personal": 6, "rating_comparison": 3, "diff": 3}


def test_valid_calls_tiene_le_valide():
    c = valid_calls([call("persona", "visitor", 5, 5), call("persona", "visitor", 0, 0, valid=False)])
    assert c[("s", "a", "persona", "visitor")]["rating_personal"] == 5


def test_twin_trials_media_dei_due_ordini_e_solo_complete():
    rows = [call("persona", "visitor", 7, 5), call("persona", "reversed", 6, 6),
            call("neutral", "visitor", 5, 5), call("neutral", "reversed", 5, 4),
            call("persona", "visitor", 7, 5, art="b")]
    t = twin_trials(valid_calls(rows), [TRIAL, TRIAL | {"artwork_id": "b"}])
    assert len(t["persona"]) == 1 and t["persona"][0]["diff"] == 1.0
    assert t["neutral"][0]["diff"] == 0.5 and t["persona"][0]["orders"] == [2, 0]


def test_agreement():
    a = agreement([{"diff": 1.0}, {"diff": 0.0}], [{"diff": 3}, {"diff": -1}])
    assert a["stesso_segno"] == 1
    assert a["tabella"]["visitatore_flat"]["gemello_pari"] == 1


def test_order_effect():
    # personalizzata in Guida 1: ordine visitatore +2 (G1 meglio), inverso +2 (G2 meglio)
    t = [{"session_id": "s", "personal_guide": 1, "orders": [2, 2]}]
    o = order_effect(t)
    assert o["guida1_meno_guida2"] == 0.0 and o["stesso_segno_nei_due_ordini"] == 1


def test_twin_themes_conta_per_commento():
    st = [{"session_id": "s", "artwork_id": "a", "mode": "persona", "tema": "domande", "guida": "personal", "segno": 1},
          {"session_id": "s", "artwork_id": "b", "mode": "persona", "tema": "domande", "guida": "personal", "segno": 1},
          {"session_id": "s", "artwork_id": "a", "mode": "neutral", "tema": "lezione", "guida": "flat", "segno": -1}]
    t = twin_themes(st)
    assert t["persona"]["domande"]["commenti"] == 2
    assert t["persona"]["domande"]["per_guida"] == {"personal+": 2}
    assert t["neutral"]["lezione"]["per_guida"] == {"flat-": 1}


def test_kappa():
    assert kappa([("a", "a"), ("b", "b")]) == 1.0
    # accordo pari al caso: 2 su 4, marginali 50/50
    assert kappa([("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")]) == 0.0
    # chi dice sempre la stessa cosa non ha accordo oltre il caso
    assert kappa([("a", "a"), ("b", "a"), ("a", "a")]) == 0.0
