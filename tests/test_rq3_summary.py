from museumbot.rq3_visitors import summary


def _row(sid, art, personal="explorer", rp=6, rf=4, guide=1, pos=1):
    return {"session_id": sid, "artwork_id": art, "personal": personal, "rating_personal": rp,
            "rating_comparison": rf, "diff": rp - rf, "personal_guide": guide, "position": pos}


def test_esiti():
    assert summary.esiti([2, 0, -1, 3]) == {"personal": 2, "pari": 1, "flat": 1}


def test_cluster_ci_contiene_la_media_e_tiene_insieme_i_visitatori():
    lo, hi = summary.cluster_ci([1, 1, -1, -1], ["a", "a", "b", "b"], reps=500)
    assert lo == -1 and hi == 1  # solo due visitatori: la media e' -1, 0 o 1


def test_describe():
    d = summary.describe([_row("a", "x", rp=6, rf=4), _row("b", "x", rp=3, rf=5)])
    assert (d["prove"], d["visitatori"], d["diff"]) == (2, 2, 0.0)
    assert d["esiti"] == {"personal": 1, "pari": 0, "flat": 1}


def test_per_category_pesa_mezzo_i_misti():
    rows = [_row("a", "x", "explorer", 6, 4), _row("b", "x", "explorer+recharger", 3, 5)]
    c = summary.per_category(rows)
    assert c["explorer"]["diff"] == 0.0
    assert c["explorer"]["diff_pesata"] == round((2 * 1 - 2 * 0.5) / 1.5, 2)
    assert c["recharger"]["puri"]["prove"] == 0 and c["recharger"]["misti"]["diff"] == -2


def test_guide_order_e_prima_opera():
    rows = [_row("a", "x", rp=6, rf=4, guide=1, pos=1), _row("a", "y", rp=6, rf=4, guide=2, pos=2)]
    g = summary.guide_order(rows)
    assert g["guida1_meno_guida2"] == 0.0 and g["esiti_guida1"] == {"personal": 1, "pari": 0, "flat": 1}
    p = summary.first_artwork(rows)
    assert p["prima"]["prove"] == 1 and p["prima"]["diff_corretta"] == 0.0


def test_themes_conta_i_visitatori_una_volta():
    st = [{"session_id": "a", "tema": "unire"}, {"session_id": "a", "tema": "unire"},
          {"session_id": "b", "tema": "goethe", "guida": "flat", "segno": -1}]
    t = summary.themes(st, soglia=2)
    assert t["unire"]["visitatori"] == 1 and not t["unire"]["di_massa"]
    assert t["goethe"]["per_guida"] == {"flat-": 1}
    assert t["unire"]["per_opera"] == {"generale": 1}


def test_theme_votes_verso_del_voto():
    rows = [_row("a", "x", rp=6, rf=4), _row("b", "x", rp=3, rf=5), _row("c", "x", rp=5, rf=5)]
    st = [{"session_id": "a", "artwork_id": "x", "tema": "t", "guida": "personal", "segno": 1},
          {"session_id": "b", "artwork_id": "x", "tema": "t", "guida": "flat", "segno": -1},
          {"session_id": "c", "artwork_id": "x", "tema": "t", "guida": "flat", "segno": 1},
          {"session_id": "c", "artwork_id": None, "tema": "t", "guida": None, "segno": None}]
    v = summary.theme_votes(st, rows)["t"]
    assert v["voto_e_commento"] == {"stesso": 1, "pari": 1, "opposto": 1}
    assert v["prove"] == 3 and v["esiti"] == {"personal": 1, "pari": 1, "flat": 1}


def test_themes_by_group():
    sessions = [{"session_id": "a", "categories": "explorer+recharger"},
                {"session_id": "b", "categories": "explorer"}]
    st = [{"session_id": "a", "tema": "unire"}, {"session_id": "a", "tema": "unire"}]
    g = summary.themes_by_group(st, sessions)
    assert g["explorer"] == {"visitatori": 2, "con_commenti": 1, "temi": {"unire": 1}}
    assert g["misti"]["temi"] == {"unire": 1} and g["puri"]["con_commenti"] == 0


def test_fit():
    s = [{"category_fit": "explorer:6 recharger:3"}, {"category_fit": "facilitator:4"}]
    assert summary.fit(s) == {"sessioni": 2, "almeno_5": 1}


def test_coding_shares_divide_le_frasi_miste():
    rows = [{"source": "wiki", "artwork_id": "x", "codes": ["V"], "words": [6]},
            {"source": "wiki", "artwork_id": "x", "codes": ["A", "V"], "words": [2, 2]}]
    s = summary.coding_shares(rows)["wiki"]["x"]
    assert s["parole"] == 10 and s["si_vede"] == 80 and s["artista_contesto"] == 20


def test_questions():
    gens = [{"condition": "explorer", "text": "a? b?"}, {"condition": "explorer+recharger", "text": "c?"}]
    q = summary.questions(gens)
    assert q == {"explorer": 2.0, "explorer (misto)": 1.0, "recharger (misto)": 1.0}
