from museumbot.rq3_visitors import dataset


def _trial(i, art, first="personal", ratings=(6, 3), heard=(90, 80)):
    return {"index": i, "artwork_id": art, "personal": "explorer", "comparison": "flat",
            "first": first, "plays": [1, 2], "ratings": list(ratings), "heard_sec": list(heard),
            "why": " bello "}


def _rec(sid, trials, voice=None):
    return {"session_id": sid, "device": "ipad", "lang": "de", "started_at": "2026-11-20T10:00:00Z",
            "categories": ["explorer"], "category_fit": {"explorer": 6}, "completed": True,
            "trials": trials, "voice": voice}


def test_trial_problem():
    assert dataset.trial_problem(_trial(0, "a")) is None
    assert dataset.trial_problem(_trial(0, "a", ratings=(None, None))) == "senza voti"
    assert dataset.trial_problem(_trial(0, "a", heard=(0, 0))) == "opera saltata (0 s di ascolto)"
    assert dataset.trial_problem(_trial(0, "a", heard=(84, 0))) == "una guida non ascoltata (0 s)"


def test_trial_row_riporta_i_voti_alla_personalizzata():
    r = dataset.trial_row(_rec("s1", []), _trial(2, "a", first="comparison", ratings=(6, 3), heard=(80, 95)))
    assert (r["personal_guide"], r["rating_personal"], r["rating_comparison"], r["diff"]) == (2, 3, 6, -3)
    assert (r["heard_personal"], r["heard_comparison"], r["position"]) == (95, 80, 3)
    assert r["comment"] == "bello" and r["oral_text"] == ""


def test_build_esclusioni_e_commenti():
    recs = [
        _rec("prova", [_trial(0, "a")]),
        _rec("s1", [_trial(0, "a"), _trial(1, "b", heard=(0, 0)), _trial(2, "c", heard=(50, 0))],
             voice={"sec": 30, "mime": "audio/mp4", "recorded_at": "x"}),
        _rec("s2", [_trial(0, "a", ratings=(None, None))]),
        _rec("s3", [_trial(0, "a")], voice={"skipped": True, "no_mic": True}),
    ]
    transcripts = {"s1.m4a": {"file": "s1.m4a", "text": "Hallo", "text_it": "Ciao"},
                   "s1_opera1.m4a": {"file": "s1_opera1.m4a", "text": "Gut", "text_it": "Bene"},
                   "s3.m4a": {"file": "s3.m4a", "text": "Ja"}}
    extra = {"s1_opera1.m4a": {"file": "s1_opera1.m4a", "session_id": "s1", "artwork_id": "a"},
             "s3.m4a": {"file": "s3.m4a", "session_id": "s3", "source": "fuori dall'app"}}
    trials, sessions, excluded = dataset.build(recs, {"prova": "prova"}, transcripts, extra)
    assert [(t["session_id"], t["artwork_id"]) for t in trials] == [("s1", "a"), ("s3", "a")]
    assert (trials[0]["oral_file"], trials[0]["oral_text_it"]) == ("s1_opera1.m4a", "Bene")
    assert trials[1]["oral_file"] == ""
    assert [(e["session_id"], e["artwork_id"], e["reason"]) for e in excluded] == [
        ("prova", "", "prova"), ("s1", "b", "opera saltata (0 s di ascolto)"),
        ("s1", "c", "una guida non ascoltata (0 s)"), ("s2", "a", "senza voti"),
        ("s2", "", "nessuna opera valida")]
    s1, s3 = sessions
    assert (s1["trials_valid"], s1["trials_total"], s1["voice"], s1["final_text_it"]) == (1, 3, "registrata nell'app", "Ciao")
    assert (s3["voice"], s3["final_text"], s3["final_note"]) == ("registrata fuori dall'app", "Ja", "fuori dall'app")


def test_build_app_060_registrazione_per_opera_e_tasto_salta():
    voice = {"sec": 20, "mime": "audio/mp4", "recorded_at": "x"}
    t0 = {**_trial(0, "a"), "voice": voice}
    t1 = {**_trial(1, "b"), "skipped": True, "ratings": [None, None], "heard_sec": [10, 0]}
    t2 = {**_trial(2, "c"), "voice": {"skipped": True}}
    recs = [_rec("s1", [t0, t1, t2], voice={"skipped": True})]
    transcripts = {"s1_1.m4a": {"file": "s1_1.m4a", "text": "Die zweite", "text_it": "La seconda"},
                   "s1_9.webm": {"file": "s1_9.webm", "text": "altro"}}
    trials, sessions, excluded = dataset.build(recs, {}, transcripts, {})
    assert [(t["artwork_id"], t["oral_file"], t["oral_text_it"]) for t in trials] == [
        ("a", "s1_1.m4a", "La seconda"), ("c", "", "")]
    assert excluded == [{"session_id": "s1", "artwork_id": "b", "reason": "opera saltata col tasto"}]
    assert (sessions[0]["voice"], sessions[0]["final_file"]) == ("saltata", "")
