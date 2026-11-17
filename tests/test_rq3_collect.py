from museumbot.rq3_visitors.collect import load_file, merge, validate


def rec(sid, completed=False, answered=1, updated="2026-12-01T10:00"):
    trials = [{"index": i, "artwork_id": f"a{i}", "personal": "explorer", "comparison": "flat",
               "first": "personal",
               "choice": "personal" if i < answered else None,
               "reasons": ["interesting"] if i < answered else []}
              for i in range(2)]
    return {"session_id": sid, "device": "ipad1", "app_version": "1", "manifest_version": "m1",
            "seed": 7, "lang": "en", "started_at": "2026-12-01T09:00", "updated_at": updated,
            "completed": completed, "last_step": "trial", "trials": trials}


def test_merge_tiene_il_record_piu_avanti():
    a = rec("s1", completed=False, answered=1, updated="2026-12-01T10:00")
    b = rec("s1", completed=True, answered=2, updated="2026-12-01T10:05")
    c = rec("s2", completed=False, answered=0, updated="2026-12-01T11:00")
    out = merge([b, c, a])
    assert [r["session_id"] for r in out] == ["s1", "s2"] and out[0]["completed"]


def test_merge_a_parita_vince_il_piu_recente():
    a = rec("s1", answered=1, updated="2026-12-01T10:00")
    b = rec("s1", answered=1, updated="2026-12-01T10:09")
    assert merge([a, b])[0]["updated_at"] == "2026-12-01T10:09"


def test_validate_campi_e_scelte():
    assert validate(rec("s1")) == []
    bad = rec("s1")
    del bad["seed"]
    bad["trials"][0]["choice"] = "left"
    assert len(validate(bad)) == 2


def test_validate_campi_facoltativi():
    # chosen, player_errors, aborted_by_operator possono mancare o esserci
    r = rec("s1")
    r.update(chosen=["recharger", "explorer"], aborted_by_operator=True,
             player_errors=[{"trial_index": 0, "position": 1, "src": "x", "message": "m", "at": "t"}])
    assert validate(r) == [] and validate(rec("s2")) == []


def test_validate_motivi_fuori_elenco():
    bad = rec("s1")
    bad["trials"][0]["reasons"] = ["boh"]
    assert len(validate(bad)) == 1


def test_load_file_riga_malformata(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n{rotto\n\n')
    righe, errori = load_file(p)
    assert righe == [{"a": 1}] and len(errori) == 1 and "x.jsonl:2" in errori[0]
