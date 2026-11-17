import csv
from pathlib import Path

from museumbot.rq3_visitors.collect import COLUMNS, load_file, merge, validate, write_csv

# Export d'esempio scritto da toCsv (rq3_app/logic.js) con rq3_app/tests/sample_sessions.js.
FIXTURE = Path(__file__).parent / "data" / "rq3_export.csv"


def rec(sid, completed=False, answered=1, updated="2026-12-01T10:00"):
    trials = [{"index": i, "artwork_id": f"a{i}", "personal": "explorer", "comparison": "flat",
               "first": "personal",
               "choice": "personal" if i < answered else None,
               "why": "piu' vivace" if i < answered else ""}
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


def test_validate_why_non_testo():
    bad = rec("s1")
    bad["trials"][0]["why"] = ["boh"]
    assert len(validate(bad)) == 1


def test_colonne_come_nell_app():
    with open(FIXTURE, encoding="utf-8-sig", newline="") as fh:
        assert tuple(next(csv.reader(fh))) == COLUMNS


def test_load_file_ricostruisce_le_sessioni():
    recs, errori = load_file(FIXTURE)
    assert errori == [] and [r["session_id"] for r in recs] == ["abc123", "zz9"]
    a, z = recs
    assert a["seed"] == 42 and a["completed"] is True and a["finished_at"] == "2026-12-01T09:20:00.000Z"
    assert a["categories"] == {"main": "explorer", "secondary": "facilitator"}
    assert a["chosen"] == ["facilitator", "explorer"] and len(a["description_order"]) == 5
    assert a["player_errors"][0]["message"] == "MediaError 4"
    t0, t1 = a["trials"]
    assert t0["why"] == 'Lebendiger, "persönlicher"\nund kürzer'
    assert t0["plays"] == [1, 2] and t0["heard_sec"] == [201.5, 88.0]
    assert t0["reached_end"] == [True, False] and t0["choice_position"] == 2 and t0["strength"] == 6
    assert t1["choice"] is None and t1["strength"] is None and t1["answered_at"] is None
    assert z["trials"] == [] and z["categories"] is None and z["aborted_by_operator"] is True
    assert all(validate(r) == [] for r in recs)


def test_write_csv_e_rilettura(tmp_path):
    recs, _ = load_file(FIXTURE)
    out = tmp_path / "responses.csv"
    write_csv(out, recs)
    again, errori = load_file(out)
    assert errori == [] and again == recs


def test_load_file_colonne_mancanti_e_celle_sbagliate(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("session_id,device\ns1,ipad\n")
    recs, errori = load_file(p)
    assert recs == [] and "colonne mancanti" in errori[0]
    righe = FIXTURE.read_text(encoding="utf-8-sig").replace(",true,end,", ",forse,end,")
    q = tmp_path / "y.csv"
    q.write_text(righe, encoding="utf-8")
    recs, errori = load_file(q)
    assert [r["session_id"] for r in recs] == ["zz9"] and "abc123" in errori[0]


def test_load_file_contatori_vuoti(tmp_path):
    # sessioni registrate prima che l'app contasse l'ascolto: celle vuote, non errori
    testo = FIXTURE.read_text(encoding="utf-8-sig").replace(",1,0,12.3,0,false,false,", ",1,0,,,,,")
    p = tmp_path / "x.csv"
    p.write_text(testo, encoding="utf-8")
    recs, errori = load_file(p)
    t1 = recs[0]["trials"][1]
    assert errori == [] and t1["heard_sec"] == [None, None] and t1["reached_end"] == [None, None]
