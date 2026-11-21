import json
from pathlib import Path

from museumbot.rq3_visitors import transcribe


def test_session_of_toglie_il_suffisso():
    assert transcribe.session_of(Path("cgofvw.m4a")) == "cgofvw"
    assert transcribe.session_of(Path("x/cgofvw_opera1.m4a")) == "cgofvw"


def test_to_do_salta_prove_esclusi_e_fatti():
    files = [Path(f"{n}.m4a") for n in ["a11111", "b22222", "c33333", "d44444"]] + [Path("a11111_opera1.m4a")]
    langs = {"a11111": "de", "b22222": "en", "c33333": "de"}   # d44444: sessione cancellata
    out = transcribe.to_do(files, langs, exclude={"c33333"}, done={"b22222.m4a"})
    assert [(f.name, l) for f, l in out] == [("a11111.m4a", "de"), ("a11111_opera1.m4a", "de")]


def test_load_done_e_load_langs(tmp_path):
    out = tmp_path / "t.jsonl"
    assert transcribe.load_done(out) == set()
    out.write_text(json.dumps({"file": "a.m4a"}) + "\n\n", encoding="utf-8")
    assert transcribe.load_done(out) == {"a.m4a"}
    exp = tmp_path / "e.json"
    exp.write_text(json.dumps([{"session_id": "a11111", "lang": "de"}]), encoding="utf-8")
    assert transcribe.load_langs(exp) == {"a11111": "de"}


def test_to_do_lingua_del_file_e_inutilizzabili():
    files = [Path("a11111_operatore.m4a"), Path("a11111_1.m4a"), Path("a11111.m4a")]
    out = transcribe.to_do(files, {"a11111": "de"}, set(), set(),
                           lang_of={"a11111_operatore.m4a": "it"}, skip={"a11111_1.m4a"})
    assert [(f.name, l) for f, l in out] == [("a11111.m4a", "de"), ("a11111_operatore.m4a", "it")]


def test_load_extra_e_load_unusable(tmp_path):
    (tmp_path / "extra.json").write_text(json.dumps([{"file": "x.m4a", "lang": "it"}]), encoding="utf-8")
    assert transcribe.load_extra(tmp_path) == {"x.m4a": {"file": "x.m4a", "lang": "it"}}
    assert transcribe.load_extra(tmp_path / "nessuna") == {}
    ex = tmp_path / "ex.json"
    ex.write_text(json.dumps({"sessions": {}, "voice": {"y.m4a": "cuffie"}}), encoding="utf-8")
    assert transcribe.load_unusable(ex) == {"y.m4a": "cuffie"}
