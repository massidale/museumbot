import shutil
import subprocess

import pytest

from museumbot.rq3_visitors.build_manifest import (
    all_conditions, build, condition_key, duration)


def test_condition_key_canonica():
    assert condition_key("recharger+professional_hobbyist") == "professional_hobbyist+recharger"
    assert condition_key("flat") == "flat" and condition_key("explorer") == "explorer"
    for bad in ["explorer+explorer", "tourist", "explorer+tourist"]:
        with pytest.raises(ValueError):
            condition_key(bad)


def test_all_conditions_come_in_js():
    c = all_conditions()
    assert len(c) == 16 and c[0] == "flat" and "experience_seeker+recharger" in c
    assert c[1:6] == ["explorer", "facilitator", "experience_seeker",
                      "professional_hobbyist", "recharger"]
    assert c[6] == "explorer+facilitator"


def _crea(audio_root, lang, art, nome, ext="mp3"):
    d = audio_root / lang / art
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{nome}.{ext}").write_bytes(b"x")


def _completo(audio_root):
    for lang in ("en", "de"):
        for c in all_conditions():
            _crea(audio_root, lang, "a1", c)


def _opera():
    return [{"id": "a1", "image": "img/a1.jpg", "title": {"en": "T", "de": "T"}, "artist": "A"}]


def test_build_elenca_tutti_i_mancanti(tmp_path):
    audio_root = tmp_path / "rq3_app" / "audio"
    for lang in ("en", "de"):
        for c in all_conditions():
            if (lang, c) in {("en", "explorer+facilitator"), ("de", "recharger")}:
                continue
            _crea(audio_root, lang, "a1", c)
    with pytest.raises(ValueError) as e:
        build(_opera(), audio_root, tmp_path / "rq3_app", "v1", dur=lambda p: 90.0)
    msg = str(e.value)
    assert "explorer+facilitator" in msg and "de" in msg and "recharger" in msg


def test_build_normalizza_e_rende_relativi(tmp_path):
    audio_root = tmp_path / "rq3_app" / "audio"
    for lang in ("en", "de"):
        for c in all_conditions():
            nome = "recharger+professional_hobbyist" if c == "professional_hobbyist+recharger" else c
            _crea(audio_root, lang, "a1", nome)
    m = build(_opera(), audio_root, tmp_path / "rq3_app", "v1", dur=lambda p: 91.5)
    a = m["artworks"][0]["audio"]["en"]["professional_hobbyist+recharger"]
    assert a == {"src": "audio/en/a1/recharger+professional_hobbyist.mp3", "sec": 91.5}
    assert m["version"] == "v1" and len(m["artworks"][0]["audio"]["de"]) == 16
    assert m["categories"][0] == "explorer"
    assert "sample" not in m


@pytest.mark.parametrize("a,b", [("explorer+facilitator.mp3", "facilitator+explorer.mp3"),
                                 ("flat.mp3", "flat.m4a")])
def test_build_rifiuta_i_doppioni(tmp_path, a, b):
    audio_root = tmp_path / "rq3_app" / "audio"
    _completo(audio_root)
    d = audio_root / "de" / "a1"
    for nome in (a, b):
        (d / nome).write_bytes(b"x")
    with pytest.raises(ValueError) as e:
        build(_opera(), audio_root, tmp_path / "rq3_app", "v1", dur=lambda p: 8.0)
    assert a in str(e.value) and b in str(e.value)


def test_build_condizione_ignota_col_percorso(tmp_path):
    audio_root = tmp_path / "rq3_app" / "audio"
    _completo(audio_root)
    (audio_root / "en" / "a1" / "tourist.mp3").write_bytes(b"x")
    with pytest.raises(ValueError, match=r"en/a1/tourist\.mp3"):
        build(_opera(), audio_root, tmp_path / "rq3_app", "v1", dur=lambda p: 8.0)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="serve ffmpeg")
def test_duration_con_ffprobe(tmp_path):
    p = tmp_path / "x.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anullsrc", "-t", "2", str(p)],
                   check=True)
    assert abs(duration(p) - 2.0) < 0.1
