import pytest

from museumbot.generation import tts

ROW = {"artwork_id": "Q1", "title": "The Girl with the Wine Glass", "condition": "explorer",
       "text": "Hello.", "text_de": "Hallo."}


def test_nome_file():
    assert tts.audio_name(ROW) == "the-girl-with-the-wine-glass/explorer"
    mix = {**ROW, "condition": "explorer+recharger"}
    assert tts.audio_name(mix) == "the-girl-with-the-wine-glass/explorer+recharger"


def test_jobs_salta_esistenti_e_vuole_il_tedesco(tmp_path):
    (tmp_path / "en" / "the-girl-with-the-wine-glass").mkdir(parents=True)
    (tmp_path / "en" / "the-girl-with-the-wine-glass" / "explorer.mp3").write_bytes(b"x")
    todo = tts.jobs([ROW], tmp_path)
    assert [lang for _, lang, _ in todo] == ["de"]
    with pytest.raises(ValueError):
        tts.jobs([{**ROW, "text_de": ""}], tmp_path)


def test_speak_manda_modello_e_voce(monkeypatch):
    monkeypatch.setattr(tts, "openrouter_key", lambda: "k")
    seen = {}

    class R:
        content = b"mp3"
        headers = {"content-type": "audio/mpeg"}

        def raise_for_status(self):
            pass

    def post(url, headers, json, timeout):
        seen.update(json)
        return R()

    assert tts.speak("Hallo.", "george", post=post) == b"mp3"
    assert seen["model"] == "elevenlabs/eleven-v4" and seen["voice"] == "george"
    monkeypatch.setattr(tts, "pcm_to_mp3", lambda pcm, rate: b"conv%d" % rate)
    R.headers = {"content-type": "audio/pcm;rate=24000;channels=1"}
    assert tts.speak("Hallo.", "gareth", "heygen/heygen-voice-1-turbo:free", post=post) == b"conv24000"
    assert seen["model"] == "heygen/heygen-voice-1-turbo:free" and seen["response_format"] == "pcm"


def test_prezzo_modello_ignoto_e_il_piu_caro():
    assert tts.price("heygen/heygen-voice-1-turbo:free") == 0.0
    assert tts.price("x/ignoto") == max(tts.PRICES.values())
