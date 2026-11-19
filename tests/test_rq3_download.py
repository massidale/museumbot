import json
from datetime import datetime
from pathlib import Path

import pytest

from museumbot.rq3_visitors import download
from museumbot.rq3_visitors.collect import load_file


class FakeResp:
    def __init__(self, data, status=200):
        self._data, self.status = data, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self._data

    @property
    def content(self):
        return self._data


def test_out_path_con_data_fissa():
    p = download.out_path(Path("raw"), datetime(2026, 12, 3, 9, 5))
    assert p == Path("raw") / "online_20261203-0905.json"


def test_fetch_manda_il_token_e_restituisce_la_lista():
    visti = {}

    def get(url, headers, timeout):
        visti.update(url=url, headers=headers, timeout=timeout)
        return FakeResp([{"session_id": "abc123"}])

    out = download.fetch("https://x/api/sessions", "tok", get=get)
    assert out == [{"session_id": "abc123"}]
    assert visti["headers"]["Authorization"] == "Bearer tok" and visti["timeout"]


def test_fetch_rifiuta_risposta_non_lista():
    with pytest.raises(ValueError):
        download.fetch("u", "t", get=lambda u, headers, timeout: FakeResp({"a": 1}))


def test_fetch_propaga_errore_http():
    with pytest.raises(RuntimeError):
        download.fetch("u", "t", get=lambda u, headers, timeout: FakeResp([], 401))


def test_il_file_scritto_si_rilegge_con_collect(tmp_path):
    rec = {"session_id": "abc123", "device": "qr", "app_version": "0.2.0",
           "manifest_version": "m1", "seed": 1, "lang": "en", "started_at": "a",
           "updated_at": "b", "completed": False, "last_step": "trial", "trials": []}
    path = download.save([rec], tmp_path / "raw", datetime(2026, 12, 3, 9, 5))
    assert path.name == "online_20261203-0905.json"
    assert path.read_text().endswith("\n")
    righe, errori = load_file(path)
    assert righe == [rec] and errori == []


def test_token_mancante_esce_con_messaggio(monkeypatch):
    monkeypatch.delenv("RQ3_EXPORT_TOKEN", raising=False)
    with pytest.raises(SystemExit) as e:
        download.export_token()
    assert "RQ3_EXPORT_TOKEN" in str(e.value)


def test_voice_name_dal_tipo():
    assert download.voice_name({"session_id": "abc123", "type": "audio/mp4"}) == "abc123.m4a"
    assert download.voice_name({"session_id": "abc123", "type": "audio/webm;codecs=opus"}) == "abc123.webm"
    assert download.voice_name({"session_id": "abc123", "type": ""}) == "abc123.bin"


def test_fetch_voice_scarica_solo_le_mancanti(tmp_path):
    (tmp_path / "old111.m4a").write_bytes(b"x")
    lista = [{"session_id": "old111", "type": "audio/mp4"},
             {"session_id": "new222", "type": "audio/webm;codecs=opus"}]
    chiesti = []

    def get(url, headers, timeout):
        chiesti.append(url)
        assert headers["Authorization"] == "Bearer tok"
        return FakeResp(lista) if "?id=" not in url else FakeResp(b"audio")

    n = download.fetch_voice("https://x/api/voice", "tok", tmp_path, get=get)
    assert n == 1
    assert chiesti == ["https://x/api/voice", "https://x/api/voice?id=new222"]
    assert (tmp_path / "new222.webm").read_bytes() == b"audio"
