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
