from museumbot.common.prompts import FALK_CATEGORIES, FOCUS, build_block, wiki_messages
from museumbot.generation.vllm_gen import generate_rows, make_row
from museumbot.generation.vllm_wiki import ALPHA, mix_jobs, pairs, pure_jobs, wiki_mix_row

ART = {"id": "Q1", "title": "Opera", "artist": "Autore", "source_text": "Voce intera."}


def test_wiki_messages_system_comune_e_categoria_in_coda():
    a = wiki_messages("explorer", "T", "A", "voce")
    b = wiki_messages("recharger", "T", "A", "voce")
    assert a[0] == b[0]
    assert a[1]["content"].endswith(build_block("explorer") + "\nFocus: " + FOCUS["explorer"])
    # prefisso comune fino al blocco di categoria, voce compresa
    head = a[1]["content"].split("Listener:")[0]
    assert "voce" in head and b[1]["content"].startswith(head)


def test_focus_per_ogni_categoria():
    assert set(FOCUS) == set(FALK_CATEGORIES)


def test_job_puri_e_misti():
    assert len(pure_jobs([ART])) == 5
    assert len(pairs()) == 10 and len(mix_jobs([ART, ART])) == 20
    assert all(al == ALPHA for _, _, al in mix_jobs([ART]))


def test_righe_con_variante_wiki():
    pure = generate_rows(pure_jobs([ART])[:1], lambda reqs: [("Un testo.", {})] * len(reqs))
    assert pure[0]["variant"] == "wiki" and pure[0]["condition"] == "explorer"
    mix = generate_rows(mix_jobs([ART])[:1], lambda reqs: [("Un testo.", {"diag": []})] * len(reqs),
                        make=wiki_mix_row)
    assert mix[0]["variant"] == "wiki" and mix[0]["alpha"] == 0.5
    assert mix[0]["weights"] == {"explorer": 0.5, "facilitator": 0.5}


def test_flat_senza_coda_e_job_con_flat():
    f = wiki_messages("flat", "T", "A", "voce")
    e = wiki_messages("explorer", "T", "A", "voce")
    assert f[0] == e[0] and "Listener:" not in f[1]["content"]
    assert e[1]["content"].startswith(f[1]["content"] + "\n\nListener:")
    assert [c for _, c, _ in pure_jobs([ART], flat=True)][-1] == "flat"
    assert len(pure_jobs([ART], flat=True)) == 6
