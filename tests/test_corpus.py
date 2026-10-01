import json

from museumbot.common.corpus import drop_truncated, load_corpus


def study_row(a, c, method, text="Look at the sky.", raw=None):
    usage = {"raw": raw} if raw else {}
    return {"artwork_id": a, "title": "T", "artist": "A", "condition": c, "method": method,
            "model": "m", "text": text, "words": len(text.split()), "usage": usage}


def write(tmp_path, rows):
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_main_maps_methods_and_drops_chain(tmp_path):
    p = write(tmp_path, [study_row("Q1", "explorer", "single_a"),
                         study_row("Q1", "explorer", "single_b"),
                         study_row("Q1", "explorer", "chain"),
                         study_row("Q1", "flat", "flat")])
    rows = load_corpus("main", p, verbose=False)
    assert sorted((r["condition"], r["variant"]) for r in rows) == [
        ("explorer", "full"), ("explorer", "full_rep"), ("flat", "full")]


def test_main_cleans_from_original_text(tmp_path):
    raw = "Here is the audio guide:\n\nLook at *the* sky."
    p = write(tmp_path, [study_row("Q1", "flat", "flat", text="Look at the sky.", raw=raw)])
    (r,) = load_corpus("main", p, verbose=False)
    assert r["text"] == "Look at the sky." and r["words"] == 4


def test_ablation_drops_whole_artwork_with_a_truncated_text(tmp_path):
    rows = [{"artwork_id": a, "title": "T", "artist": "A", "condition": c, "variant": v,
             "model": "m", "text": t, "words": 1, "usage": {}}
            for a, c, v, t in [("Q1", "flat", "full", "Fine."),
                               ("Q1", "explorer", "no_def", "Cut in the"),
                               ("Q2", "flat", "full", "Fine."),
                               ("Q2", "explorer", "no_def", "Fine.")]]
    out = load_corpus("ablation", write(tmp_path, rows), verbose=False)
    assert {r["artwork_id"] for r in out} == {"Q2"}


def test_drop_truncated_reports_artworks():
    rows = [{"artwork_id": "Q1", "text": "ok."}, {"artwork_id": "Q2", "text": "cut"}]
    kept, bad = drop_truncated(rows)
    assert bad == ["Q2"] and [r["artwork_id"] for r in kept] == ["Q1"]
