from museumbot.common.config import ROOT, emb_path, meta_path

EMB = ROOT / "data" / "emb"


def test_paths_default_to_main_corpus_and_full():
    assert emb_path("qwen") == EMB / "main" / "qwen_full.npy"
    assert meta_path() == EMB / "main" / "meta_full.csv"


def test_paths_by_corpus_and_variant():
    assert emb_path("bge-m3", "no_style", "ablation") == EMB / "ablation" / "bge-m3_no_style.npy"
    assert meta_path("name_only", "ablation") == EMB / "ablation" / "meta_name_only.csv"
