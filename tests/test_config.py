from config import ROOT, emb_path, meta_path


def test_full_paths_are_legacy_names():
    assert emb_path("qwen") == ROOT / "data" / "emb_qwen.npy"
    assert emb_path("qwen", "full") == ROOT / "data" / "emb_qwen.npy"
    assert meta_path() == ROOT / "data" / "meta.csv"


def test_variant_paths_are_suffixed():
    assert emb_path("bge-m3", "no_style") == ROOT / "data" / "emb_bge-m3_no_style.npy"
    assert meta_path("name_only") == ROOT / "data" / "meta_name_only.csv"
