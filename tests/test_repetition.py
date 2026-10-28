from museumbot.rq1_embeddings.repetition import gram_set, jaccard, mean_overlap


def test_gram_set_keeps_n_apart_and_drops_function_words():
    s = gram_set("take a deep breath", ns=(1, 2))
    assert "1:breath" in s and "2:deep breath" in s
    assert "1:a" not in s


def test_jaccard():
    assert jaccard({"a", "b"}, {"b", "c"}) == 1 / 3
    assert jaccard(set(), set()) == 0.0


def test_mean_overlap_skips_same_artwork_pairs():
    texts = [("x", "quiet room"), ("x", "quiet room"), ("y", "loud hall")]
    assert mean_overlap(texts, ns=(1,)) == 0.0


def test_formulaic_texts_overlap_more():
    rep = [("a", "take a deep breath and look at the sky"),
           ("b", "take a deep breath and look at the sea")]
    var = [("a", "the sky swirls over the village"),
           ("b", "a fisherman mends nets on the shore")]
    assert mean_overlap(rep) > mean_overlap(var)
