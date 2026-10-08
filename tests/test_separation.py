import numpy as np
import pytest

from museumbot.rq1_embeddings.separation import (
    cross_artwork_similarity, formula_coverage, top_formulas, trigrams,
)


def test_trigrams_lowercase_and_ignore_punctuation():
    assert trigrams("Take a deep, breath. Take") == {("take", "a", "deep"), ("a", "deep", "breath"),
                                                      ("deep", "breath", "take")}


def test_formula_coverage_counts_shared_trigrams():
    docs = ["take a deep breath now", "take a deep breath here", "something else entirely here"]
    # "take a deep" e "a deep breath" sono in 2/3 dei testi: formule con soglia 0.5
    cov = formula_coverage(docs, min_df=0.5)
    assert cov[0] == pytest.approx(2 / 3) and cov[2] == 0.0


def test_top_formulas_by_document_frequency():
    docs = ["take a deep breath now", "take a deep breath here", "take a deep look"]
    top = top_formulas(docs, k=2)
    assert top[0] == ("take a deep", 1.0) and top[1][0] == "a deep breath"


def test_cross_artwork_similarity_excludes_same_artwork():
    E = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    arts = ["a", "a", "b"]
    assert cross_artwork_similarity(E, arts) == pytest.approx(0.0)
