from museumbot.rq2_judge.mixed_analyze import rates, summarize_pair

A = [0.0, 0.25, 0.5, 0.75, 1.0]


def row(typ, persona, alpha, chose, pair=("x", "y"), valid=True):
    return {"type": typ, "persona": persona, "alpha": alpha, "chose_target": chose,
            "pair": list(pair), "valid": valid}


def test_rates_by_persona_and_alpha_skip_invalid():
    rows = [row("curve", "x", 0.0, True), row("curve", "x", 0.0, False),
            row("curve", "x", 0.0, True, valid=False), row("curve", "y", 0.0, True)]
    r = rates(rows, "curve")
    assert r[("x", 0.0)] == 0.5 and r[("y", 0.0)] == 1.0


def test_summarize_pair_curve_and_double_visitor_minimum():
    rows = []
    for al, px, py in zip(A, [1, 1, 1, 0.5, 0.5], [0.5, 0.5, 1, 1, 1]):
        for k in range(4):
            rows.append(row("curve", "x", al, k < 4 * px))
            rows.append(row("curve", "y", al, k < 4 * py))
            rows.append(row("double", "x", al, k < 4 * (1 - al)))
            rows.append(row("double", "y", al, k < 4 * al))
    rows += [row("coherence", "neutral", 0.5, k % 2 == 0) for k in range(4)]
    s = summarize_pair(rows, ("x", "y"))
    assert s["curve"]["rate_x"] == [1.0, 1.0, 1.0, 0.5, 0.5]
    assert s["double"]["min"] == [0.0, 0.25, 0.5, 0.25, 0.0]
    assert s["coherence"] == 0.5
