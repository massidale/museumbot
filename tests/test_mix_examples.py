from museumbot.rq1_embeddings.mix_examples import CAP, colorize


def test_colorize_neutral_tokens_stay_plain_and_text_is_escaped():
    out = colorize(["Take", " a", " <b>"], [0.0, 0.05, 0.0])
    assert out == "Take a &lt;b&gt;"


def test_colorize_x_blue_y_red_with_intensity_proportional_and_capped():
    out = colorize(["One", " two", " three"], [CAP / 2, -CAP * 3, 0.0])
    assert 'rgba(42,120,214,' in out and 'rgba(214,69,58,' in out
    a_one = float(out.split("rgba(42,120,214,")[1].split(")")[0])
    a_two = float(out.split("rgba(214,69,58,")[1].split(")")[0])
    assert a_two == 2 * a_one  # il secondo satura al tetto, il primo e' a meta'
    assert ">One<" in out and "> two<" in out and out.endswith(" three")


def test_colorize_keeps_paragraph_breaks_outside_spans():
    out = colorize(["One", ".", "\n\n", "Two"], [3.0, 0.0, 0.0, -3.0])
    assert "\n\n" in out and "<span" not in out.split("\n\n")[0][:0]
    assert out.split("\n\n")[1].startswith("<span")
