from museumbot.common.clean import clean_guide, is_truncated

# Inizi reali del corpus di settembre: "Here is" apre gia' la descrizione dell'opera.
DESCRIPTIVE = [
    "Here is a painting that rewards a second look. This is Cutting the Stone, painted around "
    "1494 by Hieronymus Bosch.\n\nAt first glance, it seems like a simple scene.",
    "Here is a moment that helped shape the story of modern art. This is Claude Monet's "
    "Le Déjeuner sur l'herbe, a context worth knowing.\n\nYou might know the scandal.",
    "Here is a moment of quiet intensity, captured in oil.\n\nThis is The Bellelli Family.",
]


def test_descriptive_here_is_is_kept():
    for t in DESCRIPTIVE:
        assert clean_guide(t) == t


def test_preamble_naming_the_text_is_removed():
    raw = ("Here is the audio guide text for The Three Philosophers by Giorgione.\n\n"
           "Look closely at this painting.")
    assert clean_guide(raw) == "Look closely at this painting."
    raw = "Here is a 250-word version for you.\n\nLook closely at this painting."
    assert clean_guide(raw) == "Look closely at this painting."


def test_preamble_with_typographic_apostrophe_is_removed():
    assert clean_guide("Here’s your audio guide.\n\nYou’re standing before it.") == (
        "You’re standing before it.")
    keep = "Here’s a great one to share with your group. This painting is by Bosch."
    assert clean_guide(keep) == keep


def test_preamble_ending_with_colon_is_removed():
    assert clean_guide("Sure, here it is:\n\nStand before the canvas.") == "Stand before the canvas."


def test_preamble_followed_by_separator_is_removed():
    raw = "Here is something special for you.\n\n---\n\nStand before the canvas."
    assert clean_guide(raw) == "Stand before the canvas."


def test_markdown_and_stray_asterisks_are_removed():
    raw = "Look at *The Starry Night* and its **swirling** sky, as they were two centuries ago.*"
    assert clean_guide(raw) == ("Look at The Starry Night and its swirling sky, "
                                "as they were two centuries ago.")


def test_unclosed_italic_is_removed():
    raw = "Its inclusion in the BBC series *100 Great Paintings cemented its fame."
    assert clean_guide(raw) == "Its inclusion in the BBC series 100 Great Paintings cemented its fame."


def test_plain_text_with_inner_asterisk_is_untouched():
    plain = "Stand back a little.\n\nThe painting measures 73 by 92 cm, 3*4 grid aside."
    assert clean_guide(plain) == plain


def test_clean_is_idempotent():
    raw = ("Here is a 250-word audio guide for *The Starry Night*, written for the **Explorer**.\n\n"
           "---\n\nPause here. Look at *The Starry Night*.\n\n(Fade out)")
    once = clean_guide(raw)
    assert once == "Pause here. Look at The Starry Night."
    assert clean_guide(once) == once
    for t in DESCRIPTIVE:
        assert clean_guide(clean_guide(t)) == clean_guide(t)


def test_is_truncated():
    assert not is_truncated("It hangs in the Musée d'Orsay.")
    assert not is_truncated("What would it feel like to sit in that garden right now?")
    assert not is_truncated('He wrote: "Beware, the Lord sees."')
    assert not is_truncated("Let the swirls speak.\n")
    assert is_truncated("one of the most visited museums in")
    assert is_truncated("Titian painted it in the summer of 1548, just")
