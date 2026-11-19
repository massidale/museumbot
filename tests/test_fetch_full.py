from museumbot.corpus.fetch_full import strip_page


def test_strip_page_toglie_coda_e_titoli_vuoti():
    raw = ("Intro.\n\n== History ==\nFatti.\n\n=== Gallery ===\n\n== Style ==\n"
           "=== Colour ===\nRosso.\n\n== See also ==\nAltro\n\n== References ==\nX")
    assert strip_page(raw) == ("Intro.\n\n== History ==\nFatti.\n\n== Style ==\n"
                               "=== Colour ===\nRosso.")


def test_strip_page_tiene_titolo_con_testo_solo_nelle_sottosezioni():
    raw = "A.\n== Analysis ==\n=== Light ===\nLuce.\n== Notes ==\nn"
    assert strip_page(raw) == "A.\n== Analysis ==\n=== Light ===\nLuce."


def test_strip_page_toglie_coda_tedesca():
    raw = "Intro.\n== Das Werk ==\nBild.\n== Literatur ==\nBuch\n== Weblinks ==\nx"
    assert strip_page(raw) == "Intro.\n== Das Werk ==\nBild."


def test_strip_page_toglie_coda_italiana_e_spagnola():
    it = "Intro.\n== Descrizione ==\nQuadro.\n== Note ==\nn\n== Bibliografia ==\nb"
    es = "Intro.\n== Descripción ==\nCuadro.\n== Referencias ==\nr"
    assert strip_page(it) == "Intro.\n== Descrizione ==\nQuadro."
    assert strip_page(es) == "Intro.\n== Descripción ==\nCuadro."
