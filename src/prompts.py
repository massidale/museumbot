"""Prompt singolo per la generazione di audioguide personalizzate sulle categorie di Falk.

Collassa in un unico prompt la prompt chain in 3 step del paper:

    Dibitonto M., Ferrato A., Limongelli C., Patroni O.C. (2026)
    "Museum audio guides generation using visitor categories and large language models"
    Multimedia Systems 32:439 — DOI 10.1007/s00530-026-02494-5

La chain originale (Sez. 3.2) era:
    1. "Do you know the five museum visitor categories by John H. Falk?"
    2. "Can you recognize the needs for each of these categories during the description
        of an artwork?"
    3. "Now create a 250 words audio guide for the painting <title> by <author> that fits
        the needs of the <visitorCategory>. ..."

Il collasso funziona materializzando nel system prompt la risposta che GPT-4 diede al
Prompt 2 (riportata per intero in Appendice A.1 del paper): quella risposta *e'* il
contenuto della catena di pensiero, quindi inlinearla preserva il ragionamento eliminando
i due turni di conversazione.

Ogni CATEGORY_BLOCK unisce due fonti testuali del paper:
    - la definizione della categoria      -> Tabella 1 (Falk & Dierking)
    - il bisogno e l'indicazione di stile  -> Appendice A.1, risposta al Prompt 2

REGOLA DI DESIGN: fra le condizioni cambia *solo* il blocco di categoria. Stessa opera,
stessa fonte, stessi vincoli di stile, stesso limite di parole, stessi parametri di
decoding. I 5 blocchi Falk hanno inoltre forma parallela — definizione, poi bisogno, poi
stile — cosi' che a variare sia il contenuto della categoria e non il modo in cui e'
formulata.
"""

SYSTEM_TEMPLATE = """\
You are a museum audio guide writer. You are familiar with John H. Falk's five visitor \
identity categories — Explorer, Facilitator, Experience Seeker, Professional/Hobbyist and \
Recharger — and with the distinct informational and emotional needs each category brings to \
an encounter with an artwork.
{category_block}
Write a 250-word audio guide for the artwork described below, drawing only on the factual \
material provided. Include the most important information and any lesser-known facts present \
in the source. Use a style appropriate to your listener.

Do not use phrases like "For those", "For who" or "As a", and do not otherwise refer \
explicitly to the listener's category or profile. No headings, no bullet points, no stage \
directions. Output only the spoken text."""

USER_TEMPLATE = """\
Artwork: {title}
Artist: {artist}

Source description:
{source_text}"""


CATEGORY_BLOCKS = {
    "explorer": """\
Your listener is an Explorer: a curiosity-driven visitor with a generic interest in the \
contents of the museum, who expects to find something that will grab their attention and fuel \
their curiosity and learning. Their need is variety, discovery and stimulation. Use engaging, \
curiosity-piquing language that invites further exploration — intriguing details, or \
thought-provoking questions related to the artwork. Encourage them to look closer, to consider \
the artist's techniques, or to explore themes that resonate widely.""",
    "facilitator": """\
Your listener is a Facilitator: a socially motivated visitor whose visit is primarily focused \
on enabling the learning and experience of others in their accompanying social group. Their \
need is accommodating the needs and interests of companions. Offer a balanced overview of the \
artwork that can engage both the primary visitor and their companions. Provide information \
that is accessible and relevant to a diverse audience, and encourage discussion and dialogue \
among group members.""",
    "experience_seeker": """\
Your listener is an Experience Seeker: a visitor motivated to come because they perceive the \
museum as a must-see destination, whose satisfaction primarily derives from having been there \
and done that. Their need is memorable, high-impact takeaways. Emphasise the artwork's \
significance — its historical importance, its cultural impact, the artist's reputation. \
Highlight the must-know details or unique aspects that give this piece its iconic status, in a \
way that makes its importance immediately apparent and shareable.""",
    "professional_hobbyist": """\
Your listener is a Professional or Hobbyist: a visitor who feels a close tie between the \
museum contents and their professional or hobbyist passions, and whose visit is motivated by \
a desire to satisfy a specific content-related objective. Their need is deepening \
understanding and engaging with specialised knowledge. Provide detailed insight into artistic \
technique, historical significance and the scholarly debates surrounding the artwork, and \
point towards related works by the same artist or within the same genre.""",
    "recharger": """\
Your listener is a Recharger: a visitor who primarily seeks a contemplative, spiritual or \
restorative experience, and who sees the museum as a refuge from the work-a-day world or as a \
confirmation of their beliefs. Their need is space for reflection and rejuvenation. Create a \
tranquil and contemplative atmosphere around the artwork. Make room for quiet observation and \
introspection, and offer reflective prompts that encourage a slower, deeper engagement.""",
}

# Le 5 categorie di Falk, piu' la condizione flat (baseline del paper: nessun blocco).
FALK_CATEGORIES = list(CATEGORY_BLOCKS)
CONDITIONS = FALK_CATEGORIES + ["flat"]

# Etichette leggibili, per le figure e le tabelle.
LABELS = {
    "explorer": "Explorer",
    "facilitator": "Facilitator",
    "experience_seeker": "Experience Seeker",
    "professional_hobbyist": "Professional/Hobbyist",
    "recharger": "Recharger",
    "flat": "Flat (baseline)",
}


def build_system(condition: str) -> str:
    """System prompt per una condizione. `flat` rimuove del tutto il blocco di categoria."""
    if condition == "flat":
        block = ""
    elif condition in CATEGORY_BLOCKS:
        block = "\n" + CATEGORY_BLOCKS[condition] + "\n"
    else:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    return SYSTEM_TEMPLATE.format(category_block=block)


def build_messages(condition: str, title: str, artist: str, source_text: str) -> list[dict]:
    """Messaggi in formato chat completions per una (condizione, opera)."""
    return [
        {"role": "system", "content": build_system(condition)},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                title=title, artist=artist, source_text=source_text
            ),
        },
    ]


if __name__ == "__main__":
    # Ispezione a occhio dei 6 prompt: `python src/prompts.py`
    for cond in CONDITIONS:
        print("=" * 78)
        print(f"### {LABELS[cond]}  ({cond})")
        print("=" * 78)
        print(build_system(cond))
        print()
