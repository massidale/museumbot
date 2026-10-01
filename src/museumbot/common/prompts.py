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

ABLAZIONE: ogni blocco e' spezzato in tre parti (def, need, style) e ricomposto secondo una
variante (vedi VARIANTS). La variante `full` riproduce il prompt originale byte per byte;
`name_only` lascia soltanto il nome della categoria.
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


# Ogni blocco e' spezzato nelle tre parti che l'ablazione manipola. Le stringhe sono
# esattamente quelle del blocco originale: `build_block(c, "full")` le ricompone byte per byte.
CATEGORY_PARTS = {
    "explorer": {
        "name": "an Explorer",
        "def": "a curiosity-driven visitor with a generic interest in the contents of the "
               "museum, who expects to find something that will grab their attention and fuel "
               "their curiosity and learning.",
        "need": "Their need is variety, discovery and stimulation.",
        "style": "Use engaging, curiosity-piquing language that invites further exploration — "
                 "intriguing details, or thought-provoking questions related to the artwork. "
                 "Encourage them to look closer, to consider the artist's techniques, or to "
                 "explore themes that resonate widely.",
    },
    "facilitator": {
        "name": "a Facilitator",
        "def": "a socially motivated visitor whose visit is primarily focused on enabling the "
               "learning and experience of others in their accompanying social group.",
        "need": "Their need is accommodating the needs and interests of companions.",
        "style": "Offer a balanced overview of the artwork that can engage both the primary "
                 "visitor and their companions. Provide information that is accessible and "
                 "relevant to a diverse audience, and encourage discussion and dialogue among "
                 "group members.",
    },
    "experience_seeker": {
        "name": "an Experience Seeker",
        "def": "a visitor motivated to come because they perceive the museum as a must-see "
               "destination, whose satisfaction primarily derives from having been there and "
               "done that.",
        "need": "Their need is memorable, high-impact takeaways.",
        "style": "Emphasise the artwork's significance — its historical importance, its "
                 "cultural impact, the artist's reputation. Highlight the must-know details or "
                 "unique aspects that give this piece its iconic status, in a way that makes "
                 "its importance immediately apparent and shareable.",
    },
    "professional_hobbyist": {
        "name": "a Professional or Hobbyist",
        "def": "a visitor who feels a close tie between the museum contents and their "
               "professional or hobbyist passions, and whose visit is motivated by a desire to "
               "satisfy a specific content-related objective.",
        "need": "Their need is deepening understanding and engaging with specialised knowledge.",
        "style": "Provide detailed insight into artistic technique, historical significance and "
                 "the scholarly debates surrounding the artwork, and point towards related "
                 "works by the same artist or within the same genre.",
    },
    "recharger": {
        "name": "a Recharger",
        "def": "a visitor who primarily seeks a contemplative, spiritual or restorative "
               "experience, and who sees the museum as a refuge from the work-a-day world or as "
               "a confirmation of their beliefs.",
        "need": "Their need is space for reflection and rejuvenation.",
        "style": "Create a tranquil and contemplative atmosphere around the artwork. Make room "
                 "for quiet observation and introspection, and offer reflective prompts that "
                 "encourage a slower, deeper engagement.",
    },
}

# Varianti dell'ablazione: flag (def, need, style). Disegno fattoriale 2x2x2.
# `full` e' il prompt originale; `name_only` lascia solo "Your listener is <name>."
VARIANTS = {
    "full":       (True,  True,  True),
    "no_def":     (False, True,  True),
    "no_need":    (True,  False, True),
    "no_style":   (True,  True,  False),
    "def_only":   (True,  False, False),
    "need_only":  (False, True,  False),
    "style_only": (False, False, True),
    "name_only":  (False, False, False),
}
PARTS = ("def", "need", "style")

# Repliche: stesso prompt di una variante, generazione indipendente. Non entrano nel
# disegno fattoriale. `full_rep` calibra il coseno con il full: il coseno fra due run del
# medesimo prompt a T = 0.7 e' il tetto che nessuna variante puo' superare.
REPLICATES = {"full_rep": "full"}


def prompt_variant(variant: str) -> str:
    """Variante il cui prompt va usato: per una replica, quella che replica."""
    return REPLICATES.get(variant, variant)


def build_block(category: str, variant: str = "full") -> str:
    """Blocco di categoria per una variante: nome sempre presente, parti solo se attive."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {list(VARIANTS)}")
    if category not in CATEGORY_PARTS:
        raise ValueError(f"unknown category {category!r}")
    p = CATEGORY_PARTS[category]
    active = [p[part] for part, on in zip(PARTS, VARIANTS[variant]) if on]
    if not active:
        return f"Your listener is {p['name']}."
    return f"Your listener is {p['name']}: " + " ".join(active)


# Alias di compatibilita': i blocchi completi, come prima del refactor.
CATEGORY_BLOCKS = {c: build_block(c, "full") for c in CATEGORY_PARTS}

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


def build_system(condition: str, variant: str = "full") -> str:
    """System prompt per una (condizione, variante). `flat` non ha blocco e ignora la variante."""
    if condition == "flat":
        block = ""
    elif condition in CATEGORY_PARTS:
        block = "\n" + build_block(condition, variant) + "\n"
    else:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    return SYSTEM_TEMPLATE.format(category_block=block)


def build_messages(
    condition: str, title: str, artist: str, source_text: str, variant: str = "full"
) -> list[dict]:
    """Messaggi in formato chat completions per una (condizione, opera, variante)."""
    return [
        {"role": "system", "content": build_system(condition, variant)},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                title=title, artist=artist, source_text=source_text
            ),
        },
    ]


if __name__ == "__main__":
    # Ispezione a occhio dei 6 prompt: `python -m museumbot.common.prompts`
    for cond in CONDITIONS:
        print("=" * 78)
        print(f"### {LABELS[cond]}  ({cond})")
        print("=" * 78)
        print(build_system(cond))
        print()

    print("\n" + "=" * 78)
    print("### Varianti dell'ablazione (Explorer)")
    for v in VARIANTS:
        print(f"--- {v}\n{build_block('explorer', v)}\n")
