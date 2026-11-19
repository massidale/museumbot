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
REPLICATES = {"full_rep": "full", "full_rep2": "full", "full_rep3": "full"}


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


# Modelli in locale (parte 2): una riga in piu' in coda al system prompt, uguale per tutte
# le condizioni, contro Markdown e preamboli. SYSTEM_TEMPLATE resta quello della parte 1.
LOCAL_SUFFIX = ("Reply with the audio guide text only, as plain text: no Markdown, "
                "no preamble, no closing remarks.")


def local_messages(
    condition: str, title: str, artist: str, source_text: str, variant: str = "full"
) -> list[dict]:
    """`build_messages` con LOCAL_SUFFIX in coda al system prompt."""
    msgs = build_messages(condition, title, artist, source_text, variant)
    msgs[0] = {**msgs[0], "content": msgs[0]["content"] + "\n\n" + LOCAL_SUFFIX}
    return msgs


def _cap_first(s: str) -> str:
    """Maiuscola solo l'iniziale: "an Explorer" -> "An Explorer"."""
    return s[:1].upper() + s[1:]


def pct_block(x: str, y: str, alpha: float) -> str:
    """Blocco a percentuale: ascoltatore (1 - alpha) di x e alpha di y, con le due definizioni."""
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha!r}")
    for c in (x, y):
        if c not in CATEGORY_PARTS:
            raise ValueError(f"unknown category {c!r}")
    if x == y:
        raise ValueError(f"the two categories must differ, got {x!r} twice")
    px, py = round(100 * (1 - alpha)), round(100 * alpha)
    nx, ny = CATEGORY_PARTS[x]["name"], CATEGORY_PARTS[y]["name"]

    def riga(c: str) -> str:
        p = CATEGORY_PARTS[c]
        return f"{_cap_first(p['name'])} is {p['def']} {p['need']} {p['style']}"

    return "\n".join([
        f"Your listener is {px}% {nx} and {py}% {ny}.",
        riga(x),
        riga(y),
        "Weigh the two profiles in the guide in this proportion: "
        f"{px}% {LABELS[x]}, {py}% {LABELS[y]}.",
    ])


def pct_messages(
    x: str, y: str, alpha: float, title: str, artist: str, source_text: str
) -> list[dict]:
    """Messaggi come `local_messages`, con il blocco a percentuale al posto del blocco puro."""
    msgs = build_messages("flat", title, artist, source_text)
    system = SYSTEM_TEMPLATE.format(category_block="\n" + pct_block(x, y, alpha) + "\n")
    msgs[0] = {**msgs[0], "content": system + "\n\n" + LOCAL_SUFFIX}
    return msgs


# Prova "pagina intera" (docs/report-pagina-intera.md): la fonte e' tutta la voce Wikipedia e
# il prompt dice su quali aspetti concentrarsi per categoria. Il system prompt e' uguale per
# tutte le categorie; blocco di categoria e indicazione stanno in coda al messaggio utente,
# dopo la voce, cosi' i due esperti di un testo misto condividono il prefisso (voce compresa)
# nella cache di vLLM.
WIKI_SYSTEM = """\
You are a museum audio guide writer. You are familiar with John H. Falk's five visitor \
identity categories — Explorer, Facilitator, Experience Seeker, Professional/Hobbyist and \
Recharger — and with the distinct informational and emotional needs each category brings to \
an encounter with an artwork.

Write a 250-word audio guide for the artwork described below, drawing only on the factual \
material provided. The source is the full encyclopedia article on the artwork and holds far \
more than 250 words can cover: select from it the material that suits your listener, \
following the focus given at the end of the message. Use a style appropriate to your listener.

Do not use phrases like "For those", "For who" or "As a", and do not otherwise refer \
explicitly to the listener's category or profile. No headings, no bullet points, no stage \
directions. Output only the spoken text.

""" + LOCAL_SUFFIX

WIKI_USER = """\
Artwork: {title}
Artist: {artist}

Source (full encyclopedia article):
{source_text}"""

# Coda del messaggio utente per le categorie di Falk; il flat non ce l'ha.
WIKI_LISTENER = """

Listener:
{block}
Focus: {focus}"""

# Aspetti della voce su cui concentrarsi, per categoria: dal bisogno della categoria (paper,
# Appendice A.1) alle sezioni tipiche di una voce su un dipinto.
FOCUS = {
    "explorer": "pick the surprising and lesser-known material: details in the picture that "
                "are easy to miss, curious episodes in its making or later history, open "
                "questions and competing readings. Point the listener's eye to specific "
                "things to look for.",
    "facilitator": "pick what a group of companions of different ages and backgrounds can "
                   "see and share: who and what is depicted and the story behind the scene, a "
                   "few clear facts about the artist and the time, details everyone can spot. "
                   "Leave out technical analysis and specialist debates, and close with a "
                   "question the group can discuss together.",
    "experience_seeker": "pick why this work is famous: its place in art history, the "
                         "reactions it caused, its influence on later artists and on popular "
                         "culture, and the one or two facts no visitor should leave without.",
    "professional_hobbyist": "pick the specialist material: technique, materials and "
                             "composition, dating and attribution, sources and models, "
                             "provenance and condition, and the scholarly interpretations "
                             "and debates the article reports, with related works by the "
                             "artist or by others.",
    "recharger": "pick the mood of the picture: light, colour, stillness or emotion, and the "
                 "human or spiritual themes it opens. Use few dates, names and figures, and "
                 "leave out provenance, prices and scholarly disputes.",
}


def wiki_messages(
    condition: str, title: str, artist: str, source_text: str, variant: str = "full"
) -> list[dict]:
    """Messaggi della prova a pagina intera: system comune, in coda all'utente il blocco di
    categoria completo e l'indicazione FOCUS. Il flat ha lo stesso system e il messaggio
    utente senza la coda (come nei corpus principali cambia solo la parte di categoria).
    `variant` c'e' per la firma di `local_messages` ed e' ignorata."""
    if condition != "flat" and condition not in FOCUS:
        raise ValueError(f"unknown condition {condition!r}")
    user = WIKI_USER.format(title=title, artist=artist, source_text=source_text)
    if condition != "flat":
        user += WIKI_LISTENER.format(block=build_block(condition, "full"),
                                     focus=FOCUS[condition])
    return [{"role": "system", "content": WIKI_SYSTEM}, {"role": "user", "content": user}]


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
