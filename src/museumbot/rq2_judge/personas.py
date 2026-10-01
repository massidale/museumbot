"""Persona del giudice: nome della categoria di Falk e definizione, niente bisogno ne' stile.

Bisogno e stile sono istruzioni per chi scrive il testo: darle al giudice trasformerebbe il
compito in un riconoscimento lessicale del prompt di generazione. La definizione dice chi
e' il visitatore e perche' viene al museo, come farebbe una persona vera.
"""

from museumbot.common.prompts import CATEGORY_PARTS

NEUTRAL = "neutral"  # baseline: giudice senza persona
OPENING = "You are visiting an art museum."


def persona_prompt(persona: str) -> str:
    """System prompt del giudice per una categoria di Falk o per la baseline neutra."""
    if persona == NEUTRAL:
        return OPENING
    if persona not in CATEGORY_PARTS:
        raise ValueError(f"persona sconosciuta {persona!r}")
    p = CATEGORY_PARTS[persona]
    return f"{OPENING} You are {p['name']}: {p['def']}"
