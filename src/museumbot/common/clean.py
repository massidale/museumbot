"""Pulizia del testo generato: resta solo il testo parlato dell'audioguida.

Una sola funzione per tutti i corpus, applicata sempre al testo originale. Toglie il
preambolo ("Here is the audio guide for ..."), i separatori, il markdown, i conteggi di
parole, le intestazioni da copione ("Audio Guide Script (approx. 250 words):") e le
didascalie di regia ("(Soft, inviting tone)", "(Fade out)").

Il preambolo si toglie solo se e' davvero un'introduzione al testo: alcuni testi aprono la
descrizione con "Here is a painting that rewards a second look.", e quella riga va tenuta.
"""

import re

PREAMBLE = re.compile(r"\A\s*(?:here is|here['’]s|below is|sure\b)[^\n]*\n+", re.I)
# Parole che dicono che la riga parla del testo, non dell'opera ("context" non conta).
PREAMBLE_META = re.compile(
    r"\b(?:audio[ -]?guides?|guide|script|text|transcript|narration|version|\d+[- ]words?)\b",
    re.I,
)
RULE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", re.M)
WORD_COUNT = re.compile(r"\n\s*[(\[]?\s*word count\b[^\n]*\s*\Z", re.I)
HEADER = re.compile(r"\A\s*audio ?guide\b[^\n]*\n+", re.I)
STAGE_LINE = re.compile(r"^\s*[(\[][^\n]*[)\]]\s*$", re.M)
STAGE_INLINE = re.compile(
    r"\s*[(\[](?:[^)\]]*\b(?:pause|pauses|tone|voice|music|silence|fade|fades|beat)\b"
    r"|end\b)[^)\]]*[)\]]", re.I)
# Fine testo regolare: punteggiatura di chiusura, virgolette o parentesi.
CLOSED = re.compile(r"[.!?…\"”’)\]]\s*\Z")


def is_preamble(text: str) -> re.Match | None:
    """Il preambolo iniziale, se la prima riga e' un'introduzione al testo."""
    m = PREAMBLE.match(text)
    if not m:
        return None
    line = m.group(0).strip()
    first_sentence = re.split(r"(?<=[.!?:])\s", line, maxsplit=1)[0]
    next_line = text[m.end():].split("\n", 1)[0]
    if (PREAMBLE_META.search(first_sentence) or line.endswith(":")
            or RULE.fullmatch(next_line)):
        return m
    return None


def clean_guide(text: str) -> str:
    """Testo parlato dell'audioguida: senza preambolo, separatori, markdown, conteggi."""
    m = is_preamble(text)
    if m:
        text = text[m.end():]
    text = RULE.sub("", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.M)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", text)
    # asterischi spaiati (corsivi aperti e mai chiusi): attaccati a una parola da un lato
    # e a uno spazio o a un estremo di riga dall'altro; "3*4" resta
    text = re.sub(r"(?:(?<=\s)|^)\*+(?=\S)", "", text, flags=re.M)
    text = re.sub(r"(?<=\S)\*+(?=\s|$)", "", text, flags=re.M)
    text = WORD_COUNT.sub("", text.strip())  # dopo il markdown: "*(Word count: 250)*"
    text = HEADER.sub("", text, count=1)      # dopo il markdown: "**Audio Guide: ...**"
    text = STAGE_LINE.sub("", text)
    text = STAGE_INLINE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def is_truncated(text: str) -> bool:
    """Testo interrotto a meta' frase: non finisce con punteggiatura di chiusura."""
    return not CLOSED.search(text)
