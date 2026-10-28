"""Confronti per il giudice sui testi misti (passo 4 del profilo continuo).

Per ogni opera e coppia (X, Y), con alpha in ALPHAS (0 e 1 sono testi puri del corpus
`local`, gli interni testi misti):

    curve      persona X: S_x(alpha) vs Y puro; persona Y: S_y(alpha) vs X puro.
               S_x(1) e S_y(0) sono la replica (`|rep`): mai un testo contro se stesso.
    double     S(alpha) vs flat, persona X e persona Y separatamente (doppio visitatore)
    coherence  testo misto (alpha interni) vs X puro e vs Y puro, giudice neutro:
               "quale suona come una voce sola"

Etichette dei testi: "<categoria>" (full), "<categoria>|rep" (full_rep), "flat",
"<X>+<Y>|<alpha>" (misto). Un solo ordine A/B per confronto, scelto dall'hash del
pair_id (bilanciato in media); la posizione si analizza come in RQ2.
"""

import hashlib

from museumbot.rq2_judge.personas import NEUTRAL

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
INNER = (0.25, 0.5, 0.75)
ORDERS = ("target_first", "target_second")


def series(pair, alpha: float, side: str | None = None) -> str:
    """Testo della serie in alpha. side="x": la replica sta in alpha=1, side="y" in 0."""
    x, y = pair
    if alpha == 0:
        return f"{x}|rep" if side == "y" else x
    if alpha == 1:
        return f"{y}|rep" if side == "x" else y
    return f"{x}+{y}|{alpha:g}"


def order_of(pair_id: str) -> str:
    return ORDERS[int(hashlib.sha256(pair_id.encode()).hexdigest(), 16) % 2]


def item(pid, a, typ, persona, target, other, **extra) -> dict:
    return {"pair_id": pid, "artwork_id": a, "type": typ, "persona": persona,
            "target": [a, target], "other": [other[0], other[1]] if isinstance(other, tuple)
            else [a, other], "order": order_of(pid), **extra}


def build_mixed_pairs(artworks, pairs) -> list[dict]:
    out = []
    for i, a in enumerate(artworks):
        for x, y in pairs:
            p, name = (x, y), f"{x}+{y}"
            for al in ALPHAS:
                out.append(item(f"{a}|{name}|curve|{x}|{al:g}", a, "curve", x,
                                series(p, al, "x"), y, alpha=al, pair=list(p)))
                out.append(item(f"{a}|{name}|curve|{y}|{al:g}", a, "curve", y,
                                series(p, al, "y"), x, alpha=al, pair=list(p)))
                for persona in p:
                    out.append(item(f"{a}|{name}|double|{persona}|{al:g}", a, "double",
                                    persona, series(p, al), "flat", alpha=al, pair=list(p)))
            for al in INNER:
                for pure in p:
                    out.append(item(f"{a}|{name}|coherence|{pure}|{al:g}", a, "coherence",
                                    NEUTRAL, series(p, al), pure, alpha=al, pair=list(p)))
    return out

