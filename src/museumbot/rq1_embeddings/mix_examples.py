"""Esempi di testi misti per un'opera: tutti gli alpha per ogni coppia, colorati per token.

Per ogni coppia di `mixed_judge.PAIRS` e alpha = 0, 0.25, 0.5, 0.75, 1 (vertici = testi puri
del corpus `local`): parole, posizione sul segmento v_X -> v_Y (come `mixed.py`),
copertura e quota di token preferiti da ciascun esperto. Nei testi misti ogni token ha
uno sfondo blu se lo preferiva la prima categoria (X), rosso se la seconda (Y), con
opacita' proporzionale al margine |log p_X - log p_Y| (satura a CAP); al passaggio del
mouse si leggono le due log-probabilita'.

Uso: python -m museumbot.rq1_embeddings.mix_examples --title "Starry Night"
Scrive docs/analisi/testi-misti-<titolo>.md (HTML inline: colori nell'anteprima di VS Code,
non su GitHub) e la stessa pagina in .html.
"""

import argparse
import html
import json
import re

import numpy as np

from museumbot.common.config import ROOT
from museumbot.common.corpus import load_corpus
from museumbot.common.prompts import CONDITIONS, LABELS
from museumbot.generation.vllm_gen import MODEL
from museumbot.rq1_embeddings.analyze import FLAT, load, steering
from museumbot.rq1_embeddings.mixed import MIXED, embed_mixed, position
from museumbot.rq2_judge.mixed_judge import PAIRS

CAP, AMAX, NEUTRAL = 5.0, 0.62, 0.1
BLUE, RED = "42,120,214", "214,69,58"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
DOCS = ROOT / "docs" / "analisi"


def span(text: str, m: float, tip: str = "") -> str:
    if abs(m) < NEUTRAL:
        return html.escape(text)
    a = round(AMAX * min(abs(m), CAP) / CAP, 3)
    t = f' title="{html.escape(tip)}"' if tip else ""
    return f'<span style="background:rgba({BLUE if m > 0 else RED},{a})"{t}>{html.escape(text)}</span>'


def colorize(toks: list[str], margins, tips=None) -> str:
    """Testo con uno sfondo per token; gli a capo restano fuori dagli span."""
    out = []
    for k, (t, m) in enumerate(zip(toks, margins)):
        tip = tips[k] if tips else ""
        parts = t.split("\n")
        for i, p in enumerate(parts):
            if i:
                out.append("\n")
            if p:
                out.append(span(p, m, tip))
    return "".join(out)


def legend(x: str, y: str) -> str:
    steps = [CAP, CAP / 2, CAP / 5, 0.0, -CAP / 5, -CAP / 2, -CAP]
    cells = " ".join(span(f"{v:+g}", v) if v else "0" for v in steps)
    return f"{LABELS[x]} (blu) ← {cells} → {LABELS[y]} (rosso)"


def sections(title_query: str, alias: str = "qwen") -> dict:
    """Dati del documento: opera e, per coppia, righe di tabella e testi colorati."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    local = [r for r in load_corpus("local", verbose=False) if r["variant"] == "full"]
    pure = {(r["artwork_id"], r["condition"]): r for r in local}
    art = next(r for r in local if title_query.lower() in r["title"].lower())
    aid = art["artwork_id"]
    mixed_all = [r for r in map(json.loads, MIXED.open()) if not r["usage"]["failed"]]
    E = embed_mixed(alias, mixed_all)
    emb = {(r["artwork_id"], r["condition"], r["alpha"]): E[i] for i, r in enumerate(mixed_all)}
    Xp, arts, _ = load(alias, "full", "local")
    a_i, V = arts.index(aid), steering(Xp)
    vec = lambda c: V[CONDITIONS.index(c) - (CONDITIONS.index(c) > FLAT)]
    out = {"title": art["title"], "artist": art["artist"], "pairs": []}
    for x, y in PAIRS:
        name, rows, texts = f"{x}+{y}", [], []
        for al in ALPHAS:
            if al in (0.0, 1.0):
                c = x if al == 0 else y
                r, e = pure[(aid, c)], Xp[a_i, CONDITIONS.index(c)]
                cov = share = "—"
                body = html.escape(r["text"])
                head = f"{LABELS[c]} (testo puro)"
            else:
                r = next(m for m in mixed_all if m["artwork_id"] == aid
                         and m["condition"] == name and m["alpha"] == al)
                e = emb[(aid, name, al)]
                ids = tok.encode(r["usage"]["raw"], add_special_tokens=False)
                d = r["usage"]["diag"]
                marg = np.array([t["logp"][0] - t["logp"][1] for t in d])
                tips = [f"log p {LABELS[x]} {t['logp'][0]:.2f} · log p {LABELS[y]} "
                        f"{t['logp'][1]:.2f} · margine {mg:+.2f}"
                        + ("" if all(t["covered"]) else " · stimato")
                        for t, mg in zip(d, marg)]
                body = (colorize([tok.decode([i]) for i in ids], marg, tips)
                        if len(ids) == len(d) else html.escape(r["text"]))
                cov = f"{r['usage']['covered_all']:.1%}"
                share = (f"{np.mean(marg >= 0.5):.0%} / {np.mean(marg <= -0.5):.0%} / "
                         f"{np.mean(np.abs(marg) < 0.5):.0%}")
                head = "testo misto"
            pos = float(position(e - Xp[a_i, FLAT], vec(x), vec(y)))
            rows.append([f"{al:g}", str(r["words"]), f"{pos:.2f}", cov, share])
            texts.append((f"α = {al:g} — {head}", body.strip()))
        out["pairs"].append({"x": x, "y": y, "rows": rows, "texts": texts})
    return out


INTRO = (
    "Per ogni coppia di categorie, i testi da α = 0 (prima categoria, X) a α = 1 (seconda, "
    "Y); i vertici sono i testi puri. Ogni testo misto è generato token per token dalla "
    "media dei logit dei due prompt di categoria, con pesi (1 − α, α). Nei testi misti "
    "ogni token ha uno sfondo <b>blu</b> se lo preferiva X, <b>rosso</b> se lo preferiva Y; "
    "l'intensità è il margine |log p_X − log p_Y| del token scelto (satura a 5, ~150 volte "
    "più probabile). Senza sfondo: i due esperti erano d'accordo. Passando sopra un token "
    "si leggono le due log-probabilità. <b>Posizione</b>: proiezione dell'embedding (Qwen3, "
    "centrato sul flat dell'opera) sul segmento fra gli steering vector delle due categorie, "
    "0 su X e 1 su Y. <b>Token</b>: quota di token preferiti da X, da Y (|margine| ≥ 0.5) "
    "o da nessuno dei due.")
SOURCE = ("Generato da <code>rq1_embeddings/mix_examples.py</code> sui dati di "
          "<code>data/local/</code> (Gemma 4 31B). Metodo e risultati aggregati in <code>docs/report-gemma.md</code>, §4.")
COLS = ["α", "parole", "posizione", "copertura", "token X / Y / accordo"]


def render_md(doc: dict) -> str:
    L = [f"# Testi misti su *{doc['title']}* ({doc['artist']})", "", f"*{SOURCE}*", "", INTRO,
         "", "*I colori si vedono nell'anteprima Markdown di VS Code o nella versione .html "
         "accanto a questo file; GitHub non mostra gli stili.*", ""]
    for p in doc["pairs"]:
        x, y = p["x"], p["y"]
        L += [f"## {LABELS[x]} → {LABELS[y]}", "", legend(x, y), "",
              "| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS),
              *["| " + " | ".join(r) + " |" for r in p["rows"]], ""]
        for head, body in p["texts"]:
            L += [f"### {head}", "", body, ""]
    return "\n".join(L) + "\n"


def render_html(doc: dict) -> str:
    def para(body):
        return "<p>" + body.replace("\n\n", "</p><p>").replace("\n", "<br>") + "</p>"

    parts = [f"<h1>Testi misti su <i>{html.escape(doc['title'])}</i> "
             f"({html.escape(doc['artist'])})</h1>", f"<p class='src'>{SOURCE}</p>",
             f"<p>{INTRO}</p>"]
    for p in doc["pairs"]:
        x, y = p["x"], p["y"]
        rows = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in p["rows"])
        parts += [f"<h2>{LABELS[x]} → {LABELS[y]}</h2>", f"<p class='leg'>{legend(x, y)}</p>",
                  "<table><tr>" + "".join(f"<th>{c}</th>" for c in COLS) + f"</tr>{rows}</table>"]
        for head, body in p["texts"]:
            parts += [f"<h3>{html.escape(head)}</h3>", f"<div class='txt'>{para(body)}</div>"]
    css = ("body{font-family:system-ui,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;"
           "line-height:1.6;color:#1f1f1e;background:#fcfcfb}h2{margin-top:2.5rem}"
           ".txt{line-height:1.95;border:1px solid #ddd;border-radius:10px;padding:.5rem 1rem}"
           "table{border-collapse:collapse}td,th{border-bottom:1px solid #ddd;padding:4px 10px;"
           "text-align:left}.src,.leg{color:#666;font-size:.9rem}span[title]{cursor:help}")
    return (f"<!doctype html><html lang='it'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>Testi misti — "
            f"{html.escape(doc['title'])}</title><style>{css}</style></head><body>"
            + "\n".join(parts) + "</body></html>\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--title", default="Starry Night")
    ap.add_argument("--model", default="qwen")
    args = ap.parse_args()
    doc = sections(args.title, args.model)
    slug = re.sub(r"[^a-z0-9]+", "-", doc["title"].lower()).strip("-")
    DOCS.mkdir(parents=True, exist_ok=True)
    for ext, text in (("md", render_md(doc)), ("html", render_html(doc))):
        path = DOCS / f"testi-misti-{slug}.{ext}"
        path.write_text(text)
        print(path)


if __name__ == "__main__":
    main()
