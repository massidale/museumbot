"""Esplorazione dei dati da terminale: una base DuckDB derivata dai file del progetto.

I file in `data/` e `results/` restano la fonte; `data/explore.duckdb` e' una vista
rigenerabile (`build`) con una riga per testo e le grandezze per testo che `analyze.py`
calcola al volo e non salva. Tabelle:

    artworks   le opere con il testo sorgente
    prompts    (condizione, variante) -> system prompt, ricostruito da prompts.py
    texts      un testo per riga, per ogni corpus, pulito come nelle analisi
    scores     per testo e modello di embedding: norma dello shift rispetto al flat
               dell'opera, coseno con lo steering vector della propria categoria (cos_own)
               e con quelli di tutte le categorie (cos_<categoria>)
    judgments  i giudizi di RQ2

Uso:
    python -m museumbot.explore build
    python -m museumbot.explore sql "SELECT ... FROM scores s JOIN texts t USING (text_id) ..."
"""

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from museumbot.common.config import ROOT, emb_path
from museumbot.common.corpus import CORPORA, SOURCES, load_corpus
from museumbot.common.prompts import (
    CONDITIONS, FALK_CATEGORIES, REPLICATES, VARIANTS, build_system, prompt_variant,
)
from museumbot.rq1_embeddings.analyze import FALK_IDX, center, load, steering

DB = ROOT / "data" / "explore.duckdb"
MODELS = ("qwen", "bge-m3")
TEXT_COLS = ("text", "system_prompt", "source_text", "reason")


def text_id(corpus: str, variant: str, artwork_id: str, condition: str) -> str:
    return f"{corpus}|{variant}|{artwork_id}|{condition}"


def score_rows(X, arts: list[str], model: str) -> list[dict]:
    """Grandezze per testo: shift rispetto al flat dell'opera contro gli steering vector."""
    Xc = center(X, "flat")
    V = steering(X)
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    out = []
    for a, art in enumerate(arts):
        for k, ci in enumerate(FALK_IDX):
            e = Xc[a, ci]
            norm = float(np.linalg.norm(e))
            cos = (Vn @ e) / norm if norm > 0 else np.zeros(len(FALK_IDX))
            row = {"artwork_id": art, "condition": CONDITIONS[ci], "model": model,
                   "norm": norm, "cos_own": float(cos[k])}
            row.update({f"cos_{c}": float(cos[j]) for j, c in enumerate(FALK_CATEGORIES)})
            out.append(row)
    return out


def prompt_rows() -> list[dict]:
    """System prompt per ogni (condizione, variante), repliche comprese."""
    variants = list(VARIANTS) + list(REPLICATES)
    return [{"condition": c, "variant": v, "system_prompt": build_system(c, prompt_variant(v))}
            for c in CONDITIONS for v in variants]


def text_rows(corpus: str, path: Path | None = None) -> list[dict]:
    """Righe pulite di un corpus, con il corpus e un identificatore per testo."""
    out = []
    for r in load_corpus(corpus, path, verbose=False):
        out.append({"text_id": text_id(corpus, r["variant"], r["artwork_id"], r["condition"]),
                    "corpus": corpus, **r})
    return out


def artwork_rows() -> list[dict]:
    rows = [json.loads(l) for l in (ROOT / "data" / "artworks.jsonl").open()]
    return [{"artwork_id": r["qid"], "title": r["title"], "artist": r["artist"],
             "museum": r.get("museum"), "source_text": r["source_text"]} for r in rows]


def judgment_rows() -> list[dict]:
    path = ROOT / "data" / "judgments.jsonl"
    if not path.exists():
        return []
    out = []
    for l in path.open():
        r = json.loads(l)
        out.append({"pair_id": r["pair_id"], "artwork_id": r["artwork_id"], "type": r["type"],
                    "persona": r["persona"], "target": "|".join(r["target"]),
                    "other": "|".join(r["other"]), "order": r.get("order"),
                    "model": r.get("model"), "chose_target": r.get("chose_target"),
                    "valid": r.get("valid"), "reason": r.get("reason")})
    return out


def corpus_scores(corpus: str, models: tuple[str, ...]) -> list[dict]:
    """Punteggi di ogni variante di cui esistono gli embedding, per i modelli dati."""
    out = []
    variants = list(VARIANTS) + list(REPLICATES)
    for model in models:
        for v in variants:
            if not emb_path(model, v, corpus).exists():
                continue
            X, arts, _ = load(model, v, corpus)
            for r in score_rows(X, arts, model):
                r["text_id"] = text_id(corpus, v, r.pop("artwork_id"), r.pop("condition"))
                out.append(r)
    return out


def build(db: Path = DB, corpora: tuple[str, ...] = CORPORA,
          models: tuple[str, ...] = MODELS) -> None:
    """Rigenera la base da zero."""
    if db.exists():
        db.unlink()
    corpora = tuple(c for c in corpora if SOURCES[c].exists())  # salta i corpus non generati
    texts = [r for c in corpora for r in text_rows(c)]
    scores = [r for c in corpora for r in corpus_scores(c, models)]
    tables = {
        "artworks": pd.DataFrame(artwork_rows()),
        "prompts": pd.DataFrame(prompt_rows()),
        "texts": pd.DataFrame(texts),
        "scores": pd.DataFrame(scores),
        "judgments": pd.DataFrame(judgment_rows()),
    }
    con = duckdb.connect(str(db))
    for name, df in tables.items():
        if len(df) == 0:
            continue
        con.register("df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM df")
        con.unregister("df")
    con.close()
    for name, df in tables.items():
        print(f"{name:10s} {len(df):6d} righe")


def query(db: Path, sql: str) -> pd.DataFrame:
    con = duckdb.connect(str(db), read_only=True)
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def format_df(df: pd.DataFrame) -> str:
    """Colonne brevi in tabella; le colonne di testo per intero, un blocco per riga."""
    long = [c for c in df.columns if c in TEXT_COLS]
    short = [c for c in df.columns if c not in long]
    if not long:
        return df.to_string(index=False, float_format=lambda x: f"{x:.3f}")
    blocks = []
    for i, r in df.iterrows():
        head = "  ".join(f"{c}={r[c]:.3f}" if isinstance(r[c], float) else f"{c}={r[c]}"
                         for c in short)
        body = "\n\n".join(f"[{c}]\n{r[c]}" for c in long)
        blocks.append(f"--- {i + 1} --- {head}\n{body}")
    return "\n\n".join(blocks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="rigenera data/explore.duckdb dai file del progetto")
    q = sub.add_parser("sql", help="esegue una query e stampa i testi per intero")
    q.add_argument("query")
    args = ap.parse_args()
    if args.cmd == "build":
        build()
    else:
        if not DB.exists():
            raise SystemExit(f"{DB} non esiste: prima `python -m museumbot.explore build`")
        print(format_df(query(DB, args.query)))


if __name__ == "__main__":
    main()
