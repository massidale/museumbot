"""Embedding dei testi generati, in locale.

OpenRouter non espone modelli di embedding (verificato: 0 su ~500), quindi si gira in
locale — che e' anche gratis e riproducibile.

Nessuna instruction prefix: Qwen3-Embedding la supporta, ma un prefisso diverso per
condizione introdurrebbe esattamente lo shift che l'esperimento vuole misurare. Il prefisso
resta assente per tutti, in modo uniforme.

Output: data/emb_{alias}.npy + data/meta.csv per la variante `full`; per le altre varianti
data/emb_{alias}_{variante}.npy + data/meta_{variante}.csv. Le righe flat vengono sempre
dalla variante `full`, anche per le repliche (es. `full_rep`).
"""

import argparse
import json

import numpy as np
import pandas as pd

from museumbot.common.config import ROOT, emb_path, meta_path
from museumbot.common.prompts import REPLICATES, VARIANTS

GEN = ROOT / "data" / "generations.jsonl"

MODELS = {
    "qwen": "Qwen/Qwen3-Embedding-0.6B",
    "bge-m3": "BAAI/bge-m3",
}


def select_rows(all_rows: list[dict], generator_model: str, variant: str) -> list[dict]:
    """Righe di una variante. Le righe flat vengono sempre dalla variante `full`,
    perche' il flat non ha blocco e quindi non ha varianti. Righe storiche senza campo
    `variant` valgono come `full`. Ordine deterministico (opera, condizione)."""
    rows = []
    for r in all_rows:
        if r.get("model") != generator_model:
            continue
        v = r.get("variant", "full")
        if r["condition"] == "flat":
            if v == "full":
                rows.append(r)
        elif v == variant:
            rows.append(r)
    keys = [(r["artwork_id"], r["condition"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit(f"righe duplicate per (opera, condizione) in variante {variant!r}")
    rows.sort(key=lambda r: (r["artwork_id"], r["condition"]))
    return rows


def embed_rows(model, rows: list[dict], alias: str, variant: str, batch_size: int) -> None:
    emb = model.encode(
        [r["text"] for r in rows],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    out = emb_path(alias, variant)
    np.save(out, emb)
    pd.DataFrame(
        [
            {
                "artwork_id": r["artwork_id"],
                "title": r["title"],
                "artist": r["artist"],
                "condition": r["condition"],
                "variant": r.get("variant", "full"),
                "words": r["words"],
                "generator_model": r["model"],
            }
            for r in rows
        ]
    ).to_csv(meta_path(variant), index=False)

    norms = np.linalg.norm(emb, axis=1)
    print(f"\n{out.relative_to(ROOT)}  shape {emb.shape}")
    print(f"  norme L2: min {norms.min():.4f} max {norms.max():.4f}   NaN: {int(np.isnan(emb).sum())}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--embedding-model", "--model", dest="embedding_model",
        default="qwen", choices=list(MODELS),
        help="modello locale usato per calcolare gli embedding",
    )
    ap.add_argument(
        "--generator-model",
        default="deepseek/deepseek-v4-flash",
        help="usa soltanto i testi prodotti da questo modello generativo",
    )
    ap.add_argument("--variant", default="full", choices=[*VARIANTS, *REPLICATES])
    ap.add_argument("--all-variants", action="store_true",
                    help="embedda ogni variante presente in generations.jsonl")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    if args.all_variants and args.variant != "full":
        print("--variant ignorato perche' e' impostato --all-variants")

    all_rows = [json.loads(l) for l in GEN.open()]
    if args.all_variants:
        present = {r.get("variant", "full") for r in all_rows if r.get("model") == args.generator_model}
        variants = [v for v in [*VARIANTS, *REPLICATES] if v in present]
    else:
        variants = [args.variant]

    from sentence_transformers import SentenceTransformer

    name = MODELS[args.embedding_model]
    print(f"carico {name} su {args.device} ...")
    m = SentenceTransformer(name, device=args.device)

    for v in variants:
        rows = select_rows(all_rows, args.generator_model, v)
        if not rows:
            print(f"[{v}] nessun testo, salto")
            continue
        n_art = len({r["artwork_id"] for r in rows})
        print(f"\n[{v}] {len(rows)} testi da {n_art} opere [generatore: {args.generator_model}]")
        embed_rows(m, rows, args.embedding_model, v, args.batch_size)


if __name__ == "__main__":
    main()
