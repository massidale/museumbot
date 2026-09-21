"""Embedding dei testi generati, in locale.

OpenRouter non espone modelli di embedding (verificato: 0 su ~500), quindi si gira in
locale — che e' anche gratis e riproducibile.

Nessuna instruction prefix: Qwen3-Embedding la supporta, ma un prefisso diverso per
condizione introdurrebbe esattamente lo shift che l'esperimento vuole misurare. Il prefisso
resta assente per tutti, in modo uniforme.

Output: data/emb_{alias}.npy (float32, L2-normalizzato) + data/meta.csv, allineati riga
per riga.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "data" / "generations.jsonl"

MODELS = {
    "qwen": "Qwen/Qwen3-Embedding-0.6B",
    "bge-m3": "BAAI/bge-m3",
}


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
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    all_rows = [json.loads(l) for l in GEN.open()]
    rows = [r for r in all_rows if r.get("model") == args.generator_model]
    if not rows:
        available = sorted({r.get("model", "<mancante>") for r in all_rows})
        raise SystemExit(
            f"nessun testo per {args.generator_model!r}; modelli disponibili: {available}"
        )

    keys = [(r["artwork_id"], r["condition"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit(
            f"righe duplicate per (opera, condizione) nel modello {args.generator_model!r}"
        )
    # ordine deterministico: l'analisi assume un reshape (opera, condizione)
    rows.sort(key=lambda r: (r["artwork_id"], r["condition"]))
    print(
        f"{len(rows)} testi da {len({r['artwork_id'] for r in rows})} opere "
        f"[generatore: {args.generator_model}]"
    )

    from sentence_transformers import SentenceTransformer

    name = MODELS[args.embedding_model]
    print(f"carico {name} su {args.device} ...")
    m = SentenceTransformer(name, device=args.device)

    emb = m.encode(
        [r["text"] for r in rows],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    out = ROOT / "data" / f"emb_{args.embedding_model}.npy"
    np.save(out, emb)
    pd.DataFrame(
        [
            {
                "artwork_id": r["artwork_id"],
                "title": r["title"],
                "artist": r["artist"],
                "condition": r["condition"],
                "words": r["words"],
                "generator_model": r["model"],
            }
            for r in rows
        ]
    ).to_csv(ROOT / "data" / "meta.csv", index=False)

    norms = np.linalg.norm(emb, axis=1)
    print(f"\n{out.relative_to(ROOT)}  shape {emb.shape}")
    print(f"  norme L2: min {norms.min():.4f} max {norms.max():.4f}")
    print(f"  NaN: {int(np.isnan(emb).sum())}")


if __name__ == "__main__":
    main()
