# Prompt Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure which part of the Falk category prompt block (definition, need, style) produces the latent shift, via a 2×2×2 factorial ablation plus a name-only variant.

**Architecture:** Add a `variant` dimension orthogonal to `condition` across the existing pipeline (prompts → generate → embed → analyze). `full` reproduces the current prompt byte-for-byte so the 600 existing generations are reused. A new `src/ablation.py` runs the existing metrics per variant and adds cosine-with-full and a factorial effects table.

**Tech Stack:** Python 3.11, requests (OpenRouter), sentence-transformers (local embeddings), numpy/pandas/scikit-learn, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-21-prompt-ablation-design.md`

## Global Constraints

- All scripts run from repo root with `.venv/bin/python src/<script>.py`; scripts import siblings by bare name (`from prompts import ...`), so tests must put `src/` on `sys.path`.
- `build_system(c, "full")` must equal the current output for all 6 conditions, byte for byte (snapshot test).
- `flat` has no block and no variants: its 100 texts from the `full` run are shared by every variant.
- Generator model `deepseek/deepseek-v4-flash`, temperature 0.7, same `BANNED` regex, for every variant.
- Existing rows in `data/generations.jsonl` have no `variant` field and are read as `full`. The file is never rewritten.
- For variant `full`, embedding output paths stay `data/emb_{alias}.npy` and `data/meta.csv`. Other variants use `data/emb_{alias}_{variant}.npy` and `data/meta_{variant}.csv`.
- The repo is **not a git repository**. Task 0 initialises one; if the user declines, skip every "Commit" step.
- All user-facing docstrings and comments in Italian, matching the existing code.

---

### Task 0: Git init and pytest

**Files:**
- Create: `.git/` (via `git init`), `tests/conftest.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Initialise the repository**

```bash
cd /Users/massimo/Documents/dev/museumbot
git init
git add -A
git commit -m "chore: initial snapshot before prompt ablation"
```

`.gitignore` already excludes `.env`, `.venv/`, `__pycache__/`. `data/*.npy` (2.4 MB each) and `data/generations.jsonl` (1 MB) are small enough to commit.

- [ ] **Step 2: Add pytest as dev dependency**

In `pyproject.toml`, append after the `dependencies` list:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Install pytest**

```bash
.venv/bin/pip install "pytest>=8.0"
```

- [ ] **Step 4: Create `tests/conftest.py`**

```python
"""Mette src/ sul path: gli script importano i moduli fratelli per nome semplice."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
```

- [ ] **Step 5: Verify pytest runs**

Run: `.venv/bin/pytest -q`
Expected: `no tests ran` (exit code 5), no import errors.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml tests/conftest.py
git commit -m "chore: add pytest scaffolding"
```

---

### Task 1: Snapshot the current system prompts

This snapshot is the oracle that guarantees `full` never drifts. It must be generated **before** `prompts.py` is touched.

**Files:**
- Create: `tests/snapshots/system_full.json`, `tests/test_prompts.py`

- [ ] **Step 1: Generate the snapshot with the current code**

```bash
mkdir -p tests/snapshots
.venv/bin/python -c "
import sys, json; sys.path.insert(0, 'src')
from prompts import CONDITIONS, build_system
json.dump({c: build_system(c) for c in CONDITIONS}, open('tests/snapshots/system_full.json', 'w'), indent=2, ensure_ascii=False)
"
```

- [ ] **Step 2: Verify the snapshot has 6 entries and the flat one has no block**

```bash
.venv/bin/python -c "
import json; d = json.load(open('tests/snapshots/system_full.json'))
assert set(d) == {'explorer','facilitator','experience_seeker','professional_hobbyist','recharger','flat'}
assert 'Your listener' not in d['flat'] and 'Your listener is an Explorer' in d['explorer']
print('ok')
"
```

- [ ] **Step 3: Write the snapshot test**

`tests/test_prompts.py`:

```python
import json
from pathlib import Path

import pytest

from prompts import CONDITIONS, build_system

SNAP = json.loads((Path(__file__).parent / "snapshots" / "system_full.json").read_text())


@pytest.mark.parametrize("cond", CONDITIONS)
def test_full_matches_snapshot(cond):
    """La variante full deve riprodurre byte per byte il prompt originale."""
    assert build_system(cond) == SNAP[cond]
```

- [ ] **Step 4: Run it, expect pass (code unchanged so far)**

Run: `.venv/bin/pytest tests/test_prompts.py -v`
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add tests/snapshots/system_full.json tests/test_prompts.py
git commit -m "test: snapshot current system prompts before ablation refactor"
```

---

### Task 2: Split category blocks into parts and add variants

**Files:**
- Modify: `src/prompts.py:54-99` (replace `CATEGORY_BLOCKS`) and `src/prompts.py:113-135` (`build_system`, `build_messages`)
- Test: `tests/test_prompts.py`

**Interfaces:**
- Produces:
  - `CATEGORY_PARTS: dict[str, dict[str, str]]` — keys `name`, `def`, `need`, `style`
  - `VARIANTS: dict[str, tuple[bool, bool, bool]]` — `(def, need, style)` flags, 8 entries, `"full"` first
  - `build_block(category: str, variant: str = "full") -> str`
  - `build_system(condition: str, variant: str = "full") -> str`
  - `build_messages(condition, title, artist, source_text, variant="full") -> list[dict]`
  - `CATEGORY_BLOCKS` kept as computed alias of `build_block(c, "full")`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_prompts.py`)

```python
from prompts import CATEGORY_PARTS, FALK_CATEGORIES, VARIANTS, build_block


def test_variants_table():
    assert list(VARIANTS) == [
        "full", "no_def", "no_need", "no_style",
        "def_only", "need_only", "style_only", "name_only",
    ]
    assert VARIANTS["full"] == (True, True, True)
    assert VARIANTS["name_only"] == (False, False, False)
    assert VARIANTS["no_style"] == (True, True, False)
    assert VARIANTS["need_only"] == (False, True, False)


@pytest.mark.parametrize("cat", FALK_CATEGORIES)
def test_full_block_equals_legacy_block(cat):
    from prompts import CATEGORY_BLOCKS
    assert build_block(cat, "full") == CATEGORY_BLOCKS[cat]


@pytest.mark.parametrize("cat", FALK_CATEGORIES)
def test_name_only_has_no_parts(cat):
    p = CATEGORY_PARTS[cat]
    block = build_block(cat, "name_only")
    assert block == f"Your listener is {p['name']}."
    for part in ("def", "need", "style"):
        assert p[part] not in block


@pytest.mark.parametrize("cat", FALK_CATEGORIES)
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_parts_present_iff_active_and_in_order(cat, variant):
    p = CATEGORY_PARTS[cat]
    flags = dict(zip(("def", "need", "style"), VARIANTS[variant]))
    block = build_block(cat, variant)
    positions = []
    for part in ("def", "need", "style"):
        if flags[part]:
            assert p[part] in block
            positions.append(block.index(p[part]))
        else:
            assert p[part] not in block
    assert positions == sorted(positions)
    assert block.startswith(f"Your listener is {p['name']}")


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_flat_ignores_variant(variant):
    assert build_system("flat", variant) == build_system("flat", "full")


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        build_block("explorer", "bogus")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_prompts.py -q`
Expected: ImportError on `CATEGORY_PARTS` / `VARIANTS` / `build_block`.

- [ ] **Step 3: Replace `CATEGORY_BLOCKS` in `src/prompts.py`**

Replace the whole `CATEGORY_BLOCKS = {...}` dict (lines 54–99) with:

```python
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
```

Keep the existing `FALK_CATEGORIES = list(CATEGORY_BLOCKS)`, `CONDITIONS`, `LABELS` lines unchanged below it.

- [ ] **Step 4: Thread `variant` through `build_system` and `build_messages`**

Replace the two functions with:

```python
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
```

Also update the module docstring: append a paragraph:

```
ABLAZIONE: ogni blocco e' spezzato in tre parti (def, need, style) e ricomposto secondo una
variante (vedi VARIANTS). La variante `full` riproduce il prompt originale byte per byte;
`name_only` lascia soltanto il nome della categoria.
```

And extend the `__main__` block to print one category across all variants:

```python
    print("\n" + "=" * 78)
    print("### Varianti dell'ablazione (Explorer)")
    for v in VARIANTS:
        print(f"--- {v}\n{build_block('explorer', v)}\n")
```

- [ ] **Step 5: Run all prompt tests**

Run: `.venv/bin/pytest tests/test_prompts.py -q`
Expected: all PASSED, including the 6 snapshot tests. If a snapshot test fails, diff the two strings: a part string was transcribed with a different space or dash. The original blocks use `—` (em dash) and `'` (ASCII apostrophe) in `artist's`.

- [ ] **Step 6: Commit**

```bash
git add src/prompts.py tests/test_prompts.py
git commit -m "feat(prompts): split category blocks into parts, add ablation variants"
```

---

### Task 3: Shared path helpers in `config.py`

**Files:**
- Modify: `src/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `emb_path(alias: str, variant: str = "full") -> Path`
  - `meta_path(variant: str = "full") -> Path`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:

```python
from config import ROOT, emb_path, meta_path


def test_full_paths_are_legacy_names():
    assert emb_path("qwen") == ROOT / "data" / "emb_qwen.npy"
    assert emb_path("qwen", "full") == ROOT / "data" / "emb_qwen.npy"
    assert meta_path() == ROOT / "data" / "meta.csv"


def test_variant_paths_are_suffixed():
    assert emb_path("bge-m3", "no_style") == ROOT / "data" / "emb_bge-m3_no_style.npy"
    assert meta_path("name_only") == ROOT / "data" / "meta_name_only.csv"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -q`
Expected: ImportError on `emb_path`.

- [ ] **Step 3: Implement** (append to `src/config.py`)

```python
def emb_path(alias: str, variant: str = "full") -> Path:
    """File degli embedding per (modello, variante). `full` mantiene il nome storico."""
    suffix = "" if variant == "full" else f"_{variant}"
    return ROOT / "data" / f"emb_{alias}{suffix}.npy"


def meta_path(variant: str = "full") -> Path:
    """Metadati allineati riga per riga con `emb_path`. `full` mantiene il nome storico."""
    suffix = "" if variant == "full" else f"_{variant}"
    return ROOT / "data" / f"meta{suffix}.csv"
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/pytest tests/test_config.py -q`
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat(config): path helpers for per-variant embeddings"
```

---

### Task 4: `generate.py` with `--variant` and `--all-variants`

**Files:**
- Modify: `src/generate.py:105-169` (`main`)
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `VARIANTS`, `build_messages(..., variant=)` from Task 2
- Produces:
  - `plan_jobs(artworks, variants, done, model) -> list[tuple[dict, str, str]]` — `(artwork, condition, variant)` triples still to generate
  - `read_done(path) -> set[tuple[str, str, str, str]]` — `(artwork_id, condition, model, variant)`
  - rows in `generations.jsonl` gain `"variant": <str>`

- [ ] **Step 1: Write the failing tests**

`tests/test_generate.py`:

```python
import json

from generate import plan_jobs, read_done


def test_read_done_defaults_missing_variant_to_full(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(
        json.dumps({"artwork_id": "Q1", "condition": "explorer", "model": "m"}) + "\n"
        + json.dumps({"artwork_id": "Q1", "condition": "explorer", "model": "m",
                      "variant": "no_style"}) + "\n"
    )
    assert read_done(p) == {("Q1", "explorer", "m", "full"), ("Q1", "explorer", "m", "no_style")}


def test_plan_jobs_skips_done_and_flat_for_non_full():
    arts = [{"id": "Q1"}, {"id": "Q2"}]
    done = {("Q1", "explorer", "m", "no_style")}
    jobs = plan_jobs(arts, ["no_style"], done, "m")
    keys = {(a["id"], c, v) for a, c, v in jobs}
    assert ("Q1", "explorer", "no_style") not in keys
    assert ("Q2", "explorer", "no_style") in keys
    assert not any(c == "flat" for _, c, _ in jobs)
    assert len(jobs) == 2 * 5 - 1


def test_plan_jobs_full_includes_flat():
    arts = [{"id": "Q1"}]
    jobs = plan_jobs(arts, ["full"], set(), "m")
    assert ("Q1", "flat", "full") in {(a["id"], c, v) for a, c, v in jobs}
    assert len(jobs) == 6
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest tests/test_generate.py -q`
Expected: ImportError on `plan_jobs`.

- [ ] **Step 3: Add the two helpers to `src/generate.py`** (above `main`)

```python
def read_done(path: Path) -> set[tuple[str, str, str, str]]:
    """Chiavi gia' generate. Le righe storiche senza `variant` valgono come `full`."""
    done = set()
    if path.exists():
        for line in path.open():
            r = json.loads(line)
            done.add((r["artwork_id"], r["condition"], r["model"], r.get("variant", "full")))
    return done


def plan_jobs(artworks, variants, done, model) -> list[tuple[dict, str, str]]:
    """Terne (opera, condizione, variante) da generare. `flat` esiste solo in `full`."""
    return [
        (a, c, v)
        for v in variants
        for a in artworks
        for c in CONDITIONS
        if not (c == "flat" and v != "full")
        and (a["id"], c, model, v) not in done
    ]
```

- [ ] **Step 4: Rewrite `main` to use them**

Replace the body of `main` from the argparse block down to the `todo` computation and the `work` function with:

```python
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, help="usa solo le prime N opere (smoke test)")
    ap.add_argument("--cap", type=float, default=1.00, help="tetto di spesa in dollari")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--variant", default="full", choices=list(VARIANTS),
                    help="variante del blocco di categoria (ablazione)")
    ap.add_argument("--all-variants", action="store_true",
                    help="genera tutte le varianti diverse da full in un solo run")
    args = ap.parse_args()

    artworks = [json.loads(l) for l in ARTWORKS.open()]
    if args.limit:
        artworks = artworks[: args.limit]

    variants = [v for v in VARIANTS if v != "full"] if args.all_variants else [args.variant]
    done = read_done(OUT)
    todo = plan_jobs(artworks, variants, done, args.model)
    print(f"[{args.model}] {len(artworks)} opere, varianti {variants}")
    print(f"  da generare {len(todo)}")
    if not todo:
        return
```

and change `work` to:

```python
    def work(job):
        art, cond, variant = job
        msgs = build_messages(cond, art["title"], art["artist"], art["source_text"], variant)
        text, usage = generate_one(session, args.model, msgs, budget, args.temperature)
        row = {
            "artwork_id": art["id"],
            "title": art["title"],
            "artist": art["artist"],
            "condition": cond,
            "variant": variant,
            "model": args.model,
            "temperature": args.temperature,
            "text": text,
            "words": len(text.split()),
            "usage": usage,
        }
```

Update the import line to `from prompts import CONDITIONS, VARIANTS, build_messages`. Everything else in `main` (budget, session, executor, summary prints) stays as is.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/test_generate.py -q`
Expected: 3 PASSED

- [ ] **Step 6: Check idempotence on existing data (no API call)**

Run: `.venv/bin/python src/generate.py --variant full`
Expected: `da generare 0` and exit. This proves the 600 legacy rows are recognised as `full`.

- [ ] **Step 7: Smoke test one variant on 2 artworks (≈ 10 API calls, < 0.01 $)**

Run: `.venv/bin/python src/generate.py --variant name_only --limit 2`
Expected: 10 rows appended, each with `"variant": "name_only"`. Verify:

```bash
tail -1 data/generations.jsonl | .venv/bin/python -c "import json,sys; r=json.loads(sys.stdin.read()); print(r['variant'], r['condition'], r['words'])"
```

- [ ] **Step 8: Commit**

```bash
git add src/generate.py tests/test_generate.py
git commit -m "feat(generate): --variant / --all-variants for prompt ablation"
```

---

### Task 5: `embed.py` per variant

**Files:**
- Modify: `src/embed.py` (whole `main`)
- Test: `tests/test_embed.py`

**Interfaces:**
- Consumes: `emb_path`, `meta_path` (Task 3); `VARIANTS` (Task 2)
- Produces:
  - `select_rows(all_rows, generator_model, variant) -> list[dict]` — rows of that variant plus the `full` flat rows, sorted by `(artwork_id, condition)`, deduplicated check
  - `embed_rows(model, rows, alias, variant, batch_size)` — writes the two files

- [ ] **Step 1: Write the failing test**

`tests/test_embed.py`:

```python
import pytest

from embed import select_rows


def row(a, c, v, model="m"):
    return {"artwork_id": a, "condition": c, "variant": v, "model": model,
            "title": "t", "artist": "x", "words": 1, "text": "hi"}


def test_select_rows_full_treats_missing_variant_as_full():
    rows = [{"artwork_id": "Q1", "condition": "flat", "model": "m", "title": "t",
             "artist": "x", "words": 1, "text": "hi"}, row("Q1", "explorer", "full")]
    out = select_rows(rows, "m", "full")
    assert [(r["artwork_id"], r["condition"]) for r in out] == [("Q1", "explorer"), ("Q1", "flat")]


def test_select_rows_variant_borrows_flat_from_full():
    rows = [row("Q1", "flat", "full"), row("Q1", "explorer", "full"),
            row("Q1", "explorer", "no_style")]
    out = select_rows(rows, "m", "no_style")
    assert {(r["condition"], r["variant"]) for r in out} == {("flat", "full"), ("explorer", "no_style")}


def test_select_rows_rejects_duplicates():
    rows = [row("Q1", "explorer", "full"), row("Q1", "explorer", "full")]
    with pytest.raises(SystemExit):
        select_rows(rows, "m", "full")


def test_select_rows_rejects_other_model():
    assert select_rows([row("Q1", "explorer", "full", model="other")], "m", "full") == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/test_embed.py -q`
Expected: ImportError on `select_rows`.

- [ ] **Step 3: Rewrite `src/embed.py`**

Replace everything from `def main()` to the end with:

```python
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
    ap.add_argument("--variant", default="full", choices=list(VARIANTS))
    ap.add_argument("--all-variants", action="store_true",
                    help="embedda ogni variante presente in generations.jsonl")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    all_rows = [json.loads(l) for l in GEN.open()]
    if args.all_variants:
        present = {r.get("variant", "full") for r in all_rows if r.get("model") == args.generator_model}
        variants = [v for v in VARIANTS if v in present]
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
```

Update imports at the top: add `from config import ROOT, emb_path, meta_path` and `from prompts import VARIANTS`; remove the local `ROOT = ...` line (keep `GEN = ROOT / "data" / "generations.jsonl"`).

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_embed.py -q`
Expected: 4 PASSED

- [ ] **Step 5: Regression check: re-embedding `full` reproduces the existing file**

```bash
cp data/emb_qwen.npy /tmp/emb_qwen_before.npy
.venv/bin/python src/embed.py --embedding-model qwen --variant full
.venv/bin/python -c "
import numpy as np; a=np.load('/tmp/emb_qwen_before.npy'); b=np.load('data/emb_qwen.npy')
print(a.shape, b.shape, 'max abs diff', float(np.abs(a-b).max()))"
```

Expected: same shape (600, 1024), max abs diff below 1e-4 (float noise from MPS). `data/meta.csv` now has an extra `variant` column, all `full`.

- [ ] **Step 6: Commit**

```bash
git add src/embed.py tests/test_embed.py data/meta.csv
git commit -m "feat(embed): per-variant embeddings, flat shared from full"
```

---

### Task 6: `analyze.py` reads any variant

**Files:**
- Modify: `src/analyze.py:63-80` (`load`), `src/analyze.py:289-296` (argparse), `src/analyze.py:368-371` (top_words gens filter)

**Interfaces:**
- Consumes: `emb_path`, `meta_path` (Task 3)
- Produces: `load(model: str, variant: str = "full") -> (X, arts, meta)` — unchanged return contract

- [ ] **Step 1: Change `load`**

```python
def load(model: str, variant: str = "full"):
    emb = np.load(emb_path(model, variant))
    meta = pd.read_csv(meta_path(variant))
    assert len(meta) == len(emb), f"meta {len(meta)} != emb {len(emb)}"
    ...  # resto invariato
```

Add `from config import emb_path, meta_path` to the imports.

- [ ] **Step 2: Add `--variant` to `main` and use it**

```python
    ap.add_argument("--variant", default="full")
    ...
    tag = args.model if args.variant == "full" else f"{args.model}_{args.variant}"
    X, arts, meta = load(args.model, args.variant)
```

In the top-words section, filter generations by variant (flat borrowed from full):

```python
    gens = [json.loads(l) for l in (ROOT / "data" / "generations.jsonl").open()]
    gens = [g for g in gens if g.get("variant", "full") == args.variant
            or (g["condition"] == "flat" and g.get("variant", "full") == "full")]
```

- [ ] **Step 3: Regression: `full` output is unchanged**

```bash
cp results/metrics_qwen.json /tmp/metrics_qwen_before.json
.venv/bin/python src/analyze.py --model qwen
.venv/bin/python -c "
import json; a=json.load(open('/tmp/metrics_qwen_before.json')); b=json.load(open('results/metrics_qwen.json'))
print('probe', a['probe']['accuracy'], b['probe']['accuracy']); print('p', a['permutation']['p_value'], b['permutation']['p_value'])"
```

Expected: identical probe accuracy and p-value (RNG seed is fixed; embeddings from Task 5 differ at most by float noise).

- [ ] **Step 4: Commit**

```bash
git add src/analyze.py results/metrics_qwen.json figures/
git commit -m "feat(analyze): --variant, paths via config helpers"
```

---

### Task 7: Generate and embed all variants

No code. Real runs, real cost (≈ 0.35 $, cap 1.00 $).

- [ ] **Step 1: Generate the 7 non-full variants**

Run: `.venv/bin/python src/generate.py --all-variants`
Expected: `da generare 3490` (7 × 500 minus the 10 smoke-test rows), ends with cost around 0.35 $. If the run stops on the cap or a network error, rerun the same command: it resumes.

- [ ] **Step 2: Verify counts**

```bash
.venv/bin/python -c "
import json, collections
c = collections.Counter((r.get('variant','full'), r['condition']) for r in map(json.loads, open('data/generations.jsonl')))
for v in ['full','no_def','no_need','no_style','def_only','need_only','style_only','name_only']:
    print(v, sum(n for (vv, _), n in c.items() if vv == v))"
```

Expected: full 600, every other variant 500.

- [ ] **Step 3: Embed all variants with both models**

```bash
.venv/bin/python src/embed.py --embedding-model qwen --all-variants
.venv/bin/python src/embed.py --embedding-model bge-m3 --all-variants
ls data/emb_*
```

Expected: 16 `.npy` files (8 variants × 2 models) and 8 `meta*.csv`.

- [ ] **Step 4: Commit data**

```bash
git add data/
git commit -m "data: ablation generations and embeddings for 8 variants"
```

---

### Task 8: `ablation.py` metrics per variant

**Files:**
- Create: `src/ablation.py`
- Test: `tests/test_ablation.py`

**Interfaces:**
- Consumes: `load`, `flatten`, `probe`, `split_half` from `analyze.py`; `VARIANTS`, `PARTS`, `FALK_CATEGORIES`, `CONDITIONS`, `LABELS` from `prompts.py`
- Produces:
  - `steering(X) -> np.ndarray` — `(C, D)` per-artwork-centred mean shift
  - `cos_with_full(V, V_full) -> np.ndarray` — `(C,)`
  - `factorial_effects(table: dict[str, float]) -> dict` — `table` maps variant → scalar; returns `{"main": {part: eff}, "interaction": {"def×need": eff, ...}}`
  - `results/ablation_{model}.json`

- [ ] **Step 1: Write the failing tests**

`tests/test_ablation.py`:

```python
import numpy as np
import pytest

from ablation import cos_with_full, factorial_effects, steering


def synthetic(rng, A=10, C=6, D=8, scale=1.0):
    """Ogni opera ha un offset proprio; ogni condizione uno shift fisso scalato."""
    art = rng.normal(size=(A, 1, D))
    shifts = rng.normal(size=(1, C, D)) * scale
    return art + shifts + rng.normal(size=(A, C, D)) * 0.01, shifts[0]


def test_steering_recovers_shifts_up_to_condition_mean():
    rng = np.random.default_rng(0)
    X, shifts = synthetic(rng)
    V = steering(X)
    expected = shifts - shifts.mean(axis=0, keepdims=True)
    assert np.allclose(V, expected, atol=0.02)


def test_cos_with_full_is_one_for_identical():
    rng = np.random.default_rng(1)
    X, _ = synthetic(rng)
    V = steering(X)
    assert np.allclose(cos_with_full(V, V), 1.0)


def test_cos_with_full_drops_when_shift_removed():
    rng = np.random.default_rng(2)
    X_full, _ = synthetic(rng, scale=1.0)
    X_half, _ = synthetic(np.random.default_rng(2), scale=0.0)  # stessa opera, nessuno shift
    c = cos_with_full(steering(X_half), steering(X_full))
    assert np.all(np.abs(c) < 0.6)


def test_factorial_main_effects_signs():
    # metrica = 1 se style presente, 0 altrimenti -> effetto di style = 1, altri = 0
    table = {"full": 1, "no_def": 1, "no_need": 1, "no_style": 0,
             "def_only": 0, "need_only": 0, "style_only": 1, "name_only": 0}
    eff = factorial_effects(table)
    assert eff["main"]["style"] == pytest.approx(1.0)
    assert eff["main"]["def"] == pytest.approx(0.0)
    assert eff["main"]["need"] == pytest.approx(0.0)
    assert all(abs(v) < 1e-9 for v in eff["interaction"].values())


def test_factorial_interaction():
    # metrica = 1 solo se def E need presenti -> interazione def×need positiva
    table = {"full": 1, "no_def": 0, "no_need": 0, "no_style": 1,
             "def_only": 0, "need_only": 0, "style_only": 0, "name_only": 0}
    eff = factorial_effects(table)
    # (d,n)=1 ; (d,¬n)=0 ; (¬d,n)=0 ; (¬d,¬n)=0  ->  1 - 0 - 0 + 0
    assert eff["interaction"]["def×need"] == pytest.approx(1.0)
    assert eff["main"]["def"] == pytest.approx(0.5)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest tests/test_ablation.py -q`
Expected: ImportError on `ablation`.

- [ ] **Step 3: Create `src/ablation.py`** with the core functions (figures come in Task 9)

```python
"""Ablazione del prompt: quale parte del blocco di categoria produce lo shift latente?

Per ogni variante (vedi prompts.VARIANTS) si ricalcolano le metriche di analyze.py e si
aggiunge la metrica chiave: il coseno fra lo steering vector della variante e quello di
`full`, per categoria. Dice se la parte conserva la *direzione* dello shift, non soltanto
la separabilita'.

Chiude un'analisi fattoriale 2x2x2 sulle tre parti (def, need, style): effetti principali
e interazioni a due vie, calcolati direttamente sulle medie delle 8 celle.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from analyze import load, probe, split_half
from config import emb_path
from prompts import CONDITIONS, FALK_CATEGORIES, LABELS, PARTS, VARIANTS

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

N_FALK = len(FALK_CATEGORIES)  # le prime 5 di CONDITIONS; flat e' l'ultima


def steering(X: np.ndarray) -> np.ndarray:
    """(A, C, D) -> (C, D): media sulle opere degli embedding centrati per opera."""
    return (X - X.mean(axis=1, keepdims=True)).mean(axis=0)


def cos_with_full(V: np.ndarray, V_full: np.ndarray) -> np.ndarray:
    num = np.sum(V * V_full, axis=1)
    den = np.linalg.norm(V, axis=1) * np.linalg.norm(V_full, axis=1) + 1e-12
    return num / den


def factorial_effects(table: dict[str, float]) -> dict:
    """Effetti principali e interazioni a due vie da una tabella variante -> scalare.

    main[p]        = media(celle con p) - media(celle senza p)
    inter[p×q]     = media(p,q) - media(p,¬q) - media(¬p,q) + media(¬p,¬q), ciascuna
                     mediata sulle 2 celle che restano libere sulla terza parte.
    """
    cells = {VARIANTS[v]: table[v] for v in VARIANTS}  # (def, need, style) -> valore
    idx = {p: i for i, p in enumerate(PARTS)}

    def mean_where(**fixed):
        vals = [val for flags, val in cells.items()
                if all(flags[idx[p]] == on for p, on in fixed.items())]
        return float(np.mean(vals))

    main = {p: mean_where(**{p: True}) - mean_where(**{p: False}) for p in PARTS}
    inter = {}
    for p, q in itertools.combinations(PARTS, 2):
        inter[f"{p}×{q}"] = (
            mean_where(**{p: True, q: True}) - mean_where(**{p: True, q: False})
            - mean_where(**{p: False, q: True}) + mean_where(**{p: False, q: False})
        )
    return {"main": main, "interaction": inter}


def analyse_variant(model: str, variant: str, V_full: np.ndarray | None) -> dict:
    X, arts, meta = load(model, variant)
    Xc = X - X.mean(axis=1, keepdims=True)
    V = steering(X)
    acc, cm, _, _ = probe(Xc[:, :N_FALK, :])          # 5 classi: flat escluso
    sh_mu, sh_sd = split_half(X)
    norms = np.linalg.norm(V, axis=1)
    words = meta.groupby("condition")["words"].mean()
    out = {
        "variant": variant,
        "n_artworks": int(X.shape[0]),
        "probe5_accuracy": float(acc),
        "probe5_recall": {LABELS[c]: float(cm[i, i] / cm[i].sum()) for i, c in enumerate(FALK_CATEGORIES)},
        "norm": {LABELS[c]: float(norms[i]) for i, c in enumerate(FALK_CATEGORIES)},
        "norm_mean": float(norms[:N_FALK].mean()),
        "split_half": {LABELS[c]: float(sh_mu[i]) for i, c in enumerate(FALK_CATEGORIES)},
        "words": {LABELS[c]: float(words[c]) for c in FALK_CATEGORIES if c in words.index},
    }
    if V_full is not None:
        cos = cos_with_full(V, V_full)
        out["cos_with_full"] = {LABELS[c]: float(cos[i]) for i, c in enumerate(FALK_CATEGORIES)}
        out["cos_with_full_mean"] = float(cos[:N_FALK].mean())
    return out, V


def run(model: str) -> dict:
    variants = [v for v in VARIANTS if emb_path(model, v).exists()]
    missing = [v for v in VARIANTS if v not in variants]
    if missing:
        print(f"  varianti senza embedding, saltate: {missing}")
    if "full" not in variants:
        raise SystemExit("serve la variante full per il confronto")

    per_variant = {}
    full, V_full = analyse_variant(model, "full", None)
    full["cos_with_full"] = {LABELS[c]: 1.0 for c in FALK_CATEGORIES}
    full["cos_with_full_mean"] = 1.0
    per_variant["full"] = full
    for v in variants:
        if v == "full":
            continue
        per_variant[v], _ = analyse_variant(model, v, V_full)

    factorial = {}
    if len(per_variant) == len(VARIANTS):
        for metric in ("norm_mean", "cos_with_full_mean", "probe5_accuracy"):
            factorial[metric] = factorial_effects({v: per_variant[v][metric] for v in VARIANTS})

    return {"model": model, "variants": per_variant, "factorial": factorial}


def print_summary(res: dict) -> None:
    print(f"\n{'variante':12} {'probe5':>7} {'||v|| media':>12} {'cos·full':>9} {'split-half':>11}")
    rows = sorted(res["variants"].values(), key=lambda r: -r["cos_with_full_mean"])
    for r in rows:
        sh = np.mean(list(r["split_half"].values()))
        print(f"{r['variant']:12} {r['probe5_accuracy']:7.1%} {r['norm_mean']:12.4f} "
              f"{r['cos_with_full_mean']:9.3f} {sh:11.3f}")
    if res["factorial"]:
        print("\neffetti principali (con parte - senza parte):")
        for metric, eff in res["factorial"].items():
            m = "  ".join(f"{p}: {e:+.3f}" for p, e in eff["main"].items())
            print(f"  {metric:20} {m}")
        print("interazioni a due vie:")
        for metric, eff in res["factorial"].items():
            m = "  ".join(f"{k}: {e:+.3f}" for k, e in eff["interaction"].items())
            print(f"  {metric:20} {m}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)

    res = run(args.model)
    print_summary(res)
    out = RESULTS / f"ablation_{args.model}.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nrisultati -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_ablation.py -q`
Expected: 5 PASSED

- [ ] **Step 5: Run on real data**

Run: `.venv/bin/python src/ablation.py --model qwen`
Expected: a table with 8 rows, `full` at cos·full 1.000 and probe5 around 0.96–0.97 (the 6-class probe was 94.5%, and flat was the confused class). Then the two factorial blocks. `results/ablation_qwen.json` written.

- [ ] **Step 6: Commit**

```bash
git add src/ablation.py tests/test_ablation.py results/ablation_qwen.json
git commit -m "feat(ablation): per-variant metrics, cosine with full, factorial effects"
```

---

### Task 9: Ablation figures

**Files:**
- Modify: `src/ablation.py` (add `fig_bars`, `fig_cosine_heatmap`, call them from `main`)

**Interfaces:**
- Consumes: `res` dict from `run()` (Task 8)
- Produces: `figures/ablation_{model}.png`, `figures/ablation_cosine_{model}.png`

- [ ] **Step 1: Add the two figure functions** (below `factorial_effects`, above `main`)

```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze import COLORS


def fig_bars(res: dict, path: Path) -> None:
    """Due pannelli: (a) probe a 5 classi e norma media; (b) coseno con full per categoria."""
    vs = [v for v in VARIANTS if v in res["variants"]]
    R = res["variants"]
    x = np.arange(len(vs))
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    a.bar(x - 0.2, [R[v]["probe5_accuracy"] for v in vs], 0.4, color="#444", label="probe (5 classi)")
    a.axhline(1 / N_FALK, color="#444", ls=":", lw=1)
    a.set_ylabel("accuracy")
    a.set_ylim(0, 1.05)
    a2 = a.twinx()
    a2.bar(x + 0.2, [R[v]["norm_mean"] for v in vs], 0.4, color="#E8743B", label="||v|| media")
    a2.set_ylabel("norma media steering")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, frameon=False, loc="lower left", fontsize=9)
    a.set_title("Ablazione del prompt — separabilita' e ampiezza dello shift", fontsize=11)

    w = 0.8 / N_FALK
    for i, c in enumerate(FALK_CATEGORIES):
        b.bar(x + (i - (N_FALK - 1) / 2) * w, [R[v]["cos_with_full"][LABELS[c]] for v in vs],
              w, color=COLORS[c], label=LABELS[c])
    b.axhline(0, color="#ccc", lw=0.8)
    b.set_ylabel("cos(v_variante, v_full)")
    b.set_ylim(-0.2, 1.05)
    b.set_xticks(x, vs, rotation=30, ha="right")
    b.legend(frameon=False, fontsize=8, ncol=5, loc="lower left")
    b.set_title("Quanto ogni variante conserva la direzione del blocco completo", fontsize=11)
    for ax in (a, b):
        ax.spines[["top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_cosine_heatmap(res: dict, path: Path) -> None:
    vs = [v for v in VARIANTS if v in res["variants"]]
    M = np.array([[res["variants"][v]["cos_with_full"][LABELS[c]] for c in FALK_CATEGORIES] for v in vs])
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(N_FALK), [LABELS[c] for c in FALK_CATEGORIES], rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(vs)), vs, fontsize=9)
    for i in range(len(vs)):
        for j in range(N_FALK):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(M[i, j]) > 0.55 else "black")
    ax.set_title("Coseno fra v_variante[c] e v_full[c]", fontsize=11)
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
```

In `main`, after `print_summary(res)`:

```python
    fig_bars(res, FIGS / f"ablation_{args.model}.png")
    fig_cosine_heatmap(res, FIGS / f"ablation_cosine_{args.model}.png")
    print(f"figure -> figures/ablation_{args.model}.png, figures/ablation_cosine_{args.model}.png")
```

- [ ] **Step 2: Run for both embedding models**

```bash
.venv/bin/python src/ablation.py --model qwen
.venv/bin/python src/ablation.py --model bge-m3
```

Expected: 4 PNGs in `figures/`, 2 JSONs in `results/`. Open the PNGs and check the `full` bars are at 1.0 on the cosine panel.

- [ ] **Step 3: Run the whole test suite**

Run: `.venv/bin/pytest -q`
Expected: all PASSED.

- [ ] **Step 4: Commit**

```bash
git add src/ablation.py figures/ablation_*.png results/ablation_*.json
git commit -m "feat(ablation): figures"
```

---

### Task 10: Report section

**Files:**
- Modify: `docs/report-2026-09-21-stato-progetto.md` (section 5)

- [ ] **Step 1: Replace section 5 with results**

Replace "## 5. Passo successivo: ablazione del prompt" and its paragraph with a section that:

1. restates the 8-variant table from the spec;
2. embeds `![](../figures/ablation_qwen.png)` and `![](../figures/ablation_cosine_qwen.png)`;
3. has a markdown table with one row per variant: probe5, norma media, cos·full media, split-half media — copy the numbers from the terminal summary of `ablation.py --model qwen`, with bge-m3 in parentheses;
4. has a table of main effects and interactions for `cos_with_full_mean` and `norm_mean`;
5. ends with three sentences of interpretation, answering: which part is sufficient, which is necessary, how much survives with the name alone. Do not write the interpretation before seeing the numbers.

- [ ] **Step 2: Commit**

```bash
git add docs/report-2026-09-21-stato-progetto.md
git commit -m "docs: ablation results in project report"
```

---

## Self-review

**Spec coverage.** §2 variants → Task 2. §3.1 → Task 2. §3.2 → Task 4. §3.3 → Tasks 3, 5. §3.4 → Task 6. §3.5 metrics 1–6 → Task 8; factorial → Task 8; figures → Task 9. §4 tests → Tasks 1, 2, 8 (`tests/test_ablation.py` covers the synthetic-tensor requirement). §5 execution order → Tasks 1–10 in that order. §6 out of scope → nothing planned.

**Type consistency.** `VARIANTS` values are `(def, need, style)` tuples everywhere; `PARTS` order matches. `load(model, variant)` returns `(X, arts, meta)` as before. `probe` returns 4 values, `split_half` returns `(mu, sd)`, both over all C conditions with flat last; `ablation.py` slices `[:N_FALK]`. `emb_path`/`meta_path` used by embed, analyze, ablation with the same signature.

**Known simplification.** `split_half` in `analyze.py` uses the module-level `RNG`, so calling it 8 times in one process draws different splits per variant. That is fine for a mean over 200 repetitions; if exact reproducibility per variant is wanted later, reseed before each call.
