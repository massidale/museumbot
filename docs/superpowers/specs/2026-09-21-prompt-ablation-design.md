# Prompt ablation — design

Data: 2026-09-21. Stato: approvato in chat, da implementare.

## 1. Domanda

Ogni blocco categoria in `src/prompts.py` è fatto di tre frasi in ordine fisso:
**definizione** (chi è il visitatore), **bisogno** ("Their need is …"), **stile** (come
scrivere). Lo shift latente misurato in `results/metrics_qwen.json` è prodotto da tutto il
blocco. Quale parte lo produce? Quale parte è necessaria, quale è sufficiente, e quanto
resta se si lascia soltanto il nome della categoria (conoscenza parametrica di Falk)?

## 2. Varianti

Disegno fattoriale 2×2×2 sulle tre parti, più la variante con il blocco vuoto:

| variante | def | need | style |
|---|---|---|---|
| `full` | ✓ | ✓ | ✓ |
| `no_def` | | ✓ | ✓ |
| `no_need` | ✓ | | ✓ |
| `no_style` | ✓ | ✓ | |
| `def_only` | ✓ | | |
| `need_only` | | ✓ | |
| `style_only` | | | ✓ |
| `name_only` | | | |

Invarianti fra le varianti:

- La frase iniziale "Your listener is an Explorer:" resta sempre; in `name_only` diventa
  "Your listener is an Explorer." (punto invece dei due punti, nessuna parte dopo).
- La cornice Falk nel `SYSTEM_TEMPLATE` ("You are familiar with John H. Falk's …") resta
  sempre. Non è oggetto di questa ablazione.
- La condizione `flat` non ha blocco, quindi non ha varianti: i 100 testi flat esistenti
  valgono per tutte.
- `full` deve riprodurre **byte per byte** il prompt attuale: i 600 testi in
  `data/generations.jsonl` restano validi come variante `full`. Un test lo garantisce.
- Stesso modello (DeepSeek V4 Flash), stessa temperatura (0.7), stesso filtro `BANNED`.

Costo: 7 varianti × 500 testi ≈ 0.35 $ (la generazione esistente è costata 0.09 $ per 600).

## 3. Modifiche al codice

### 3.1 `src/prompts.py`

- `CATEGORY_BLOCKS` viene sostituito da `CATEGORY_PARTS: dict[str, dict[str, str]]` con
  chiavi `name`, `def`, `need`, `style` per categoria. Le stringhe sono ricavate spezzando
  i blocchi attuali sulle frasi già presenti; non si riscrive nulla.
- `VARIANTS: dict[str, tuple[bool, bool, bool]]` con le 8 righe della tabella.
- `build_block(category, variant) -> str` assembla `"Your listener is a(n) <name>: <def> <need> <style>"`
  includendo solo le parti attive; per `name_only` produce `"Your listener is a(n) <name>."`.
- `build_system(condition, variant="full")` e `build_messages(..., variant="full")`
  acquistano il parametro con default `full`, così i chiamanti esistenti non cambiano.
- `CATEGORY_BLOCKS` resta come alias calcolato (`build_block(c, "full")`) per compatibilità.

### 3.2 `src/generate.py`

- Nuovo argomento `--variant` (default `full`). Per `flat` la variante è forzata a `full`,
  perché il flat non varia.
- Ogni riga di `generations.jsonl` acquista il campo `variant`. Le righe esistenti senza
  campo vengono lette come `full` (retrocompatibilità in lettura, nessuna riscrittura del
  file).
- La chiave di dedup per il resume diventa `(artwork_id, condition, model, variant)`.
- Nuovo flag `--all-variants` che itera su tutte le varianti non-`full` in un solo run,
  con il tetto di spesa condiviso (`--cap` default alzato a 1.00 $).

### 3.3 `src/embed.py`

- Nuovo argomento `--variant` (default `full`), che filtra le righe. Output
  `data/emb_{alias}_{variant}.npy` e `data/meta_{variant}.csv`; per `full` i nomi restano
  quelli attuali (`emb_qwen.npy`, `meta.csv`) così `analyze.py` non cambia.
- Nuovo flag `--all-variants` che embedda tutte le varianti presenti nel file.
- Per le varianti non-`full`, le righe flat vengono prese dalla variante `full` (sono
  condivise), così il tensore (opera, condizione) resta completo a 6 condizioni.

### 3.4 `src/analyze.py`

- Le funzioni di calcolo (`load`, `probe`, `split_half`, `permutation_test`) vengono
  parametrizzate su variante e restano invariate nella logica. `load(model, variant="full")`
  risolve i nomi file secondo 3.3. Nessun cambiamento all'output per `full`.

### 3.5 Nuovo `src/ablation.py`

Per ogni variante e per l'embedding scelto (`--model`, default `qwen`):

1. carica il tensore, centra per opera, calcola V_variant (6 × D);
2. **probe a 5 classi** (flat escluso, perché identico fra varianti), GroupKFold per opera;
3. **norma** di v[c] per categoria;
4. **coseno con full**: cos(v_variant[c], v_full[c]) per categoria. È la metrica principale:
   dice se la parte conserva la *direzione* dello shift, non solo la separabilità;
5. **split-half** per categoria;
6. **lunghezza media** per categoria.

Poi l'**analisi fattoriale** sulle 8 varianti: per ciascuna delle tre parti, effetto
principale sulla norma media di v[c] e sul coseno medio con full (differenza fra media delle
4 varianti con la parte e media delle 4 senza), e le tre interazioni a due vie. Calcolo
diretto sulle medie, senza modello lineare: 8 celle, una osservazione aggregata per cella.

Output:

- `results/ablation_{model}.json` con tutte le metriche per variante e la tabella fattoriale;
- `figures/ablation_{model}.png`: barre raggruppate, varianti sull'asse x, due pannelli
  (probe accuracy + norma media; coseno con full per categoria);
- `figures/ablation_cosine_{model}.png`: heatmap varianti × categorie del coseno con full.

Stampa a terminale una tabella riassuntiva ordinata per coseno medio con full.

## 4. Test

`tests/test_prompts.py` (pytest, da aggiungere a `pyproject.toml` come dev dependency):

- `build_system(c, "full")` è identico, per ogni categoria, al system prompt salvato prima
  della modifica (snapshot delle 6 stringhe in `tests/snapshots/system_full.json`,
  generato una volta con il codice attuale prima di toccare `prompts.py`);
- `build_block(c, "name_only")` non contiene nessuna delle tre parti e termina con `.`;
- per ogni variante, le parti presenti nel blocco sono esattamente quelle attive nella
  tabella, e in ordine def → need → style;
- `build_system("flat", v)` è identico per ogni `v`.

`tests/test_ablation.py`:

- su un tensore sintetico (A=10, C=6, D=8) con shift noti, `cos_with_full` restituisce 1
  quando la variante coincide con full, e i segni attesi degli effetti principali quando
  una parte viene azzerata.

## 5. Ordine di esecuzione

1. snapshot dei system prompt attuali → test;
2. refactor `prompts.py` → test verdi;
3. `generate.py` con `--variant` / `--all-variants` → smoke test con `--limit 2`, poi run
   completo (≈ 0.35 $);
4. `embed.py` con `--variant` / `--all-variants` (qwen, poi bge-m3);
5. `ablation.py` → risultati e figure;
6. aggiornare il report in `docs/` con una sezione risultati dell'ablazione.

## 6. Fuori scope

- Varianti con nome incrociato (nome di A, descrizione di B) e rimozione della cornice
  Falk: scartate in fase di design, eventualmente in una seconda ablazione.
- Controllo del confondente lunghezza: resta un punto aperto separato.
- Feature demografiche del visitatore: passo successivo, con spec propria.
