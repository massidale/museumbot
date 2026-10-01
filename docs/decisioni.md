# Registro delle decisioni

Correzioni e modifiche agli esperimenti degne di nota, in ordine cronologico. Per ogni voce:
la data, cosa si è deciso, perché, e cosa è cambiato nei risultati. Lo stato attuale del
progetto è in `docs/report-stato-progetto.md`, che non contiene storia.

---

## 2026-09-21 — Embedding Qwen3 ricalcolati

**Decisione.** Embedding Qwen3 dello studio principale ricalcolati dopo un cambio di
ambiente.
**Effetto.** Accuracy del probe da 94.5% a 94.7%; nessuna conclusione cambiata.

---

## 2026-09-29 — Steering vector rispetto al flat, non alla media

**Decisione.** Gli steering vector `v[c]` sono calcolati rispetto al testo flat dell'opera
(`e[a,c] − e[a,flat]`), non rispetto alla media delle 6 condizioni. La media resta per il
probe, il test di permutazione e la scomposizione della varianza. Il centering è in una
sola funzione, `analyze.center(X, ref)`.

**Motivo.**
- Con la media i 6 `v[c]` sommano a zero per costruzione, e il coseno medio fra categorie
  è spinto verso −1/(C−1) = −0.2: i coseni negativi erano in parte un artefatto.
- Il flat risponde alla domanda di ricerca: quanto e in che direzione la categoria sposta
  il testo rispetto alla descrizione neutra.
- Nell'ablazione il flat è lo stesso testo in tutte le varianti, quindi il riferimento è
  fisso; la media invece include le 5 categorie della variante e si sposta con essa.
- Il probe e la permutazione lavorano sul singolo testo con le etichette: lì tutte le
  condizioni devono essere scambiabili, e sottrarre un solo testo flat aggiungerebbe il suo
  rumore a ogni opera.

**Effetto.**
- Matrice dei coseni ora 5×5. La lettura "Professional/Hobbyist e Recharger quasi opposti"
  (−0.53) è ritirata: con il riferimento flat è −0.20, e la distanza fra le due viene
  soprattutto dall'ampiezza dei due shift. Explorer–Facilitator sale da +0.33 a +0.61.
- Experience Seeker non è la categoria più vicina al flat (lo suggeriva il coseno +0.41
  con il riferimento media); le più vicine sono Explorer e Facilitator.
- Ablazione: le medie per variante cambiano al più di 0.04 e le conclusioni sullo stile
  restano. Cambiano i valori per categoria: in `name_only` Explorer passa da 0.56 a 0.71 ed
  Experience Seeker da 0.62 a 0.46 (la categoria ambigua è Experience Seeker, non Explorer);
  in `need_only` Experience Seeker passa da 0.50 a 0.25.
- La scala della heatmap dell'ablazione dipende ora dal minimo osservato (prima era fissa
  a 0.4 e tagliava lo 0.25).

## 2026-09-29 — Split-half riportato rispetto al flat

**Decisione.** La consistenza split-half principale è calcolata sugli steering vector
rispetto al flat; quella rispetto alla media resta nel JSON come limite superiore e per la
stabilità del flat stesso.
**Motivo.** Lo split-half deve misurare l'affidabilità dei vettori effettivamente
riportati. Con un solo testo flat come riferimento il rumore è maggiore che con la media di
6 testi, quindi i valori sono più bassi (Qwen3 0.85–0.94 invece di 0.88–0.96; Facilitator
su BGE-M3 0.59 invece di 0.76).

## 2026-09-29 — Distanza cross-validata per la separazione fra categorie

**Decisione.** La separazione fra condizioni si misura con la distanza euclidea quadrata
cross-validata (`d²[a,b] = (v¹[a] − v¹[b]) · (v²[a] − v²[b])`, metà disgiunte di opere,
200 divisioni), cioè la crossnobis senza normalizzazione per la covarianza.
**Motivo.** Il coseno dipende dal riferimento; la distanza no, perché usa solo differenze.
La distanza semplice è gonfiata dal rumore, quella cross-validata è imparziale (≈ 0 se due
condizioni non differiscono). Scartato il probe binario per coppia: con il probe a 6 classi
già al 95% satura quasi ovunque al 100%. Scartata la crossnobis completa: la covarianza
del rumore in 1 024 dimensioni con 100 opere richiede una regolarizzazione difficile da
giustificare.

## 2026-09-29 — Numeri del report allineati ai risultati

**Decisione.** Corretti nel report valori che non corrispondevano ai JSON: recall di
Explorer (0.96 → 0.97), Experience Seeker (0.97 → 0.96) e Flat (0.84 → 0.85), quota di
varianza dell'opera (69.9% → 70.0%).
**Motivo.** Il report deve riportare esattamente i risultati del codice.

---

## 2026-10-01 — Calibrazione del coseno con una replica del full

**Decisione.** Generata `full_rep`: stesso prompt del full, 500 testi, stesso flat del
full. È registrata in `REPLICATES` (`common/prompts.py`), fuori dal disegno fattoriale.
`ablation.py` riporta per ogni variante il coseno con il full diviso per il tetto
(`cos_rel`); il full è escluso perché il suo coseno vale 1 per costruzione.
**Motivo.** Senza il coseno fra due run dello stesso prompt i coseni dell'ablazione si
leggevano solo in ordine relativo.
**Effetto.** Tetto 0.96–0.98 (Qwen3), 0.88–0.97 (BGE-M3); `no_def` e `no_need` a 1.00 del
tetto. Con la riserva della voce successiva.

## 2026-10-01 — Deriva di provider su OpenRouter

**Scoperta.** Generando `full_rep` i token di prompt erano diversi a parità di prompt
(450 → 463), i token generati più numerosi per testi più corti, il costo per token 1.4
volte più alto e le rigenerazioni per il filtro 12% invece di 4%. OpenRouter distribuisce
DeepSeek V4 Flash su circa 15 provider, con quantizzazioni diverse (fp4, fp8, non
dichiarata) e comportamento diverso sul ragionamento (Venice ragiona di default, DeepInfra
no). `generate.py` non salvava il provider.

**Decisioni.**
- Il corpus di settembre (studio principale e ablazione) resta com'è ed è dichiarato come
  limite: miscela di provider non tracciata.
- Le nuove generazioni usano un provider fisso senza fallback (`--provider
  deepinfra/fp8`) con ragionamento attivato esplicitamente, e i confronti si fanno fra
  testi della stessa sessione.
- `generate.py` salva `usage.provider` e `usage.reasoning_tokens`.

**Motivo.** Un tetto di rumore calcolato fra run su provider diversi include anche la
deriva: è più basso del vero e fa sembrare equivalenti cose che non lo sono. DeepInfra fp8:
quantizzazione dichiarata, provider stabile, costo basso. Ragionamento attivo: è il
comportamento con cui è stata generata la maggior parte del corpus.

**Effetto.** La calibrazione dell'ablazione è provvisoria finché lo studio chain vs
singolo non misura il rumore su provider fisso.

## 2026-10-01 — Studio chain vs singolo: disegno e criterio

**Decisione.** Sostituito il confronto esplorativo (`compare_chain.py`, 3 × 3, un testo
per metodo) con uno studio formale (`chain_study.py`, `chain_noise.py`): per 100 opere × 5
categorie due run del singolo e una della chain, più un flat per opera, tutto nella stessa
sessione su DeepInfra fp8 con i job mescolati fra metodi. Rapporto di dissimilarità
`R = mean(1 − between) / mean(1 − within)`, IC con bootstrap sulle opere.
**Criterio fissato prima di vedere i dati:** equivalenti se il limite superiore dell'IC al
95% di R è sotto 1.10.
**Motivo.** Con un testo per metodo non si separa l'effetto del metodo dal rumore. Una
sola run della chain invece di due: l'assunzione "stesso rumore del singolo" può solo far
sembrare i metodi più diversi, quindi un esito "equivalenti" resta valido; una seconda run
della chain serve solo se l'esito è "non equivalenti". Il prefisso della chain (turni 1–2)
è generato una volta e condiviso, come nel paper.

## 2026-10-01 — Pulizia simmetrica dei testi nello studio chain

**Decisione.** `clean_guide` toglie preambolo, separatori `---`, markdown e note sul
conteggio delle parole, ed è applicata a tutti i metodi dello studio prima del filtro sui
riferimenti espliciti. Il testo originale resta in `usage.raw`. Le stesse regole valgono per
i testi `full`/`full_rep` usati nel confronto sulla deriva.
**Motivo.** La chain apre sempre con "Here is a 250-word audio guide … for the Recharger":
il filtro la rifiutava 4 volte su 4 e la chain non passava mai. Si valuta l'audioguida, non
la presentazione; pulire solo la chain introdurrebbe un'asimmetria fra metodi.
**Aperto.** Il corpus principale non è pulito: 135 testi su 4 548 hanno un preambolo e la
maggior parte usa markdown.

## 2026-10-01 — Robustezza della generazione

**Decisione.** `generate_one` aspetta fra un tentativo e l'altro dopo un errore 429/5xx
(prima riprovava subito) e riporta il motivo dell'ultimo fallimento; `chain_study.py`
salta e segnala le celle che falliscono 4 tentativi invece di fermare il run.
**Motivo.** Una sola cella rifiutata 4 volte dal filtro fermava l'intero run (successo a
152 testi su 1 600).
