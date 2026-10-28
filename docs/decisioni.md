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

## 2026-10-01 — Rimossa la figura t-SNE

**Decisione.** `analyze.py` non produce più `figures/tsne_{qwen,bge-m3}.png`; le figure di
proiezione restano PCA e LDA.
**Motivo.** RQ1 chiede in che direzione e di quanto la categoria sposta il testo. La t-SNE
conserva solo i vicini locali: non ha direzioni (niente steering vector), le distanze e le
dimensioni dei cluster non sono interpretabili, dipende da perplexity e seed e non permette
di proiettare opere tenute fuori. La PCA mostra la stessa separazione senza usare le
etichette e con le frecce dei v[c]; la LDA held-out la affianca come controparte visiva del
probe.
**Effetto sui risultati.** Nessuno: la figura era supplementare e nessuna metrica ne
dipendeva.

## 2026-10-01 — Pulizia estesa al formato copione della chain

**Decisione.** `clean_guide` toglie anche le intestazioni ("Audio Guide Script (approx. 250
words):", "Audio Guide: <titolo>") e le didascalie di regia ("(Soft, inviting tone)",
"(Fade out)", "(pause)"); `chain_noise.py` la riapplica ai testi già salvati.
**Motivo.** Dopo la prima analisi è emerso che 240 testi della chain su 500 avevano
un'intestazione da copione e 98 didascalie su righe a sé; nei singoli e nel flat nessuna.
Si confronta il testo parlato, non il formato. La pulizia è stata estesa *dopo* aver visto i
primi risultati; il criterio di equivalenza no.
**Effetto.** Nessuno sulle conclusioni: R da 1.42 a 1.43 (Qwen3), da 1.43 a 1.45 (BGE-M3);
shift relativo della chain da 0.70 a 0.73 (Qwen3), da 0.69 a 0.80 (BGE-M3).

## 2026-10-01 — Cache degli embedding dello studio chain per modello

**Correzione.** `chain_noise.py` usava un solo file di metadati per entrambi i modelli di
embedding: dopo il ricalcolo con Qwen3, BGE-M3 trovava i metadati aggiornati e riusava i
propri embedding vecchi. Ora la cache è per modello (`meta_<modello>_chain_study.csv`).
**Effetto.** Il primo risultato BGE-M3 dopo la pulizia estesa era quello vecchio; i valori
riportati sopra sono quelli ricalcolati.

## 2026-10-01 — Chain vs singolo: esito

**Esito.** Con il criterio fissato prima dei dati i due metodi **non sono equivalenti**:
R = 1.43 [1.39, 1.49] (Qwen3), 1.45 [1.40, 1.50] (BGE-M3). Lo shift di categoria della
chain conserva in media 0.73 (0.80) di quello del singolo, rispetto al tetto; Experience
Seeker e Facilitator sono le categorie più diverse.
**Conseguenza.** I risultati di RQ1 valgono per il prompt singolo e non si trasferiscono
automaticamente alla chain del paper. La riserva prevista dal disegno (chain più rumorosa
del singolo) richiederebbe una seconda run della chain; non generata, in attesa di
decisione.

## 2026-10-01 — Tetto severo per la calibrazione dell'ablazione

**Decisione.** `ablation.py` riporta il coseno con il full rispetto a due tetti: quello di
`full_rep` (indulgente, include la deriva di provider) e quello fra `single_a` e
`single_b` dello studio chain (severo, stesso provider). Il tetto vero sta in mezzo.
**Motivo.** La deriva di provider misurata sui testi (D = 1.19 Qwen3, 1.17 BGE-M3) rende il
tetto di `full_rep` troppo basso; il tetto su provider fisso non ha deriva ma viene da un
setup un po' diverso (un solo provider, testi puliti), quindi è usato come limite, non come
nuovo tetto.
**Effetto.** Con Qwen3 i due tetti quasi coincidono e `no_def`/`no_need` restano a 1.00.
Con BGE-M3 e il tetto severo scendono a 0.97–0.98.

---

## 2026-10-02 — Corpus principale di RQ1 su provider fisso

**Decisione.** L'analisi principale di RQ1 (§3 del report) usa i testi a prompt singolo
dello studio chain: `single_a` come `full`, `single_b` come replica `full_rep`, più il flat
della stessa sessione (`common/corpus.py`, corpus `main`). Il corpus di settembre resta solo
per l'ablazione (corpus `ablation`). Gli embedding stanno ora in `data/emb/<corpus>/`; il
tetto severo dell'ablazione viene dalla sezione `ceiling` di `results/metrics_*.json`.
**Motivo.** Il corpus di settembre è una miscela non tracciata di provider; lo studio chain
ha già, su DeepInfra fp8 e nella stessa sessione, due run del prompt singolo e il flat: è
un corpus completo con la replica per il tetto, senza nuove generazioni.
**Effetto.** Nessuna conclusione cambia. Qwen3: probe da 94.7% a 95.3%, varianza
opera/categoria da 70.0%/7.4% a 73.3%/6.5%, split-half sempre 0.85–0.94. BGE-M3: probe
90.3% in entrambi i casi; split-half di Facilitator da 0.59 a 0.65.

## 2026-10-02 — Pulizia unica dei testi e regola del preambolo corretta

**Decisione.** `clean_guide` passa da `chain_study.py` a `common/clean.py` ed è applicata a
tutti i corpus, sempre dal testo originale; `generate.py` la applica anche alle nuove
generazioni. Il preambolo si toglie solo se la prima frase nomina il testo stesso
("audio guide", "script", "text", "N words"…), se la riga finisce con `:` o se è seguita
da `---`. Riconosce anche l'apostrofo tipografico ("Here’s") e toglie i corsivi lasciati
aperti.
**Motivo.** La regola precedente toglieva qualunque prima riga iniziasse con "Here is":
nel corpus di settembre 4 testi aprono la descrizione così ("Here is a painting that
rewards a second look. …") e avrebbero perso l'intero primo paragrafo. In più non
riconosceva "Here’s your audio guide:" e lasciava asterischi di corsivi non chiusi. Il
corpus di settembre non era pulito affatto.
**Effetto.** Corpus di settembre: 35 preamboli tolti, markdown tolto da 3 382 testi su
4 600. Corpus principale: 5 preamboli, markdown da 847 testi su 1 100. Chain: 498
preamboli su 500. Studio chain vs singolo: R invariato (1.43 / 1.45); deriva di routing da
1.19 a 1.16 (Qwen3) e da 1.17 a 1.16 (BGE-M3), anche per l'esclusione sotto.

## 2026-10-02 — Ablazione senza le 10 opere con testi troncati

**Decisione.** Le opere con almeno un testo che, dopo la pulizia, non finisce con
punteggiatura di chiusura sono escluse per intero dal corpus di ablazione: 10 opere, ne
restano 90. Regola in `common.clean.is_truncated`, applicata da `load_corpus`.
**Motivo.** 10 testi del corpus di settembre si interrompono a metà frase (8 hanno
esaurito i 2 000 token di output, ragionamento compreso; 2 si fermano prima). Cadono in 10
opere diverse: escludere solo le celle avrebbe obbligato le metriche a gestire opere
incomplete (centering sulla media, permutazione entro opera); escludere le opere tiene
appaiati i confronti fra varianti. Rigenerarli avrebbe mescolato un'altra sessione e un
altro provider nel corpus. Nel corpus principale la regola non trova testi troncati.
Decisa prima di vedere i risultati.
**Effetto.** Nessuna conclusione dell'ablazione cambia. Qwen3: `no_def` e `no_need` a 1.00
del tetto indulgente e 0.99 di quello severo; effetto principale dello stile sul coseno
+0.197 (prima +0.182); `name_only` conserva 0.70 del coseno (prima 0.73).

## 2026-10-02 — RQ2: giudice GLM-5.3 su Ollama Cloud, persona con definizione

**Decisione.** RQ2 usa un solo giudice, GLM-5.3 su Ollama Cloud (FP8), in confronto a coppie
con scelta forzata. La persona è nome della categoria più definizione di Falk, senza
bisogno né stile. Coppie A (k vs flat), B (k vs j), C (j vs flat), baseline N senza persona
e controllo di attenzione, in entrambi gli ordini, sul corpus principale. Disegno e soglie
in `docs/plans/2026-10-01-rq2-giudice-design.md`, scritto prima dei giudizi.
**Motivo.** GLM-5.3 è il miglior giudice a pesi aperti su Judgemark v4 (72.6) fra quelli
disponibili su Ollama, ed è di una famiglia diversa dal generatore (Qwen3.8 non è su Ollama
Cloud, Kimi K3 costa circa 5 volte di più). Il confronto a coppie evita la compressione dei
voti su scala assoluta. Bisogno e stile sono istruzioni per lo scrittore: darli al giudice
renderebbe il compito un riconoscimento lessicale del prompt. Il piano di tesi prevedeva
più giudici e DeepSeek come controllo dell'auto-preferenza: con un solo generatore e un
giudice di un'altra famiglia, Massimo ha scelto un giudice solo.
**Effetto.** Nessun risultato precedente da confrontare.

## 2026-10-02 — RQ2: `reasoning_effort: "low"` invece di ragionamento spento

**Decisione.** Il giudice gira con `reasoning_effort: "low"`, non `"none"`.
**Motivo.** Nello smoke test, con `"none"` (e con `think: false` nell'API nativa) Ollama
non spegne il ragionamento di GLM-5.3 ma lo scrive dentro la risposta; con `"low"` la
risposta è JSON pulito e il ragionamento interno è minimo (21 caratteri in media, 56 token
di output per giudizio). Il parsing prende comunque l'ultimo oggetto JSON della risposta.
**Effetto.** Nessuno sui criteri: 4 200 risposte valide su 4 200.

## 2026-10-02 — RQ2: pilota e giro completo

**Decisione.** Pilota (controllo di attenzione su 100 opere, coppie su 20 opere, 1 000
giudizi, $1.25), poi giro completo sulle 100 opere (altri 3 200 giudizi, $4.07; totale
$5.32 di crediti Ollama).
**Esito dei criteri fissati prima.** Risposte valide 100% (soglia 98%), controllo di
attenzione 100% (soglia 95%): giudice utilizzabile, nessuna modifica al prompt.
**Risultato.** A = 0.967 [0.953, 0.979], B = 0.991 [0.984, 0.996], C = 0.207, N = 0.432,
A − C = 0.760 [0.718, 0.799], A − N = 0.535 [0.495, 0.573]. Consistenza fra i due ordini
0.88, scelte "A" 53%.

## 2026-10-02 — Dataset separati per scopo

**Decisione.** I dati generati sono separati per scopo: `data/main/` (studio chain: corpus
principale, chain e prefisso) e `data/ablation/` (corpus di settembre). Il corpus di
settembre è usato solo dall'ablazione. La misura della deriva di routing (`full` contro
`full_rep` del corpus di settembre, rispetto al rumore su provider fisso) passa da
`chain_noise.py` ad `ablation.py` (`routing_drift`), dove calibra il tetto indulgente; dal
corpus principale usa solo i suoi coseni interni.
**Motivo.** Tenere il corpus a routing libero isolato nell'unica analisi che ne ha bisogno:
lo studio chain vs singolo ora usa solo testi della stessa sessione e dello stesso provider.
Si è valutato di rigenerare l'ablazione su provider fisso (circa $0.60) o di usare per le
varianti di settembre il flat della sessione fissa; la seconda opzione è stata scartata
perché il flat è l'origine degli steering vector e la differenza sistematica fra i corpus
si sarebbe sommata a tutte le varianti, proprio dove l'ablazione ha i risultati nulli
(no_def, no_need). Massimo ha scelto di lasciare l'ablazione sul corpus di settembre.
**Effetto.** Nessuno sui valori: D = 1.16 (Qwen3 e BGE-M3) come prima. Gli IC bootstrap
dello shift relativo della chain cambiano alla seconda cifra (sequenza casuale diversa).

## 2026-10-02 — Ablazione calibrata con un solo tetto

**Decisione.** Tolto il tetto "indulgente" (coseno fra `full` e `full_rep` del corpus di
settembre) e la riga `full_rep` dalla tabella dell'ablazione. Resta un solo tetto, quello
del corpus principale (due run sullo stesso provider); `cos_rel` è ora il rapporto con quel
tetto (prima `cos_rel_strict`). La deriva di routing resta, come indicazione che il
rapporto è conservativo.
**Motivo.** Il tetto indulgente mescolava rumore di generazione e deriva di provider, e
serviva solo a dare un intervallo; con il corpus di settembre isolato per l'ablazione, il
tetto su provider fisso è l'unico riferimento pulito.
**Effetto.** I valori del rapporto sono quelli che prima comparivano come "severi": Qwen3
`no_def`/`no_need` 0.99, BGE-M3 0.97; nessuna conclusione cambia.

## 2026-10-02 — Effetti principali e interazioni tolti dal report

**Decisione.** Il report non riporta più la scomposizione dell'ablazione in effetti
principali e interazioni a due vie (ex §5.2); la lettura usa i confronti diretti fra
varianti (`style_only` vs `full`, `no_style`, `name_only` → `def_only`, `style_only` →
`no_need`). `ablation.py` continua a calcolarla (`factorial` nel JSON).
**Motivo.** Non aggiunge conclusioni rispetto alla tabella delle 8 varianti, e il nome
"analisi fattoriale" si confonde con la factor analysis.
**Effetto.** Nessuno sui risultati.

## 2026-10-07 — Modello aperto della parte 2: Gemma 4 31B su Kaggle

**Decisione.** Il generatore della parte 2 (profilo continuo) è Gemma 4 31B QAT a 4 bit
(`google/gemma-4-31B-it-qat-w4a16-ct`) con vLLM 0.31.0 su Kaggle (2× T4, float16, tensor
parallel 2), invece di Gemma 4 12B a 4 bit con MLX sul Mac previsto dalla spec del 6
ottobre. vLLM richiede sulle T4 una patch al kernel di attenzione Triton
(`kaggle/patch_vllm_turing.py`: blocchi da 16 token, un solo stadio) e gira senza encoder
multimodale né CUDA graph.
**Motivo.** Sul MacBook Air M2 (16 GB) il 12B produce testi corretti ma a 2.3–3.1 token/s:
oltre 30 ore per i 1 100 testi del passo 2. Su Kaggle (gratuito) il 31B fa 57 token/s in
batch, ~5 secondi a testo, ed è più vicino per qualità al generatore della parte 1. La
via sul generatore della parte 1 (miscela passo per passo su DeepInfra, stimata $4–25) è
rinviata a dopo i risultati sul modello aperto.
**Effetto.** Nessuno sui risultati esistenti. Il pilota del passo 1 (38 testi) supera i
criteri fissati nella spec. Tutti i testi della parte 2 vanno generati su questa
configurazione.

## 2026-10-07 — Recupero dei testi in violazione del corpus `local`

**Decisione.** I 13 testi Professional/Hobbyist del corpus `local` (5 `full`, 8 `full_rep`,
12 opere) rimasti in violazione del filtro dopo 4 tentativi sono stati rigenerati in una
seconda sessione Kaggle (`kaggle/retry`, `vllm_gen --retry-failed 12`), proseguendo la
sequenza di seed per testo dal tentativo 5 al 12, con la stessa configurazione (vLLM
0.31.0, stessi pesi, stessa patch). Le altre 1 087 righe non cambiano.
**Motivo.** Tutti e 13 per la formula "For those analyzing/studying…", vietata dal prompt;
il vizio è lo stesso di DeepSeek (Professional/Hobbyist rigenerati: Gemma 80 su 200,
DeepSeek 63 su 200), solo più ostinato. Escludere le 12 opere avrebbe tolto il 12% del
corpus e selezionato proprio le opere su cui la categoria scrive in quel modo; la stessa
procedura usata per DeepSeek è rigenerare finché il testo è valido.
**Effetto.** Tutti recuperati (7 al tentativo 5, gli altri fra il 7 e il 10), 199–227 parole.
Il corpus `local` ha 1 100 testi validi su 100 opere. Limite: 13 testi vengono da una
sessione diversa, con configurazione identica.

## 2026-10-08 — Miscela delle categorie passo per passo su vLLM

**Decisione.** I testi a profilo misto si generano con la miscela da fuori: a ogni passo
una chiamata a vLLM da un token per esperto con le prime 100 log-probabilità grezze, media
pesata dei log (equivalente alla media dei logit), token assenti da una lista al minimo di
quella lista. Gruppi di 6 testi misti per volta. Pilota con tutti e tre gli α generati con
questo metodo.
**Motivo.** vLLM non accoppia due sequenze a ogni passo; il ciclo esatto con
`transformers` avrebbe richiesto un altro motore e un'altra configurazione rispetto ai
testi puri. Con gruppi di 16 testi le sequenze non stanno nella cache KV (~6 600 token sul
31B con 2× T4) e il pilota ha richiesto 5 ore; con 12 sequenze il passo resta in cache.
**Effetto.** Verifica del metodo a α = 0 e 1: 1.00–1.01 del tetto rispetto ai testi puri;
copertura 97.8% a α = 0.5. I vertici del giro completo restano i testi puri del passo 2.

## 2026-10-08 — Recupero dei testi misti in violazione del filtro

**Decisione.** 7 testi misti rimasti in violazione del filtro dopo 4 tentativi (5
Facilitator–Professional/Hobbyist a α = 0.75, 2 del pilota a α = 1) rigenerati in una
seconda sessione con la stessa configurazione (`kaggle/mix-retry`, `vllm_mix
--retry-failed 12`), tentativi 5–12.
**Motivo.** Stessa politica dei testi puri (voce del 7 ottobre): tutti per "For those
analyzing/examining…", con il Professional/Hobbyist prevalente.
**Effetto.** Tutti validi; il corpus misto ha 940 testi validi.

## 2026-10-08 — Ritirata la "fusione per sezioni" dei testi misti

**Decisione.** Tolta dal report l'osservazione che a α = 0.5 la fusione è per sezioni
(apertura analitica, chiusura contemplativa), sostituita dall'analisi dei margini per
token (`generation/mix_diag.py`).
**Motivo.** L'osservazione veniva dalla lettura di due testi. Sui ~100 testi per coppia i
margini log p_X − log p_Y per frase non formano blocchi più che per caso e i confini di
frase contano poco; resta solo una tendenza nella chiusura.
**Effetto.** Il report della parte 2 (`docs/report-gemma.md`, §4) descrive un'alternanza fine dei due registri, senza sezioni,
con il profilo per posizione riportato per intero (vedi la voce del 13 ottobre).

## 2026-10-13 — Profilo per posizione dei testi misti riportato per intero

**Decisione.** Nel report di Gemma il profilo del margine per posizione (§4) è riportato per
tutti e dieci i decimi del testo, con un intervallo di confidenza bootstrap sui testi, e
marcato come esplorativo; prima il report ne mostrava solo gli ultimi tre decimi.
**Motivo.** Gli ultimi tre decimi erano stati scelti dopo aver visto i dati, dove c'era una
tendenza: una selezione che nascondeva l'apertura (Recharger nel Recharger–Professional/
Hobbyist, Professional/Hobbyist nel Facilitator–Professional/Hobbyist) e non diceva quali
differenze superano il rumore.
**Effetto.** Nessun numero cambia; si aggiungono gli intervalli. La lettura diventa:
apertura e chiusura tendono al Recharger nella coppia Recharger–Professional/Hobbyist,
l'apertura al Professional/Hobbyist nella coppia con il Facilitator, la chiusura
all'Explorer nella coppia vicina.

## 2026-10-19 — Ampiezza dello shift dalla distanza cross-validata

**Decisione.** Nel report di stato (§3.5) l'ampiezza dello shift di ogni categoria è la
radice della distanza quadrata cross-validata dal flat, non più la norma di v[c].
**Motivo.** Con il disegno bilanciato ‖v[c]‖² è la distanza quadrata semplice fra la media
della categoria e quella del flat: la stessa grandezza della distanza cross-validata, ma
gonfiata dal rumore di generazione (circa 0.0025 con Qwen3, fino al 9% per il Facilitator).
**Effetto.** L'ordine delle categorie non cambia; i valori scendono di poco (Experience
Seeker 0.21, Facilitator 0.17 con Qwen3) e si aggiungono quelli di BGE-M3.

## 2026-10-19 — Test di permutazione rimosso

**Decisione.** Il test di permutazione entro opera esce da `analyze.py` (sezione
`permutation` di `results/metrics_*.json`) e da `length.py` (`permutation_p_corrected` di
`results/length_*.json`), e dai due report.
**Motivo.** Non aggiunge informazione al probe lineare valutato su opere mai viste
(GroupKFold per opera): con accuracy 0.95–0.98 contro un caso di 0.17 il p era sempre al
minimo possibile. Inoltre il test implementato non era quello fissato nella spec del
profilo continuo ("probe sopra il caso con p < 0.01 al test di permutazione"): permutava le
etichette entro opera e confrontava la norma media degli scostamenti delle condizioni, non
l'accuracy del probe. Prima della rimozione dava p = 0.0005 (il minimo con 2 000
permutazioni) su `main` e `local` con entrambi gli embedding, e p = 0.001 (minimo con 1 000)
dopo la correzione per la lunghezza, sugli stessi corpus ed embedding.
**Effetto.** Nessun altro numero cambia. Il criterio della spec sul probe si legge come
"probe sopra il caso, valutato fuori dal fold per opera". Le figure LDA cambiano la metà di
opere usata per l'addestramento, perché il test consumava il generatore casuale condiviso;
la lettura delle figure non cambia. Il report di stato perde la sezione §3.6, i confondenti
diventano §3.6.

## 2026-10-19 — Lessico distintivo con log-odds ratio su n-grammi

**Decisione.** Il lessico per condizione esce da `analyze.py` (`top_words` nei
`results/metrics_*.json`) e passa a `rq1_embeddings/lexicon.py` (`results/lexicon.json`,
corpus `main` e `local`): log-odds ratio con prior di Dirichlet informativo (Monroe et al.
2008) contro il resto del corpus, su unigrammi, bigrammi e trigrammi, con la quota di testi
che contengono il termine e il segno di eco del blocco di categoria del prompt. Fissati
prima: 15 termini per condizione, presenza in almeno il 5% dei testi, unigrammi senza
parole funzionali. Dopo la prima uscita sono cambiate due regole: l'eco si riconosce per
parola piena a meno della flessione (prefisso di 5 lettere) invece che per n-gramma
identico, che non vedeva *your companions* o *discuss* (il blocco dice *companions*,
*discussion*); un termine si salta se condivide una parola piena con uno già scelto,
invece che solo se lo contiene, che lasciava catene come *you are standing* / *are
standing before*.
**Motivo.** `top_words` dava solo parole singole, con un rapporto di frequenze lisciato che
la docstring chiamava log-odds con prior informativo senza esserlo; il report nominava
l'eco del prompt solo come rischio. Il tf-idf con la condizione come documento è stato
scartato: con 6 documenti l'idf guarda solo in quante condizioni compare un termine, non
quanto.
**Effetto.** Nuova tabella del lessico nei due report (§3.6 del report di stato, §2 di
quello di Gemma), con la quota di eco: da 1 a 4 termini su 15 con DeepSeek, da 2 a 7 con
Gemma. Le altre metriche di `analyze.py` non cambiano.

## 2026-10-19 — Controllo di attenzione del giudice rimosso

**Decisione.** Il controllo di attenzione di RQ2 (il flat dell'opera contro il flat di
un'altra opera, giudice neutro) esce dalla generazione delle coppie (`rq2_judge/pairs.py`,
`mixed_pairs.py`), dalle analisi (`analyze.py`, `mixed_analyze.py`), da `explore.py` e dai
report. Le righe `attention` già raccolte restano nei `data/judgments*.jsonl` e si
ignorano al caricamento.
**Motivo.** Non aggiunge informazione: il giudice ha scelto il flat giusto nel 100% dei
casi su tutti e tre i corpus (200 giudizi su DeepSeek, 200 sui puri di Gemma, 100 sui
misti), e la qualità delle risposte è già coperta dalla quota di risposte valide, dalla
consistenza fra i due ordini e dal bias di posizione. Era un criterio di utilizzabilità
fissato prima dei dati nella spec di RQ2 (soglia 95%), superato prima della rimozione.
**Effetto.** I tassi di preferenza non cambiano. Cambiano i totali: 4 000 giudizi invece di
4 200 su DeepSeek ($5.08 invece di $5.32) e sui puri di Gemma, 8 100 invece di 8 200 sui
misti; la quota di scelte in posizione A si sposta di meno di 0.002.

## 2026-10-19 — RQ2: differenze A − C, A − N e Spearman fuori dai report

**Decisione.** Le righe A − C e A − N e la tabella degli Spearman fra accettazione e
geometria di RQ1 escono dai due report; restano in `results/judge_*.json`.
**Motivo.** Si leggono già dalle righe A, C, N e dalla matrice di preferenza; il legame con
RQ1 si dice con la confusione Facilitator–Explorer, comune a giudice e probe.
**Effetto.** Nessun numero cambia. A − C era l'ipotesi della spec di RQ2: IC complessivo
[0.718, 0.799] su DeepSeek, [0.72, 0.82] sui puri di Gemma, entrambi lontani da 0.

## 2026-10-19 — Analisi delle formule (`separation.py`) rimossa

**Decisione.** `rq1_embeddings/separation.py`, il suo test e `results/separation_*.json`
escono dal progetto; la tabella "formule / somiglianza sopra il flat" esce dal report di
Gemma (§2), che non la ripete più nel commento al lessico e nei limiti.
**Motivo.** Usava "formule" in un senso diverso dal lessico distintivo subito sotto
(trigrammi ricorrenti dentro la condizione, senza confronto con le altre, contro termini
distintivi rispetto al resto del corpus), e compariva solo per Gemma: la parte su `main`
restava fuori dai report. Il lessico copre già il registro di ogni categoria.
**Effetto.** Nessun altro risultato cambia. Ultimi valori (Qwen3) della quota di trigrammi
ricorrenti: Gemma 0.044 (Professional/Hobbyist) – 0.157 (Recharger), flat 0.034; DeepSeek
0.015 – 0.057, flat 0.020. Somiglianza fra opere sopra il flat, Recharger: +0.151 su Gemma,
+0.068 su DeepSeek.

## 2026-10-20 — Giudice sui misti: Spearman, incrocio, ordinamento e coerenza fuori dal report (momentaneo)

**Decisione.** Dal §6 del report di Gemma escono, per ora, lo Spearman fra α e preferenza,
l'incrocio delle due curve, l'ordinamento dei cinque testi da parte del giudice neutro e la
coerenza ("voce sola"). Restano le curve di preferenza e il doppio visitatore. Scelta
momentanea: le misure restano nel codice (`rq2_judge/mixed_analyze.py`) e in
`results/judge_mixed_glm-5.3.json` e possono tornare nel report.
**Motivo.** Report più semplice; la forma delle curve si legge già dai tassi per α.
**Effetto.** Nessun numero cambia. Valori a oggi (Recharger → Prof./Hobbyist,
Facilitator → Prof./Hobbyist, Explorer → Facilitator): Spearman |0.89|, |0.98|, |1.00|;
incrocio 0.50, 0.45, 0.66 (la coppia vicina fuori dalla soglia della spec 0.4–0.6);
ordinamento, Spearman mediano 1.00, 1.00, 0.90; coerenza 0.52, 0.57, 0.58. Le soglie della
spec su queste misure restano in `docs/plans/2026-10-06-profilo-continuo-design.md`.

## 2026-10-20 — Doppio visitatore con i puri ai vertici fuori dal report (momentaneo)

**Decisione.** Il §6.1 del report di Gemma (preferenza sul flat in funzione della quota w
della propria categoria, con la RQ2 sui puri ai vertici; `rq2_judge/combined.py`) esce per
ora dal report, con la sua figura e la soglia della spec sul doppio visitatore. Il doppio
visitatore resta nella riga in basso della figura del §6. Scelta momentanea: codice,
`results/judge_combined_glm-5.3.json` e `figures/judge_combined_glm-5.3.png` restano.
**Motivo.** La figura ripeteva i giudizi della riga in basso di `judge_mixed_glm-5.3.png`,
ordinati per w invece che per α.
**Effetto.** Nessun numero cambia. A oggi i vertici delle due stime differiscono al più di
0.03; la soglia della spec (massimo del doppio visitatore a un α intermedio) è superata
nelle due coppie contrapposte, con il massimo a 0.5, e nella coppia vicina il massimo è a
0.25 su una curva quasi piatta.

## 2026-10-26 — Lunghezza: tolte la correzione lineare degli embedding e la regressione del giudice

**Decisione.** Il controllo della lunghezza perde due analisi basate su un modello lineare:
la correzione degli embedding di `rq1_embeddings/length.py` (coefficiente per dimensione
sul log delle parole, poi probe e steering vector ricalcolati, anche nel confronto fra
generatori) e la regressione logistica della scelta del giudice sulla differenza di parole
in `rq2_judge/analyze.py`. Restano i controlli che non presuppongono una forma
dell'effetto: probe a 6 classi sulla sola lunghezza e probe senza flat (RQ1); tassi sulle
coppie con lunghezze entro il 10% e quota di coppie in cui l'elemento di interesse è il
testo più lungo (RQ2).
**Motivo.** Le due analisi assumevano un effetto lineare (nel log delle parole per gli
embedding, nella differenza di parole per il logit del giudice), e la correzione assumeva
anche che l'effetto stimato dentro le categorie valesse fra le categorie. Ipotesi difficili
da giustificare in tesi, per controlli già coperti da quelli che restano.
**Effetto.** Nessun valore che resta cambia, salvo gli IC bootstrap dei tassi sulle coppie
di lunghezza simile (terza cifra decimale, per il diverso uso del generatore casuale).
Escono da `results/length_*.json` le chiavi `*_corrected`, `steering_norm*` e
`cross_generator_corrected`; da `results/judge_glm-5.3*.json` la chiave
`length/logit_per_sd_words`. Valori a oggi delle analisi tolte: probe a 6 classi dopo la
correzione 0.95 (0.92) su `main`, 0.99 (0.95) su `local`; coseno fra steering vector
corretto e originale 0.99–1.00 (0.98–1.00) e 0.97–0.99 (0.94–0.99); logit del giudice per
30 parole +0.36 (DeepSeek) e +0.44 (Gemma). Nella spec del profilo continuo il punto 3
dell'analisi della lunghezza è ritirato.

## 2026-10-26 — Niente intervalli di confidenza; split-half e distanza su una sola divisione

**Decisione.** Tolti dal codice tutti gli intervalli di confidenza bootstrap: tassi del
giudice (`rq2_judge/analyze.py`, anche le barre d'errore della figura), R e rapporto
chain/tetto (`rq1_embeddings/chain_noise.py`), deriva di routing
(`rq1_embeddings/ablation.py`), profilo e margine medio di `generation/mix_diag.py` (anche
la banda della figura). Split-half e distanza cross-validata (`rq1_embeddings/analyze.py`)
si calcolano su una sola divisione delle opere (seed fisso, la stessa per le due misure e
per la LDA) invece che come media su 200 divisioni; la riproducibilità si verifica sul
tetto dalla replica. Le ripetizioni casuali restano solo dove stimano un atteso per caso
senza forma chiusa: i tagli casuali di `mix_diag` (100 per testo). I cambi di segno attesi
in ordine casuale passano dalla media di 200 rimescolamenti al valore esatto 2pq/(p + q).
**Motivo.** Analisi più semplici: gli IC non cambiavano nessuna conclusione, e la media su
molte divisioni aggiungeva calcolo a una misura già verificata dalla replica.
**Effetto.** I valori puntuali non cambiano (probe, tetto, R, rapporti chain/tetto, tassi
del giudice, deriva, coseni e rapporti dell'ablazione, profilo di `mix_diag`). Cambiano le
misure su una divisione: split-half di qualche centesimo nei corpus `main` e `local`
(Facilitator su BGE-M3 da 0.65 a 0.67); nell'ablazione, con 45 opere per metà, la media per
variante cambia fino a 0.08 (Qwen3 `def_only` da 0.78 a 0.70); distanze cross-validate al
più di 0.1 ×100. I criteri fissati prima dei dati che usavano un IC si leggono sul valore
puntuale: chain equivalente se R < 1.10 (era: limite superiore dell'IC sotto 1.10); effetto
del giudice presente se il tasso supera il valore nullo (era: IC che lo esclude). Gli esiti
non cambiano: R = 1.43 (1.45), A = 0.97, B = 0.99. Escono dai JSON le chiavi `ci95`,
`R_ci95`, `rel_ci95`, `profile_ci95`, `profile_mean_ci95` e gli `std` di split-half e
distanza; in `mix_diag.json` `sign_changes_shuffled` diventa `sign_changes_expected`.

## 2026-10-26 — Analisi ridotte a ciò che citano i report

**Decisione.** Tolto dal codice tutto ciò che i due report non citano, più i probe sulla
lunghezza e lo split-half con riferimento media:
- `rq1_embeddings/length.py` per intero (probe sulla sola lunghezza e probe senza flat); la
  lunghezza resta come mediana delle parole per condizione (`words_median` in
  `results/metrics_*.json`) e, per il giudice, come sensibilità sulle coppie di lunghezza
  simile;
- split-half con riferimento media in `analyze.py`;
- ablazione: effetti fattoriali, recall per classe del probe a 5 classi e la heatmap
  (`ablation_cosine_*.png`); resta una figura, `ablation_qwen.png`;
- giudice (`rq2_judge/analyze.py`): A − C e A − N, legame con RQ1 via Spearman, parole più
  frequenti nelle motivazioni, numerosità della matrice; la matrice persona × testo solo
  per il corpus principale;
- giudice sui misti: ordinamento, coerenza, Spearman, incrocio delle curve e criteri della
  spec (`mixed_analyze.py`, `mixed_pairs.py`, `mixed_judge.py`) e `rq2_judge/combined.py`;
  restano curva di preferenza e doppio visitatore;
- `rq1_embeddings/cross_generator.py` (confronto DeepSeek–Gemma, fuori dai report) e
  `generation/compare_chain.py` (prova esplorativa; i prompt della chain passano in
  `chain_study.py`), con i loro risultati;
- `mix_diag.py`: margine medio per token, quota sotto il caso, quota sopra 1, testi saltati;
- figure non citate: tutte quelle con BGE-M3 e, per il corpus `local`, confusione, LDA e
  distanza.
**Motivo.** Ridurre il progetto al minimo che serve ai risultati riportati.
**Effetto.** Nessun valore citato cambia, salvo i controlli del giudice sui misti, ora
calcolati sui soli giudizi di curva e doppio visitatore: 6 000 giudizi, posizione A 49.2%
(prima 8 100 con ordinamento e coerenza, 46.9%). Rinunce da dichiarare: senza probe la
lunghezza non è più esclusa come spiegazione dello shift (i valori ritirati erano: sola
lunghezza 0.28 su `main` e 0.35 su `local` contro 0.95 e 0.98 degli embedding); A − C > 0 era
un'ipotesi fissata prima dei dati nella spec di RQ2 (valore ritirato 0.76); per i misti le
soglie della spec su Spearman, incrocio, ordinamento e coerenza non si verificano più (valori
a oggi nella voce del 20 ottobre). I giudizi di ordinamento e coerenza restano in
`data/judgments_mixed.jsonl` e si ignorano.

## 2026-10-26 — Lunghezza tolta del tutto

**Decisione.** Esce ogni riferimento alla lunghezza come confondente: le parole mediane per
condizione (`words_median` in `results/metrics_*.json`, `by_alpha` di `results/mixed_*.json`),
le parole medie per variante dell'ablazione e per metodo nel confronto chain, e nel giudice
(`rq2_judge/analyze.py`) i tassi sulle coppie con lunghezze entro il 10%, la quota di coppie
in cui l'elemento di interesse è il più lungo e le motivazioni che citano la lunghezza. Nel
report di stato la §3.6 diventa *Lessico*.
**Motivo.** Completa la voce precedente: la lunghezza non entra più nelle analisi.
**Effetto.** Nessun altro numero cambia. Valori ritirati: mediane 253 parole
(Professional/Hobbyist) contro 218–236 su `main`, flat 178 contro 215–226 su `local`;
chain 244 parole medie contro 236 del singolo; ablazione 229–241 parole medie per variante; giudice,
tassi sulle coppie di lunghezza simile A 0.96, B 0.99, C 0.19, N 0.41 (DeepSeek) e A 0.98,
B 1.00, C 0.13, N 0.23 (Gemma), elemento di interesse più lungo nel 68% e nel 95% delle
coppie A. Da dichiarare: la lunghezza non è controllata né in RQ1 né in RQ2.

## 2026-10-27 — Giudice sui misti: torna la coerenza

**Decisione.** Torna nel codice (`mixed_pairs.py`, `mixed_judge.py`, `mixed_analyze.py`) e
nel §6 del report di Gemma la coerenza: il giudice neutro sceglie fra un misto e un puro
quale "suona come una voce sola". Spearman, incrocio e ordinamento restano fuori.
**Motivo.** Delle misure tolte il 26 ottobre è l'unica che non si legge già dalle curve o
dalla posizione negli embedding: dice che la miscela non degrada il testo, l'obiezione più
ovvia al metodo.
**Effetto.** I giudizi esistenti in `data/judgments_mixed.jsonl` bastano, nessun nuovo
giudizio. Coerenza 0.517, 0.567, 0.575 (Rech. → Prof., Fac. → Prof., Expl. → Fac.), sopra la
soglia della spec (0.4). I controlli si calcolano su 7 800 giudizi (curva, doppio visitatore
e coerenza): posizione A 46.9%.

## 2026-10-27 — Alternanza degli esperti: tolti dispersione per frase e profilo per posizione

**Decisione.** Da `generation/mix_diag.py` e dal §4 del report di Gemma escono la
dispersione dei margini per frase contro tagli casuali e il profilo del margine per decimo
del testo, con la figura `mix_diag_profile.png`. Restano i cambi di segno fra frasi
(osservati contro l'atteso esatto) e il margine medio, ora chiave `margin_mean` (era
`profile_mean`) di `results/mix_diag.json`.
**Motivo.** La dispersione arrivava alla stessa conclusione dei cambi di segno (niente
blocchi di frasi) ed era l'ultima misura che richiedeva ripetizioni casuali; il profilo era
esplorativo, fuori dai criteri della spec, e il margine medio ne è la media.
**Effetto.** Nessun valore che resta cambia. Valori a oggi delle misure tolte (Rech.–Prof.,
Fac.–Prof., Expl.–Fac.): dispersione 1.08, 1.01, 1.13; profilo, primo decimo +0.23, −0.17,
−0.03 e ultimo +0.36, −0.05, +0.20.

## 2026-10-28 — Corpus di ablazione rigenerato su provider fisso

**Decisione.** Il corpus di ablazione di settembre (routing libero, provider non registrato)
è sostituito da uno nuovo, generato con `generate.py --provider deepinfra/fp8 --reasoning`:
le 8 varianti, la replica `full_rep` e il flat, 4 600 testi, $0.59. `ablation.py` prende il
tetto dalla replica dello stesso corpus invece che dal corpus principale; la deriva di
routing (`routing_drift`) esce dal codice, dai risultati e dal report. `generate.py`
guadagna `--out` e `--reasoning`, e una risposta di provider occupato (429, 5xx) fa
attendere e riprovare lo stesso testo invece di consumare i tentativi.
**Motivo.** Il corpus di settembre era una miscela non tracciata di provider: non si
confrontava con il corpus principale e il suo tetto andava preso in prestito, con una
calibrazione solo conservativa.
**Effetto.** L'esecuzione è durata dal 27 al 28 ottobre per il sovraccarico di DeepInfra
(`engine_overloaded`, limite del pool condiviso di OpenRouter): `full` e le varianti sono
quasi tutti del 27, la replica quasi tutta del 28. Il tetto del nuovo corpus (Qwen3
0.969–0.989) è in linea con quello del corpus principale (0.957–0.990). 3 testi troncati
(contro 10), 97 opere (contro 90). Le conclusioni dell'ablazione non cambiano: lo stile da
solo conserva la direzione (coseno con il full 0.962, prima 0.951), toglierlo costa
(0.851, prima 0.828), il solo nome conserva circa il 70% (0.703, prima 0.695). Cambia
l'ordine in coda: con Qwen3 `need_only` (0.699) scende appena sotto `name_only` (0.703).
Togliere definizione o bisogno resta entro il rumore (cos/tetto 0.98–1.00 con entrambi gli
embedding; prima 0.97 con BGE-M3).

## 2026-10-28 — Ablazione: rigenerati i 3 testi troncati

**Decisione.** I 3 testi troncati del nuovo corpus di ablazione (Professional/Hobbyist:
*Garden at Sainte-Adresse* in `no_def`, *Les Grandes Baigneuses* e *The Haywain Triptych* in
`style_only`, tutti al limite di 2 000 token di output, ragionamento compreso) sono stati
tolti e rigenerati lo stesso giorno, con lo stesso provider e le stesse impostazioni;
embedding e ablazione ricalcolati per intero.
**Motivo.** Il caricamento del corpus escludeva le 3 opere da tutte le varianti; con i testi
completi l'ablazione usa tutte le 100 opere, come le altre analisi.
**Effetto.** 97 → 100 opere. I valori cambiano al più di qualche millesimo sul coseno con il
full (Qwen3 `style_only` 0.962 → 0.964, `no_style` 0.851 → 0.853, `name_only` 0.703 → 0.702,
`need_only` 0.699 → 0.701) e di qualche centesimo sullo split-half (una sola divisione delle
opere, ora su 100); nessuna conclusione cambia.

## 2026-10-28 — Alternanza degli esperti: statistiche per frase al posto del margine per decimi

**Decisione.** In `generation/mix_diag.py` ogni frase ha la somma dei margini dei suoi
token (prima la media; il segno è lo stesso). Per testo si calcolano il numero di frasi,
la somma massima e minima fra le frasi, la somma media in valore assoluto e i cambi di
segno fra frasi consecutive contro l'atteso in ordine casuale; poi la media sui testi.
Esce il margine medio per decimi del testo (`margin_mean`).
**Motivo.** Descrivere l'alternanza per frase, con l'intensità con cui una frase pende
verso un esperto, invece che con una media sul testo intero.
**Effetto.** I cambi di segno non cambiano (4.63 / 4.72, 4.33 / 4.59, 5.49 / 5.54).
Valore ritirato del margine medio: +0.13, +0.01, +0.03 (Rech.–Prof., Fac.–Prof.,
Expl.–Fac.). Nuovi valori in `results/mix_diag.json` e nel §4 del report di Gemma.

## 2026-10-28 — Alternanza degli esperti: divisione in frasi corretta

**Decisione.** `split_sentences` in `generation/mix_diag.py` chiude una frase solo se il
token seguente comincia con uno spazio o un a capo e se il punto non segue un'iniziale o
un'abbreviazione (St, Dr, Mr, Mrs, Ms, Jr, Sr, c., ca., vol., no.).
**Motivo.** Un controllo a campione ha trovato circa 50 frasi spezzate su 3 356: decimali
("81." + "5 cm"), iniziali ("Andrew W."), "St.", "D.C.". Resta un caso raro: una frase che
finisce davvero con un'iniziale (es. "Washington D.C.") si unisce alla successiva.
**Effetto.** Frasi per testo 11.2, 10.1, 12.4 → 11.0, 9.9, 12.3 (Rech.–Prof., Fac.–Prof.,
Expl.–Fac.); cambi di segno osservati / attesi 4.63 / 4.72, 4.33 / 4.59, 5.49 / 5.54 →
4.59 / 4.65, 4.23 / 4.48, 5.47 / 5.49; somme massima e minima invariate al decimo, somma
assoluta media di un decimo più alta. Conclusioni invariate.

## 2026-10-28 — Alternanza degli esperti: somma media con segno

**Decisione.** In `generation/mix_diag.py` la somma media delle frasi si calcola con il
segno (`sentence_sum_mean`) invece che in valore assoluto.
**Motivo.** Con il segno lo 0 fa da riferimento: dice verso quale esperto pende il testo; il
valore assoluto dava solo l'intensità, positiva per costruzione.
**Effetto.** Valori ritirati della somma assoluta media: 12.7, 8.4, 4.4 (Rech.–Prof.,
Fac.–Prof., Expl.–Fac.); nuovi valori con segno: +3.1, +0.3, +0.7.
