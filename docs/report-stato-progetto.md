# Museumbot — stato del progetto

*Ultimo aggiornamento: 1 ottobre 2026. Il report descrive solo lo stato attuale del codice e
dei risultati; le decisioni e le correzioni nel tempo sono in `docs/decisioni.md`.*

**Domanda di ricerca.** Quando un LLM genera un'audioguida "personalizzata" su una
categoria di visitatore di Falk, il testo cambia davvero, e in che direzione? Il paper di
riferimento (Dibitonto, Ferrato, Limongelli, Patroni, *Museum audio guides generation using
visitor categories and large language models*, Multimedia Systems 32:439, 2026) valuta le
audioguide con giudizio umano. Qui la domanda viene posta nello spazio latente: esiste, per
ogni categoria, uno **shift direzionale** misurabile e riproducibile rispetto a una
descrizione neutra?

**Risposta breve.** Sì. Con 100 opere e 6 condizioni, un probe lineare distingue la
condizione di generazione con accuracy del 94.7% (chance 16.7%), le direzioni di shift
rispetto al testo neutro sono stabili su metà disgiunte del corpus (coseno 0.85–0.94) e il
test di permutazione entro opera dà p = 0.0005. L'effetto è piccolo in termini di varianza
(7% contro il 70% spiegato dall'opera) ma è nitido, e le cinque categorie non collassano
fra loro.

---

## 1. Pipeline

| passo | modulo (in `src/museumbot/`) | output |
|---|---|---|
| corpus | `corpus/fetch_artworks.py` | `data/artworks.jsonl` — 100 dipinti da Wikidata (15 grandi musei, filtro sui sitelink), con l'intro della voce Wikipedia inglese troncata su confine di frase (67–207 parole, mediana 135) |
| prompt | `common/prompts.py` | 6 condizioni: 5 categorie Falk + `flat` (nessun blocco categoria) |
| generazione | `generation/generate.py` | `data/generations.jsonl` — 600 testi dello studio principale, 4 000 dell'ablazione (§5) e 500 della replica `full_rep` (§5.4); DeepSeek V4 Flash via OpenRouter, T = 0.7 |
| embedding | `rq1_embeddings/embed.py` | `data/emb_{qwen,bge-m3}[_variante].npy` (Qwen3-Embedding-0.6B, BGE-M3), locali, L2-normalizzati |
| analisi | `rq1_embeddings/analyze.py` | `results/metrics_{qwen,bge-m3}.json` + figure |
| ablazione | `rq1_embeddings/ablation.py` | `results/ablation_{qwen,bge-m3}.json` + figure |
| chain vs singolo | `generation/chain_study.py`, `rq1_embeddings/chain_noise.py` | `data/chain_study.jsonl`, `results/chain_noise_{qwen,bge-m3}.json` (§1.2) |

Il corpus è sbilanciato sui musei (Orsay 27, Prado 26, NGA 14, Met 13, resto < 10): non è
un problema per l'analisi, che centra per opera, ma va detto se si riporta il dataset.

**Provider.** OpenRouter distribuisce DeepSeek V4 Flash su circa 15 provider, con
quantizzazioni diverse (fp4, fp8, non dichiarata) e comportamento diverso sul
ragionamento (alcuni ragionano di default, altri no). Lo studio principale e l'ablazione
sono stati generati con routing libero e senza registrare il provider: il corpus è una
miscela non tracciata di provider, ed è un limite da dichiarare. `generate.py` ora salva il
provider e i token di ragionamento di ogni risposta e accetta `--provider` (es.
`deepinfra/fp8`) per fissarlo senza fallback. Lo studio chain vs singolo usa un provider
fisso.

**Preamboli.** `generate.py` non pulisce l'output: 135 testi su 4 548 del corpus iniziano
con un preambolo ("Here is …") e la maggior parte usa markdown (titoli in corsivo). Lo
studio chain vs singolo applica a tutti i testi la stessa pulizia (§1.2); il corpus
principale non è pulito.

### 1.1 Il prompt singolo

Il paper usa una chain di 3 turni (1: "conosci le categorie di Falk?", 2: "riconosci i bisogni
di ciascuna?", 3: "scrivi l'audioguida per la categoria X"). Qui la chain è collassata in un
unico system prompt, inserendo come contenuto la risposta che GPT-4 aveva dato al turno 2
(Appendice A.1 del paper). Ogni blocco categoria ha la stessa forma in tre parti:

1. **definizione** della categoria (Tabella 1 del paper, Falk & Dierking);
2. **bisogno** ("Their need is …", da Appendice A.1);
3. **indicazione di stile** (da Appendice A.1).

Fra le condizioni cambia *solo* il blocco. Opera, fonte, vincoli, limite di parole e
decoding sono identici. Il prompt vieta riferimenti espliciti alla categoria ("For those",
"As a …"); 23 testi su 600 hanno richiesto una rigenerazione per questo motivo.

### 1.2 Chain vs prompt singolo

**Domanda.** Collassare la chain del paper in un prompt singolo cambia i testi più di
quanto li cambi il rumore di generazione?

**Disegno** (`generation/chain_study.py`). Per ogni opera e categoria si generano tre testi
nella stessa sessione: due run indipendenti del prompt singolo (`single_a`, `single_b`) e
una della chain del paper (`chain`: turni 1–2 generati una volta e condivisi, come nel
paper, poi il Prompt 3 con la stessa fonte del singolo). Si aggiunge un testo `flat` per
opera, riferimento per gli steering vector. In totale 1 600 testi su DeepInfra fp8, con
ragionamento attivato esplicitamente; i job sono mescolati fra metodi, così nessun metodo
viene generato in un momento diverso dagli altri.

La chain apre sempre con una presentazione ("Here is a 250-word audio guide … for the
Recharger"), separata da `---`, usa markdown e in circa metà dei testi scrive in formato
copione: un'intestazione ("Audio Guide Script (approx. 250 words):", 240 testi su 500) e
didascalie di regia ("(Soft, inviting tone)", "(Fade out)"). `clean_guide` toglie
preambolo, separatori, markdown, conteggi delle parole, intestazioni e didascalie, ed è
applicata a **tutti** i metodi (sui singoli e sul flat non trova nulla da togliere oltre
al markdown); il testo originale resta nella riga.

**Analisi** (`rq1_embeddings/chain_noise.py`). Per ogni cella (opera, categoria):

```
within  = cos(S_a, S_b)                      rumore di generazione
between = ½ [cos(C, S_a) + cos(C, S_b)]      differenza fra metodi + rumore
R       = mean(1 − between) / mean(1 − within)
```

R ≈ 1 vuol dire che un testo della chain dista da un testo singolo quanto due testi singoli
fra loro. IC al 95% con bootstrap sulle opere. **Criterio, fissato prima di vedere i dati:
i due metodi sono equivalenti se il limite superiore dell'IC di R è sotto 1.10.** Assunzione:
la chain ha lo stesso rumore del singolo; se fosse più rumorosa l'eccesso verrebbe contato
come differenza, quindi un esito "equivalenti" è conservativo e un esito "non equivalenti"
va verificato con una seconda run della chain.

A livello di steering si confronta lo shift di categoria: `cos(v_C[c], v_S[c])` diviso per
il tetto `cos(v_Sa[c], v_Sb[c])`. Lo stesso `within` calcolato fra `full` e `full_rep` del
corpus principale (routing libero, settembre contro ottobre) misura quanto rumore aggiunge
il cambio di provider (§5.4).

**Risultati** (`results/chain_noise_{qwen,bge-m3}.json`; Qwen3, BGE-M3 fra parentesi).

![Dissimilarità per cella: rumore, chain vs singolo, deriva di routing (Qwen3)](../figures/chain_noise_qwen.png)

| | Qwen3 | BGE-M3 |
|---|---|---|
| cos singolo–singolo (rumore) | 0.885 | 0.920 |
| cos chain–singolo | 0.835 | 0.884 |
| **R** | **1.43** [1.39, 1.49] | **1.45** [1.40, 1.50] |

**I due metodi non sono equivalenti**: R supera la soglia di 1.10 con entrambi gli
embedding, e l'IC è lontano dalla soglia. Un testo della chain dista da un testo singolo
circa il 45% in più di quanto distino fra loro due testi singoli. R per categoria va da
1.26 (Explorer) a 1.63 (Professional/Hobbyist).

Anche lo shift di categoria cambia. Rapporto fra il coseno chain–singolo degli steering
vector e il tetto singolo–singolo (IC 95% bootstrap sulle opere):

| | tetto | chain / tetto | ‖v_chain‖ / ‖v_singolo‖ |
|---|---|---|---|
| Explorer | 0.96 (0.94) | 0.82 [0.75, 0.87] (0.89) | 0.93 (1.07) |
| Facilitator | 0.96 (0.91) | 0.71 [0.64, 0.78] (0.70) | 1.00 (1.66) |
| Experience Seeker | 0.98 (0.97) | 0.52 [0.45, 0.61] (0.69) | 0.74 (0.94) |
| Professional/Hobbyist | 0.99 (0.98) | 0.73 [0.68, 0.76] (0.77) | 0.79 (1.14) |
| Recharger | 0.99 (0.99) | 0.88 [0.84, 0.91] (0.94) | 0.94 (1.16) |
| media | | 0.73 (0.80) | |

La chain conserva la direzione di Recharger ed Explorer, molto meno quella di Experience
Seeker e Facilitator. I testi della chain sono un po' più lunghi (244 parole contro 236).

**Come leggerlo.** I risultati di RQ1 valgono per il prompt singolo, non per la chain del
paper: le due implementazioni della stessa idea producono testi e shift di categoria
diversi. Resta una riserva prevista dal disegno: se la chain fosse più rumorosa del
singolo, parte di R sarebbe rumore della chain e non differenza fra metodi. Una seconda run
della chain (`chain_rep`, 500 testi) la scioglierebbe; non è ancora stata generata.

**Deriva di routing.** Fra `full` e `full_rep` (provider misti, settembre contro ottobre) la
dissimilarità per cella è 1.19 [1.14, 1.24] (1.17 [1.13, 1.21]) volte quella fra due run
sullo stesso provider: il cambio di provider sposta i singoli testi in modo misurabile.

`generation/compare_chain.py` resta come prova esplorativa (3 opere × 3 categorie, V4
Flash e V4 Pro, un testo per metodo): non ha un termine di confronto per il rumore e non va
usato come risultato.

## 2. Metodo di analisi

Il passaggio decisivo è il **centering per opera**. Nello spazio di embedding la varianza
dominante è *quale opera* si descrive, non *per chi*. Senza centering ogni proiezione mostra
100 cluster-opera e nessuna struttura di categoria. Si sottrae a ogni opera un riferimento,
e il riferimento dipende dalla domanda:

```
media:  e'[a,c] = e[a,c] − mean_c e[a,·]     tutte le condizioni trattate allo stesso modo
flat:   e'[a,c] = e[a,c] − e[a,flat]          effetto della categoria rispetto al neutro
v[c]    = mean_a e'[a,c]                      steering vector della categoria c
```

Gli **steering vector** sono presi rispetto al **flat**: rispondono a "quanto e in che
direzione la categoria sposta il testo rispetto alla descrizione neutra". Con la media, i 6
v[c] sommerebbero a zero per costruzione e i coseni fra categorie sarebbero spinti verso
−1/(C−1) = −0.2: un artefatto del riferimento, non un'opposizione fra categorie. Il flat è
l'origine e non ha un vettore proprio.

La **media** resta per le procedure che lavorano sul singolo testo con le etichette (probe,
permutazione, scomposizione della varianza): lì tutte le condizioni devono essere
scambiabili, e sottrarre un solo testo flat aggiungerebbe il suo rumore a ogni opera.

Le metriche, in ordine di forza probatoria:

1. **probe lineare** (logistica, GroupKFold per opera; riferimento media): lo shift è
   abbastanza preciso da riconoscere la condizione su opere mai viste?
2. **consistenza split-half** (riferimento flat): v[c] stimato su due metà disgiunte di
   opere ha la stessa direzione? Se sì, è una proprietà della categoria, non del campione.
3. **geometria**: la **distanza quadrata cross-validata** fra condizioni dice quali sono
   vicine e quali lontane; coseni e norme dei v[c] (riferimento flat) dicono in che
   direzione e di quanto ciascuna categoria sposta il testo.
4. **test di permutazione entro opera** (riferimento media): si permutano le etichette
   *dentro* ogni opera, così l'effetto-opera è preservato e il null è onesto.
5. **lunghezza e lessico**: confondenti da sorvegliare.

La distanza cross-validata è la crossnobis senza normalizzazione per la covarianza del
rumore (in 1024 dimensioni con 100 opere quella stima richiederebbe una regolarizzazione
difficile da giustificare):

```
d²[a,b] = (v¹[a] − v¹[b]) · (v²[a] − v²[b])     v¹, v² da metà disgiunte di opere
```

mediata su 200 divisioni. È imparziale: vale circa 0 se due condizioni non differiscono,
mentre la distanza semplice è sempre gonfiata dal rumore. Usa solo differenze fra
condizioni, quindi non dipende dal riferimento. È preferita a un probe binario per coppia,
che con un probe a 6 classi già al 95% darebbe quasi ovunque il 100% e non distinguerebbe
le coppie.

Le figure LDA sono addestrate su metà delle opere e proiettate sull'altra metà: senza questa
separazione la figura sovrastima grossolanamente la separabilità.

## 3. Risultati (embedding Qwen3; BGE-M3 fra parentesi)

### 3.1 Varianza

| sorgente | quota di varianza |
|---|---|
| opera | 70.0% (77.1%) |
| categoria | 7.4% (3.7%) |

L'effetto di categoria è un ordine di grandezza sotto l'effetto opera. È atteso, e giustifica
il centering.

### 3.2 Probe lineare

![Matrice di confusione del probe lineare, embedding Qwen3](../figures/confusion_qwen.png)

Accuracy **94.7%** (90.3%). Recall per classe:

| condizione | Qwen3 | BGE-M3 |
|---|---|---|
| Explorer | 0.97 | 0.97 |
| Facilitator | 0.92 | 0.86 |
| Experience Seeker | 0.96 | 0.99 |
| Professional/Hobbyist | 0.98 | 1.00 |
| Recharger | 1.00 | 1.00 |
| Flat (baseline) | 0.85 | 0.60 |

Le confusioni residue coinvolgono quasi solo il flat: è coerente, il flat non è una
categoria ma il punto da cui le categorie si allontanano. Il probe lo scambia soprattutto
con Experience Seeker ed Explorer (Qwen3).

### 3.3 Stabilità delle direzioni (split-half)

Coseno fra v[c] stimati su due metà disgiunte di opere, riferimento flat.

| condizione | Qwen3 | BGE-M3 |
|---|---|---|
| Explorer | 0.85 ± 0.02 | 0.74 ± 0.02 |
| Facilitator | 0.88 ± 0.02 | 0.59 ± 0.03 |
| Experience Seeker | 0.90 ± 0.01 | 0.86 ± 0.01 |
| Professional/Hobbyist | 0.93 ± 0.01 | 0.86 ± 0.01 |
| Recharger | 0.94 ± 0.01 | 0.92 ± 0.01 |

Le direzioni sono stabili. Explorer e Facilitator sono le due più deboli in entrambi gli
embedding: sono anche le due il cui blocco di prompt è più "generico" (curiosità, gruppo) e
meno prescrittivo sullo stile. Il punto debole è Facilitator su BGE-M3 (0.59): con quel
modello la sua direzione rispetto al neutro è poco riproducibile.

Con il riferimento media i valori sono più alti (Qwen3 0.88–0.96, BGE-M3 0.76–0.95), perché
la media di 6 testi è un riferimento meno rumoroso di un solo testo flat. Sono il limite
superiore dell'affidabilità della direzione; i valori sopra sono l'affidabilità degli
steering vector effettivamente riportati. Con il riferimento media si misura anche la
stabilità del flat stesso: 0.78 (0.57).

### 3.4 Geometria

![PCA sui dati centrati per opera, con i vettori di steering](../figures/pca_qwen.png)

![LDA addestrata su metà delle opere e proiettata sull'altra metà](../figures/lda_qwen.png)

![Distanza quadrata cross-validata fra condizioni](../figures/distance_qwen.png)

![Coseno fra gli steering vector, riferimento flat](../figures/cosine_qwen.png)

**Separazione (distanza cross-validata, ×100).** Le coppie più vicine sono Explorer–Facilitator
(2.9) e poi queste due con il flat (3.5 e 4.1). Le più lontane sono Professional/Hobbyist–
Recharger (22.5) ed Experience Seeker–Recharger (19.0). Nessuna distanza è vicina a zero: le
cinque categorie non collassano. Su BGE-M3 l'ordine è lo stesso, ma Facilitator è quasi
sovrapposto al flat (0.5, contro 1.0 per Explorer–Facilitator): coerente con la sua
split-half bassa e con il recall del flat al 60%.

**Ampiezza (norme dei v[c], rispetto al flat, Qwen3).** Recharger 0.32,
Professional/Hobbyist 0.30, Experience Seeker 0.22, Facilitator 0.21, Explorer 0.20. Lo
shift più forte è quello di Recharger e Professional/Hobbyist.

**Direzione (coseno fra i v[c], rispetto al flat).** Explorer e Facilitator spostano il
testo nella stessa direzione (+0.61; +0.45 su BGE-M3), e nella LDA e nella PCA i loro
cluster si sovrappongono: sono le due categorie che rischiano di collassare l'una
sull'altra. Anche Recharger condivide in parte quella direzione (+0.32 con Explorer, +0.37
con Facilitator). Le direzioni più opposte sono Explorer–Experience Seeker (−0.36):
"guarda più da vicino" vs "must-see", ed Experience Seeker–Recharger (−0.30).
Professional/Hobbyist e Recharger sono la coppia più distante, ma le loro direzioni sono
solo moderatamente opposte (−0.20): la distanza viene soprattutto dal fatto che sono i due
shift più ampi.

### 3.5 Permutazione

Norma media osservata 0.200 contro null 0.042, **p = 0.0005** (2000 permutazioni entro opera).
Identico con BGE-M3 (0.121 vs 0.033).

### 3.6 Confondenti

**Lunghezza.** Professional/Hobbyist produce testi più lunghi (258 parole contro 224–239
delle altre condizioni). Parte del suo steering vector potrebbe essere "lunghezza" e non
registro. Va controllato, ad esempio con un probe che riceve anche la lunghezza come feature,
o sottraendo la componente correlata con la lunghezza.

**Lessico.** Le parole discriminative sono coerenti con i blocchi di prompt: Recharger
(*breath, settle, slowly, pause, rest*), Facilitator (*talk, discuss, companions, group*),
Experience Seeker (*must, iconic, legendary, unforgettable*), Professional/Hobbyist
(*scholarly, tonal, compositional, compare*), Explorer (*closer, curious, clue, puzzle*).
È una conferma, ma anche un avvertimento: parte dello shift potrebbe essere puro eco
lessicale delle istruzioni di stile, più che un cambio di registro. È esattamente la
domanda a cui risponde il passo successivo.

## 4. Conclusioni

- Il condizionamento sulla categoria Falk produce uno shift latente reale, riproducibile e
  specifico per categoria, robusto al modello di embedding.
- Le cinque categorie non collassano: quattro sono ben separate; Explorer e Facilitator
  sono le più vicine fra loro, le più vicine al flat e le più deboli, e spostano il testo
  in direzioni simili.
- Il flat si comporta come baseline: è il testo da cui le categorie si allontanano, e il
  probe lo confonde solo con le categorie a esso più vicine.
- I risultati valgono per il prompt singolo. La chain del paper produce testi diversi
  oltre il rumore e uno shift di categoria solo in parte uguale (§1.2).
- Punto aperto: il confondente lunghezza per Professional/Hobbyist.

## 5. Ablazione del prompt: quale parte del blocco produce lo shift?

Ogni blocco categoria è composto da tre parti in ordine fisso: **definizione** (chi è il
visitatore), **bisogno** ("Their need is …"), **stile** (come scrivere). L'ablazione genera
le stesse 500 audioguide (5 categorie × 100 opere) per 8 varianti del blocco, in un disegno
fattoriale 2×2×2 più la variante "solo nome". Il flat non varia. Stesso modello, stessa
temperatura, stesso filtro sui riferimenti espliciti. Costo 0.45 $; rigenerazioni per
violazione del filtro 3.9% (3.8% nel run originale).

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

Per ogni variante: probe a 5 classi (flat escluso, è identico fra varianti), norma media
degli steering vector, **coseno con il full** per categoria (la metrica chiave: dice se la
parte conserva la *direzione* dello shift, non solo la separabilità), split-half e, quando
c'è la replica `full_rep`, il **coseno rispetto al tetto** (§5.4). Gli
steering vector sono presi rispetto al flat, che è lo stesso testo in tutte le varianti:
il riferimento è quindi identico fra varianti. Con il riferimento media, invece, il
riferimento include le 5 categorie della variante e si sposta con essa, contaminando il
confronto con il full. Poi
l'analisi fattoriale: effetto principale di ciascuna parte e interazioni a due vie, sulle
medie delle 8 celle.

![Ablazione: separabilità, ampiezza e direzione per variante (Qwen3)](../figures/ablation_qwen.png)

![Coseno con il full, varianti × categorie (Qwen3)](../figures/ablation_cosine_qwen.png)

### 5.1 Risultati per variante (Qwen3; BGE-M3 fra parentesi)

Ordinate per coseno medio con il full. `cos/tetto` è il coseno con il full diviso per il
coseno fra due run del full: il primo valore usa il tetto indulgente, il secondo quello
severo (§5.4).

| variante | probe 5 classi | ‖v‖ media | cos·full | cos/tetto | split-half |
|---|---|---|---|---|---|
| `full` | 98.4% (97.0%) | 0.249 (0.149) | 1.000 (1.000) | — | 0.90 (0.79) |
| `no_need` | 98.6% (95.0%) | 0.251 (0.149) | 0.973 (0.929) | 1.00–1.00 (0.97–0.99) | 0.90 (0.79) |
| `no_def` | 98.2% (95.2%) | 0.254 (0.147) | 0.970 (0.935) | 1.00–1.00 (0.98–1.00) | 0.91 (0.79) |
| `style_only` | 97.0% (95.2%) | 0.243 (0.142) | 0.958 (0.909) | 0.98–0.99 (0.95–0.97) | 0.90 (0.77) |
| `no_style` | 91.8% (90.4%) | 0.181 (0.121) | 0.855 (0.850) | 0.88–0.88 (0.89–0.91) | 0.81 (0.72) |
| `def_only` | 93.0% (90.0%) | 0.186 (0.121) | 0.852 (0.827) | 0.87–0.88 (0.86–0.88) | 0.82 (0.69) |
| `need_only` | 85.2% (87.4%) | 0.172 (0.111) | 0.740 (0.738) | 0.76–0.76 (0.77–0.79) | 0.80 (0.66) |
| `name_only` | 77.4% (74.6%) | 0.146 (0.098) | 0.725 (0.677) | 0.74–0.75 (0.71–0.72) | 0.66 (0.53) |
| `full_rep` | 96.6% (95.4%) | 0.246 (0.144) | 0.972 (0.934) | — | 0.90 (0.78) |

Chance del probe a 5 classi: 20%. La lunghezza media resta fra 229 e 240 parole per tutte
le varianti: l'ablazione non introduce un confondente di lunghezza.

### 5.2 Analisi fattoriale

Effetti principali (media delle 4 celle con la parte meno media delle 4 senza) e
interazioni a due vie, sul coseno medio con il full e sulla norma media.

| | def | need | style | def×need | def×style | need×style |
|---|---|---|---|---|---|---|
| cos·full (Qwen3) | +0.072 | +0.014 | **+0.182** | +0.001 | **−0.100** | +0.011 |
| cos·full (BGE-M3) | +0.087 | +0.045 | **+0.170** | +0.003 | **−0.088** | +0.006 |
| ‖v‖ media (Qwen3) | +0.013 | +0.007 | **+0.078** | −0.022 | −0.023 | −0.006 |
| ‖v‖ media (BGE-M3) | +0.010 | +0.004 | **+0.034** | −0.009 | −0.012 | −0.004 |

### 5.3 Lettura

**Lo stile è la discriminante.** È sufficiente: da solo riproduce la direzione del blocco
completo (coseno 0.96 / 0.91) con la stessa ampiezza e la stessa separabilità. Ed è
necessaria: toglierlo è l'unica rimozione singola con un costo netto (0.86 / 0.85, e
la norma cala di un quarto su Qwen3 e di un quinto su BGE-M3). L'effetto principale dello
stile sul coseno è 2.5 volte quello della definizione e 13 volte quello del bisogno (Qwen3;
2.0 e 3.8 volte su BGE-M3).

**La definizione conta solo in assenza dello stile.** L'interazione def×style è negativa e
grande (−0.10 / −0.09): la definizione porta il coseno da 0.73 a 0.85 (0.71 a 0.84) quando
lo stile manca, ma da 0.96 a 0.99 (0.92 a 0.96) quando c'è. Le due parti dicono al modello
la stessa cosa e lo stile la dice in modo più operativo.

**Il bisogno da solo non basta, e in un caso devia.** `need_only` è la più debole delle
varianti a una parte, e per Experience Seeker il coseno crolla a 0.25 / 0.39: la frase
"Their need is memorable, high-impact takeaways", senza definizione né stile, spinge il
testo in una direzione quasi scorrelata da quella del blocco completo, e più debole (norma
ridotta di circa il 40%). È l'unico caso in cui una
parte del prompt non è un sottoinsieme dell'effetto totale ma un effetto diverso.

**Il nome da solo conserva circa tre quarti dello shift.** Con "Your listener is a
Recharger." e nient'altro, il probe resta al 77% / 75% contro il 20% di chance, e il
coseno medio è 0.73 / 0.68. Questa è la conoscenza parametrica di Falk del modello. Ma è
distribuita in modo molto diseguale: Recharger (0.94 / 0.94) e Professional/Hobbyist
(0.78 / 0.84) sono nomi autoesplicativi e il modello li interpreta come il blocco
completo; Explorer e Facilitator stanno nel mezzo (0.71 e 0.74 / 0.60 e 0.60); Experience
Seeker no (0.46 / 0.40), ed è anche il nome più ambiguo in inglese comune.

**Asimmetria fra categorie.** Recharger è robusto a ogni ablazione (coseno ≥ 0.93 in tutte
le varianti): la sua direzione, quella contemplativa, è così marcata che qualunque indizio
basta a evocarla. Experience Seeker è la più fragile quando manca lo stile (`need_only`,
`name_only`): la sua direzione dipende dall'istruzione esplicita, non dal nome.

### 5.4 Calibrazione del coseno

Il **tetto** è il coseno che otterrebbe una variante identica al full, dato il rumore di
generazione a T = 0.7. Quello giusto sarebbe fra due run del full con la miscela di provider
di settembre, che non è più riproducibile (§1). Due stime lo racchiudono:

- **indulgente**: coseno fra il full e `full_rep`, una seconda generazione del full a
  ottobre con routing libero e lo stesso flat (registro `REPLICATES` in
  `common/prompts.py`, fuori dal disegno fattoriale). Include anche la deriva di provider,
  quindi è più basso del vero e i rapporti sono sovrastimati;
- **severo**: coseno fra `single_a` e `single_b` dello studio chain (§1.2), due run del full
  sullo stesso provider. Solo rumore di generazione, quindi più alto del vero.

| | Explorer | Facilitator | Exp. Seeker | Prof./Hobbyist | Recharger |
|---|---|---|---|---|---|
| indulgente Qwen3 | 0.965 | 0.963 | 0.975 | 0.974 | 0.981 |
| severo Qwen3 | 0.957 | 0.958 | 0.980 | 0.985 | 0.990 |
| indulgente BGE-M3 | 0.899 | 0.875 | 0.971 | 0.958 | 0.969 |
| severo BGE-M3 | 0.937 | 0.905 | 0.974 | 0.977 | 0.985 |

Con Qwen3 i due tetti quasi coincidono: sugli steering vector, che sono medie su 100 opere,
la deriva di provider pesa poco, anche se sui singoli testi è misurabile (§1.2). Con
BGE-M3 il tetto severo è più alto per Explorer e Facilitator.

Rapportate al tetto (colonna `cos/tetto` in §5.1), `no_def` e `no_need` valgono 1.00 con
Qwen3 con entrambi i tetti: togliere la definizione o il bisogno **non lascia traccia
misurabile oltre al rumore**. Con BGE-M3 e il tetto severo valgono 0.97–0.98: una perdita
piccola ma visibile. `style_only` è a 0.98–0.99 (0.95–0.97). Togliere lo stile costa circa
il 10%, `need_only` e `name_only` un quarto, con entrambi i tetti.

### 5.5 Implicazione per il prompt

Per un sistema di produzione il blocco può ridursi all'istruzione di stile: stessa
efficacia, circa metà delle parole del blocco di categoria (il resto del system prompt non
cambia). Per lo studio scientifico l'ablazione dice che quello che
il paper chiama "adattamento alla categoria di Falk" è, nel modello, in larga parte
l'esecuzione di un'istruzione stilistica esplicita, e in misura minore l'evocazione di uno
stereotipo associato al nome. La definizione sociologica della categoria non aggiunge
nulla di misurabile quando lo stile è presente (al più il 2–3% con BGE-M3, §5.4).

## 6. Prossimi passi

1. Chain vs singolo: decidere se generare `chain_rep` per separare il rumore della chain
   dalla differenza fra metodi (§1.2).
2. Robustezza dello studio principale: rifare probe, split-half e distanze sul corpus
   dello studio chain (`single_a` + flat, provider fisso, testi puliti) e confrontarli con
   quelli di settembre. Solo calcolo locale.
3. Confondente lunghezza per Professional/Hobbyist nello studio principale (§3.6).
4. Preamboli e markdown nel corpus principale (§1): decidere se pulirli e ricalcolare.
5. RQ2 e RQ3: vedi `docs/plans/piano-progetto-tesi.md`.

## Riferimenti

- Dibitonto M., Ferrato A., Limongelli C., Patroni O.C. (2026). Museum audio guides
  generation using visitor categories and large language models. *Multimedia Systems* 32:439.
- Zheng M. et al. (2024). When "A Helpful Assistant" Is Not Really Helpful: Personas in
  System Prompts Do Not Improve Performances of LLMs. *Findings of EMNLP 2024*.
  arXiv:2311.10054.
- Lutz M. et al. (2025). The Prompt Makes the Person(a): A Systematic Evaluation of
  Sociodemographic Persona Prompting for LLMs. *Findings of EMNLP 2025*. arXiv:2507.16076.
- Sclar M. et al. (2024). Quantifying Language Models' Sensitivity to Spurious Features in
  Prompt Design. *ICLR 2024*. arXiv:2310.11324.
- Chen R. et al. (2025). Persona Vectors: Monitoring and Controlling Character Traits in
  Language Models. arXiv:2507.21509.
- Rooein D. et al. (2023). Know your audience: Do LLMs adapt to different age and education
  levels? arXiv:2312.02065.
