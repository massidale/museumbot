# Museumbot — stato del progetto al 21 settembre 2026

**Domanda di ricerca.** Quando un LLM genera un'audioguida "personalizzata" su una
categoria di visitatore di Falk, il testo cambia davvero, e in che direzione? Il paper di
riferimento (Dibitonto, Ferrato, Limongelli, Patroni, *Museum audio guides generation using
visitor categories and large language models*, Multimedia Systems 32:439, 2026) valuta le
audioguide con giudizio umano. Qui la domanda viene posta nello spazio latente: esiste, per
ogni categoria, uno **shift direzionale** misurabile e riproducibile rispetto a una
descrizione neutra?

**Risposta breve.** Sì. Con 100 opere e 6 condizioni, un probe lineare distingue la
condizione di generazione con accuracy del 94.5% (chance 16.7%), le direzioni di shift sono
stabili su metà disgiunte del corpus (coseno 0.88–0.96) e il test di permutazione entro
opera dà p = 0.0005. L'effetto è piccolo in termini di varianza (7% contro il 70% spiegato
dall'opera) ma è nitido, e le cinque categorie non collassano fra loro.

---

## 1. Pipeline

| passo | script | output |
|---|---|---|
| corpus | `src/fetch_artworks.py` | `data/artworks.jsonl` — 100 dipinti da Wikidata (15 grandi musei, filtro sui sitelink), con l'intro della voce Wikipedia inglese troncata su confine di frase (67–207 parole, mediana 135) |
| prompt | `src/prompts.py` | 6 condizioni: 5 categorie Falk + `flat` (nessun blocco categoria) |
| generazione | `src/generate.py` | `data/generations.jsonl` — 600 testi, DeepSeek V4 Flash via OpenRouter, T = 0.7, costo totale 0.09 $ |
| embedding | `src/embed.py` | `data/emb_qwen.npy` (Qwen3-Embedding-0.6B) e `data/emb_bge-m3.npy` (BGE-M3), locali, L2-normalizzati |
| analisi | `src/analyze.py` | `results/metrics_{qwen,bge-m3}.json` + figure |

Il corpus è sbilanciato sui musei (Orsay 27, Prado 26, NGA 14, Met 13, resto < 10): non è
un problema per l'analisi, che centra per opera, ma va detto se si riporta il dataset.

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

### 1.2 Chain vs prompt singolo (analisi preliminare)

`src/compare_chain.py` confronta chain e singolo su 3 opere × 3 categorie, con V4 Flash e V4
Pro, passando la stessa fonte a entrambi i metodi. Al momento il confronto si limita al
numero di parole (chain 250–318, singolo 238–285) e a una lettura qualitativa: i testi sono
equivalenti nel registro. Non c'è ancora una misura di equivalenza semantica; è un punto
aperto, non un risultato.

## 2. Metodo di analisi

Il passaggio decisivo è il **centering per opera**. Nello spazio di embedding la varianza
dominante è *quale opera* si descrive, non *per chi*. Senza centering ogni proiezione mostra
100 cluster-opera e nessuna struttura di categoria.

```
e'[a,c] = e[a,c] − mean_c e[a,·]        rimuove l'identità dell'opera
v[c]    = mean_a e'[a,c]                steering vector della categoria c
```

Le metriche, in ordine di forza probatoria:

1. **probe lineare** (logistica, GroupKFold per opera): lo shift è abbastanza preciso da
   riconoscere la condizione su opere mai viste?
2. **consistenza split-half**: v[c] stimato su due metà disgiunte di opere ha la stessa
   direzione? Se sì, è una proprietà della categoria, non del campione.
3. **geometria fra i v[c]**: coseni e norme; quali categorie collassano?
4. **test di permutazione entro opera**: si permutano le etichette *dentro* ogni opera, così
   l'effetto-opera è preservato e il null è onesto.
5. **lunghezza e lessico**: confondenti da sorvegliare.

Le figure LDA sono addestrate su metà delle opere e proiettate sull'altra metà: senza questa
separazione la figura sovrastima grossolanamente la separabilità.

## 3. Risultati (embedding Qwen3; BGE-M3 fra parentesi)

### 3.1 Varianza

| sorgente | quota di varianza |
|---|---|
| opera | 69.9% (77.1%) |
| categoria | 7.4% (3.7%) |

L'effetto di categoria è un ordine di grandezza sotto l'effetto opera. È atteso, e giustifica
il centering.

### 3.2 Probe lineare

![Matrice di confusione del probe lineare, embedding Qwen3](../figures/confusion_qwen.png)

Accuracy **94.5%** (90.3%). Recall per classe:

| condizione | Qwen3 | BGE-M3 |
|---|---|---|
| Explorer | 0.96 | 0.97 |
| Facilitator | 0.92 | 0.86 |
| Experience Seeker | 0.97 | 0.99 |
| Professional/Hobbyist | 0.98 | 1.00 |
| Recharger | 1.00 | 1.00 |
| Flat (baseline) | 0.84 | 0.60 |

Le confusioni residue coinvolgono quasi solo il flat: è coerente, il flat non è una
categoria ma il centro da cui le categorie si allontanano, quindi il probe lo confonde con
le categorie a shift più debole (Explorer, Experience Seeker).

### 3.3 Stabilità delle direzioni (split-half)

| condizione | Qwen3 | BGE-M3 |
|---|---|---|
| Explorer | 0.88 ± 0.02 | 0.81 ± 0.01 |
| Facilitator | 0.91 ± 0.01 | 0.76 ± 0.02 |
| Experience Seeker | 0.96 ± 0.01 | 0.93 ± 0.01 |
| Professional/Hobbyist | 0.96 ± 0.01 | 0.92 ± 0.01 |
| Recharger | 0.96 ± 0.01 | 0.95 ± 0.00 |
| Flat | 0.78 ± 0.03 | 0.57 ± 0.03 |

Le direzioni sono stabili. Explorer e Facilitator sono le due più deboli in entrambi gli
embedding: sono anche le due il cui blocco di prompt è più "generico" (curiosità, gruppo) e
meno prescrittivo sullo stile.

### 3.4 Geometria

![PCA sui dati centrati per opera, con i vettori di steering](../figures/pca_qwen.png)

![LDA addestrata su metà delle opere e proiettata sull'altra metà](../figures/lda_qwen.png)

![Coseno fra i vettori di steering](../figures/cosine_qwen.png)

Norme dei v[c] (Qwen3): Recharger 0.28, Professional/Hobbyist 0.27, Experience Seeker 0.24,
Facilitator 0.16, Explorer 0.15, Flat 0.10. Lo shift più forte è quello di Recharger e
Professional/Hobbyist, che sono anche quasi opposti (coseno −0.53): contemplazione vs
analisi tecnica. Experience Seeker è opposto a Explorer (−0.50): "must-see" vs "guarda più da
vicino". Explorer e Facilitator sono le uniche due con coseno positivo apprezzabile (+0.33),
e nella LDA e nella PCA i loro cluster si sovrappongono: sono le due categorie che rischiano
di collassare l'una sull'altra.

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
  sono le più vicine fra loro e le più deboli.
- Il flat si comporta come baseline: shift minimo e direzione poco stabile.
- Due punti aperti: il confondente lunghezza per Professional/Hobbyist e l'equivalenza
  chain/singolo, finora solo qualitativa.

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
parte conserva la *direzione* dello shift, non solo la separabilità), split-half. Poi
l'analisi fattoriale: effetto principale di ciascuna parte e interazioni a due vie, sulle
medie delle 8 celle.

![Ablazione: separabilità, ampiezza e direzione per variante (Qwen3)](../figures/ablation_qwen.png)

![Coseno con il full, varianti × categorie (Qwen3)](../figures/ablation_cosine_qwen.png)

### 5.1 Risultati per variante (Qwen3; BGE-M3 fra parentesi)

Ordinate per coseno medio con il full.

| variante | probe 5 classi | ‖v‖ media | cos·full | split-half |
|---|---|---|---|---|
| `full` | 98.4% (97.0%) | 0.220 (0.135) | 1.000 (1.000) | 0.93 (0.87) |
| `no_need` | 98.6% (95.0%) | 0.223 (0.136) | 0.969 (0.930) | 0.94 (0.87) |
| `no_def` | 98.2% (95.2%) | 0.220 (0.132) | 0.969 (0.930) | 0.94 (0.87) |
| `style_only` | 97.0% (95.2%) | 0.217 (0.131) | 0.960 (0.911) | 0.94 (0.86) |
| `no_style` | 91.8% (90.4%) | 0.166 (0.110) | 0.871 (0.865) | 0.89 (0.81) |
| `def_only` | 93.0% (90.0%) | 0.173 (0.113) | 0.868 (0.857) | 0.90 (0.82) |
| `need_only` | 85.2% (87.4%) | 0.148 (0.097) | 0.776 (0.754) | 0.85 (0.76) |
| `name_only` | 77.4% (74.6%) | 0.131 (0.088) | 0.741 (0.683) | 0.78 (0.69) |

Chance del probe a 5 classi: 20%. La lunghezza media resta fra 229 e 240 parole per tutte
le varianti: l'ablazione non introduce un confondente di lunghezza.

### 5.2 Analisi fattoriale

Effetti principali (media delle 4 celle con la parte meno media delle 4 senza) e
interazioni a due vie, sul coseno medio con il full e sulla norma media.

| | def | need | style | def×need | def×style | need×style |
|---|---|---|---|---|---|---|
| cos·full (Qwen3) | +0.065 | +0.020 | **+0.160** | −0.005 | **−0.091** | +0.000 |
| cos·full (BGE-M3) | +0.094 | +0.042 | **+0.153** | −0.006 | **−0.098** | +0.005 |
| ‖v‖ media (Qwen3) | +0.016 | +0.002 | **+0.065** | −0.015 | −0.027 | −0.005 |
| ‖v‖ media (BGE-M3) | +0.012 | +0.002 | **+0.031** | −0.007 | −0.014 | −0.003 |

### 5.3 Lettura

**Lo stile è la discriminante.** È sufficiente: da solo riproduce la direzione del blocco
completo (coseno 0.96 / 0.91) con la stessa ampiezza e la stessa separabilità. Ed è
necessaria: toglierlo è l'unica rimozione singola che costa qualcosa (0.87 / 0.87, e
la norma cala di un quarto). L'effetto principale dello stile sul coseno è 2.5 volte
quello della definizione e 8 volte quello del bisogno.

**La definizione conta solo in assenza dello stile.** L'interazione def×style è negativa e
grande (−0.09 / −0.10): la definizione porta il coseno da 0.74 a 0.87 quando lo stile
manca, ma da 0.96 a 0.97 quando c'è. Le due parti dicono al modello la stessa cosa e lo
stile la dice in modo più operativo.

**Il bisogno da solo non basta, e in un caso devia.** `need_only` è la più debole delle
varianti a una parte, e per Experience Seeker il coseno crolla a 0.50 / 0.53: la frase
"Their need is memorable, high-impact takeaways", senza definizione né stile, spinge il
testo in una direzione diversa da quella del blocco completo. È l'unico caso in cui una
parte del prompt non è un sottoinsieme dell'effetto totale ma un effetto diverso.

**Il nome da solo conserva circa tre quarti dello shift.** Con "Your listener is a
Recharger." e nient'altro, il probe resta al 77% / 75% contro il 20% di chance, e il
coseno medio è 0.74 / 0.68. Questa è la conoscenza parametrica di Falk del modello. Ma è
distribuita in modo molto diseguale: Recharger (0.94 / 0.95) e Professional/Hobbyist
(0.84 / 0.91) sono nomi autoesplicativi e il modello li interpreta come il blocco
completo; Explorer (0.56 / 0.45) e Experience Seeker (0.62 / 0.52) no. Sono anche le due
categorie con i nomi più ambigui in inglese comune.

**Asimmetria fra categorie.** Recharger è robusto a ogni ablazione (coseno ≥ 0.94 in tutte
le varianti): la sua direzione, quella contemplativa, è così marcata che qualunque indizio
basta a evocarla. Explorer è la più fragile, e già nello studio principale era la categoria
con lo shift più debole.

**Limite: manca la calibrazione del coseno.** I coseni si leggono solo in ordine relativo.
Non sappiamo quale sia il coseno fra due run del *medesimo* prompt full a temperatura 0.7,
quindi non possiamo dire se 0.97 (`no_def`) è "identico al full" o già una perdita
misurabile. Una seconda generazione del full su 100 opere (≈ 0.05 $) darebbe il tetto.

### 5.4 Implicazione per il prompt

Per un sistema di produzione il blocco può ridursi all'istruzione di stile: stessa
efficacia, un terzo dei token. Per lo studio scientifico l'ablazione dice che quello che
il paper chiama "adattamento alla categoria di Falk" è, nel modello, quasi interamente
l'esecuzione di un'istruzione stilistica esplicita, e in misura minore l'evocazione di uno
stereotipo associato al nome. La definizione sociologica della categoria non aggiunge
nulla quando lo stile è presente.

## 6. Prossimi passi

1. Calibrazione del coseno: seconda generazione del full, per leggere i coseni in assoluto.
2. Confondente lunghezza per Professional/Hobbyist nello studio principale (§3.6).
3. Feature demografiche del visitatore (età, sesso): stessa pipeline, con un profilo nel
   prompt in aggiunta o in alternativa alla categoria Falk, per misurare se lo shift
   demografico è indipendente o collineare con quello di categoria.

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
