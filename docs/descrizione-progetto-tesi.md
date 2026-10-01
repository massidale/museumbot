# Descrizione del progetto di tesi

**Candidato:** [Nome Cognome], matricola [000000]
**Corso di laurea:** [Corso di laurea magistrale in …]
**Relatore:** Prof.ssa [Nome Cognome]
**Correlatore:** [eventuale]
**Anno accademico:** 2026/2027

## Titolo provvisorio

**Personalizzare le descrizioni delle opere d'arte con i modelli linguistici: generazione e
valutazione secondo le categorie di visitatori di Falk**

## Contesto

Le audioguide e le didascalie museali si rivolgono di norma a un visitatore medio, ma le
persone visitano un museo con motivazioni diverse. Il modello di Falk distingue cinque
identità di visita: Explorer, Facilitator, Experience Seeker, Professional/Hobbyist e
Recharger. I modelli linguistici di grandi dimensioni (LLM) permettono di generare, a costo
contenuto, descrizioni della stessa opera adattate a ciascuna identità. Studi recenti
(Dibitonto et al., *Multimedia Systems*, 2026) hanno proposto questo tipo di generazione,
ma restano aperte due questioni: quanto l'adattamento modifichi davvero i testi e se i
destinatari lo preferiscano a una descrizione generica.

## Domande di ricerca

1. **RQ1.** Quanto un LLM è in grado di personalizzare la descrizione di un dipinto secondo
   le categorie di Falk, e quanto le descrizioni ottenute differiscono fra loro?
2. **RQ2.** Quanto un LLM usato come giudice, istruito a impersonare un visitatore di una
   data categoria, preferisce la descrizione personalizzata per quella categoria a una
   descrizione generica?
3. **RQ3.** Quanto i visitatori museali reali preferiscono la descrizione personalizzata
   per la propria categoria a una descrizione generica?

## Metodologia

**Corpus e generazione.** 100 dipinti di grandi musei internazionali, con descrizioni di
partenza tratte da Wikipedia. Per ogni dipinto un LLM genera sei testi: uno per ciascuna
categoria di Falk e uno generico, senza indicazioni sul visitatore. Tutti gli altri
parametri del prompt e della generazione restano identici.

**RQ1: analisi del testo.** I testi vengono rappresentati con modelli di embedding. Dopo
aver rimosso la componente dovuta all'opera, si misurano la direzione e la stabilità dello
spostamento prodotto da ciascuna categoria e la separabilità delle categorie (classificatore
lineare, test di permutazione). Un'ablazione del prompt identifica quale parte della
descrizione della categoria produce lo spostamento. Questa parte del lavoro è in fase
avanzata.

**RQ2: valutazione con LLM come giudice.** A un LLM che impersona un visitatore di una data
categoria vengono presentate coppie di descrizioni della stessa opera, e il modello indica
quale preferisce. Le coppie mettono a confronto il testo per la categoria del visitatore
con il testo generico e con testi pensati per altre categorie. Per limitare le distorsioni
note di questo metodo (posizione, lunghezza, preferenza per il proprio stile), si usano più
modelli giudici, entrambi gli ordini di presentazione e descrizioni del visitatore
formulate in modo indipendente dal prompt di generazione.

**RQ3: studio con visitatori.** I partecipanti indicano la propria motivazione di visita,
e da questa si ricava la categoria di Falk. Poi valutano coppie di descrizioni con lo
stesso protocollo di RQ2. I dati si analizzano con modelli a effetti misti che tengono conto
del partecipante e dell'opera. Lo studio sarà sottoposto all'approvazione del comitato
etico e rispetterà la normativa sulla protezione dei dati personali.

**Confronto fra RQ2 e RQ3.** Il confronto fra i giudizi del modello e quelli dei visitatori,
sulle stesse coppie di testi, misura quanto un LLM possa sostituire le persone nella
valutazione di contenuti museali personalizzati.

## Risultati attesi

- Una misura quantitativa dell'effetto della personalizzazione sul testo generato e
  dell'importanza delle diverse parti del prompt.
- Una stima della preferenza per i testi personalizzati da parte di giudici automatici e
  di visitatori reali.
- Una valutazione dell'affidabilità degli LLM come sostituti dei visitatori nella
  valutazione di contenuti museali.

## Firme

Luogo e data: ______________________

Il candidato ______________________

La relatrice ______________________
