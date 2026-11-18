# RQ3: visitatori reali all'Herzog Anton Ulrich-Museum

*Ultimo aggiornamento: 17 novembre 2026. Stato di RQ3: disegno deciso, stimoli testuali
pronti in inglese e tedesco (correzione degli errori da decidere), app per iPad scritta,
pubblicata con le 4 opere finte e provata nel browser. Scelta del modello vocale, audio,
prova sull'iPad, comitato etico e pilota sono da fare. Nessun dato raccolto.*

**Domanda.** Un visitatore reale preferisce l'audioguida scritta per la sua categoria di
Falk? RQ2 lo chiede a un giudice LLM con persona; RQ3 lo chiede alle persone, sulle stesse
condizioni, e prepara il confronto fra le due risposte (gemello LLM, §8).

## 1. Contesto

La raccolta si fa all'Herzog Anton Ulrich-Museum (HAUM) di Braunschweig, il museo del
progetto del robot guida Carl (HCR-Lab di Ostfalia), su iPad e con dipinti del museo invece
delle 100 opere dei corpus di RQ1 e RQ2. Il robot non si usa. Deciso il 4 novembre 2026
(`docs/plans/piano-progetto-tesi.md`, fase 3).

## 2. Disegno

Deciso il 16 novembre 2026 (`docs/decisioni.md`), con le semplificazioni del 17 novembre.

- **Classificazione.** Il visitatore legge le 5 descrizioni di Falk, in prima persona, senza
  il nome della categoria e in ordine casuale, e ne sceglie 1 o 2. Con 2 indica la principale.
- **Coppia per opera.** Due audioguide in ordine casuale, chiamate solo "Guida 1" e "Guida 2":
  - 1 categoria `k`: pura `k` contro `flat`;
  - 2 categorie, principale `k` e secondaria `j`: mista a α = 0.5 di `k` e `j` contro pura `k`.
- **Domande per opera.** Quale guida preferisce (scelta forzata), quanto (1–7), perché
  (aperta, facoltativa).
- **Opere.** Le 4 del §3, in ordine casuale; il visitatore può fermarsi dopo ognuna, quindi
  da 1 a 4 prove per sessione. Ogni sessione è indipendente (nessun bilanciamento fra
  visitatori).
- **Lingua.** Inglese o tedesco, scelta all'inizio.
- **Ascolto libero.** Play, pausa, −10 s e cursore; l'app non impone l'ordine né l'ascolto
  completo, ma registra per ogni guida `plays`, `reached_end` e `heard_sec` (secondi
  ascoltati senza i salti). Chi ha ascoltato abbastanza si decide nell'analisi, con una
  soglia da fissare prima di guardare le risposte.
- **Operatore presente.** Massimo resta accanto al visitatore, regola il volume prima di
  dare l'iPad e interviene se un audio non parte.

Ne escono **due stime separate**: personalizzato contro generico (chi sceglie 1 categoria) e
misto contro puro della principale (chi ne sceglie 2), ciascuna con la sua numerosità.

Tolti nel disegno del 17 novembre, e quindi non disponibili nell'analisi: schermata di
consenso (l'introduzione dice che la partecipazione è volontaria e anonima), prova del
volume, motivi a scelta multipla, familiarità con l'opera, domande finali (età, frequenza di
visita, lingua madre). Il campione si descrive solo per lingua e categorie scelte.

## 3. Scelta delle opere

### 3.1 Cosa offre il museo

Dal sito del museo (3landesmuseen-braunschweig.de, consultato il 16 novembre 2026):

- La pagina **Highlights** elenca 65 oggetti. Solo 9 sono dipinti della pinacoteca (numero
  di inventario GG): Vermeer, *Das Mädchen mit dem Weinglas* (GG 316); Rembrandt,
  *Familienbild* (GG 238); Giorgione, *Selbstbildnis als David* (GG 454); Rubens, *Judith
  mit dem Haupt des Holofernes* (GG 87); Elsheimer, *Aurora* (GG 550); Cranach il Vecchio,
  *Herkules bei Omphale* (GG 25); Manfredi, *Die Verleugnung Petri* (GG 495); Largillière,
  *Jean-Baptiste Tavernier* (GG 520); Monogrammista di Braunschweig, *Das Gleichnis vom
  Großen Gastmahl* (GG 165). Gli altri "dipinti" della pagina (Canaletto, Van Dyck, Franz
  Marc, Herolt, Meister mit den Keulen, Dürer, Goya, Kollwitz) sono fogli del
  Kupferstichkabinett. La pagina della pinacoteca aggiunge come highlight l'*Autoritratto* di
  Lucas van Leyden (GG 160) e le *Quattro stagioni* di Joos de Momper.
- **Disposizione.** Il sito dice solo il piano: pinacoteca al 1° piano (circa 315–320
  dipinti, allestita nel 2016, ogni sala con un tema: autoritratti, inganno in pittura,
  teatro e pittura, artisti migranti nella guerra dei Trent'anni, famiglia), arti applicate
  e sculture al 2°. Nessuna pianta delle sale con le opere, né sul sito né altrove online.
  Il Burg Dankwarderode (opere medievali, Leone originale) è chiuso fino a nuovo avviso.
- **Materiale di mediazione del museo:** un opuscolo gratuito con i testi delle sale,
  un'audioguida in tedesco e inglese su 50 opere (più un giro di 7 highlights del
  direttore) e iPad eyeVisit con schede su 70 opere. Nessuno di questi testi è online.

### 3.2 Quanto sono documentate su Wikipedia

Interrogazione di Wikidata del 16 novembre 2026: dipinti (Q3305213) con collezione o luogo
= HAUM (Q678082). Dati in `data/rq3/haum_wikidata.json` (fuori da git, come tutto
`data/rq3/`).

- **414 dipinti** su Wikidata; **11** hanno una voce Wikipedia dedicata all'opera in almeno
  una lingua, **5** in inglese.
- Lunghezza della voce intera (parole, sezioni di bibliografia comprese):

| opera | en | de | it | fr |
|---|---|---|---|---|
| Giorgione, *Selbstbildnis als David* | 110 | **686** | 233 | 154 |
| Rubens, *Judith mit dem Haupt des Holofernes* | – | **679** | – | 166 |
| Rembrandt, *Familienbild* | 166 | **576** | 61 | 84 |
| Vermeer, *Das Mädchen mit dem Weinglas* | 129 | 520 | 510 | **602** |
| Rosso Fiorentino, *Cleopatra morente* | 188 | – | 369 | 145 |
| Batoni, *Ritratto del duca di Brunswick* | 211 | – | – | – |
| Rembrandt, *Paesaggio con temporale* | – | – | – | 173 |
| Gentileschi, *Cristo coronato di spine* | – | – | – | 163 |
| Cranach, Elsheimer, Lievens | voce solo in polacco, spagnolo o ebraico |

**Le 4 opere scelte** sono le 4 con la voce tedesca più lunga, tutte fra gli highlights del
museo: *Familienbild* (Rembrandt), *Das Mädchen mit dem Weinglas* (Vermeer), *Selbstbildnis
als David* (Giorgione), *Judith mit dem Haupt des Holofernes* (Rubens). Nessun altro dipinto
del museo ha una voce di lunghezza paragonabile in una delle quattro lingue.

**Limite.** Anche queste voci sono corte per audioguide con contenuto diverso per categoria
(§4). Fonti più lunghe sono i testi del museo (eyeVisit, audioguida, opuscolo, cataloghi
della pinacoteca), da chiedere tramite il lab, e per singole opere le banche dati
specializzate (RKD per i quadri olandesi e fiamminghi, Rembrandt Database, Essential
Vermeer). Non ancora richiesti né usati.

## 4. Testi

Dettagli, tabelle dei fatti per condizione e testi completi: `docs/report-pagina-intera.md`
§6 e `docs/analisi/haum-testi.md`; fonti a fronte in `docs/analisi/haum-fonti.md`.

- **Fonti.** Voci della Wikipedia tedesca senza le sezioni di servizio
  (`fetch_full.py --lang de --titles`), tradotte in inglese a mano, alla lettera, perché
  prompt, filtro e pulizia del progetto sono in inglese.

| opera | voce tedesca | revid | parole DE | parole EN |
|---|---|---|---|---|
| Portrait of a Family | Familienbildnis (Rembrandt) | 270096678 | 432 | 483 |
| The Girl with the Wine Glass | Das Mädchen mit dem Weinglas | 250710229 | 455 | 498 |
| Self-Portrait as David | Selbstbildnis als David | 270830096 | 576 | 655 |
| Judith with the Head of Holofernes | Judith mit dem Haupt des Holofernes (Rubens) | 262538765 | 575 | 606 |

- **Generazione.** Gemma 4 31B su Kaggle (`kaggle/haum`, 16 novembre 2026), prompt della
  prova a pagina intera (voce intera e indicazione di contenuto per categoria), miscela di
  logit per i misti. Per opera: `flat`, 5 puri e 10 misti a α = 0.5, cioè i 16 testi che
  servono al disegno. In tutto **64 testi** (24 puri con flat, 40 misti), mediana **237
  parole** (212–253).
- **Tedesco.** Tutti i 64 testi tradotti a mano dall'inglese (`text_de`), mediana **234
  parole** (212–249): stessa audioguida nelle due lingue, stessi fatti e stesso ordine,
  testi confrontabili con quelli misurati in RQ1 e RQ2.
- **Cosa si vede.** Le categorie scelgono fatti diversi come sulle voci lunghe, ma con voci
  corte il nucleo comune pesa di più (nel Vermeer vetrata e ritratto sono in tutti i 16
  testi). Recharger resta quasi senza fatti.
- **Errori.** Il controllo a mano sulle voci trova errori in **13 testi** (imprecisioni,
  esagerazioni, fatti esterni; elenco in `report-pagina-intera.md` §6.3). La traduzione li
  riporta così come sono. Il tedesco va riletto da un madrelingua.

### 4.1 Errori per condizione

| condizione | testi | con errori |
|---|---|---|
| `flat` | 4 | 0 |
| puri delle categorie | 20 | 3 |
| misti a α = 0.5 | 40 | 10 |

Con questi numeri la differenza fra puri e misti può essere caso. Guardando i singoli errori:

- **ereditati da un esperto**: già nel puro di una delle due categorie, ripetuti nel misto
  (Beckett che cerca apposta il quadro, dall'Experience Seeker puro; "canvas" per la tavola di
  Rubens, dall'Explorer puro);
- **solo nei misti, ma del tipo che fanno anche i puri**: fatti sbagliati o aggiunti ("his
  wives" nel Rembrandt, "royal commission" nel Rubens, la versione di Budapest "mirroring this
  one", la collezione "since the late 17th century" nel Vermeer, Salzdahlum al presente);
  possono venire dalla miscela come essere errori casuali;
- **tipico della miscela**: "Judith's beauty and beauty", una ripetizione nel punto in cui i
  due esperti vogliono parole diverse. Nei puri non c'è niente di simile.

### 4.2 Correzione (da decidere)

Proposta, non ancora decisa né applicata: correggere prima della sintesi vocale, perché RQ3
misura la preferenza e non la fedeltà, gli errori pesano più sui misti che sul `flat` (quindi
sbilanciano il confronto) e i visitatori del museo non devono sentire fatti falsi. Regole
proposte, da scrivere in `docs/decisioni.md` prima di toccare i testi:

- si correggono solo frasi che contraddicono la voce o aggiungono fatti che la voce non ha, e
  glitch evidenti di lingua ("beauty and beauty", "canvas");
- cambio minimo (una parola o una frase), stile, ordine e formule della categoria invariati;
- stessa regola su tutti i 64 testi, `flat` compreso, in inglese e tedesco;
- l'originale resta in `text_orig`; RQ1 e RQ2, se usano questi testi, usano l'originale;
- si riporta quanti testi e quante parole sono cambiati.

La tabella del §4.1 resta come risultato: il generatore sbaglia, e un uso reale richiede una
revisione dei testi.

## 5. Audio

`generation/tts.py`: ElevenLabs via OpenRouter (`/api/v1/audio/speech`), una voce sola
(`george`) per tutti i testi e le due lingue, un mp3 per testo e lingua in
`rq3_app/audio/<lingua>/<opera>/<condizione>.mp3`. 128 file, **186 843 caratteri** (88 121 in
inglese, 98 722 in tedesco). **Non ancora eseguito**: prima vanno corretti i testi (§4.2) e
scelto il modello (§5.1), poi una prova su un file (`--limit 1`). Un testo dura circa 1,5
minuti, una prova 3,5–4 minuti, una sessione da circa 6 (1 opera) a circa 18 minuti (4 opere):
stime dalla lunghezza dei testi, da verificare con gli audio veri.

### 5.1 Scelta del modello (da fare)

Requisiti: buon tedesco, stessa voce nelle due lingue, stabilità su 1–2 minuti, nomi propri
(Holofernes, Temperantia, Salzdahlum, Giorgione). La voce è la stessa in tutte le condizioni,
quindi non falsa il confronto, ma una voce mediocre può abbassare tutti i giudizi e
schiacciare le differenze: si sceglie sulla qualità, non sul prezzo.

Costo dei 128 file al listino OpenRouter del 17 novembre 2026:

| modello | $ / carattere | costo |
|---|---|---|
| `elevenlabs/eleven-v4` (default di `tts.py`) | 0.00008 | ~15 $ |
| `elevenlabs/eleven-v4-turbo` | 0.00004 | ~7.5 $ |
| `minimax/speech-2.8-turbo` | 0.00006 | ~11 $ |
| `fish-audio/s2.1-pro` | 0.000015 | ~2.8 $ |

Recensioni trovate in rete (novembre 2026), per lo più blog secondari: v4 è primo nella
classifica di Artificial Analysis per le voci standard, con pronuncia migliore di v3; il dato
del 75% di preferenza nei confronti alla cieca è di ElevenLabs stessa. v4 Turbo è pensato per
la bassa latenza, ha le stesse lingue e costa la metà; nella stessa classifica è di poco sopra
v4 sul tedesco (scarto piccolo). Nessun test d'ascolto dedicato al tedesco.

Proposta: prova alla cieca con lo stesso paragrafo in inglese e tedesco, stessa voce, sui
quattro modelli della tabella (meno di 0.10 $), ascoltata da Massimo e da un madrelingua
tedesco. Se v4 e v4 Turbo non si distinguono, si usa Turbo. `tts.py` cambia modello con
`MODEL`; vanno controllate le voci disponibili per ciascun modello.

Oggi in `rq3_app/audio/` ci sono solo gli audio di prova delle 4 opere finte
(`demo_stimuli.py`). Mancano anche le immagini dei 4 dipinti (`rq3_app/img/<id>.jpg`).

## 6. App per iPad

Disegno e piano: `docs/plans/2026-11-16-rq3-app-ipad-design.md`,
`docs/plans/2026-11-16-rq3-app-ipad.md`.

- **Forma.** Pagina web statica (HTML, CSS e JavaScript senza build) da installare nella Home
  dell'iPad; un service worker scarica pagina, manifesto, immagini e audio, poi funziona
  offline. Scelta rispetto a un'app nativa (Apple ID, scadenza a 7 giorni) e a LimeSurvey o
  SoSci Survey (serve rete, smistamento macchinoso).
- **Percorso.** Operatore avvia la sessione → lingua → presentazione dello studio →
  classificazione (1 o 2 descrizioni, poi la principale) → istruzioni (cuffie) → per ogni
  opera immagine, titolo, Guida 1 e Guida 2, domande → "un'altra opera?" → saluti. Testi
  dell'interfaccia in inglese e tedesco, con una versione italiana di riferimento che l'app
  non mostra (`rq3_app/texts.json`, copia di `docs/plans/2026-11-16-rq3-testi-app.json`).
  Lo studio è presentato come confronto fra due audioguide scritte con l'intelligenza
  artificiale: il visitatore non vede i nomi delle categorie né sa quale guida è
  personalizzata.
- **Manifesto degli stimoli.** `rq3_visitors/build_manifest.py` costruisce
  `rq3_app/manifest.json` da `data/rq3/artworks_haum.json` (id, titolo inglese e tedesco,
  artista, immagine) e dagli audio; si ferma se un'opera non ha i 16 audio in entrambe le
  lingue o se due file danno la stessa condizione, e registra la durata di ogni file.
- **Logica** (`logic.js`, funzioni pure): coppia dalla classificazione, piano della sessione
  (ordine delle opere, delle guide, delle descrizioni) con un generatore casuale il cui seed
  va nel record, così ogni sessione si ricostruisce.
- **Salvataggio.** Un record per sessione in IndexedDB, scritto dopo ogni passo: una
  sessione interrotta resta con `completed: false` e il passo raggiunto. Nessun dato che
  identifica il visitatore.
- **Operatore.** Schermata protetta da PIN (`config.js`; si apre tenendo premuto 2 s
  l'angolo in alto a destra) con nome del dispositivo, controllo del materiale in cache ed
  export. "Nuova sessione" resta disattivata finché il nome non è impostato e il materiale non
  è tutto in cache. Lo stesso angolo, sulle schermate del visitatore, interrompe la sessione
  (`aborted_by_operator`).
- **Dove stanno i dati.** Solo sull'iPad, in IndexedDB, legati all'indirizzo da cui si è
  installata l'app: nessun dato va in rete. L'app chiede a iOS lo storage persistente (lo
  stato è nella schermata dell'operatore). Si perdono se si cancellano i dati di Safari o si
  toglie l'app dalla Home prima dell'export.
- **Export.** Un file `rq3_<device>_<AAAAMMGG-HHMM>.json` con la lista delle sessioni
  (rientrata, leggibile). "Scarica" lo salva come un download normale (Mac: Download; iPad:
  File > Download); "Condividi" apre il menu di condivisione (AirDrop, mail, Salva su File).
  I record restano sull'iPad finché l'operatore non conferma l'export, e si cancellano solo
  quelli esportati. Al museo c'è solo l'iPad: si esporta a fine giornata e si copia il file
  fuori dall'iPad (iCloud Drive, mail, AirDrop sul telefono).
- **Raccolta.** A casa, `rq3_visitors/collect.py` legge gli export (lista JSON o una riga
  per sessione), unisce le sessioni in `data/rq3/responses.jsonl`, tiene per ogni
  `session_id` la versione più avanti (completa, poi più risposte, poi più recente) e
  controlla lo schema. `data/rq3/` resta fuori da git.
- **Test.** 23 test della logica, del player e dei testi (`node --test`) e 18 test Python
  (manifesto, raccolta, sintesi vocale), tutti passati il 17 novembre 2026.
- **Pubblicazione.** Il service worker richiede HTTPS, quindi l'iPad non può usare il
  server del Mac: l'app è su Cloudflare Pages (progetto `museumbot`, indirizzo fisso
  `https://museumbot.pages.dev`), pubblicata con `wrangler pages deploy`, che carica solo i
  file cambiati. L'indirizzo è pubblico (app e audio), le risposte no. Si usa sempre
  l'indirizzo fisso: gli indirizzi dei singoli deploy sono siti diversi, ciascuno con la sua
  memoria. A ogni modifica di `rq3_app/` si ripubblica; se cambiano gli audio, con una nuova
  versione del manifesto, e sull'iPad "Scarica i mancanti".
- **Stato.** Pubblicata il 17 novembre 2026 con le 4 opere finte e provata nel browser del
  Mac: materiale in cache (132 file), service worker attivo, sessioni complete, export
  scaricato in Download da Safari e riletto da `collect.py` senza errori. Non ancora provata
  sull'iPad.

## 7. Numerosità

Stima provvisoria, da rifare dopo il pilota. Ogni opera dà un giudizio binario
(personalizzata o no); il test è contro il 50%, α = 0.05 bilaterale, potenza 0.8.
Giudizi indipendenti necessari per **ciascuna** delle due stime:

| la personalizzata vince il… | giudizi |
|---|---|
| 70% | ~47 |
| 65% | ~85 |
| 60% | ~194 |
| 55% | ~780 |

I giudizi dello stesso visitatore sono correlati. Con ipotesi da verificare nel pilota (2
opere in media per visitatore, correzione ×1,2 per la correlazione) e visitatori divisi a
metà fra 1 e 2 categorie servono circa 60 visitatori per un effetto del 70%, 100 per il 65%,
230 per il 60%. Un effetto del 55% non è raggiungibile in museo. Per singola categoria di
Falk i dati non bastano: l'analisi per categoria resta descrittiva. La forza della
preferenza (scala da −7 a +7 invece della sola scelta) aumenta un po' la potenza.

**Quanti visitatori.** L'annuario statistico di Braunschweig (tabella 13_01,
<https://www.braunschweig.de/politik_verwaltung/statistik/jahrbuch/jahrbuch/13_01_export.pdf>) dà solo il
totale dei 3Landesmuseen (HAUM, Landesmuseum, Museo di storia naturale, sedi distaccate
comprese): 162 201 nel 2023, 133 392 nel 2024, 140 272 nel 2025, circa 450 al giorno su
circa 300 giorni di apertura. Il dato del solo HAUM va chiesto al museo. Il limite non è il
flusso ma l'iPad: una sessione dura 6–18 minuti (§5), quindi un operatore con un iPad fa
circa 10–20 sessioni al giorno; 200 visitatori sono 10–20 giorni con un iPad, 5–10 con due.
Il pilota misura le sessioni al giorno, le opere per visitatore, la quota di chi sceglie 1 o
2 categorie e la correlazione fra i giudizi della stessa persona.

## 8. Analisi prevista e gemello LLM

Da fare (`rq3_visitors/analyze.py` non esiste). Per ciascuna delle due stime: scelta della
guida personalizzata (o mista) con effetti casuali per partecipante e opera, posizione della
guida (1 o 2) e differenza di durata come covariate; forza della preferenza come esito
secondario; codifica tematica delle risposte al "perché".

**Esclusioni.** I dati esportati non si correggono mai a mano. Il visitatore può cambiare
scelta e forza finché non preme "Next"; dopo, un errore (un tocco sbagliato, una risposta
data senza ascoltare) si esclude con regole fissate prima di guardare le scelte:

- ascolto insufficiente: meno di una soglia di `heard_sec` per una delle due guide;
- risposta troppo rapida: `answered_at` − `shown_at` sotto una soglia;
- sessione interrotta dall'operatore: valgono solo le opere già risposte;
- errori segnalati dal visitatore: id della sessione, numero dell'opera e motivo in un
  registro a parte (`data/rq3/esclusioni.csv`, da creare), senza toccare l'export.

Nella tesi si danno i risultati con le esclusioni e, come controllo, senza.

**Gemello LLM** (fase 4 del piano): per ogni visitatore, il giudice di RQ2 con la sola
categoria del visitatore giudica esattamente le stesse coppie, sui testi. Accordo item per
item (κ di Cohen) e tassi di vittoria per categoria, umani contro LLM.

## 9. Da fare, in ordine

1. Decidere la correzione (§4.2), scriverla in `docs/decisioni.md`, correggere i testi in
   inglese e tedesco; rilettura del tedesco da un madrelingua.
2. Immagini dei 4 dipinti in `rq3_app/img/`.
3. Sintesi vocale: prova alla cieca dei modelli (§5.1), prova su un file, ascolto, poi i 128 file.
4. `build_manifest.py` sulle 4 opere vere.
5. Prova sull'iPad (l'app è già pubblicata con le opere finte): installazione nella Home,
   circa 100 MB di audio in cache, riproduzione offline, cursore e −10 s col dito, "Scarica"
   in File > Download. Poi la stessa prova con le 4 opere vere.
6. Comitato etico: informativa, eventuale consenso esplicito o conferma della maggiore età
   (fuori dall'app o rimettendo la schermata), piano di gestione dei dati.
7. Accordi con il museo tramite il lab: giorni e posto per la raccolta, e i loro testi
   (eyeVisit, audioguida, opuscolo) come fonti più lunghe o come termine di confronto.
8. Pilota con 10–15 persone (durata, comprensione, categorie scelte), poi numerosità (§7).
9. Soglie delle esclusioni (§8) fissate e scritte prima della raccolta.
10. Raccolta, `analyze.py`, gemello LLM.

**In corso: telefoni dei visitatori e risposte online.** Design in
`docs/plans/2026-11-18-rq3-qr-online-design.md`, da approvare; niente ancora nel codice. Allo
stand, oltre all'iPad, partecipano il telefono di Massimo (con le sue cuffie) e i telefoni dei
visitatori che hanno cuffie, da un QR code (`?m=qr`, solo con Massimo presente). Tutte le
sessioni vanno anche in un database D1 su Cloudflare (Pages Functions), che sostituisce
l'export a mano per i telefoni dei visitatori; nel record `device: qr`, `consent_at`
(informativa, che torna su tutti i dispositivi) e `earlier_sessions`.

Proposta non implementata: un tasto dell'operatore che segna l'ultima risposta come errore
(nel record, senza chiudere la sessione).

## 10. File e comandi

| cosa | dove |
|---|---|
| disegno e piano dell'app | `docs/plans/2026-11-16-rq3-app-ipad-design.md`, `docs/plans/2026-11-16-rq3-app-ipad.md`; telefoni e risposte online: `docs/plans/2026-11-18-rq3-qr-online-design.md` |
| testi dell'app (en, de, it) | `docs/plans/2026-11-16-rq3-testi-app.json` → `rq3_app/texts.json` |
| opere | `data/rq3/artworks_haum.json`; indagine Wikidata in `data/rq3/haum_wikidata.json` |
| fonti | `data/wiki/haum_pages_de.jsonl` (tedesco), `data/wiki/haum_pages.jsonl` (traduzione inglese); `docs/analisi/haum-fonti.md` |
| testi generati | `kaggle/haum/` → `data/wiki/haum_generations.jsonl` (con `text_de`); `docs/analisi/haum-testi.md` |
| sintesi vocale | `src/museumbot/generation/tts.py` → `rq3_app/audio/` |
| app | `rq3_app/` (audio e `manifest.json` fuori da git); online su `https://museumbot.pages.dev` |
| manifesto, stimoli finti, raccolta | `src/museumbot/rq3_visitors/` (`build_manifest.py`, `demo_stimuli.py`, `collect.py`) |
| test | `rq3_app/tests/`, `tests/test_rq3_manifest.py`, `tests/test_rq3_collect.py`, `tests/test_tts.py` |

```bash
node --test rq3_app/tests/*.test.js
.venv/bin/python -m pytest -q tests/test_rq3_manifest.py tests/test_rq3_collect.py tests/test_tts.py
PYTHONPATH=src .venv/bin/python -m museumbot.generation.tts --in data/wiki/haum_generations.jsonl --limit 1
PYTHONPATH=src .venv/bin/python -m museumbot.rq3_visitors.build_manifest --artworks data/rq3/artworks_haum.json --app-root rq3_app --version <v>
PYTHONPATH=src .venv/bin/python -m museumbot.rq3_visitors.demo_stimuli --app-root rq3_app
python3 -m http.server 8765 -d rq3_app
npx wrangler pages deploy rq3_app --project-name museumbot --branch restructure-src --commit-dirty=true
PYTHONPATH=src .venv/bin/python -m museumbot.rq3_visitors.collect data/rq3/raw/*.json
```
