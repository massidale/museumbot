# RQ3: visitatori reali all'Herzog Anton Ulrich-Museum

*Ultimo aggiornamento: 24 novembre 2026. Stato di RQ3: disegno deciso, stimoli testuali
pronti in inglese e tedesco (senza correzione degli errori), 160 audio e immagini delle 5
opere (il `flat` è l'audioguida del museo accorciata), app (iPad, telefono di Massimo, telefoni dei visitatori da QR) con risposte salvate
anche online e domande a voce dopo ogni opera e alla fine (versione 0.6.0, pubblicata il 21
novembre 2026). Prima giornata al museo, con la versione 0.5.0
il 20 novembre 2026: 14 sessioni nel database, prove comprese, e 9 registrazioni finali
nell'app più 3 fatte a parte, tutte trascritte. Seconda giornata il 21 novembre, con la 0.6.0:
18 sessioni; 12 registrazioni dell'iPad inutilizzabili (microfono delle cuffie Bluetooth),
in parte sostituite da registrazioni a parte. Terza giornata il 22 novembre: 15 sessioni.
Dataset pulito delle tre giornate: 36 sessioni e 78 opere (§6). Analisi esplorativa delle tre
giornate fatta (§8, `docs/analisi/rq3-esplorativa.md`); gemello LLM su tutte le 78 prove, confrontato
con i visitatori (§8); comitato etico e analisi prevista sono da fare.*

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
  il nome della categoria e in ordine casuale, e ne sceglie 1 o 2. Poi dice quanto lo
  descrive ciascuna di quelle scelte, da 1 (per niente) a 7 (del tutto); nessuna principale.
- **Coppia per opera.** Due audioguide in ordine casuale, chiamate solo "Guida 1" e "Guida 2":
  - 1 categoria `k`: pura `k` contro `flat`;
  - 2 categorie `k` e `j`: mista a α = 0.5 di `k` e `j` contro `flat`.

  Nell'app `flat` è il testo "Read along" del museo per quell'opera, accorciato alla lunghezza
  dei nostri testi togliendo solo frasi intere (§3.1), non il `flat` generato da Gemma.
- **Domande per opera.** Quanto è piaciuta ciascuna guida, da 1 (per niente) a 7 (moltissimo).
  Poi, a voce in una registrazione facoltativa per quell'opera: "Le due guide di questo dipinto
  ti sono sembrate diverse? In che modo? Quale pensi che sia la più adatta a te?". Così ogni
  coppia ha il suo commento, e "la prima" o "la seconda" si riferiscono a guide note. Nessuna
  scelta forzata: la preferenza è la differenza fra i due voti (personalizzata meno `flat`),
  con pareggi possibili. Un tasto "Salta questo dipinto" passa oltre senza voti.
- **Opere.** Le 5 del §3 hanno gli audio; una sessione le mostra tutte e 5 (`MAX_ARTWORKS` in
  `app.js`), in ordine casuale o, sull'iPad, nell'ordine fisso del giro scelto dall'operatore
  (per seguire il percorso nelle sale). Il visitatore può fermarsi dopo ognuna o saltarne una,
  quindi da 1 a 5 prove per sessione. Ogni sessione è indipendente (nessun bilanciamento fra visitatori).
  L'ordine delle due guide resta casuale anche con il giro fisso.
- **Lingua.** Inglese o tedesco, scelta all'inizio.
- **Ascolto libero.** Play, pausa, −10 s e cursore; l'app non impone l'ordine né l'ascolto
  completo, ma registra per ogni guida `plays`, `reached_end` e `heard_sec` (secondi
  ascoltati senza i salti). Chi ha ascoltato abbastanza si decide nell'analisi, con una
  soglia da fissare prima di guardare le risposte.
- **Operatore presente.** Massimo resta allo stand, regola il volume prima di dare l'iPad e
  interviene se un audio non parte.
- **Dispositivi.** L'iPad con l'operatore; il telefono di Massimo (con le sue cuffie); i
  telefoni dei visitatori che hanno cuffie, da un QR code (`?m=qr`), solo con Massimo
  presente (§6).
- **Domande finali a voce.** Alla fine del giro (dopo l'ultima opera o quando il visitatore
  dice basta) tre domande a schermo, con istruzioni brevi, a cui si risponde a voce in
  un'unica registrazione, facoltativa:
  1. È stato facile scegliere le descrizioni che ti rispecchiano? Mancava qualcosa?
  2. Le audioguide erano facili da capire?
  3. Hai altri commenti, richieste o suggerimenti?

  Le risposte, come quelle per opera, servono come dati qualitativi (comprensione, categorie
  mancanti, differenza percepita fra le guide); vanno trascritte.
- **Informativa.** Prima dell'introduzione c'è una schermata con le informazioni sui dati
  (cosa si salva, registrazione facoltativa della voce, server nell'Unione europea,
  partecipazione volontaria, nessun nome) e un tasto di accordo; l'istante dell'accordo va nel
  record come `consent_at`.

Ne escono **due stime separate**: personalizzato contro generico (chi sceglie 1 categoria) e
misto contro generico (chi ne sceglie 2), ciascuna con la sua numerosità. Il riferimento è
sempre `flat`, cioè l'audioguida del museo: il confronto è fra un testo generato per la
categoria e un testo scritto dal museo per tutti, quindi cambiano insieme personalizzazione,
autore e fonte.

Tolti nel disegno del 17 novembre, e quindi non disponibili nell'analisi: prova del volume, motivi a scelta multipla, familiarità con l'opera, domande finali (età, frequenza di
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
  direttore) e iPad eyeVisit con schede su 70 opere. Per alcune opere la pagina del museo
  ha un testo "Read along" in inglese, senza autore (probabilmente di un curatore): ricostruiti
  dagli screenshot per 5 delle opere di RQ3 (Judith, Family Group, Cleopatra, Vermeer, Aurora;
  246–320 parole, 219 per Aurora senza la poesia di Goethe; i nostri testi delle stesse opere hanno mediana 233–240 parole, cioè da 12 a 85 parole in meno) in `docs/analisi/haum-testi-museo.md` e
  `data/rq3/museum_texts.jsonl`. Self-Portrait as David non ha un testo nella guida del museo.
  Ne esiste anche una versione accorciata alla lunghezza dei nostri testi (234–239 parole, campo
  `text_short`), ottenuta togliendo solo frasi intere (1–9 per opera; per Aurora restano la
  prima e l'ultima strofa della poesia); le frasi tolte sono elencate in `haum-testi-museo.md`.
  La versione accorciata è tradotta in tedesco con Claude Opus 5.5 (`text_short_de`, 197–216
  parole; per Aurora i versi originali di Goethe, "Mit Elsheimers Morgen") e ha un audio per
  lingua (§5).
  Opuscolo, schede eyeVisit e cataloghi non sono online.

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

**Le prime 4 opere scelte** sono le 4 con la voce tedesca più lunga, tutte fra gli highlights del
museo: *Familienbild* (Rembrandt), *Das Mädchen mit dem Weinglas* (Vermeer), *Selbstbildnis
als David* (Giorgione), *Judith mit dem Haupt des Holofernes* (Rubens). Nessun altro dipinto
del museo ha una voce di lunghezza paragonabile in una delle quattro lingue.

**Opere aggiunte e tolte** (18 novembre 2026): si aggiungono *The Dying Cleopatra* (Rosso
Fiorentino, voce italiana) e *Aurora* (Elsheimer, voce spagnola), che hanno un testo "Read along"
del museo; *Self-Portrait as David* resta fra i testi generati ma non ha audio, perché non è
nella guida ufficiale del museo. Gli audio sono quindi per 5 opere; una sessione le
mostra tutte e 5.

### 3.3 Le opere dell'app del museo

Le 21 opere della lista *Explore > Stops* dell'app del museo (screenshot del 18 novembre
2026), cercate su Wikidata e misurate in tutte le Wikipedia in cui hanno una voce dedicata
(interrogazione del 18 novembre 2026). Dati in `data/rq3/haum_stops_wiki.json`. Lunghezza
della voce intera in parole (in caratteri per il giapponese), sezioni di bibliografia
comprese.

**8 opere hanno una voce**, ordinate per la voce più lunga:

| opera | voci (parole) |
|---|---|
| Vermeer, *The Girl with the Wine Glass*, 1658 | **he 2588**, fr 602, de 520, pl 515, it 510, ko 464, tr 372, ru 366, nl 349, sv 319, hy 306, es 269, eu 212, ro 130, en 129, mk 118, ml 105, fa 79, arz 38, avk 10; ja 1357 car. |
| Elsheimer, *Morning Landscape (Aurora)*, c. 1606 | **es 805** |
| Rubens, *Judith with the Head of Holofernes*, c. 1616/18 | **de 679**, fr 166; ja 1331 car. |
| Rembrandt, *Family Portrait*, c. 1668 | **de 576**, pl 313, en 166, fr 84, arz 78, it 61 |
| Rosso Fiorentino, *The Dying Cleopatra*, c. 1530 | **it 369**, en 188, fr 145, arz 55 |
| Rembrandt, *Landscape with approaching Thunderstorm*, c. 1637/38 | **fr 173**; ja 1201 car. |
| Cranach il Vecchio, *Hercules and Omphale*, 1537 | **pl 173** |
| Gentileschi, *The Crowning with Thorns of Christ*, c. 1610/15 | **fr 163** |

**13 opere senza voce in nessuna lingua:** Lucas van Leyden, *Self-Portrait*; Wtewael,
*The Wedding of Peleus and Thetis* (1602); van Goyen, *Dune Landscape* (1631); Heda,
*Breakfast Still Life* (1642; l'unico Heda del museo su Wikidata, GG 802, è datato 1636);
Honthorst, *The Soldier and the Girl* (c. 1622); Boucher, *The Secret Message* (1767);
Largillierre, *Portrait of Count Konrad Detlev von Dehn* (1724; elemento GG 521 senza
etichetta; c'è solo la voce tedesca sul personaggio); Hans von Aachen, *The Three Graces*
(c. 1604); Ludger tom Ring il Giovane, *Self-Portrait* (1547); Holbein il Giovane,
*Portrait of Cyriakus Kale* (1533); Palma il Vecchio, *Adam and Eve*. Teniers il Giovane,
*The Alchemist* (c. 1644) e Hackert, *Italian Landscape* (1793) non sono neppure su Wikidata.

La voce più piena è quella ebraica del Vermeer (2588 parole, con sezioni su descrizione,
soggetto, stile, interpretazione, simbolismo, provenienza e critica), l'unica di lunghezza
molto maggiore delle voci tedesche usate per le 4 opere scelte. Fra le opere della lista
non scelte, solo l'*Aurora* di Elsheimer ha una voce più lunga di 600 parole (spagnola).

**Limite.** Anche queste voci sono corte per audioguide con contenuto diverso per categoria
(§4). Fonti più lunghe sono i testi del museo (eyeVisit, audioguida, opuscolo, cataloghi
della pinacoteca; i testi "Read along" del §3.1 sono di lunghezza simile alle voci), da
chiedere tramite il lab, e per singole opere le banche dati
specializzate (RKD per i quadri olandesi e fiamminghi, Rembrandt Database, Essential
Vermeer). Non ancora richiesti né usati.

## 4. Testi

Dettagli, tabelle dei fatti per condizione e testi completi: `docs/report-pagina-intera.md`
§6 e `docs/analisi/haum-testi.md`; fonti a fronte in `docs/analisi/haum-fonti.md`.

- **Fonti.** Voci di Wikipedia senza le sezioni di servizio (`fetch_full.py --lang <lingua>
  --titles`): tedesche per le prime 4 opere, italiana per Cleopatra, spagnola per Aurora (le
  voci più lunghe di ciascuna opera, §3.2). Tradotte in inglese con Claude Opus 5.5, alla
  lettera, perché prompt, filtro e pulizia del progetto sono in inglese.

| opera | voce | revid | parole (originale) | parole EN |
|---|---|---|---|---|
| Portrait of a Family | de: Familienbildnis (Rembrandt) | 270096678 | 432 | 483 |
| The Girl with the Wine Glass | de: Das Mädchen mit dem Weinglas | 250710229 | 455 | 498 |
| Self-Portrait as David | de: Selbstbildnis als David | 270830096 | 576 | 655 |
| Judith with the Head of Holofernes | de: Judith mit dem Haupt des Holofernes (Rubens) | 262538765 | 575 | 606 |
| The Dying Cleopatra | it: Morte di Cleopatra (Rosso Fiorentino) | 149207749 | 330 | 372 |
| Aurora | es: La Aurora (Elsheimer) | 159263723 | 699 | 672 |

- **Generazione.** Gemma 4 31B su Kaggle (`kaggle/haum`, 16 novembre 2026; `kaggle/haum2` per
  Cleopatra e Aurora, 18 novembre 2026), prompt della
  prova a pagina intera (voce intera e indicazione di contenuto per categoria), miscela di
  logit per i misti. Per opera: `flat`, 5 puri e 10 misti a α = 0.5, cioè i 16 testi che
  servono al disegno. In tutto **96 testi** (36 puri con flat, 60 misti), mediana **236
  parole** (204–253); in `data/wiki/haum_generations.jsonl` e `data/wiki/haum2_generations.jsonl`.
  I 6 `flat` generati restano come testi ma non sono nell'app, dove il `flat` è il testo del
  museo (§2).
- **Tedesco.** Tutti i 96 testi tradotti con Claude Opus 5.5 dall'inglese (`text_de`), mediana
  **234.5 parole** (200–255): stessa audioguida nelle due lingue, stessi fatti e stesso ordine,
  testi confrontabili con quelli misurati in RQ1 e RQ2.
- **Cosa si vede.** Le categorie scelgono fatti diversi come sulle voci lunghe, ma con voci
  corte il nucleo comune pesa di più (nel Vermeer vetrata e ritratto sono in tutti i 16
  testi). Recharger resta quasi senza fatti.
- **Errori.** Il controllo a mano sulle voci trova errori in **22 testi** (imprecisioni,
  esagerazioni, fatti esterni; elenchi in `report-pagina-intera.md` §6.3 e §6.5). La traduzione li
  riporta così come sono. Il tedesco va riletto da un madrelingua.

### 4.1 Errori per condizione

| condizione | testi | con errori |
|---|---|---|
| `flat` | 6 | 0 |
| puri delle categorie | 30 | 7 |
| misti a α = 0.5 | 60 | 15 |

Puri e misti sbagliano con frequenza simile (23% e 25%); il `flat` mai. Guardando i singoli
errori delle prime 4 opere:

- **ereditati da un esperto**: già nel puro di una delle due categorie, ripetuti nel misto
  (Beckett che cerca apposta il quadro, dall'Experience Seeker puro; "canvas" per la tavola di
  Rubens, dall'Explorer puro);
- **solo nei misti, ma del tipo che fanno anche i puri**: fatti sbagliati o aggiunti ("his
  wives" nel Rembrandt, "royal commission" nel Rubens, la versione di Budapest "mirroring this
  one", la collezione "since the late 17th century" nel Vermeer, Salzdahlum al presente);
  possono venire dalla miscela come essere errori casuali;
- **tipico della miscela**: "Judith's beauty and beauty", una ripetizione nel punto in cui i
  due esperti vogliono parole diverse. Nei puri non c'è niente di simile.

In Cleopatra e Aurora nessun errore di un misto viene da uno dei due esperti e non ci sono
ripetizioni; gli errori sono dello stesso tipo nei puri e nei misti (anacronismi, fatti presi
da un'altra parte della voce, esagerazioni).

### 4.2 Nessuna correzione

I testi si usano come li ha generati Gemma, senza correggere gli errori del §4.1, in inglese e
in tedesco (decisione di Massimo, 18 novembre 2026, `docs/decisioni.md`). Gli stimoli di RQ3
sono quindi l'uscita del generatore così com'è. Conseguenze da tenere presenti nell'analisi:
alcuni visitatori sentono fatti sbagliati o esagerati, e gli errori sono nei puri e nei misti
ma mai nel `flat`. La tabella del §4.1 resta come risultato: il generatore sbaglia, e un uso
reale richiederebbe una revisione dei testi.

## 5. Audio

`generation/tts.py`: sintesi via OpenRouter (`/api/v1/audio/speech`), modello scelto con
`--model` (default ElevenLabs v4), una voce sola per tutti i testi e le due lingue, un mp3 per testo e lingua in
`rq3_app/audio/<lingua>/<opera>/<condizione>.mp3`, con `--exclude` per saltare un'opera.

**Audio prodotti** (18 novembre 2026): HeyGen Voice Turbo nella versione gratuita
(`heygen/heygen-voice-1-turbo:free`), voce femminile `nadine`, per Portrait of a Family, The
Girl with the Wine Glass, Judith, The Dying Cleopatra e Aurora: 16 testi per opera in inglese e
tedesco. Self-Portrait as David è escluso perché non è fra le opere della guida
ufficiale del museo. I testi non sono corretti (§4.2).

**Audio dei testi del museo** (19 novembre 2026): la versione accorciata dei 5 testi "Read
along" (§3.1), stesso modello e stessa voce, in inglese e tedesco, 10 file in
`data/rq3/museum_audio/<lingua>/<opera>/museum.mp3` (input in `data/rq3/museum_guides.jsonl`),
copiati come `flat.mp3` in `rq3_app/audio/`. Gli audio del `flat` generato sono in
`data/rq3/flat_gemma_audio/`, fuori dall'app.

Nell'app quindi **160 file**: per opera e lingua i 15 puri e misti di Gemma e il `flat` del
museo. Durata dei 150 di Gemma: inglese mediana 85 s (75–98), tedesco 95 s (78–109), tutti fra
15.2 e 17.4 caratteri al secondo, quindi nessun audio troncato; del museo: inglese 76–87 s,
tedesco 79–86 s. Ascolto da fare.
Una prova (due guide) dura così circa 3–3,5 minuti, una sessione da circa 5 (1 opera) a circa
18 minuti (5 opere), senza contare le domande.

### 5.1 Scelta del modello

Requisiti: buon tedesco, stessa voce nelle due lingue, stabilità su 1–2 minuti, nomi propri
(Holofernes, Temperantia, Salzdahlum, Giorgione). La voce è la stessa in tutte le condizioni,
quindi non falsa il confronto, ma una voce mediocre può abbassare tutti i giudizi e
schiacciare le differenze: si sceglie sulla qualità, non sul prezzo.

Costo dei 128 file al listino OpenRouter del 18 novembre 2026:

| modello | $ / carattere | costo |
|---|---|---|
| `elevenlabs/eleven-v4` (default di `tts.py`) | 0.00008 | ~15 $ |
| `elevenlabs/eleven-v4-turbo` | 0.00004 | ~7.5 $ |
| `minimax/speech-2.8-turbo` | 0.00006 | ~11 $ |
| `fish-audio/s2.1-pro` | 0.000015 | ~2.8 $ |
| `heygen/heygen-voice-1-turbo` | 0.000004 | ~0.75 $ |
| `heygen/heygen-voice-1-turbo:free` | 0 | 0 $, gratuito fino al 30 novembre 2026 |

Recensioni trovate in rete (novembre 2026), per lo più blog secondari: v4 è primo nella
classifica di Artificial Analysis per le voci standard, con pronuncia migliore di v3; il dato
del 75% di preferenza nei confronti alla cieca è di ElevenLabs stessa. v4 Turbo è pensato per
la bassa latenza, ha le stesse lingue e costa la metà; nella stessa classifica è di poco sopra
v4 sul tedesco (scarto piccolo). Nessun test d'ascolto dedicato al tedesco.

**Prova di HeyGen Voice Turbo** (18 novembre 2026, versione gratuita). HeyGen restituisce
solo PCM a 44.1 kHz, che `tts.py` converte in mp3 con ffmpeg; 10 voci, nessuna indicazione
sulle lingue. Sintetizzato l'Explorer di Portrait of a Family in inglese e tedesco con le voci
`gareth` e `nadine`: 79–81 s in inglese, 94 s in tedesco. File in
`data/rq3/tts_prova/heygen-turbo-<voce>/` (fuori da git). Ascolto da fare.

**Scelta** (18 novembre 2026, Massimo): HeyGen Voice Turbo gratuito, voce `nadine`.
Le altre righe della tabella restano alternative; `tts.py` cambia
modello con `--model`, e le voci di ciascun modello sono in `supported_voices` dell'elenco
modelli di OpenRouter.

**Immagini.** Da Wikimedia Commons, dalla proprietà P18 di Wikidata di ogni dipinto,
ridimensionate a 1600 px al massimo, in `rq3_app/img/<id>.jpg` (fuori da git come gli audio).
Il manifesto `haum-2` (19 novembre 2026, `flat` del museo) ha 5 opere e 160 audio, costruito con `build_manifest.py`. Gli stimoli
finti si ricreano con `demo_stimuli.py` (non sono più in `rq3_app/`).

## 6. App: iPad, telefono di Massimo e telefoni dei visitatori

Disegno e piano: `docs/plans/2026-11-16-rq3-app-ipad-design.md`,
`docs/plans/2026-11-16-rq3-app-ipad.md`; telefoni e risposte online:
`docs/plans/2026-11-18-rq3-qr-online-design.md`, `docs/plans/2026-11-18-rq3-qr-online.md`.
Versione dell'app 0.6.0 (`config.js`).

- **Forma.** Pagina web statica (HTML, CSS e JavaScript senza build). Sull'iPad si installa
  nella Home e un service worker scarica pagina, manifesto, immagini e audio, poi funziona
  offline. Scelta rispetto a un'app nativa (Apple ID, scadenza a 7 giorni) e a LimeSurvey o
  SoSci Survey (serve rete, smistamento macchinoso). Il design grafico (`style.css`) è lo
  stesso su iPad e telefono. La scelta della lingua è la parete verde della galleria con un
  cartellino per lingua (nome della lingua e "Continue in English" / "Auf Deutsch weiter").
- **Modi.**
  - *Operatore* (iPad e telefono di Massimo): schermata in inglese, protetta da PIN (`config.js`; si apre
    tenendo premuto 2 s l'angolo in alto a destra) con nome del dispositivo, ordine del giro,
    controllo del materiale in cache, stato dell'invio online ed export. L'ordine del giro è
    casuale per ogni sessione oppure fisso: l'operatore spunta "Use a fixed order" e sposta le
    opere con le frecce; vale per tutte le nuove sessioni di quell'iPad (impostazione
    `tour_order` in IndexedDB, `tourOrder` in `logic.js`). Nel modo QR l'ordine è sempre casuale. "New session", a fondo pagina, resta disattivata
    finché il nome non è impostato e il materiale non è tutto in cache. Lo stesso angolo, sulle
    schermate del visitatore, interrompe la sessione (`aborted_by_operator`).
  - *QR* (telefoni dei visitatori, `https://museumbot.pages.dev/?m=qr`): si parte subito dalla
    scelta della lingua, senza angolo dell'operatore e senza service worker; il modo resta
    dopo un ricaricamento. Il telefono scarica solo le due guide di ogni opera, in anticipo
    (`prefetch` in `player.js`): il `flat` della prima opera appena scelta la lingua, la sua
    personalizzata dopo i voti delle categorie, e le due guide dell'opera dopo mentre si
    risponde alla precedente. Si usa solo con Massimo presente. Il service worker, che all'install
    scarica tutti i 160 audio (circa 200 MB), c'è solo sui dispositivi che hanno aperto
    l'indirizzo dell'operatore: aperto lì, controlla anche il modo QR.
- **Percorso.** (Operatore avvia la sessione →) lingua → informativa → presentazione dello
  studio → classificazione (1 o 2 descrizioni, poi un voto 1–7 per ciascuna) → istruzioni (cuffie) → per
  ogni opera immagine, titolo, Guida 1 e Guida 2 ciascuna col suo voto 1–7 subito sotto, domanda
  a voce sulla coppia; "Avanti" o "Salta questo dipinto" → "un'altra opera?" → domande finali a
  voce → saluti. In fondo a ogni schermata del visitatore c'è l'id della
  sessione (`ID abc123`), per annotare errori ed esclusioni (§8).
  Testi dell'interfaccia in inglese e tedesco, con una versione italiana di riferimento che
  l'app non mostra (`rq3_app/texts.json`, copia di `docs/plans/2026-11-16-rq3-testi-app.json`).
  Lo studio è presentato come confronto fra due audioguide scritte con l'intelligenza
  artificiale: il visitatore non vede i nomi delle categorie né sa quale guida è
  personalizzata.
- **Informativa.** Schermata `info_*` di `texts.json`: cosa si salva (risposte e tempo di
  ascolto; non chiediamo il nome e non salviamo l'indirizzo IP; Cloudflare, che ospita il sito,
  registra gli indirizzi IP nei suoi log tecnici, che non conserviamo né usiamo), la
  registrazione facoltativa della voce dopo ogni opera e alla fine (ascoltata e trascritta solo per la tesi, non
  data ad altri; si chiede di non dire il proprio nome), dati su server Cloudflare nell'Unione
  europea e usati solo per la tesi, partecipazione volontaria, domande allo stand. "Accetto, continua" salva
  `consent_at` nel record. Il testo va approvato dal comitato etico (§9).
- **Manifesto degli stimoli.** `rq3_visitors/build_manifest.py` costruisce
  `rq3_app/manifest.json` da `data/rq3/artworks_haum.json` (id, titolo inglese e tedesco,
  artista, immagine) e dagli audio; si ferma se un'opera non ha i 16 audio in entrambe le
  lingue o se due file danno la stessa condizione, e registra la durata di ogni file.
- **Logica** (`logic.js`, funzioni pure): coppia dalla classificazione, piano della sessione
  (ordine delle opere, delle guide, delle descrizioni) con un generatore casuale il cui seed
  va nel record, così ogni sessione si ricostruisce.
- **Record.** Uno per sessione, scritto dopo ogni passo: una sessione interrotta resta con
  `completed: false` e il passo raggiunto. Campi obbligatori (`validate.js`, gli stessi di
  `collect.py`): `session_id` (6 caratteri minuscoli e cifre), `device`, `app_version`,
  `manifest_version`, `seed`, `lang`, `started_at`, `updated_at`, `completed`, `last_step`,
  `trials` (al massimo 5); `completed` è vero o falso, `started_at` e `updated_at` sono
  stringhe, `device` ha da 1 a 40 lettere, cifre, `-` o `_`. Con l'informativa `consent_at`; nel modo QR `device: qr` ed
  `earlier_sessions`, il numero di sessioni già fatte da quel browser (0 la prima volta).
  `tour_order` dice se le opere erano nell'ordine casuale (`random`) o in quello
  dell'operatore (`operator`); l'ordine effettivo si legge dagli `artwork_id` delle prove. Per
  prova `ratings` (voti 1–7 di Guida 1 e Guida 2, nell'ordine di ascolto come `plays` e
  `heard_sec`), `voice` (la registrazione sulla coppia, nella stessa forma di quella finale qui
  sotto) e `answered_at`; un'opera saltata ha `skipped: true`, `skipped_at`, voti nulli e niente
  `answered_at`. Quale guida è la personalizzata si legge da `first`. Le sessioni della 0.5.0
  hanno al posto di `voice` il commento scritto `why` (al massimo 2000 caratteri), che il
  server accetta ancora.
  Dopo la classificazione `categories` (lista delle 1 o 2 categorie) e `category_fit` (voto
  1–7 per ciascuna). Alla fine `voice`: `{sec, mime, recorded_at}` se c'è la registrazione,
  `{skipped: true}` se il visitatore conclude senza (con `no_mic: true` se il microfono non era
  disponibile), assente se non ci è arrivato. `completed` diventa vero alla fine del giro, prima
  delle domande a voce (`last_step: voice`, poi `end`). Nessun nome; la voce è l'unico dato
  personale, ed è facoltativa. L'ora dell'app è strettamente
  crescente, così due salvataggi ravvicinati hanno sempre `updated_at` diversi.
- **Invio online** (`sync.js`). Ogni sessione, da qualsiasi dispositivo, è inviata a
  `POST /api/session` dopo ogni salvataggio (senza farlo aspettare), ogni 30 s, quando torna la
  rete, quando la pagina torna visibile e quando si chiude (`keepalive`), un giro alla volta,
  le più vecchie prima. Ogni sessione ha una chiave di scrittura casuale di 32 caratteri
  esadecimali (intestazione `X-Write-Key`), salvata sul dispositivo. Un invio conta come fatto
  solo se il server risponde 200 con `{"ok": true}`: un reindirizzamento (per esempio la pagina
  di accesso di un Wi-Fi), una risposta d'altro tipo o 15 s senza risposta fermano il giro, che
  si ritenta più tardi. Gli invii rifiutati (400, 403) non si ritentano; una sessione oltre il
  suo limite (429) si salta in quel giro e si passa alla successiva. Lo stato (inviate e rifiutate) è in IndexedDB e la
  schermata dell'operatore mostra quante sessioni e quante registrazioni sono state inviate.
- **Registrazione** (`recorder.js`, `voiceRecorder` in `app.js`, lo stesso registratore per
  le opere e per la fine). Microfono con `getUserMedia` e `MediaRecorder` (AAC in `audio/mp4`
  su Safari, Opus altrove), "Inizia a registrare" → "Ferma" → "Registra di nuovo" se serve; si
  ferma da sola dopo 10 minuti. Sulla pagina dell'opera la registrazione si salva con
  "Avanti", che resta disattivato mentre si registra; "Salta questo dipinto" la scarta. Alla
  fine "Salva e concludi" o "Concludi senza registrare". Senza microfono (permesso negato o
  browser senza registrazione) i pulsanti di registrazione spariscono e la `voice` è
  `{skipped: true, no_mic: true}`. Ogni registrazione ha un id (`voiceId` in `validate.js`):
  `<session_id>` per le domande finali, `<session_id>_<n>` per l'opera n del giro. Va in
  IndexedDB (store `voice`, chiave = id) e, dopo le sessioni, `sync.js` la invia a
  `PUT /api/voice?id=<id>` con la chiave della sua sessione, senza `keepalive` e con 120 s
  di timeout. 404 o 409 (il record con `voice` non è ancora sul server) la fanno saltare a un
  giro successivo. Su un telefono la registrazione parte solo finché la pagina resta aperta;
  se il visitatore chiude subito, resta sul telefono e parte alla prossima apertura dell'indirizzo.
- **Server** (`rq3_pages/`). Cloudflare Pages Functions con database D1 `rq3` (giurisdizione
  UE, binding `DB`, tabella `sessions` in `schema.sql`: record intero più chiave, dispositivo,
  completezza, numero di prove con risposta (`answered_at`), `created_at`, `received_at`, `writes`). Le regole
  sono in `lib/server.js` (funzioni pure, testate). `POST /api/session` accetta un corpo di
  al massimo 65536 byte che passa `validateRecord`, controllato prima di interrogare il
  database; una sessione già presente si aggiorna solo
  con la sua chiave (altrimenti 403) e solo se la versione nuova non è più indietro (completa,
  poi più risposte, poi più recente, come `collect.py`); l'aggiornamento non cambia mai la
  chiave. Limiti: 300 sessioni nuove in un'ora
  in tutto il sito e 300 scritture per sessione (429). `GET /api/sessions` esporta tutti i
  record solo con il token `EXPORT_TOKEN` (secret del progetto, in `.env` come
  `RQ3_EXPORT_TOKEN`); senza token 401. Le registrazioni vanno nel bucket R2
  `museumbot-voice` (giurisdizione UE, binding `VOICE`, chiave = id della registrazione):
  `PUT /api/voice` accetta solo audio (`audio/mp4`, `webm`, `ogg`, `mpeg`) fino a 25 MB, per una
  sessione già nel database, con la sua chiave (altrimenti 403) e il cui record dichiara quella
  registrazione, nella `voice` finale o in quella della prova n (altrimenti 409); `GET /api/voice` elenca le registrazioni e `GET /api/voice?id=…` ne
  dà una, solo col token.
- **Dove stanno i dati.** Ogni sessione è in due posti: sul dispositivo, in IndexedDB legata
  all'indirizzo da cui si è installata l'app (si perde se si cancellano i dati di Safari o si
  toglie l'app dalla Home prima dell'export; l'app chiede a iOS lo storage persistente), e nel
  database D1. I dati della sessione partono dal dispositivo a ogni salvataggio, quindi la
  copia online è completa appena c'è rete. Lo stesso per la registrazione (IndexedDB e R2); con
  "Delete exported sessions" si cancellano anche le registrazioni di quelle sessioni già
  online, mentre quelle non ancora inviate restano sul dispositivo.
- **Export (alternativa).** Se manca la rete, dalla schermata dell'operatore un file
  `rq3_<device>_<AAAAMMGG-HHMM>.json` con la lista delle sessioni (rientrata, leggibile).
  "Download" lo salva come un download normale (Mac: Download; iPad: File > Download). I record restano sul
  dispositivo finché l'operatore non conferma l'export, e si cancellano solo quelli esportati.
- **Raccolta.** `rq3_visitors/download.py` scarica le sessioni online da
  `https://museumbot.pages.dev/api/sessions` in `data/rq3/raw/online_<AAAAMMGG-HHMM>.json`
  (token da `RQ3_EXPORT_TOKEN`) e le registrazioni che mancano in
  `data/rq3/raw/voice/<id>.m4a` (o `.webm`), con l'id della registrazione. `rq3_visitors/collect.py` legge gli export (online e dell'app;
  lista JSON o una riga per sessione), unisce le sessioni in `data/rq3/responses.jsonl`, tiene
  per ogni `session_id` la versione più avanti (completa, poi più risposte, poi più recente) e
  controlla lo schema. `rq3_visitors/transcribe.py` trascrive sul Mac le registrazioni di
  `data/rq3/raw/voice/` con Whisper (`openai/whisper-large-v3-turbo` con `transformers`, sulla
  GPU del Mac; nessun servizio esterno), nella lingua della sessione, in
  `data/rq3/transcripts.jsonl` (un file per riga; salta i file già fatti, quelli di sessioni
  che non sono nell'export, quelli in `--exclude` e quelli inutilizzabili). `<session_id>_<n>` è
  la registrazione dell'opera n; un altro suffisso (`cgofvw_opera1`, `dyu9bg_recupero`,
  `is449e_operatore`) è una registrazione fatta fuori dall'app. `voice/extra.json` dice per
  questi file e per quelli da spostare (dispositivi scambiati) a quale sessione vanno,
  eventualmente a quale opera, perché, e la lingua se non è quella della sessione (`it` per i
  riassunti dell'operatore). I file inutilizzabili sono in `data/rq3/exclusions.json`
  (`voice`, con il motivo) e non si trascrivono. Le trascrizioni con la traduzione italiana, l'ordine delle guide e i voti di
  ogni opera sono in `data/rq3/trascrizioni.md`; la traduzione è anche in `text_it` di
  `transcripts.jsonl`. `data/rq3/` resta fuori da git.
- **Dataset pulito.** `rq3_visitors/dataset.py` unisce tutti gli export online (come
  `collect.py`) e scrive in `data/rq3/clean/` tre CSV:
  - `trials.csv`, una riga per opera valida: sessione, giorno, dispositivo, lingua, categorie,
    posizione nel giro, opera, quale guida era la personalizzata (`personal_guide`, 1 o 2),
    voti e secondi di ascolto della personalizzata e del confronto, `diff` (personalizzata
    meno confronto), commento scritto (`comment`, solo nella 0.5.0) e, se c'è, il commento a
    voce su quell'opera (`oral_file`, `oral_text`, `oral_text_it`, `oral_note`: il file
    `<session_id>_<n>` o quello indicato in `extra.json`);
  - `sessions.csv`, una riga per sessione con almeno un'opera valida o con un commento finale
    utilizzabile: categorie e `category_fit`, opere valide e totali, com'è andata la
    registrazione finale nell'app (`voice`) e le trascrizioni finali (`final_file`,
    `final_text`, `final_text_it`, `final_note`: registrazione dell'app, registrazioni fatte a
    parte, riassunti dell'operatore, con il motivo dei file inutilizzabili);
  - `excluded.csv`: ogni sessione o opera tolta, con il motivo.

  `data/rq3/operator.json` tiene le annotazioni scritte dall'operatore: pareri riferiti da chi
  non voleva essere registrato (`notes`, in `oral_text_it` o `final_text_it`) e i voti di
  un'opera saltata per errore col tasto (`ratings`, colonna `note` di `trials.csv`). Queste e i riassunti
  a voce dell'operatore portano la nota "annotazione dell'operatore, tradotta dall'inglese: da
  non citare letteralmente".

  Fuori: le sessioni in `data/rq3/exclusions.json` (prove, scritte a mano con il motivo), le
  opere saltate col tasto, quelle senza i due voti, quelle con una guida a 0 s di ascolto (saltate, o una guida votata
  senza ascoltarla) e le sessioni rimaste senza opere. È solo la pulizia minima: la soglia di
  ascolto e le altre esclusioni del §8 sono ancora da fissare.
- **Test.** 83 test Node (`node --test rq3_app/tests/*.test.js rq3_pages/tests/*.test.js`:
  logica, player, registrazione, testi, validazione, invio, QR, regole del server) e 37 test Python
  (manifesto, raccolta, download, trascrizione, dataset, sintesi vocale), tutti passati il 21
  novembre 2026.
- **Pubblicazione.** Il service worker richiede HTTPS, quindi l'iPad non può usare il server
  del Mac: l'app è su Cloudflare Pages (progetto `museumbot`, indirizzo fisso
  `https://museumbot.pages.dev`), pubblicata da `rq3_pages/` con `wrangler pages deploy`, che
  carica `../rq3_app` (solo i file cambiati) e le funzioni. L'indirizzo è pubblico (app e
  audio), le risposte no. Si usa sempre l'indirizzo fisso: gli indirizzi dei singoli deploy
  sono siti diversi, ciascuno con la sua memoria. A ogni modifica di `rq3_app/` si
  ripubblica; se cambiano gli audio, con una nuova versione del manifesto, e sull'iPad
  "Download missing files". Un file che manca dà 404 (`rq3_app/404.html` toglie a Pages il ritorno
  alla pagina principale, che subito dopo un deploy faceva arrivare HTML al posto di un modulo
  e lasciava la pagina bianca); se dopo un deploy la pagina resta bianca, si ricarica.
- **Stato.** Versione 0.6.0 pubblicata il 21 novembre 2026 con il manifesto `haum-2` (5 opere,
  160 audio, `flat` del museo); controllati online la versione del manifesto e i 10 `flat`,
  uguali ai file locali, e dopo la 0.6.0 la versione, i testi nuovi e il 400 del server per un
  id di registrazione non ammesso.
  Provata in locale con `wrangler pages dev` (sessioni inviate; una sessione salvata a server
  spento inviata dal timer di 30 s al ritorno; modo QR, ricaricamento, `earlier_sessions` 0 poi
  1) e in produzione con una sessione QR completa a 375×667, arrivata in D1, scaricata da
  `download.py` e letta da `collect.py` (1 sessione, 1 completa, 0 non valide). Non ancora
  provati: modo operatore in produzione, iPad, telefono con dati mobili. In produzione un
  file mancante dà 404, un corpo non valido 400, e l'informativa compare nel modo QR. I voti
  1–7 sono provati in locale con una sessione QR a una categoria e una a due: nel record
  `categories`, `category_fit`, i `ratings` della prova e il confronto con `flat`. Le domande a
  voce sono provate in locale nel browser con un microfono simulato (inglese a tutta larghezza,
  tedesco a 375 px): registrazione salvata in IndexedDB, record con `voice`, `PUT /api/voice`
  200 sull'R2 locale; "Concludi senza registrare" scrive `{skipped: true}`. La 0.6.0 è provata
  in locale con `wrangler pages dev`, modo QR e microfono simulato: un'opera con registrazione
  (`voice` nella prova, file `<id>_1` in IndexedDB e `PUT /api/voice` 200), una saltata col tasto
  (`skipped: true`, voti nulli), una senza registrazione (`{skipped: true}`), poi la
  registrazione finale; "Avanti" disattivato mentre si registra; pagina dell'opera a 375 px
  (tedesco) e a 1180×820.
- **Prima giornata al museo** (20 novembre 2026, versione 0.5.0). Nel database 14 sessioni
  (`ipad-max` 9, `Ibrahim` 3, `iphone` 1, `qr` 1; 12 in tedesco, 2 in inglese), 13 complete,
  38 opere votate. Nel dataset pulito restano 11 sessioni e 31 opere. Tolte: la prova
  `kut5mg`; `qux99m` (modo QR, aperta e lasciata alla prima opera senza voti); 5 opere saltate
  (0 s di ascolto per tutte e due le guide); l'unica opera di `swo9jg`, che ha ascoltato la
  prima guida e se n'è dovuto andare prima della seconda (votata 1 senza ascoltarla, come dice il
  commento dell'operatore). Registrazioni finali: 9 in R2, tutte trascritte; quella di `6youru` è muta
  (l'iPad ha registrato dal microfono delle cuffie Bluetooth rimaste collegate, in mano
  all'operatore: prima delle domande a voce le cuffie vanno staccate o date al visitatore). Registrate a parte, fuori dall'app, e
  collegate alla sessione dall'ora: le risposte finali di `zxut5q` (registrazione saltata
  nell'app) e di `ujakv7` (microfono negato per sbaglio) e il parere di `cgofvw` dopo la prima
  opera. In R2
  restano anche `q1pb6j` e `vvprxv`, registrazioni di prova la cui sessione non è più nel
  database.
- **Seconda giornata al museo** (21 novembre 2026, versione 0.6.0). Nel database 18 sessioni
  (`ipad` 15, `iphone` 3; 15 in tedesco, 3 in inglese), tutte complete; 21 opere votate e 9
  saltate col tasto. Nel dataset pulito 14 sessioni e 22 opere: le altre 4 sessioni hanno solo
  opere saltate (`3ap10u`, `ydlgvr`, `lxtczf`, `unzq8y`). Aurora di `zk9nrd`, saltata per errore
  col tasto durante lo scambio dei dispositivi dopo l'ascolto, rientra con i voti 6 / 4
  riportati dall'operatore. Registrazioni nell'app: 21, di cui le 12 dell'iPad fino alle 14:25
  (ora locale) inutilizzabili, registrate dal microfono delle cuffie Bluetooth (sessioni
  `nv7x8b`, `mndv3z`, `dyu9bg`, `oodswo`). Al posto di quelle di `dyu9bg` e `oodswo` ci sono
  le risposte dei visitatori registrate di nuovo sull'iPhone dell'operatore; tre visitatori che
  non volevano essere registrati (`is449e`, `9p0r1b`, `m3p4yx`) hanno un riassunto in italiano
  dell'operatore. Madre e figlia (`tyetwa` e `zk9nrd`, solo Aurora) si sono scambiate i
  dispositivi alla fine: le due registrazioni finali sono attribuite alla persona che parla.
  Trascritte 14 registrazioni (9 dell'app, 5 a parte); commenti per opera utilizzabili: 4.
- **Terza giornata al museo** (22 novembre 2026, versione 0.6.0). Nel database 15 sessioni
  (`ipad` 14, `iphone` 1). Nel dataset pulito 11 sessioni e 25 opere, 24 registrazioni
  trascritte. Tolte: `lz29rf` e `hiooyf` (nessuna opera valida); `zaba33` e `hf9n4p`, stessa
  persona di `tfhvt3` tornata per le altre opere con categorie diverse (lo dicono le
  trascrizioni). Aurora di `u5oi3e`, saltata per sbaglio col tasto dopo l'ascolto, rientra con
  i voti 6 / 6 annotati dall'operatore; il suo commento a voce non è stato salvato. `kp8igf` e
  `r56v63` (16:09) non volevano essere registrati: hanno le annotazioni scritte dell'operatore.

## 7. Numerosità

Stima provvisoria, da rifare dopo il pilota. L'esito di ogni opera è la differenza fra i due
voti; la stima qui sotto è per il suo segno (personalizzata votata più in alto o più in
basso, pareggi esclusi), con test contro il 50%, α = 0.05 bilaterale, potenza 0.8. Giudizi
senza pareggio necessari per **ciascuna** delle due stime:

| la personalizzata votata più in alto nel… | giudizi |
|---|---|
| 70% | ~47 |
| 65% | ~85 |
| 60% | ~194 |
| 55% | ~780 |

I giudizi dello stesso visitatore sono correlati. Con ipotesi da verificare nel pilota (2
opere in media per visitatore, correzione ×1,2 per la correlazione) e visitatori divisi a
metà fra 1 e 2 categorie servono circa 60 visitatori per un effetto del 70%, 100 per il 65%,
230 per il 60%. Un effetto del 55% non è raggiungibile in museo. Per singola categoria di
Falk i dati non bastano: l'analisi per categoria resta descrittiva. I pareggi tolgono
giudizi; la differenza usata per intero invece del solo segno aumenta la potenza. La stima
per la differenza dei voti si fa dopo il pilota, con la quota di pareggi e la varianza
misurate.

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

**Analisi esplorativa** delle giornate del 20–22 novembre: `docs/analisi/rq3-esplorativa.md`,
con i numeri in `results/rq3_summary.json` (`rq3_visitors/summary.py`), i temi dei commenti
codificati a mano in `data/rq3/temi.json` e le voci Wikipedia e i testi del museo classificati
frase per frase in `data/wiki/haum_codifica.jsonl`. Nel complesso diff +0.24 [−0.17, +0.66];
puri −0.23 [−0.96, +0.33], misti +0.55 [+0.06, +1.08]; Aurora +1.09, Cleopatra −1.27. La
discussione nasce da questi dati e va verificata su dati nuovi.

**Analisi prevista:** da fare (`rq3_visitors/analyze.py` non esiste). Per ciascuna delle due stime: differenza fra
i voti 1–7 della guida personalizzata (o mista) e di `flat`, con effetti casuali per
partecipante e opera, posizione della guida personalizzata (1 o 2), differenza di durata e
voti delle categorie scelte (`category_fit`) come covariate; segno della differenza come
esito secondario; codifica tematica delle risposte aperte.

**Esclusioni.** I dati esportati non si correggono mai a mano. Il visitatore può cambiare
i voti finché non preme "Next"; dopo, un errore (un tocco sbagliato, una risposta
data senza ascoltare) si esclude con regole fissate prima di guardare i voti:

- ascolto insufficiente: meno di una soglia di `heard_sec` per una delle due guide;
- risposta troppo rapida: `answered_at` − `shown_at` sotto una soglia;
- sessione interrotta dall'operatore: valgono solo le opere già risposte;
- errori segnalati dal visitatore: id della sessione, numero dell'opera e motivo in un
  registro a parte, senza toccare l'export. Per ora `data/rq3/exclusions.json` registra solo
  sessioni intere (le prove), letto da `dataset.py`.

Nella tesi si danno i risultati con le esclusioni e, come controllo, senza.

**Gemello LLM** (`rq3_visitors/twin.py`, disegno in `docs/decisioni.md`, 23 novembre 2026):
GLM-5.3 su Ollama Cloud, con le impostazioni del giudice di RQ2 (`judge_one`), rifà ogni
prova del dataset pulito.

- Persona: le descrizioni dell'app scelte dal visitatore, nell'ordine di scelta, con il voto
  di aderenza 1–7; niente nome della categoria.
- Coppia: titolo e artista, testo di Gemma e testo del museo accorciato (`text_short`, o
  `text_short_de` per le sessioni in tedesco), chiamati come nell'app ("Audioguide 1" / "Guide 1").
  Tutto il prompt è nella lingua della sessione.
- Domande: la domanda a voce dell'app, poi "Wie hat Ihnen Audioguide 1 gefallen?" (e per la 2)
  da 1 a 7, pareggi ammessi; risposta in JSON (`comment`, `guide1`, `guide2`).
- Due ordini (quello del visitatore e l'inverso) e due modi (`persona` e `neutral`, senza
  descrizioni): 4 chiamate per prova, una per opera, senza memoria.
- Uscita in `data/rq3/twin.jsonl`, una riga per (sessione, opera, modo, ordine), con
  `rating_personal` e `rating_comparison` già riportati alla guida.

Giro su tutte le 78 prove: 312 risposte valide, spesa $0.61. Le chiamate
fallite per il limite mensile di Ollama Cloud (HTTP 429) restano nel file con `valid: false` e
sono state rifatte; l'analisi tiene solo le valide, una per (sessione, opera, modo, ordine).
Risposte nella lingua del prompt (un commento tedesco passa al "du").

**Risultati** (`rq3_visitors/twin_summary.py` → `results/rq3_twin.json`; voti del gemello = media
dei due ordini; 78 prove, le stesse dei visitatori):

| | voto personalizzata | voto museo | diff [IC 95%] | personalizzata / pari / museo |
|---|---|---|---|---|
| visitatori | 5.46 | 5.22 | +0.24 [−0.17, +0.66] | 36 / 13 / 29 |
| gemello con persona | 6.33 | 5.13 | +1.19 [+1.00, +1.41] | 70 / 6 / 2 |
| gemello senza persona | 5.73 | 5.76 | −0.03 [−0.28, +0.23] | 33 / 19 / 26 |

- La preferenza del gemello viene dalla persona: con persona meno senza, +1.22 [+0.88, +1.61].
- Puri e misti vanno al contrario dei visitatori: il gemello con persona dà +1.53 ai
  puri e +0.97 ai misti, i visitatori −0.23 e +0.55.
  Su Cleopatra il gemello dà +1.53, i visitatori −1.27.
- Segno prova per prova (personalizzata / pari / museo): il gemello con persona coincide con il
  visitatore in 32 prove su 78, κ di Cohen −0.05; nelle 29 prove in cui il visitatore preferisce
  il museo è d'accordo 0 volte. Senza persona coincide in 33, κ 0.10. Senza le prove pari per
  uno dei due: con persona 32 su 59, κ 0.00 (il gemello dice sempre "personalizzata"); senza
  persona 29 su 50, κ 0.15.
- Posizione: Guida 1 meno Guida 2 nelle singole chiamate −0.22 [−0.36, −0.08] con persona,
  −0.42 [−0.52, −0.34] senza; stesso segno nei due ordini in 60 prove su
  78 con persona, 31 senza; chiamate pari 16 e 37 su 156.
- Commenti: il gemello nota le stesse differenze dei visitatori (una guida più informativa,
  l'altra che fa cercare dettagli o invita alla calma) e sceglie quella che corrisponde alla
  descrizione ("passt besser zu mir, weil…").
- Temi dei commenti (`data/rq3/temi_gemello.json`: 579 affermazioni sui 156 commenti nell'ordine
  del visitatore, con i temi dei visitatori e tre nuovi). I visitatori si contano per persona
  (commenti, domande a voce e finali), il gemello per commento (uno per prova e modo, 78 per modo):
  - `adatta_a_me` (nuovo): con persona la guida personalizzata "passt zu mir" in 73 commenti su 78.
    I visitatori che dicono che una guida "passt besser" lo motivano con il testo ("weil er
    weniger stark wertet", qz12x0), non con la propria motivazione o con chi è con loro.
  - Domande e inviti a guardare: lodati nella personalizzata in 44 commenti con persona e
    35 senza; fra i visitatori ne parlano 3, con giudizi discordi.
  - Più informazioni: 8 visitatori le trovano nella personalizzata; il gemello loda le
    informazioni del museo (22 commenti con persona, 33 senza).
  - Racconto: 4 visitatori lo lodano nel museo; il gemello nella personalizzata (13 e 12).
  - Goethe (Aurora): 4 visitatori su 5 contro; il gemello a favore in 14 commenti in entrambi i modi.
  - `lezione` (nuovo): il testo del museo sembra una lezione in 14 commenti con persona e 8 senza.
  - Emozioni e calma: senza persona il gemello le critica nella personalizzata in 9 commenti
    ("schwülstig"), con persona mai.
  - Mancano nel gemello i temi `unire` (8 visitatori), voce e scelta delle categorie.

## 9. Da fare, in ordine

1. Rilettura del tedesco da un madrelingua.
2. Sintesi vocale: ascolto dei 160 audio di HeyGen (§5).
3. Prova sull'iPad: installazione nella Home, circa 100 MB di audio in cache, riproduzione
   offline, cursore e −10 s col dito, "Download" in File > Download, invio online dalla
   modalità operatore, anche con la rete tolta e poi tornata.
4. Prova sul telefono di Massimo con i dati mobili (modo `?m=qr`, audio, invio).
   Su iPad e telefono anche le domande a voce: permesso del microfono (sull'iPad conviene
   "Consenti" fisso nelle impostazioni del sito), qualità della registrazione allo stand, arrivo
   in R2 e download con `download.py`.
5. Museo: Wi-Fi allo stand, cartello con il QR code.
6. Comitato etico: informativa e uso di Cloudflare (D1 e R2 in giurisdizione UE, Pages
   Functions) per i dati, registrazione della voce (dato personale: conservazione e
   cancellazione dopo la trascrizione), eventuale conferma della maggiore età, piano di
   gestione dei dati.
7. Cancellare dal database remoto le sessioni delle prove (`DELETE FROM sessions`, §6) e
   dal bucket `museumbot-voice` le loro registrazioni prima della raccolta.
   Durante la raccolta, trascrivere le registrazioni nuove con `transcribe.py` e rifare il
   dataset pulito con `dataset.py`.
8. Accordi con il museo tramite il lab: giorni e posto per la raccolta, e i loro testi
   (eyeVisit, audioguida, opuscolo) come fonti più lunghe o come termine di confronto.
9. Pilota con 10–15 persone (durata, comprensione, categorie scelte), poi numerosità (§7).
10. Soglie delle esclusioni (§8) fissate e scritte prima della raccolta.
11. Raccolta, `analyze.py`; rilettura della codifica dei commenti del gemello (§8).

Proposta non implementata: un tasto dell'operatore che segna l'ultima risposta come errore
(nel record, senza chiudere la sessione).

## 10. File e comandi

| cosa | dove |
|---|---|
| disegno e piano dell'app | `docs/plans/2026-11-16-rq3-app-ipad-design.md`, `docs/plans/2026-11-16-rq3-app-ipad.md`; telefoni e risposte online: `docs/plans/2026-11-18-rq3-qr-online-design.md`, `docs/plans/2026-11-18-rq3-qr-online.md` |
| testi dell'app (en, de, it) | `docs/plans/2026-11-16-rq3-testi-app.json` → `rq3_app/texts.json` |
| opere | `data/rq3/artworks_haum.json`; indagine Wikidata in `data/rq3/haum_wikidata.json`; opere dell'app del museo in `data/rq3/haum_stops_wiki.json` |
| fonti | `data/wiki/haum_pages_de.jsonl` (tedesco), `data/wiki/haum_pages.jsonl` (traduzione inglese); `docs/analisi/haum-fonti.md` |
| testi del museo ("Read along") | `docs/analisi/haum-testi-museo.md`, `data/rq3/museum_texts.jsonl` (anche accorciati e in tedesco); screenshot in `data/rq3/museo_readalong/`; audio in `data/rq3/museum_audio/` (input `data/rq3/museum_guides.jsonl`) |
| testi generati | `kaggle/haum/` → `data/wiki/haum_generations.jsonl` (con `text_de`); `docs/analisi/haum-testi.md` |
| sintesi vocale | `src/museumbot/generation/tts.py` → `rq3_app/audio/` |
| app | `rq3_app/` (audio, immagini e `manifest.json` fuori da git; `sync.js`, `qr.js`, `validate.js`, `recorder.js`); online su `https://museumbot.pages.dev` |
| server delle risposte | `rq3_pages/` (`wrangler.toml`, `schema.sql`, `functions/api/session.js`, `sessions.js` e `voice.js`, `lib/server.js`, `tests/`) |
| manifesto, stimoli finti, raccolta | `src/museumbot/rq3_visitors/` (`build_manifest.py`, `demo_stimuli.py`, `collect.py`, `download.py`, `transcribe.py`, `dataset.py`) |
| analisi esplorativa | `src/museumbot/rq3_visitors/summary.py` → `results/rq3_summary.json`; temi in `data/rq3/temi.json`; codifica delle fonti in `data/wiki/haum_codifica.jsonl`; testo in `docs/analisi/rq3-esplorativa.md` |
| gemello LLM | `src/museumbot/rq3_visitors/twin.py` → `data/rq3/twin.jsonl`; confronto `twin_summary.py` → `results/rq3_twin.json`; temi dei commenti in `data/rq3/temi_gemello.json` |
| test | `rq3_app/tests/`, `rq3_pages/tests/`, `tests/test_rq3_manifest.py`, `tests/test_rq3_collect.py`, `tests/test_rq3_download.py`, `tests/test_rq3_transcribe.py`, `tests/test_rq3_dataset.py`, `tests/test_rq3_summary.py`, `tests/test_rq3_twin.py`, `tests/test_rq3_twin_summary.py`, `tests/test_tts.py` |

```bash
node --test rq3_app/tests/*.test.js rq3_pages/tests/*.test.js
.venv/bin/python -m pytest -q tests/test_rq3_manifest.py tests/test_rq3_collect.py tests/test_rq3_download.py tests/test_rq3_transcribe.py tests/test_rq3_dataset.py tests/test_rq3_summary.py tests/test_rq3_twin.py tests/test_rq3_twin_summary.py tests/test_tts.py
PYTHONPATH=src .venv/bin/python -m museumbot.rq3_visitors.dataset        # dataset pulito -> data/rq3/clean/
PYTHONPATH=src .venv/bin/python -m museumbot.rq3_visitors.summary        # analisi esplorativa -> results/rq3_summary.json
PYTHONPATH=src .venv/bin/python -m museumbot.rq3_visitors.twin --limit 5 # gemello LLM (senza --limit: tutte le prove) -> data/rq3/twin.jsonl
PYTHONPATH=src .venv/bin/python -m museumbot.rq3_visitors.twin_summary  # gemello contro visitatori -> results/rq3_twin.json
PYTHONPATH=src .venv/bin/python -m museumbot.generation.tts --in data/wiki/haum_generations.jsonl --limit 1
PYTHONPATH=src .venv/bin/python -m museumbot.rq3_visitors.build_manifest --artworks data/rq3/artworks_haum.json --app-root rq3_app --version <v>
PYTHONPATH=src .venv/bin/python -m museumbot.rq3_visitors.demo_stimuli --app-root rq3_app
python3 -m http.server 8765 -d rq3_app
cd rq3_pages && npx wrangler pages dev                                   # app e server in locale (D1 locale)
cd rq3_pages && npx wrangler pages deploy --project-name museumbot --branch restructure-src --commit-dirty=true   # carica ../rq3_app
cd rq3_pages && npx wrangler d1 execute rq3 --remote --file schema.sql    # schema (una volta)
cd rq3_pages && npx wrangler d1 execute rq3 --remote --command "SELECT session_id, device, completed, answered FROM sessions"
PYTHONPATH=src .venv/bin/python -m museumbot.rq3_visitors.download                # sessioni online -> data/rq3/raw/online_*.json (RQ3_EXPORT_TOKEN in .env)
PYTHONPATH=src .venv/bin/python -m museumbot.rq3_visitors.collect data/rq3/raw/*.json
```
