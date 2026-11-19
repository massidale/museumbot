# RQ3: visitatori reali all'Herzog Anton Ulrich-Museum

*Ultimo aggiornamento: 19 novembre 2026. Stato di RQ3: disegno deciso, stimoli testuali
pronti in inglese e tedesco (senza correzione degli errori), 160 audio e immagini delle 5
opere (il `flat` è l'audioguida del museo accorciata), app (iPad, telefono di Massimo, telefoni dei visitatori da QR) con risposte salvate
anche online, pubblicata il 19 novembre 2026 e provata nel browser. Prova sull'iPad e sul
telefono con dati mobili, ascolto degli audio, comitato etico e pilota sono da fare. Nessun
dato di visitatori raccolto.*

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
- **Domande per opera.** Quanto è piaciuta ciascuna guida, da 1 (per niente) a 7 (moltissimo),
  e cosa è piaciuto o no delle due (aperta, facoltativa). Nessuna scelta forzata: la
  preferenza è la differenza fra i due voti (personalizzata meno `flat`), con pareggi possibili.
- **Opere.** Le 5 del §3 hanno gli audio; una sessione ne mostra fino a 4, in ordine casuale
  (`MAX_ARTWORKS` in `app.js`). Il visitatore può fermarsi dopo ognuna, quindi da 1 a 4
  prove per sessione. Ogni sessione è indipendente (nessun bilanciamento fra visitatori).
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
- **Informativa.** Prima dell'introduzione c'è una schermata con le informazioni sui dati
  (cosa si salva, server nell'Unione europea, partecipazione volontaria, nessuna identità) e
  un tasto di accordo; l'istante dell'accordo va nel record come `consent_at`.

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
nella guida ufficiale del museo. Gli audio sono quindi per 5 opere; una sessione ne
mostra fino a 4, in ordine casuale.

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
15 minuti (4 opere), senza contare le domande.

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
Versione dell'app 0.4.4 (`config.js`).

- **Forma.** Pagina web statica (HTML, CSS e JavaScript senza build). Sull'iPad si installa
  nella Home e un service worker scarica pagina, manifesto, immagini e audio, poi funziona
  offline. Scelta rispetto a un'app nativa (Apple ID, scadenza a 7 giorni) e a LimeSurvey o
  SoSci Survey (serve rete, smistamento macchinoso). Il design grafico (`style.css`) è lo
  stesso su iPad e telefono. La scelta della lingua è la parete verde della galleria con un
  cartellino per lingua (nome della lingua e "Continue in English" / "Auf Deutsch weiter").
- **Modi.**
  - *Operatore* (iPad e telefono di Massimo): schermata in inglese, protetta da PIN (`config.js`; si apre
    tenendo premuto 2 s l'angolo in alto a destra) con nome del dispositivo, controllo del
    materiale in cache, stato dell'invio online ed export. "New session", a fondo pagina, resta disattivata
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
  aperta → "un'altra opera?" → saluti. In fondo a ogni schermata del visitatore c'è l'id della
  sessione (`ID abc123`), per annotare errori ed esclusioni (§8).
  Testi dell'interfaccia in inglese e tedesco, con una versione italiana di riferimento che
  l'app non mostra (`rq3_app/texts.json`, copia di `docs/plans/2026-11-16-rq3-testi-app.json`).
  Lo studio è presentato come confronto fra due audioguide scritte con l'intelligenza
  artificiale: il visitatore non vede i nomi delle categorie né sa quale guida è
  personalizzata.
- **Informativa.** Schermata `info_*` di `texts.json`: cosa si salva (risposte e tempo di
  ascolto, nessun nome né indirizzo IP; Cloudflare, che ospita il sito, registra gli indirizzi IP
  nei suoi log tecnici, che non conserviamo né usiamo), dati su server Cloudflare nell'Unione
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
  `trials` (al massimo 4); `completed` è vero o falso, `started_at` e `updated_at` sono
  stringhe, `device` ha da 1 a 40 lettere, cifre, `-` o `_`. Con l'informativa `consent_at`; nel modo QR `device: qr` ed
  `earlier_sessions`, il numero di sessioni già fatte da quel browser (0 la prima volta). Per
  prova `ratings` (voti 1–7 di Guida 1 e Guida 2, nell'ordine di ascolto come `plays` e
  `heard_sec`), `why` (al massimo 2000 caratteri) e `answered_at`; quale guida è la
  personalizzata si legge da `first`.
  Dopo la classificazione `categories` (lista delle 1 o 2 categorie) e `category_fit` (voto
  1–7 per ciascuna). Nessun dato che identifica il visitatore. L'ora dell'app è strettamente
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
  schermata dell'operatore mostra quante sessioni sono state inviate.
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
  `RQ3_EXPORT_TOKEN`); senza token 401.
- **Dove stanno i dati.** Ogni sessione è in due posti: sul dispositivo, in IndexedDB legata
  all'indirizzo da cui si è installata l'app (si perde se si cancellano i dati di Safari o si
  toglie l'app dalla Home prima dell'export; l'app chiede a iOS lo storage persistente), e nel
  database D1. I dati della sessione partono dal dispositivo a ogni salvataggio, quindi la
  copia online è completa appena c'è rete.
- **Export (alternativa).** Se manca la rete, dalla schermata dell'operatore un file
  `rq3_<device>_<AAAAMMGG-HHMM>.json` con la lista delle sessioni (rientrata, leggibile).
  "Download" lo salva come un download normale (Mac: Download; iPad: File > Download). I record restano sul
  dispositivo finché l'operatore non conferma l'export, e si cancellano solo quelli esportati.
- **Raccolta.** `rq3_visitors/download.py` scarica le sessioni online da
  `https://museumbot.pages.dev/api/sessions` in `data/rq3/raw/online_<AAAAMMGG-HHMM>.json`
  (token da `RQ3_EXPORT_TOKEN`). `rq3_visitors/collect.py` legge gli export (online e dell'app;
  lista JSON o una riga per sessione), unisce le sessioni in `data/rq3/responses.jsonl`, tiene
  per ogni `session_id` la versione più avanti (completa, poi più risposte, poi più recente) e
  controlla lo schema. `data/rq3/` resta fuori da git.
- **Test.** 67 test Node (`node --test rq3_app/tests/*.test.js rq3_pages/tests/*.test.js`:
  logica, player, testi, validazione, invio, QR, regole del server) e 25 test Python
  (manifesto, raccolta, download, sintesi vocale), tutti passati il 19 novembre 2026.
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
- **Stato.** Versione 0.4.4 pubblicata il 19 novembre 2026 con il manifesto `haum-2` (5 opere,
  160 audio, `flat` del museo); controllati online la versione del manifesto e i 10 `flat`,
  uguali ai file locali.
  Provata in locale con `wrangler pages dev` (sessioni inviate; una sessione salvata a server
  spento inviata dal timer di 30 s al ritorno; modo QR, ricaricamento, `earlier_sessions` 0 poi
  1) e in produzione con una sessione QR completa a 375×667, arrivata in D1, scaricata da
  `download.py` e letta da `collect.py` (1 sessione, 1 completa, 0 non valide). Non ancora
  provati: modo operatore in produzione, iPad, telefono con dati mobili. In produzione un
  file mancante dà 404, un corpo non valido 400, e l'informativa compare nel modo QR. I voti
  1–7 sono provati in locale con una sessione QR a una categoria e una a due: nel record
  `categories`, `category_fit`, i `ratings` della prova e il confronto con `flat`. Il database di
  produzione è vuoto.

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

Da fare (`rq3_visitors/analyze.py` non esiste). Per ciascuna delle due stime: differenza fra
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
  registro a parte (`data/rq3/esclusioni.csv`, da creare), senza toccare l'export.

Nella tesi si danno i risultati con le esclusioni e, come controllo, senza.

**Gemello LLM** (fase 4 del piano): per ogni visitatore, il giudice di RQ2 con la persona
delle categorie del visitatore giudica esattamente le stesse coppie, sui testi (con il testo del
museo accorciato, `text_short` in `data/rq3/museum_texts.jsonl`, come `flat`). Accordo item per
item fra la scelta del giudice e il segno della differenza dei voti del visitatore (κ di
Cohen, pareggi a parte) e tassi di vittoria per categoria, umani contro LLM.

## 9. Da fare, in ordine

1. Rilettura del tedesco da un madrelingua.
2. Sintesi vocale: ascolto dei 160 audio di HeyGen (§5).
3. Prova sull'iPad: installazione nella Home, circa 100 MB di audio in cache, riproduzione
   offline, cursore e −10 s col dito, "Download" in File > Download, invio online dalla
   modalità operatore, anche con la rete tolta e poi tornata.
4. Prova sul telefono di Massimo con i dati mobili (modo `?m=qr`, audio, invio).
5. Museo: Wi-Fi allo stand, cartello con il QR code.
6. Comitato etico: informativa e uso di Cloudflare (D1 in giurisdizione UE, Pages Functions)
   per i dati, eventuale conferma della maggiore età, piano di gestione dei dati.
7. Cancellare dal database remoto le sessioni delle prove (`DELETE FROM sessions`, §6) prima
   della raccolta.
8. Accordi con il museo tramite il lab: giorni e posto per la raccolta, e i loro testi
   (eyeVisit, audioguida, opuscolo) come fonti più lunghe o come termine di confronto.
9. Pilota con 10–15 persone (durata, comprensione, categorie scelte), poi numerosità (§7).
10. Soglie delle esclusioni (§8) fissate e scritte prima della raccolta.
11. Raccolta, `analyze.py`, gemello LLM.

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
| app | `rq3_app/` (audio, immagini e `manifest.json` fuori da git; `sync.js`, `qr.js`, `validate.js`); online su `https://museumbot.pages.dev` |
| server delle risposte | `rq3_pages/` (`wrangler.toml`, `schema.sql`, `functions/api/session.js` e `sessions.js`, `lib/server.js`, `tests/`) |
| manifesto, stimoli finti, raccolta | `src/museumbot/rq3_visitors/` (`build_manifest.py`, `demo_stimuli.py`, `collect.py`, `download.py`) |
| test | `rq3_app/tests/`, `rq3_pages/tests/`, `tests/test_rq3_manifest.py`, `tests/test_rq3_collect.py`, `tests/test_rq3_download.py`, `tests/test_tts.py` |

```bash
node --test rq3_app/tests/*.test.js rq3_pages/tests/*.test.js
.venv/bin/python -m pytest -q tests/test_rq3_manifest.py tests/test_rq3_collect.py tests/test_rq3_download.py tests/test_tts.py
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
