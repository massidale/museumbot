# Prova a pagina intera: voce Wikipedia completa e indicazioni di contenuto per categoria

*Ultimo aggiornamento: 18 novembre 2026. Prova qualitativa su 3 opere del corpus (§1–5) e
sulle 6 opere del museo di Braunschweig per RQ3 (§6), fuori dai corpus di RQ1 e RQ2.
Generatore, miscela e filtro sono quelli della parte 2 (`docs/report-gemma.md`); qui si
descrive solo ciò che cambia. Testi completi: `docs/analisi/pagina-intera-testi.md` e
`docs/analisi/haum-testi.md`.*

**Domanda.** Nei corpus `main` e `local` la fonte è l'intro della voce Wikipedia, troncata a
~1 200 caratteri: tutte le categorie ricevono gli stessi pochi fatti e possono cambiare quasi
solo il registro. Se il modello riceve la voce intera e il prompt gli dice su quali aspetti
concentrarsi per ciascuna categoria, i testi cambiano anche nel *contenuto*, cioè nei fatti
scelti? E i testi misti a α = 0.5 restano leggibili e prendono qualcosa da entrambe le
categorie?

## 1. Opere e fonte

Tre opere del corpus (`data/artworks.jsonl`), scelte per avere tre secoli, tre musei e tipi
di voce diversi (soggetto sacro, allegoria, scandalo moderno), con voci abbastanza corte da
stare nel contesto disponibile su Kaggle (§3).

| opera | artista | museo | parole | token (Gemma) | revid |
|---|---|---|---|---|---|
| The Descent from the Cross | Rogier van der Weyden | Prado | 2 276 | 2 970 | 1364130677 |
| The Art of Painting | Johannes Vermeer | Kunsthistorisches Museum | 1 449 | 1 933 | 1381703681 |
| Olympia | Édouard Manet | Musée d'Orsay | 2 283 | 3 345 | 1383452137 |

`corpus/fetch_full.py` scarica il testo semplice della voce inglese (API MediaWiki,
`explaintext`) e toglie le sezioni finali di servizio (References, Notes, See also, External
links, Bibliography, …) e i titoli rimasti senza testo (gallerie). I titoli di sezione restano
nella forma `== Titolo ==`. Output: `data/wiki/full_pages.jsonl`, con `revid` della revisione
letta. Le voci (1 449–2 283 parole) sono da 7 a 11 volte l'intro dei corpus principali
(~200 parole).

## 2. Prompt

`prompts.wiki_messages`. Tre differenze rispetto a `prompts.local_messages`:

1. **Fonte**: la voce intera, introdotta come *Source (full encyclopedia article)*.
2. **Indicazione di contenuto** (`prompts.FOCUS`), una per categoria, aggiunta al blocco di
   categoria completo (definizione, bisogno, stile). Il system prompt dice che la voce
   contiene molto più di quanto stia in 250 parole e che va scelto il materiale adatto
   all'ascoltatore, seguendo l'indicazione.
3. **Posizione**: system prompt uguale per tutte le categorie; blocco di categoria e
   indicazione stanno in coda al messaggio utente, dopo la voce. Così i due esperti di un
   testo misto hanno in comune tutto il prefisso fino al blocco (2 178–3 590 token secondo
   l'opera, su prompt di 2 312–3 743) e vLLM tiene la voce una volta sola nella cache.

Il resto è invariato: 250 parole, solo i fatti della fonte, divieto di riferimenti espliciti
alla categoria, niente Markdown.

Indicazioni per categoria (testo esatto, dopo `Focus:`):

| categoria | indicazione |
|---|---|
| Explorer | pick the surprising and lesser-known material: details in the picture that are easy to miss, curious episodes in its making or later history, open questions and competing readings. Point the listener's eye to specific things to look for. |
| Facilitator | pick what a group of companions of different ages and backgrounds can see and share: who and what is depicted and the story behind the scene, a few clear facts about the artist and the time, details everyone can spot. Leave out technical analysis and specialist debates, and close with a question the group can discuss together. |
| Experience Seeker | pick why this work is famous: its place in art history, the reactions it caused, its influence on later artists and on popular culture, and the one or two facts no visitor should leave without. |
| Professional/Hobbyist | pick the specialist material: technique, materials and composition, dating and attribution, sources and models, provenance and condition, and the scholarly interpretations and debates the article reports, with related works by the artist or by others. |
| Recharger | pick the mood of the picture: light, colour, stillness or emotion, and the human or spiritual themes it opens. Use few dates, names and figures, and leave out provenance, prices and scholarly disputes. |

Le indicazioni traducono il bisogno di ciascuna categoria (paper, Appendice A.1) nelle sezioni
tipiche di una voce su un dipinto (descrizione, storia, provenienza, ricezione, influenza).

## 3. Generazione

`generation/vllm_wiki.py`, notebook `kaggle/wiki` (sessione del 9 novembre 2026).

- **Modello e decoding**: come la parte 2. Gemma 4 31B QAT a 4 bit, vLLM 0.31.0 su 2× T4,
  temperatura 0.7, ragionamento disattivato, un seed per testo (`rows.row_seed`).
- **Testi**: per ogni opera 5 puri (uno per categoria) e 10 misti a α = 0.5 (tutte le coppie
  di categorie), 45 in tutto. Puri con il backend di `vllm_gen`; misti con la miscela
  classica di `vllm_mix` (media geometrica a pesi 0.5/0.5 sulle prime 100 log-probabilità di
  ciascun esperto, nessun ingresso né debito della minoranza). Tutte le righe hanno
  `variant = "wiki"`; i misti hanno `pair` e `alpha` e la diagnostica per token.
- **Contesto**: `max_model_len` 5 120 invece di 4 096 (prompt fino a 3 743 token più 800 di
  testo). vLLM riporta 6 637 token di cache KV. Un testo misto alla volta (`GROUP = 1`):
  voce condivisa più due code da ~650 token stanno nella cache; due testi insieme arrivano
  al limite e vLLM ricalcolerebbe il prompt a ogni passo.
- **Codice sul notebook**: il notebook clona `restructure-src` al commit 65c8c7e e ci
  scrive sopra i file della prova (`prompts.py`, `vllm_gen.py`, `vllm_mix.py`,
  `vllm_wiki.py`, `full_pages.jsonl`), incorporati nello script, perché non sono ancora sul
  branch remoto. `vllm_gen.vllm_backend` e `vllm_mix.vllm_mixer` hanno un parametro
  `messages` (predefinito `local_messages`, quindi gli altri corpus non cambiano).
- **Tempi**: puri 250 s per tutti e 15; misti in media 111 s di generazione per testo
  (~300 token generati), 61 minuti per tutti e 30.

| | testi | parole (mediana, min–max) | rigenerati per il filtro | falliti |
|---|---|---|---|---|
| puri | 15 | 236 (222–253) | 0 | 0 |
| misti α = 0.5 | 30 | 240 (222–255) | 2 | 0 |

I due rigenerati sono misti con Professional/Hobbyist su The Art of Painting ("For those",
"as a professional"), come nella parte 2. Nei misti il token scelto è fra le prime 100 di
entrambi gli esperti nel 99.0% dei passi (minimo 97.6% per testo).

## 4. Risultati

### 4.1 Testi puri: il contenuto cambia con la categoria

Quali fatti della voce compaiono in ciascun testo puro (x), e in quanti dei 10 misti della
stessa opera (conteggio per parola chiave sul testo, controllato a mano).

**The Descent from the Cross**

| fatto | Expl | Fac | ExpS | P/H | Rech | misti /10 |
|---|---|---|---|---|---|---|
| svenimento della Vergine | x | x | x | x | x | 10 |
| corpo di Cristo come balestra (committenti) | x | x | x | x | . | 7 |
| data (1435) | . | x | x | x | . | 8 |
| scala impossibile, chiodo davanti alla cornice | x | . | . | x | . | 6 |
| lacrime dipinte | . | x | . | . | x | 5 |
| guerra civile, viaggio in Svizzera | x | . | x | x | . | 4 |
| "il miglior quadro del mondo" (1551) | . | . | x | . | . | 4 |
| studiosi nominati (Campbell, de Vos, Kempis) | . | . | . | x | . | 0 |

**The Art of Painting**

| fatto | Expl | Fac | ExpS | P/H | Rech | misti /10 |
|---|---|---|---|---|---|---|
| mappa delle Province | x | x | x | x | x | 10 |
| lampadario | x | x | x | x | x | 10 |
| mai venduto da Vermeer | x | x | x | x | . | 8 |
| miniera di sale | x | x | x | x | . | 8 |
| Hitler | x | x | x | x | . | 7 |
| alloro e tromba (Clio, Fama) | x | . | . | x | . | 5 |
| attribuzione a de Hooch, firma falsa | . | . | x | x | . | 4 |
| Visscher, 1636 | . | x | . | x | . | 2 |

**Olympia**

| fatto | Expl | Fac | ExpS | P/H | Rech | misti /10 |
|---|---|---|---|---|---|---|
| sguardo frontale | x | x | x | x | x | 10 |
| gatto nero | x | x | x | x | x | 10 |
| domestica (Laure) | x | x | x | x | . | 10 |
| Salon del 1865 | x | x | x | x | . | 9 |
| Venere di Urbino di Tiziano | x | . | x | x | . | 4 |
| Monet e la sottoscrizione | x | x | . | x | . | 4 |
| omaggi (Cézanne, Magritte, Basquiat) | . | . | x | x | . | 4 |
| sguardo "oppositivo" della domestica (O'Grady) | x | . | . | x | . | 4 |

Ogni opera ha un nucleo di 2–4 fatti che compare in quasi tutti i testi (lo svenimento, la
mappa e il lampadario, lo sguardo e il gatto): sono i dettagli visivi centrali, che ogni
categoria può usare. Intorno al nucleo le scelte seguono l'indicazione:

- **Explorer**: enigmi e dettagli da cercare (il chiodo che esce dalla cornice, la piega
  della mappa, l'aquila bicipite sul lampadario, il gatto al posto del cane di Tiziano) e
  domande aperte sulle letture contrastanti ("who is the man in red?"). Chiude con una
  domanda al visitatore in due opere su tre.
- **Facilitator**: chi è chi nel quadro (Giuseppe d'Arimatea, Nicodemo, Giovanni, la
  Maddalena), dettagli
  facili da trovare insieme ("Can you spot…"), niente dibattiti specialistici. Chiude con una
  domanda per il gruppo in tutte e tre le opere, come chiede l'indicazione.
- **Experience Seeker**: fama, scandalo, record e storia avventurosa (il "miglior quadro del
  mondo", Hitler e 1.82 milioni di Reichsmark, l'uproar del Salon), con la formula di
  chiusura "If there is one thing to remember" in due opere su tre (nella terza "If there is
  one image…"), che riprende quasi alla lettera l'indicazione.
- **Professional/Hobbyist**: l'unica categoria che nomina gli studiosi e le posizioni in
  disaccordo (Campbell contro de Vos sull'uomo in rosso, O'Grady sulla domestica, la datazione
  1665–1668 contro 1670–1675), la tecnica (repoussoir, luce senza mezzitoni, Cabanel come
  termine di confronto), la provenienza e il restauro del 1992.
- **Recharger**: quasi nessun fatto. Nessuna data in nessuna delle tre opere; il Vermeer
  non contiene neanche un'informazione storica (luce, silenzio, mappa e lampadario come
  oggetti da guardare). L'indicazione "use few dates, names and figures" viene presa alla
  lettera fino a svuotare il testo: è il testo più riconoscibile ma anche il meno
  informativo, e quello che potrebbe adattarsi meglio a un altro quadro.

Rispetto ai testi a intro troncata la differenza è netta. Su Olympia, Explorer e
Professional/Hobbyist del corpus `local` usano gli stessi fatti (Meurent, Laure, il titolo
suggerito da Astruc, Monet e il 1890) e cambiano solo registro; con la voce intera i due
testi hanno in comune solo il nucleo (sguardo, gatto, 1865, Laure) e per il resto scelgono
materiale diverso (Explorer: lo sguardo "oppositivo" della domestica come enigma, il
quasi-duello per la sottoscrizione; Professional/Hobbyist: Cabanel, i mezzitoni, la mano che
blocca invece di invitare, O'Grady, il bianco e nero come alternativa al chiaroscuro).

### 4.2 Testi misti a α = 0.5

Margine per token fra i due esperti (log p_X − log p_Y del token scelto, dalla diagnostica);
un passo è *conteso* se |margine| ≥ 3. Media sulle tre opere.

| coppia (X + Y) | passi contesi per testo | quota dei token | vinti da X |
|---|---|---|---|
| Explorer + Facilitator | 23.0 | 7.7% | 0.52 |
| Explorer + Professional/Hobbyist | 24.3 | 8.1% | 0.48 |
| Explorer + Experience Seeker | 30.3 | 10.6% | 0.43 |
| Facilitator + Experience Seeker | 34.7 | 11.6% | 0.56 |
| Experience Seeker + Professional/Hobbyist | 37.0 | 11.6% | 0.49 |
| Facilitator + Recharger | 37.3 | 13.3% | 0.43 |
| Facilitator + Professional/Hobbyist | 43.7 | 13.4% | 0.60 |
| Explorer + Recharger | 46.0 | 15.4% | 0.60 |
| Professional/Hobbyist + Recharger | 54.7 | 18.0% | 0.44 |
| Experience Seeker + Recharger | 55.0 | 19.4% | 0.47 |

Le coppie più contese sono quelle con Recharger (13–19% dei token; solo Facilitator +
Professional/Hobbyist arriva allo stesso livello), le meno contese quelle fra Explorer e
Facilitator o Professional/Hobbyist (8%): Recharger è la categoria più lontana dalle altre
anche qui. Nessuna coppia ha un vincitore netto: la quota dei passi contesi vinti da X sta fra 0.43
e 0.60.

**Lettura dei testi.** I misti sono testi normali, scorrevoli, senza salti di registro a metà
frase, e in quasi tutti si riconoscono tutte e due le categorie, di solito per divisione
dei compiti fra le parti del testo:

- **Coppie vicine** (Explorer, Facilitator, Experience Seeker fra loro): il testo prende il
  contenuto della categoria più "informativa" e la cornice dell'altra. Explorer + Facilitator
  su The Descent: i dettagli nascosti dell'Explorer (le balestre negli angoli, il chiodo) e
  la domanda al gruppo del Facilitator in chiusura. Facilitator + Experience Seeker sulla
  stessa opera: apertura da Experience Seeker ("You are standing before one of the most
  influential paintings"), chiusura con domanda sulle figure.
- **Con Professional/Hobbyist**: il lessico specialistico passa quasi sempre (tableau vivant,
  spasimo, repoussoir, Visscher, O'Grady), anche quando l'altra categoria guida il tono.
  Experience Seeker + Professional/Hobbyist è la combinazione più riuscita: fama e
  dibattito degli studiosi nello stesso testo. Facilitator + Professional/Hobbyist su The Art
  of Painting, alla lettura, pende verso Professional/Hobbyist anche se i margini sono in
  equilibrio, e chiude con una frase poco coerente ("the
  artist's face is hidden from us, but the scene is an illusion of his own space").
- **Con Recharger**: apertura e chiusura contemplative ("Take a moment…", "What does this
  stir within you?") intorno a un corpo centrale che porta i fatti dell'altra categoria. Il
  misto fa quello che il Recharger puro non fa: Explorer + Recharger su Olympia tiene la
  calma del Recharger ma recupera la domestica, la pennellata, la sottoscrizione di Monet.
  Professional/Hobbyist + Recharger su The Descent tiene spasimo, Campin e la musica
  polifonica in un registro raccolto. Professional/Hobbyist + Recharger su Olympia apre
  con un ibrido letterale, "Take a moment to analyze the composition": *analyze* è al di fuori
  delle prime 100 del Recharger e lo porta Professional/Hobbyist.

### 4.3 Errori

Controllo a mano delle affermazioni specifiche sulla voce (date, nomi, cifre, episodi).
La grande maggioranza dei fatti è corretta e tratta dalla voce, anche nei dettagli
(1.82 milioni di Reichsmark, il vagone chiuso a chiave per Vienna, il restauro del 1992, il
quasi-duello fra Monet e Antonin Proust). Gli errori trovati:

| testo | errore | tipo |
|---|---|---|
| Olympia, Experience Seeker (puro) | "To the Victorian public" per il pubblico parigino del 1865 | etichetta sbagliata |
| Olympia, Facilitator + Experience Seeker | omaggi "from Picasso to Jean-Michel Basquiat": Picasso non è nella voce | fatto esterno alla fonte |
| Olympia, Explorer + Recharger | "condemned as vulgar and immortal" (per *immoral*) | parola sbagliata |
| Olympia, Explorer + Recharger | "nearly destroyed by the administration": la voce dice che fu l'amministrazione a proteggerlo | senso rovesciato |
| The Descent, Explorer + Experience Seeker | "the most important painting of the entire 15th century": la voce dice "most influential Netherlandish painting of Christ's crucifixion" | esagerazione |
| The Art of Painting, Experience Seeker + Recharger | "the extraordinary same-time presence of the two figures" | composto innaturale |
| The Descent, Professional/Hobbyist + Recharger | "the same-shaped curves of the bodies" | composto innaturale |

Nei misti alcuni errori vengono da token poco probabili per *entrambi* gli esperti: *same* in
"same-time" ha log-probabilità −7.05 e −8.46 nei due esperti, in "same-shaped" −5.83 e −5.06;
*immortal* −3.96 e −3.76. Sono token di coda campionati a T = 0.7, che la media geometrica non
filtra perché nessuno dei due esperti ha una preferenza forte in quel punto. *Picasso* invece
viene dal Facilitator (−1.85 contro −5.58): è conoscenza del modello, non un effetto della
miscela. Con un solo testo per condizione non si può dire se i misti sbagliano più dei puri.

## 5. Valutazione

- **L'indicazione di contenuto funziona.** Con la voce intera e una lista di aspetti per
  categoria, le categorie scelgono fatti diversi, non solo un registro diverso; le scelte
  seguono l'indicazione in modo riconoscibile in tutte e tre le opere. È il cambiamento più
  visibile rispetto ai corpus a intro troncata.
- **Il nucleo visivo resta comune.** I 2–4 dettagli centrali di ogni quadro compaiono in quasi
  tutti i testi. È ragionevole per un'audioguida (sono le cose davanti agli occhi del
  visitatore) e limita quanto i testi possono divergere.
- **Recharger va calibrato.** L'indicazione attuale produce testi quasi privi di
  informazioni. Un'indicazione che chieda uno o due fatti scelti per il loro peso emotivo
  (le lacrime dipinte, Vermeer che non vendette mai il quadro) darebbe testi meno generici.
- **Formule riprese dal prompt.** "If there is one thing to remember" (Experience Seeker) e la
  domanda finale al gruppo (Facilitator) tornano in quasi tutti i testi. Con più opere
  diventerebbero ripetitive, come le formule già misurate in `repetition.py`.
- **I misti a α = 0.5 sono buoni testi.** Leggibili, con tratti di entrambe le categorie,
  spesso divisi fra cornice (apertura e chiusura) e corpo. Con Recharger il misto è spesso
  migliore del Recharger puro, perché la seconda categoria riporta i fatti.
- **Fedeltà alla fonte.** Alta ma non perfetta: con una fonte lunga aumentano i dettagli e
  quindi le occasioni di errore (7 errori in 6 testi su 45, di cui 3 lessicali). Il vincolo
  "drawing only on the factual material provided" non impedisce del tutto i fatti esterni.

## 6. Opere del museo di Braunschweig (RQ3)

Stesso approccio e stesso prompt su 4 dipinti dell'Herzog Anton Ulrich-Museum, il museo di
RQ3, con in più il flat. I testi sono poi tradotti in tedesco, per la sintesi vocale nelle due
lingue. Perché queste 4 opere, disegno e app di RQ3: `docs/report-rq3.md`.

### 6.1 Fonti

Voci della Wikipedia **tedesca** (`fetch_full.py --lang de --titles`), senza le sezioni di
servizio (Literatur, Weblinks, Einzelnachweise e simili). Anteprima completa, tedesco e
inglese a fronte: `docs/analisi/haum-fonti.md`.

| opera | artista | voce tedesca | revid | parole DE | parole EN | token EN |
|---|---|---|---|---|---|---|
| Portrait of a Family (Familienbildnis) | Rembrandt | Familienbildnis (Rembrandt) | 270096678 | 432 | 483 | 660 |
| The Girl with the Wine Glass (Das Mädchen mit dem Weinglas) | Johannes Vermeer | Das Mädchen mit dem Weinglas | 250710229 | 455 | 498 | 624 |
| Self-Portrait as David (Selbstbildnis als David) | Giorgione | Selbstbildnis als David | 270830096 | 576 | 655 | 877 |
| Judith with the Head of Holofernes | Peter Paul Rubens | Judith mit dem Haupt des Holofernes (Rubens) | 262538765 | 575 | 606 | 756 |

**Le voci sono tradotte in inglese con Claude Opus 5.5**, alla lettera e paragrafo per paragrafo,
con i titoli di sezione tradotti. La fonte data a Gemma è la traduzione; il tedesco resta in
`source_text_de`. Motivo: prompt, filtro dei riferimenti espliciti, pulizia e tutti gli altri
testi del progetto sono in inglese. Con la fonte tedesca e il prompt inglese il modello
dovrebbe tradurre e scegliere i fatti nello stesso passo, e nel testo entrerebbero termini e
citazioni tedesche. Le voci sono corte: 3–4 volte l'intro dei corpus principali, contro le
7–11 volte delle voci inglesi del §1.

### 6.2 Flat e generazione

**Flat**: stesso system prompt delle categorie (`WIKI_SYSTEM`), messaggio utente senza la coda
con blocco di categoria e indicazione (`WIKI_LISTENER`). È la regola dei corpus principali (fra
le condizioni cambia solo la parte di categoria); resta quindi nel system prompt il rimando
alla "focus given at the end of the message", che nel flat non c'è.

Notebook `kaggle/haum` (16 novembre 2026), stessa configurazione del §3: per opera 6 puri (5
categorie e flat) e 10 misti a α = 0.5, 64 testi. Con voci corte i prompt sono di 873–1 277
token e i misti si generano 3 alla volta (`--group 3`). Tempi: puri 243 s per tutti e 24, misti
44 minuti per tutti e 40.

| | testi | parole (mediana, min–max) | rigenerati per il filtro | falliti |
|---|---|---|---|---|
| puri (con flat) | 24 | 231 (212–243) | 1 | 0 |
| misti α = 0.5 | 40 | 239.5 (218–253) | 0 | 0 |

Il rigenerato è il Professional/Hobbyist di Self-Portrait as David ("For those"). Nei misti il
token scelto è fra le prime 100 di entrambi gli esperti nel 98.9% dei passi (minimo 96.2%).

### 6.3 Risultati

Fatti della voce per testo puro (x) e in quanti dei 10 misti (stesso conteggio del §4.1).

**Portrait of a Family**

| fatto | Expl | Fac | ExpS | P/H | Rech | flat | misti /10 |
|---|---|---|---|---|---|---|---|
| impasto, coltello da pittore | x | . | x | x | . | x | 7 |
| firma sul cesto di fiori | x | x | x | . | . | . | 8 |
| lutti di Rembrandt | x | x | . | x | . | . | 7 |
| "primo pittore moderno" | . | . | x | x | . | x | 7 |
| Lovis Corinth, "gioiello" | x | . | x | . | . | . | 6 |
| La sposa ebrea | . | . | . | x | . | . | 3 |

**The Girl with the Wine Glass**

| fatto | Expl | Fac | ExpS | P/H | Rech | flat | misti /10 |
|---|---|---|---|---|---|---|---|
| Temperantia nella vetrata | x | x | x | x | x | x | 10 |
| ritratto alla parete | x | x | x | x | x | x | 10 |
| agrumi, afrodisiaco | x | . | . | x | . | x | 6 |
| duca Anton Ulrich, ~1696 | x | . | . | x | . | . | 4 |
| francobollo del 2017 | x | . | x | . | . | . | 2 |
| mostra di Dresda | . | . | . | . | . | . | 1 |

**Self-Portrait as David**

| fatto | Expl | Fac | ExpS | P/H | Rech | flat | misti /10 |
|---|---|---|---|---|---|---|---|
| tela tagliata | x | x | x | x | . | x | 9 |
| testa di Golia | x | x | x | x | . | x | 9 |
| morte per la peste | x | x | x | . | . | x | 8 |
| acquaforte di Hollar (1650) | x | . | x | x | . | x | 6 |
| Beckett | x | x | x | x | . | x | 6 |
| attribuzioni a Raffaello e Dosso Dossi | x | . | . | . | . | . | 6 |
| Vasari | . | . | x | x | . | . | 4 |
| seconda versione a Budapest | . | . | . | x | . | . | 2 |

**Judith with the Head of Holofernes**

| fatto | Expl | Fac | ExpS | P/H | Rech | flat | misti /10 |
|---|---|---|---|---|---|---|---|
| candela della serva | x | x | x | x | x | x | 10 |
| tavola semplice, studio | x | . | x | x | . | x | 7 |
| tavola ingrandita con pezzi aggiunti | x | . | x | x | . | x | 7 |
| armatura sullo sfondo | x | x | . | . | . | x | 5 |
| Betulia | . | . | . | . | . | x | 4 |
| altre scene del ciclo, Varsavia | x | . | . | x | . | x | 3 |
| Salzdahlum | . | . | . | . | . | . | 2 |

- **Le categorie si comportano come sulle voci inglesi.** Explorer cerca enigmi e dettagli
  (la vetrata della Temperanza, i pezzi di legno aggiunti alla tavola), Facilitator dice chi è
  chi e chiude con una domanda al gruppo in tutte e quattro le opere, Professional/Hobbyist è
  il solo con provenienza e storia delle attribuzioni (Salzdahlum, Hollar, Justi e Wickhoff,
  Budapest) e ha più date (2–6 per testo), Experience Seeker insiste su fama e "capolavoro".
- **Con una voce corta il nucleo comune pesa di più.** Nel Vermeer vetrata e ritratto sono in
  tutti e 16 i testi, nel Rubens la candela; le voci hanno pochi fatti oltre la descrizione, e
  le categorie hanno meno da scegliere che sulle voci lunghe.
- **Recharger resta quasi senza fatti.** Nessuna data in nessuna opera; nel Giorgione non dice
  che la tela è tagliata né cita Golia, cioè non spiega il titolo. Nel Vermeer tiene la storia
  della scena (vetrata, ritratto), perché la descrizione è il contenuto della voce.
- **Il flat è informativo e "medio".** Riprende i fatti principali di più categorie
  (tecnica, storia della tela, ciclo di Rubens, Betulia) senza un registro marcato, e apre
  spesso con le stesse formule delle categorie ("Take a moment…", "Observe…").
- **I misti** dividono di nuovo il lavoro fra cornice e corpo, come nel §4.2.

**Errori** (controllo a mano sulla voce):

| testo | errore | tipo |
|---|---|---|
| Rembrandt, Explorer + Experience Seeker; Experience Seeker + Professional/Hobbyist | "the deaths of his wives": Rembrandt ebbe una moglie (Saskia) e una compagna (Hendrickje) | imprecisione |
| Giorgione, Experience Seeker | "Nobel Prize-winning writer", e Beckett "visited this museum specifically to see this piece": nessuno dei due è nella voce | fatto esterno, esagerazione |
| Giorgione, Experience Seeker + Recharger | Beckett "sought out this very painting" | esagerazione |
| Giorgione, Professional/Hobbyist + Recharger | la versione di Budapest "mirroring this one": è speculare l'acquaforte, non la seconda versione | fatto sbagliato |
| Vermeer, Experience Seeker | "official German postage stamps as a national treasure": un solo francobollo | esagerazione |
| Vermeer, Facilitator + Professional/Hobbyist | "since the late 17th century … part of the Herzog Anton Ulrich Museum's collection": il museo è posteriore | anacronismo |
| Rubens, Explorer | "the edges of this canvas … the surface seems uneven": è una tavola, e la voce non parla di superficie irregolare | fatto sbagliato |
| Rubens, Professional/Hobbyist + Recharger | "across the canvas" per la tavola | parola sbagliata |
| Rubens, Explorer + Experience Seeker; Explorer + Recharger | "not a royal commission", "for a king or a church"; "you can see where he expanded" la tavola: dettagli non nella voce | fatto esterno |
| Rubens, Explorer + Facilitator | "Judith's beauty and beauty" | ripetizione |
| Rubens, Experience Seeker + Professional/Hobbyist | "has been part of the Salzdahlum Palace gallery since around 1700": oggi è al museo | tempo sbagliato |

La traduzione tedesca riporta gli errori così come sono (§6.4). Per RQ3 i testi si usano senza
correzione (`report-rq3.md` §4.2).

### 6.4 Traduzione in tedesco

Tutti i 64 testi sono tradotti in tedesco con Claude Opus 5.5, dal testo inglese, nel campo
`text_de` di `data/wiki/haum_generations.jsonl`; testi a fronte in
`docs/analisi/haum-testi.md`. Si traduce invece di rigenerare in tedesco perché così le due
lingue sono la stessa audioguida (stessi fatti, stesso ordine), il prompt resta quello del
resto della prova, il filtro dei riferimenti espliciti (solo inglese) resta valido, e i testi
di RQ3 restano quelli misurabili con RQ1 e RQ2. Regole: traduzione fedele (nessun fatto
aggiunto o tolto, errori compresi), stessi paragrafi, formule della categoria mantenute,
visitatore con il "Sie", lessico della voce tedesca (Kürass, Wams, Malmesser, Temperantia,
Schloss Salzdahlum), titoli tedeschi del museo. Controllo automatico: stessi paragrafi e stessi
numeri in tutti i 64 testi (le sole differenze sono "seventeenth century" → "17. Jahrhundert").
Mediana 237 parole in inglese, 234 in tedesco. Una lettura di un campione da parte di una
persona di madrelingua tedesca è da fare.

### 6.5 The Dying Cleopatra e Aurora

Altre due opere del museo, con lo stesso metodo: *The Dying Cleopatra* (Rosso Fiorentino), dalla
voce **italiana** "Morte di Cleopatra (Rosso Fiorentino)" (revid 149207749, 330 parole), e
*Aurora* (Adam Elsheimer), dalla voce **spagnola** "La Aurora (Elsheimer)" (revid 159263723,
699 parole). `fetch_full.py` toglie anche le sezioni di servizio italiane e spagnole (Note,
Bibliografia, Referencias, Enlaces externos e simili). Le voci sono tradotte in inglese a mano
come le altre (372 e 672 parole; numeri e sezioni uguali all'originale); originale in
`source_text_orig` di `data/wiki/haum2_pages.jsonl`, anteprima in `docs/analisi/haum-fonti.md`.
La voce spagnola dà il numero di inventario 549 (il museo: GG 550); la traduzione lo riporta
com'è.

Notebook `kaggle/haum2` (18 novembre 2026), stesso codice di `kaggle/haum`, misti a 2 per volta
(`--group 2`, perché la voce di Aurora è la più lunga). Prompt di 723–1 253 token. Puri 94 s,
misti 712 s per Cleopatra e 744 s per Aurora.

| | testi | parole (mediana, min–max) | rigenerati per il filtro | falliti |
|---|---|---|---|---|
| puri (con flat) | 12 | 233.5 (204–240) | 0 | 0 |
| misti α = 0.5 | 20 | 237.5 (232–250) | 0 | 0 |

Copertura dei misti 98.7% (minimo 96.4%). Testi in `data/wiki/haum2_generations.jsonl`, con la
traduzione tedesca a mano (`text_de`, stesse regole del §6.4; mediana 235.5 parole); testi a
fronte in `docs/analisi/haum-testi.md`.

**The Dying Cleopatra**

| fatto | Expl | Fac | ExpS | P/H | Rech | flat | misti /10 |
|---|---|---|---|---|---|---|---|
| fanciulla da sinistra | x | x | x | x | x | x | 10 |
| maschera sotto i cuscini | x | x | x | x | x | x | 10 |
| Arianna dormiente scambiata per Cleopatra | x | . | x | x | . | x | 9 |
| bracciale a serpente | x | x | . | x | . | . | 8 |
| panni cangianti | . | x | x | x | x | x | 8 |
| tratti "alla greca" | x | . | . | x | . | x | 8 |
| primo soggetto profano, Fontainebleau | . | x | x | x | . | x | 8 |
| creduta del Seicento | x | . | x | x | . | x | 7 |
| Deposizione di Sansepolcro | . | . | . | x | . | . | 4 |
| fra le più copiate | x | . | x | . | . | x | 3 |
| capezzoli | . | . | . | x | . | . | 3 |
| Cagnacci | . | . | . | x | . | . | 2 |
| Giulio II, Belvedere | x | . | . | x | . | . | 1 |
| Laocoonte | . | . | . | . | . | . | 1 |

**Aurora**

| fatto | Expl | Fac | ExpS | P/H | Rech | flat | misti /10 |
|---|---|---|---|---|---|---|---|
| pastore solitario | x | x | . | x | x | x | 10 |
| rame | x | x | x | x | . | x | 10 |
| alba che illumina il primo piano, una novità | x | x | x | x | . | x | 10 |
| Villa di Mecenate | x | x | . | x | . | x | 8 |
| aspetto di smalto | x | x | x | x | . | x | 6 |
| Aci e Galatea | x | . | . | x | . | . | 5 |
| Goudt, incisione del 1613 | x | x | . | x | . | . | 4 |
| influenza su Rubens, Rembrandt, Lorrain | x | x | x | x | . | x | 4 |
| 17 x 22,5 cm | x | . | . | x | . | . | 4 |
| Moser e Altdorfer | . | . | . | x | . | . | 0 |
| tenebrismo, Caravaggio | . | . | . | x | . | . | 0 |
| Milizia | . | . | . | . | . | . | 0 |

- Le categorie si comportano come nelle altre opere: Professional/Hobbyist è il solo con
  Cagnacci, Moser e Altdorfer, tenebrismo; Recharger non ha date e quasi nessun fatto (in
  Aurora nemmeno il rame); il flat riprende i fatti principali.
- Il nucleo visivo è in quasi tutti i testi (la fanciulla e la maschera in Cleopatra, il
  pastore e la luce dell'alba in Aurora).
- La citazione di Milizia, un terzo della voce di Aurora, non entra in nessun testo come
  citazione; ne entrano però le immagini (le bande dorate e argentee, i vapori), a volte come
  se descrivessero il quadro (vedi sotto).
- La voce italiana parla del nudo ("i turgidi capezzoli"); lo riprendono Professional/Hobbyist
  e 3 misti.

**Errori** (controllo a mano sulla voce):

| testo | errore | tipo |
|---|---|---|
| Cleopatra, Explorer; Experience Seeker + Professional/Hobbyist | la statua "then held by Pope Julius II" quando Rosso dipinge (c. 1525): Giulio II la acquista nel 1512 e muore nel 1513 | anacronismo |
| Cleopatra, Experience Seeker | "doubted the painting's authenticity": la voce parla di attribuzione | imprecisione |
| Cleopatra, Explorer + Experience Seeker | "for centuries, its true origin was a mystery" | esagerazione |
| Cleopatra, Explorer + Recharger | "Rosso Fiorentino embraced this mistake", come se sapesse dell'errore; "across the canvas" per la tavola | fatto sbagliato, parola sbagliata |
| Cleopatra, Facilitator + Professional/Hobbyist | la Deposizione di Sansepolcro come "earlier work": è posteriore (1527–28), e la voce non la data | fatto sbagliato |
| Aurora, Professional/Hobbyist; Facilitator + Recharger | bande "vermilion and gold" e "gold and silver" nel cielo del quadro: vengono dalla descrizione generica dell'alba di Milizia | fatto attribuito al quadro |
| Aurora, Explorer | "this tiny panel paved the way for masters like Rembrandt and Claude Lorrain": la voce parla dell'opera di Elsheimer in generale | esagerazione |

### 6.6 Sintesi vocale

`generation/tts.py` tramite OpenRouter (`/api/v1/audio/speech`), modello con `--model`: gli
audio di RQ3 sono fatti con HeyGen Voice Turbo gratuito, voce `nadine`, un mp3 per testo e
lingua in `rq3_app/audio/<lingua>/<opera>/<condizione>.mp3`, la struttura che legge l'app di
RQ3. L'id dell'opera è il titolo inglese con i trattini (`tts.artwork_slug`). 160 file per 5
opere (Self-Portrait as David escluso con `--exclude`). Scelta del modello, durate e controlli
in `report-rq3.md` §5.

## 7. Limiti e passi successivi

Prova qualitativa: 9 opere in tutto, un testo per condizione, un solo seed, valutazione di una
sola persona su testi letti a mano. Nessuna misura di RQ1 (embedding) né di RQ2 (giudice) su
questi testi. Per farne un esperimento servirebbero più opere (con il limite di contesto delle
T4: voci sotto ~3 500 token, o una GPU con più memoria), le repliche per il tetto, e le
stesse analisi della parte 2 (probe, split-half, posizione dei misti, giudice). Non avviati.

## 8. File e comandi

| cosa | dove |
|---|---|
| voci intere | `corpus/fetch_full.py` → `data/wiki/full_pages.jsonl`; con `--lang de --titles` → `data/wiki/haum_pages_de.jsonl` |
| voci di Braunschweig tradotte | `data/wiki/haum_pages.jsonl` (traduzione inglese in `source_text`, tedesco in `source_text_de`), `data/wiki/haum2_pages.jsonl` (originale italiano e spagnolo in `source_text_orig`); anteprima in `docs/analisi/haum-fonti.md` |
| prompt | `common/prompts.py`: `WIKI_SYSTEM`, `WIKI_USER`, `WIKI_LISTENER`, `FOCUS`, `wiki_messages` |
| generazione | `generation/vllm_wiki.py`, `kaggle/wiki/` → `data/wiki/generations.jsonl`; `kaggle/haum/` → `data/wiki/haum_generations.jsonl`, `kaggle/haum2/` → `data/wiki/haum2_generations.jsonl` (con `text_de`) |
| sintesi vocale | `generation/tts.py` → `rq3_app/audio/` |
| opere per l'app di RQ3 | `data/rq3/artworks_haum.json` |
| testi completi | `docs/analisi/pagina-intera-testi.md`, `docs/analisi/haum-testi.md` |
| test | `tests/test_fetch_full.py`, `tests/test_vllm_wiki.py`, `tests/test_tts.py` |

```bash
PYTHONPATH=src .venv/bin/python -m museumbot.corpus.fetch_full Q568847 Q588695 Q737062
KAGGLE_API_TOKEN=$KAGGLE_KEY .venv/bin/kaggle kernels push -p kaggle/wiki
KAGGLE_API_TOKEN=$KAGGLE_KEY .venv/bin/kaggle kernels output massidale/museumbot-wiki -p <dir>
PYTHONPATH=src .venv/bin/python -m museumbot.corpus.fetch_full --lang de --titles "Familienbildnis (Rembrandt)" \
    "Das Mädchen mit dem Weinglas" "Selbstbildnis als David" "Judith mit dem Haupt des Holofernes (Rubens)" \
    --out data/wiki/haum_pages_de.jsonl
KAGGLE_API_TOKEN=$KAGGLE_KEY .venv/bin/kaggle kernels push -p kaggle/haum
KAGGLE_API_TOKEN=$KAGGLE_KEY .venv/bin/kaggle kernels push -p kaggle/haum2
PYTHONPATH=src .venv/bin/python -m museumbot.generation.tts --in data/wiki/haum_generations.jsonl \
    --model heygen/heygen-voice-1-turbo:free --voice nadine --exclude self-portrait-as-david
PYTHONPATH=src .venv/bin/python -m museumbot.generation.tts --in data/wiki/haum2_generations.jsonl \
    --model heygen/heygen-voice-1-turbo:free --voice nadine
```
