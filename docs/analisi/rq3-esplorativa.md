# RQ3: analisi esplorativa delle giornate al museo

*Ultimo aggiornamento: 24 novembre 2026. Dati: dataset pulito delle giornate del 20, 21 e 22
novembre 2026 (`data/rq3/clean/`, 36 visitatori e 78 opere; regole in `docs/report-rq3.md`
§6). Numeri da `results/rq3_summary.json`
(`PYTHONPATH=src .venv/bin/python -m museumbot.rq3_visitors.summary`); temi dei commenti
codificati a mano in `data/rq3/temi.json`.*

L'analisi è esplorativa: le giornate e le versioni dell'app sono unite, e la discussione
(parte 5) nasce leggendo questi stessi dati. Va verificata su dati nuovi.

`diff` = voto della guida personalizzata − voto del `flat` (l'audioguida del museo accorciata),
voti da 1 a 7. Quando un visitatore parla di
"prima" e "seconda" guida, qui è già tradotto in personalizzata o `flat`.

## 1. Personalizzata, mista e `flat`

**Nel complesso non c'è una preferenza netta.** La personalizzata prende in media 5.46, il
`flat` 5.22: diff +0.24. Vince la personalizzata 36 volte, pari 13, vince il
`flat` 29.

**Le personalizzate miste vanno meglio delle pure.** È il risultato più chiaro.

| personalizzata | prove | visitatori | voto pers. | voto flat | diff | pers. / pari / flat |
|---|---|---|---|---|---|---|
| pura (1 categoria) | 31 | 18 | 5.00 | 5.23 | −0.23 | 12 / 3 / 16 |
| mista (2 categorie) | 47 | 18 | 5.77 | 5.21 | **+0.55** | 24 / 10 / 13 |

Il `flat` prende lo stesso voto nei due gruppi (5.23 e 5.21): la differenza viene tutta dal
voto della personalizzata. Le pure perdono contro il `flat`, le miste vincono.

**Cosa dicono.** Nessun visitatore si accorge che la sua guida mescola due profili. Chi ha
una guida mista parla soprattutto di quanto la guida è vicina a ciò che si vede (7 dei 17 che
hanno commentato) e di unire le due guide (5); chi ha una guida pura parla soprattutto della
quantità di informazioni (7 su 16), in tutte e due le direzioni. Le critiche più dure alla
personalizzata, cioè che dà troppo pochi fatti, vengono da guide pure (facilitator e
recharger, parte 2). Una guida pura spinge un solo registro fino in fondo (le domande
dell'explorer, le emozioni del recharger, la tecnica del professional_hobbyist); la miscela lo
attenua e aggiunge i contenuti dell'altra categoria. Un esempio misurabile: un testo explorer
puro ha in media 3.8 domande, un misto con explorer 1.6, il `flat` 0.4.

## 2. Per categoria

Ogni prova in cui la categoria compare, separando le guide pure da quelle miste. Per ogni
categoria, i commenti più frequenti di chi l'ha scelta (un visitatore con due categorie conta
in tutte e due).

| categoria | visitatori | prove | guide pure: diff (prove) | guide miste: diff (prove) |
|---|---|---|---|---|
| explorer | 22 | 51 | −0.14 (14) | +0.51 (37) |
| facilitator | 10 | 21 | −0.86 (7) | +1.07 (14) |
| experience_seeker | 9 | 26 | +1.50 (4) | +0.41 (22) |
| professional_hobbyist | 6 | 12 | −3.00 (1) | +0.82 (11) |
| recharger | 7 | 15 | −0.40 (5) | 0.00 (10) |

Con 1–8 visitatori per cella sono descrizioni, non confronti. Fra i misti più frequenti:
explorer + experience_seeker +0.18 (17 prove, 5 visitatori), explorer + professional_hobbyist
+1.00 (8, 3), explorer + facilitator +1.29 (7, 4).

**Explorer (22 visitatori; pure −0.14, miste +0.51).** La guida explorer fa domande e invita a
guardare dettagli precisi. I commenti più frequenti: la guida vicina a ciò che si vede (8),
più informazioni (7), unire le due guide (5). Gli explorer puri si dividono in due. Chi cerca
ciò che si vede sceglie la guida explorer: cgofvw e dyu9bg dicono di preferire ogni volta
quella più vicina al quadro. Chi cerca una storia o il contesto sceglie il `flat`: keamg8
sulla Giuditta (4 contro 7), scdpaq su Cleopatra e Aurora (5 contro 6, 4 contro 7), tyetwa per
la mitologia di Aurora (5 contro 7). Le domande coinvolgono qualcuno (jq4q04) ma non bastano:
9p0r1b vuole essere indirizzato verso il significato senza che gli venga imposto.

**Experience_seeker (9 visitatori; pure +1.50 su 4 prove, miste +0.41).** I commenti più
frequenti: più informazioni (5) e unire le due guide (4). La guida experience_seeker spiega
perché l'opera è importante, e questo piace: 9oudwc impara che la Ragazza è finita su un
francobollo (7 contro 5), is449e premia le informazioni contro le domande del `flat` (6 contro
4, riferito dall'operatore). Con l'explorer danno voti alti a tutte e due le guide (5.71 e
5.53) e chiedono di unirle (snnp09, zxut5q).

**Facilitator (10 visitatori; pure −0.86, miste +1.07).** Le guide facilitator pure sono le
meno commentate: dei 4 visitatori, 2 non hanno lasciato commenti. 2646hr dice che mancano le
informazioni interessanti, la luce e il pittore (3 contro 6); oodswo non sente grandi
differenze fra le due guide. Le miste con facilitator invece piacciono; i commenti di chi le
ha scelte riguardano soprattutto quanto la guida è vicina a ciò che si vede (4) e Goethe su
Aurora (3), cioè l'effetto delle opere (parte 3).

**Professional_hobbyist (6 visitatori; una sola prova pura, −3, miste +0.82).** Il commento più
frequente: rimandi a cose che non si vedono o non vengono spiegate (3 dei 5 che hanno
commentato): la *Sposa ebrea* (qz12x0), altri quadri di Rubens (jxgtc0: "diventa astratto", 3
contro 6), la Temperanza nello stemma (tfhvt3). Poi le interpretazioni imposte (2): qz12x0
vuole una guida oggettiva e premia la personalizzata quando lo è (7 contro 4 su Aurora, 7
contro 3 sulla Giuditta). vd92rm nota che la categoria non distingue chi studia da chi lavora
nel campo. Chi si dichiara esperto vuole l'inquadramento storico-artistico, ma legato a ciò che
ha davanti.

**Recharger (7 visitatori; pure −0.40, miste 0.00).** È l'unica categoria le cui guide
miste non battono il `flat` (pari in media). Il commento più frequente: le emozioni (4 dei 6 che hanno commentato). La guida
con recharger è sentita come più emotiva: piace quando dà atmosfera (9ad18s, 7 contro 5 sulla
Ragazza) o un modo diverso di guardare (44x0vx); non piace quando prende il posto dei fatti
(m3p4yx, 3 contro 6, riferito dall'operatore). Per cdq93g "dipende dal quadro": una guida dà
più sentimento, l'altra più fatti (due opere a testa).

## 3. Per opera

**L'opera conta più della categoria.**

| opera | prove | voto pers. | voto flat | diff | pers. / pari / flat |
|---|---|---|---|---|---|
| Aurora | 22 | 5.86 | 4.77 | **+1.09** | 15 / 3 / 4 |
| Ragazza col bicchiere | 19 | 5.63 | 5.16 | +0.47 | 9 / 6 / 4 |
| Ritratto di famiglia | 10 | 5.40 | 5.00 | +0.40 | 6 / 0 / 4 |
| Giuditta | 12 | 5.42 | 5.33 | +0.08 | 5 / 2 / 5 |
| Cleopatra | 15 | 4.73 | 6.00 | **−1.27** | 1 / 2 / 12 |

Il voto della personalizzata varia poco da un'opera all'altra (4.7–5.9); quello del `flat`
molto di più (4.8–6.0). Le differenze fra opere vengono soprattutto dal testo del museo.

**Cosa dicono.** Qui solo i commenti riferiti esplicitamente a
un'opera, con il voto dato in quel confronto (personalizzata contro `flat`).

- **Cleopatra: il `flat` vince quasi sempre, ed è un limite della nostra fonte.** Il testo del
  museo parte dalla scena centrale: Cleopatra morente sui cuscini, il morso del serpente sul
  braccio, il corpo nudo, la luce che lo fa sembrare di marmo. La voce Wikipedia da cui partono
  le guide personalizzate parla soprattutto della statua di Arianna, di Vasari e
  dell'attribuzione, e della scena dà solo dettagli (la ragazza che entra da sinistra, la
  maschera sotto i cuscini, le pennellate). Le guide personalizzate fanno lo stesso: nessuna
  parla del serpente sul braccio. I commenti che confrontano le due guide vanno tutti in questa
  direzione, e tutti con il voto al `flat`: per snnp09 il `flat` parla del soggetto e la
  personalizzata di tecnica e artista (5 contro 7); per kp8igf e r56v63, riferito
  dall'operatore, la personalizzata parla di fatti e contesto invece che dell'immagine (2 contro
  6, 4 contro 5); per 9ad18s il `flat` è più avvincente e si collega meglio a ciò che si guarda
  (5 contro 7); per scdpaq il `flat` entra subito nel vivo con una storia (5 contro 6).
- **Aurora: la personalizzata vince, perché il testo del museo descrive poco.** Anche la voce
  Wikipedia descrive poco il quadro, ma il testo del museo è in gran parte la poesia di Goethe.
  La poesia divide, e il voto segue sempre il commento: contro ujakv7 (6 contro 5), fjlq2w (6
  contro 4), h7v6kx (7 contro 3) e qz12x0 (7 contro 4), a favore mzoiec (6 contro 7). Le guide
  personalizzate parlano del paesaggio, del pastore, dell'edificio e della luce: cgofvw (6
  contro 4) e h7v6kx ("ho notato perfino l'edificio in basso a destra") le preferiscono per
  questo. Le eccezioni: vd92rm, che trova noioso l'inizio con le misure del quadro e preferisce
  il racconto del `flat` (3 contro 6), e tyetwa, a cui piace la mitologia del `flat` (5 contro
  7).
- **Ragazza col bicchiere: le due guide si completano, ed è l'opera con più pari.** Le due guide
  leggono in modo diverso gli stessi dettagli (il limone: afrodisiaco o innocenza ferita); chi
  propone di unirle, o dice che hanno entrambe dei vantaggi, parla quasi sempre di quest'opera
  (tapn00 5 contro 5, snnp09 6 contro 6, tfhvt3 6 contro 6). Quando qualcuno sceglie, premia la
  personalizzata per i dettagli, fra gli altri mzoiec (7 contro 5), 6youru (7 contro 6), 9oudwc
  (7 contro 5) e kp8igf, riferito dall'operatore, perché fa vedere l'opera intera prima dei
  dettagli, cioè racconta in modo più omogeneo (6 contro 3). Le critiche alla personalizzata riguardano due guide pure: alla
  facilitator mancano la luce e il pittore (2646hr, 3 contro 6), la recharger è troppo legata
  all'emozione e dice poco dell'artista (m3p4yx, riferito dall'operatore, 3 contro 6). tfhvt3
  vorrebbe spiegata la Temperanza nello stemma.
- **Giuditta: dipende da chi conosce la storia.** Il testo del museo racconta la storia biblica.
  A keamg8 serve (4 contro 7); per qz12x0 chi la conosce "stacca dopo due secondi", e in più il
  `flat` giudica (7 contro 3); snnp09 apprezza l'inizio biblico del `flat` ma premia la
  personalizzata perché descrive cosa succede (6 contro 4). jxgtc0 boccia la
  professional_hobbyist pura: troppa tecnica e rimandi a quadri che non ci sono (3 contro 6).
- **Ritratto di famiglia: dipende dal testo personalizzato.** L'explorer parte dalla superficie
  (l'impasto, la firma nel cesto) e cgofvw la trova più vicina all'opera (6 contro 5); per
  snnp09, con il misto explorer + experience_seeker, il `flat` dice di più su cosa succede (5
  contro 7). A qz12x0 danno fastidio il "guarda adesso qui" e il rimando alla *Sposa ebrea*, ma
  vota comunque la personalizzata (6 contro 5).

**L'ordine delle guide pesa un po'.** La Guida 1 prende in media +0.27 rispetto
alla Guida 2. Al netto dell'opera, la personalizzata fa +0.26 quando è la Guida 1 e −0.26
quando è la Guida 2.

## 4. Cosa dicono i visitatori in generale

Qui i temi generali, non legati a un'opera (quelli legati a un'opera sono nella parte 3). Per
ogni tema: quanti visitatori lo citano, le loro preferenze su tutte le opere che hanno
giudicato e, quando il commento dice quale guida è migliore in un confronto preciso, se il
voto va nello stesso verso. Un tema è "di massa" se lo citano almeno 4 visitatori diversi
(soglia fissata dopo aver letto i commenti).

**Le categorie si scelgono bene (12 visitatori).** Dicono che è stato facile scegliere; 33
visitatori su 36 danno almeno 5 (su 7) a una delle categorie scelte. Su tutte le loro opere
preferiscono la personalizzata: +0.57 su 37 confronti (20 / 5 / 12). I due che hanno trovato
la scelta difficile (snnp09; vd92rm, per cui professional_hobbyist non distingue chi studia da
chi lavora nel campo) preferiscono il `flat`: −0.50 su 6 confronti.

**Si preferisce la guida che parla di ciò che si vede (10 visitatori).** È il criterio
dichiarato più spesso, anche in generale (cgofvw: la guida più vicina al quadro; dyu9bg:
"ogni volta" quella riferita direttamente al quadro, non quella con informazioni di contorno).
Quando una guida parla di contesto, di artista o di misure invece che dell'immagine, perde. Il
voto va nel verso del commento in 11 confronti su 12. Quale guida soddisfa il criterio dipende
dall'opera (parte 3): per questo, nel complesso, questi visitatori sono vicini al pari (+0.35
su 26 confronti, 13 / 4 / 9).

**Si premiano i fatti e i dettagli (11 visitatori).** Più informazioni, più dettagli, cose che
non si sapevano. Il voto va nel verso del commento in 10 confronti su 11 (l'altro è un pari).
Di solito è la personalizzata a darne di più (8 visitatori); le critiche opposte riguardano
guide pure che ne danno troppo pochi. Su tutte le loro opere: +0.17 su 18 confronti (9 / 4 /
5).

**Le due guide si completano (8 visitatori).** 6 propongono di unirle, 2 dicono che valeva la
pena sentirle tutte e due; 3 visitatori descrivono la guida ideale a livelli, prima l'opera
intera o un breve contesto e poi i dettagli (zxut5q, con l'esempio di un'audioguida a più
livelli in un castello; qe1301; kp8igf). Chi lo dice dà più pari degli altri e preferisce
comunque la personalizzata: +0.76 su 25 confronti (12 / 7 / 6).

**Un racconto coinvolge (4 visitatori).** Apprezzano una guida che entra subito nella storia e
la lega a ciò che si vede; tutti e 4 lo dicono del `flat` e lo votano: −1.67 su 6 confronti (1
/ 0 / 5). Le guide personalizzate tendono a mettere in fila fatti e dettagli; i testi del museo
raccontano.

**Si vuole interpretare da sé (4 visitatori).** Non vogliono giudizi né interpretazioni già
pronte, ma fatti e spunti per arrivare da soli al significato (qz12x0, scdpaq, 9p0r1b,
tfhvt3). In 4 confronti il commento dice che la personalizzata interpreta meno del `flat`; in 3
la personalizzata ha anche il voto piu' alto (tutti e tre di qz12x0: Ragazza, Aurora, Giuditta),
nel quarto scdpaq critica il `flat` di Cleopatra per troppa interpretazione ma gli da' il voto
piu' alto. Su tutte le loro opere: +1.00 su 9 confronti (5 / 2 / 2).

**Le emozioni piacciono se c'è anche altro (6 visitatori).** La personalizzata è sentita come
più emotiva. Piace quando dà atmosfera (9ad18s) o un modo diverso di guardare (44x0vx); non
piace quando prende il posto dei fatti (m3p4yx); per cdq93g "dipende dal quadro". Su tutte le
loro opere: +0.63 su 19 confronti (11 / 2 / 6).

**La voce (5 visitatori l'accento, 5 il ritmo).** Accento inglese, soprattutto nei nomi;
pause troppo lunghe fra le parole o troppo brevi fra un argomento e l'altro; parlato da
velocizzare. La voce è la stessa per le due guide, e chi la nota non preferisce il `flat`
(accento +0.44 su 16 confronti, ritmo +0.82 su 17).

**Temi minori** (3 visitatori o meno), con il voto che spesso non segue il commento:

- domande e inviti a guardare: a jq4q04 piacciono ma vota il `flat`, a qz12x0 danno fastidio ma
  vota la personalizzata (nel verso del commento 1 volta su 3);
- rimandi a cose che non si vedono o non vengono spiegate (altre opere, simboli): jxgtc0,
  qz12x0, tfhvt3 (nel verso del commento 1 volta su 2);
- guide poco diverse (oodswo, u5oi3e, jq4q04): −0.12 su 8 confronti, con 3 pari.

## 5. Discussione

**Le categorie di Falk dicono perché il visitatore è al museo, non che informazioni vuole
sentire.** Lo mostrano i commenti su come si vuole ricevere l'interpretazione. Alcuni
visitatori vogliono interpretare da sé: fatti oggettivi, niente giudizi né letture già pronte
(qz12x0, scdpaq, 9p0r1b, tfhvt3). Altri vogliono un racconto che leghi i fatti a ciò che si
vede (keamg8, scdpaq, 9ad18s, vd92rm; scdpaq vuole una storia che resti oggettiva). Dei 7
visitatori che ne parlano, 6 hanno scelto
explorer: la stessa motivazione per la visita, attese opposte sulla guida. Il caso più chiaro
è quello di qz12x0 e vd92rm: stessa coppia di categorie (explorer + professional_hobbyist),
stesso testo di Aurora, giudizi opposti. Per qz12x0 è "più oggettiva" (7 contro 4), per vd92rm
"comincia subito con le misure, noioso" (3 contro 6). Personalizzare sulla motivazione della
visita non basta a decidere il registro della guida; servirebbe chiedere anche questo.

**Conta quanto la guida parla del quadro, e lo decide la fonte.** La personalizzata è scritta
a partire dalla voce di Wikipedia; il `flat` è il testo del museo. Le voci della Ragazza e della
Giuditta hanno una vera descrizione del dipinto; quella del Ritratto descrive le figure e la
pennellata; quelle di Cleopatra e di Aurora parlano soprattutto dell'artista e del contesto. I
testi del museo, a loro volta, hanno ciascuno uno stile diverso: la Giuditta racconta la storia
biblica, la Ragazza interpreta, Cleopatra descrive la scena, Aurora è in gran parte la poesia di
Goethe. La personalizzata perde nettamente solo dove la fonte non descrive la scena e il testo
del museo sì (Cleopatra), e vince nettamente dove il testo del museo descrive poco (Aurora):
conta la descrizione della fonte rispetto a quella del `flat`. Con 5 opere è una lettura caso
per caso.

**Il confronto a coppie misura anche la complementarità.** Molti giudicano la seconda guida per
quello che aggiunge alla prima (tapn00: "la prima la conoscevo già"), e la Guida 1 prende in
media +0.27. Una guida a livelli, prima l'opera intera e poi i dettagli a richiesta (zxut5q,
qe1301, kp8igf), risponderebbe a entrambe le cose.

## 6. Limiti

- 36 visitatori, 5 opere; 1–8 visitatori per categoria o coppia di categorie.
- Una sola voce sintetica per tutte le guide, con un accento notato da 5 visitatori.
- Temi codificati da un solo codificatore, e lettura delle fonti qualitativa; le note dell'operatore
  (is449e, 9p0r1b, m3p4yx, kp8igf, r56v63) sono parafrasi, non parole del visitatore.
- La discussione nasce da questi dati.
