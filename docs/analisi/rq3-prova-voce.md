# RQ3: prova d'ascolto delle voci Azure

Scopo: scegliere una voce inglese e una tedesca per le audioguide di RQ3, prima di generare
gli audio. Si ascoltano gli stessi tre testi di un'opera con 3 voci neurali standard per
lingua, più una voce HD come confronto.

## Come fare

1. Account Azure, poi una risorsa **Speech** con piano gratuito **F0** (500k caratteri
   neurali al mese).
2. Speech Studio → **Audio Content Creation** (speech.microsoft.com): nuovo file, incolla un
   testo, scegli la voce, ascolta. Lascia velocita' e tono di default.
3. Tre testi × 4 voci × 2 lingue ≈ 32k caratteri: meno del 7% della quota gratuita.

## Voci

| lingua | voce | tipo |
|---|---|---|
| en | `en-GB-RyanNeural` | standard, M |
| en | `en-GB-SoniaNeural` | standard, F |
| en | `en-US-AvaMultilingualNeural` | standard multilingue, F |
| en | `en-GB-Sonia:DragonHDLatestNeural` | HD, solo confronto |
| de | `de-DE-ConradNeural` | standard, M |
| de | `de-DE-KatjaNeural` | standard, F |
| de | `de-DE-SeraphinaMultilingualNeural` | standard multilingue, F |
| de | `de-DE-Seraphina:DragonHDLatestNeural` | HD, solo confronto |

## Cosa ascoltare

- **Pronuncia**: "Bruegel", "Kunsthistorisches Museum" nella voce inglese, "BBC", "1565".
- **Tenuta su 1,5 minuti**: si ascolta senza fatica? Pause fra i paragrafi naturali?
- **Tono costante fra i tre testi**: la voce legge `flat`, puro e misto allo stesso modo?
  Con le HD e' il punto da controllare: se il tono cambia col contenuto, la voce aggiunge
  una differenza fra le due guide della coppia.
- Annota per ogni voce un voto da 1 a 5 e una riga di commento.

## Testi (The Hunters in the Snow, Bruegel; corpus `local`, Gemma 4 31B)

I testi inglesi vengono da `data/local/generations.jsonl` (`flat`, `facilitator`, variante
`full`) e da `data/local/mixed.jsonl` (`facilitator+professional_hobbyist`, α = 0.5). I
testi tedeschi sono una mia traduzione, **solo per questa prova**: non sono stimoli
dell'esperimento.

### EN — flat

Step closer to this 1565 masterpiece by Pieter Bruegel the Elder, titled The Hunters in the Snow, or The Return of the Hunters. This oil-on-wood painting is a definitive example of the Northern Renaissance, capturing a stark landscape set in the depths of winter during December and January.

Bruegel created this scene as part of a larger series designed to depict the different seasons of the year. While many of these works have been lost to time, this is one of only five surviving paintings from the original set. Its enduring influence and technical brilliance have earned it a place among the BBC's 100 Great Paintings.

As you look across the frozen terrain, notice how Bruegel captures the biting cold and the stillness of the winter air. The composition draws you into the daily struggle and rhythm of life in the 16th century. Today, this evocative work is held in the permanent collection of the Kunsthistorisches Museum in Vienna, Austria. Take a moment to immerse yourself in the atmosphere of the season and the quiet journey of the hunters returning home.

### EN — facilitator

Take a moment to look around and gather your group before we dive into this masterpiece. You are looking at The Hunters in the Snow, painted in 1565 by Pieter Bruegel the Elder. This oil-on-wood work is a quintessential example of the Northern Renaissance and is currently housed here in the Kunsthistorisches Museum in Vienna.

This scene captures the depths of winter, specifically the cold transition between December and January. It is actually part of a larger series created by Bruegel to depict the different seasons of the year, and this is one of only five paintings from that series that still survive today. Because of its lasting impact and composition, it has even been named one of the BBC's 100 Great Paintings.

As you explore the details with your companions, notice how Bruegel captures the atmosphere of a winter landscape. You might ask those you are with what they notice first about the hunters' return or how the painting makes them feel about the cold. Does the scene feel quiet, or can you almost hear the sounds of the village in the background? Encourage everyone to share their impressions of the landscape and discuss how the artist managed to convey the specific mood of mid-winter. It is a wonderful piece for comparing perspectives, as there is so much for everyone in your group to discover together.

### EN — facilitator + professional_hobbyist (α = 0.5)

Pieter Bruegel the Elder's 1565 masterpiece, The Hunters in the Snow, also known as The Return of the Hunters, is a definitive example of Northern Renaissance art. This oil-on-wood painting is part of a larger series depicting the cycle of the seasons, of which five works still survive today. Set in the depths of winter during December or January, the scene captures a specific atmospheric mood and a precise moment in time.

Currently held in the collection of the Kunsthistorisches Museum in Vienna, this work is recognized globally for its composition and detail, earning a place among the BBC's 100 Great Paintings. To engage with this piece, consider how Bruegel uses the landscape to convey the harshness of the season and the fatigue of the figures. The painting is not merely a landscape but a study of human interaction with nature during the coldest months of the year.

As you examine the work, you might compare it with the other surviving seasonal paintings in the series to see how Bruegel adapts his palette and subject matter to reflect the changing environment. This provides a wonderful opportunity to observe the artist's mastery of perspective and his ability to evoke a sense of place. Take a moment to notice the subtle details in the snow-covered terrain and the posture of the hunters, which reflect the physical toll of their journey.

### DE — flat

Treten Sie näher an dieses Meisterwerk von Pieter Bruegel dem Älteren aus dem Jahr 1565, Die Jäger im Schnee, auch bekannt als Die Heimkehr der Jäger. Das Ölgemälde auf Holz ist ein herausragendes Beispiel der nordischen Renaissance und zeigt eine karge Landschaft mitten im tiefsten Winter, zwischen Dezember und Januar.

Bruegel schuf diese Szene als Teil einer größeren Serie, die die verschiedenen Jahreszeiten darstellen sollte. Viele dieser Werke sind im Laufe der Zeit verloren gegangen; dies ist eines von nur fünf erhaltenen Gemälden der ursprünglichen Reihe. Wegen seines anhaltenden Einflusses und seiner technischen Meisterschaft zählt es zu den 100 großen Gemälden der BBC.

Wenn Ihr Blick über das gefrorene Gelände wandert, achten Sie darauf, wie Bruegel die schneidende Kälte und die Stille der Winterluft einfängt. Die Komposition zieht Sie hinein in die tägliche Mühe und den Rhythmus des Lebens im 16. Jahrhundert. Heute befindet sich dieses eindrucksvolle Werk in der Dauerausstellung des Kunsthistorischen Museums in Wien. Nehmen Sie sich einen Moment Zeit, um in die Stimmung der Jahreszeit und den stillen Heimweg der Jäger einzutauchen.

### DE — facilitator

Nehmen Sie sich einen Moment Zeit, schauen Sie sich um und versammeln Sie Ihre Gruppe, bevor wir in dieses Meisterwerk eintauchen. Sie sehen Die Jäger im Schnee, 1565 von Pieter Bruegel dem Älteren gemalt. Das Ölgemälde auf Holz ist ein Musterbeispiel der nordischen Renaissance und befindet sich heute im Kunsthistorischen Museum in Wien.

Die Szene zeigt den tiefsten Winter, genauer den kalten Übergang von Dezember zu Januar. Sie gehört zu einer größeren Serie, mit der Bruegel die verschiedenen Jahreszeiten darstellte, und ist eines von nur fünf Gemälden dieser Serie, die bis heute erhalten sind. Wegen seiner nachhaltigen Wirkung und seiner Komposition wurde es sogar zu einem der 100 großen Gemälde der BBC gewählt.

Während Sie mit Ihren Begleitern die Details erkunden, achten Sie darauf, wie Bruegel die Atmosphäre einer Winterlandschaft einfängt. Fragen Sie die anderen, was ihnen an der Heimkehr der Jäger zuerst auffällt oder welches Gefühl der Kälte das Bild in ihnen weckt. Wirkt die Szene still, oder hört man fast die Geräusche des Dorfes im Hintergrund? Ermuntern Sie alle, ihre Eindrücke von der Landschaft zu teilen, und sprechen Sie darüber, wie der Künstler die besondere Stimmung des Hochwinters vermittelt. Es ist ein wunderbares Bild, um Sichtweisen zu vergleichen, denn es gibt so viel, das Ihre Gruppe gemeinsam entdecken kann.

### DE — facilitator + professional_hobbyist (α = 0.5)

Pieter Bruegels des Älteren Meisterwerk von 1565, Die Jäger im Schnee, auch bekannt als Die Heimkehr der Jäger, ist ein herausragendes Beispiel der Kunst der nordischen Renaissance. Das Ölgemälde auf Holz gehört zu einer größeren Serie über den Kreislauf der Jahreszeiten, von der heute noch fünf Werke erhalten sind. Die Szene spielt im tiefsten Winter, im Dezember oder Januar, und hält eine bestimmte atmosphärische Stimmung und einen genauen Augenblick fest.

Das Werk befindet sich heute in der Sammlung des Kunsthistorischen Museums in Wien und ist weltweit für seine Komposition und seine Details bekannt; es zählt zu den 100 großen Gemälden der BBC. Um sich mit dem Bild auseinanderzusetzen, achten Sie darauf, wie Bruegel die Landschaft nutzt, um die Härte der Jahreszeit und die Erschöpfung der Figuren zu vermitteln. Das Gemälde ist nicht bloß eine Landschaft, sondern eine Studie über das Verhältnis des Menschen zur Natur in den kältesten Monaten des Jahres.

Wenn Sie das Werk betrachten, können Sie es mit den anderen erhaltenen Jahreszeitenbildern der Serie vergleichen und sehen, wie Bruegel Palette und Motiv an die wechselnde Umgebung anpasst. Das bietet eine wunderbare Gelegenheit, die perspektivische Meisterschaft des Künstlers und seine Fähigkeit zu beobachten, ein Gefühl für den Ort zu wecken. Achten Sie auf die feinen Details im verschneiten Gelände und auf die Haltung der Jäger, die die körperlichen Strapazen ihres Weges widerspiegeln.
