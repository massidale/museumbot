"""Museumbot: personalizzazione delle audioguide secondo le categorie di Falk.

Un sottopacchetto per fase del progetto; gli esperimenti non si importano fra loro, tranne
che da `common` (configurazione e prompt condivisi) e, dentro RQ1, `ablation` da `analyze`.

    common/          configurazione, percorsi, categorie e prompt di generazione
    corpus/          costruzione del corpus di opere (Wikidata + Wikipedia)
    generation/      generazione delle audioguide; confronto chain vs prompt singolo
    rq1_embeddings/  RQ1: embedding, analisi dello shift latente, ablazione del prompt

Ogni script si lancia come modulo dalla root del repository, ad esempio
`python -m museumbot.rq1_embeddings.analyze --model qwen`.
"""
