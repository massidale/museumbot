# Scaletta — presentazione breve dello stato della tesi (5 minuti)

*Bozza del 27 ottobre 2026. Numeri da `docs/report-stato-progetto.md` (parte 1, DeepSeek V4
Flash) e `docs/report-gemma.md` (parte 2, Gemma 4 31B), embedding Qwen3. 6 slide, circa
50 secondi l'una.*

---

**1. Domanda e setup** (30 s)
- Un LLM che scrive un'audioguida per una categoria di visitatore di Falk cambia davvero il
  testo? E serve a qualcuno? RQ1 embedding · RQ2 giudice LLM · RQ3 visitatori reali.
- 100 dipinti, 5 categorie + `flat` (nessuna categoria). Ogni testo diventa un vettore;
  lo shift di una categoria è la sua distanza dal flat della stessa opera.

**2. RQ1: lo shift esiste** (1 min)
- Figura: `figures/pca_qwen.png`
- Un probe lineare riconosce la categoria su opere mai viste nel **95%** dei casi (caso 17%).
- Rigenerando tutti i testi, la direzione di ogni categoria torna uguale (coseno 0.96–0.99).

**3. RQ1: dove di più, dove di meno** (1 min)
- **Forte:** Recharger e Professional/Hobbyist, lontani da tutto.
- **Debole:** Explorer e Facilitator, vicini al flat e fra loro.
- Lo guida lo **stile** del prompt; il solo nome della categoria ne conserva circa il 70%.
- Nel testo: il Recharger ripete le stesse formule il doppio del flat (*let, quiet,
  breathe*).

**4. RQ2 e testi misti** (1 min)
- Giudice LLM con persona: sceglie il testo della propria categoria contro il flat nel
  **97%** dei casi; con la persona sbagliata il testo personalizzato perde.
- Parte 2, product of experts: con Gemma si generano testi *fra* due categorie con un peso
  α. Negli embedding si spostano davvero fra le due, e il giudice vede il gradiente.
- Figura: `figures/mixed_qwen_recharger+professional_hobbyist.png`

**5. RQ3: visitatori reali** (1 min)
- Il giudice LLM simula un visitatore: va verificato che ciò che produce l'LLM sia ciò che
  vogliono le persone.
- **Ideale: una visita al museo.** Le categorie di Falk sono motivazioni della visita; in
  una simulazione, a "perché sei qui?" la risposta vera è "perché me l'hai chiesto tu".
- Questionario: il visitatore sceglie la coppia di categorie in cui si riconosce e quanto
  (solo A / più A / uguale / più B / solo B, cioè α = 0…1). Poi, davanti all'opera, due
  audioguide: la sua (pura, o mista al suo α) contro flat o un'altra categoria.
- Validazione sulla parte 1; sulla parte 2 se possibile.

**6. Sintesi** (20 s)
- Lo shift c'è ed è specifico per categoria; un giudice LLM lo riconosce e lo preferisce.
  Ora serve sapere se lo preferiscono anche i visitatori.
