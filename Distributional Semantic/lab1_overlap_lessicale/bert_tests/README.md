L'obiettivo era di questo sub-lab era quello di verificare se tramite BERT sarebbe stato possibile effettuare il confronto tra due definizioni.
All'interno di 'bert_sym' è contenuto il succo del discorso:
- Problema: ho voluto allenare da 0 BERT direttamente su quel mini corpus di frasi
- L'allenamento quindi ha previsto l'utilizzo del MLM (15% di parole mascherate) etc con l'unico obiettivo
  di poter utilizzare in seguito il modello solo per fare Embedding..
- Avendo solo una sessantina di definizioni a disposizione purtroppo neanche aumentando epoche etc, c'è modo di migliorare le 
  performance del modello -> Servono più dati! (mi sono però fermato qua e non ho estratto altri sinonimi o definizioni)

p.s. ho eliminato il modello dalla cartella bert_from_scratch (pesava 400 MB)