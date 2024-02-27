# Information-Retrieval-Progetto-esame

Il progetto di Information Retrieval (IR) descritto nel notebook `notebook_progetto_antique.ipynb` mira a sviluppare un motore di ricerca in grado di restituire i documenti più rilevanti rispetto a una query inserita, utilizzando il dataset "Antique" (una raccolta di domande e risposte di Yahoo). Il progetto prevede diverse fasi: pre-processing del dataset, sviluppo del motore di ricerca, e infine il re-ranking dei documenti più rilevanti utilizzando i modelli KNRM e BERT.

## Tecnologie e Passaggi Utilizzati

1. **Pre-processing del Dataset**: 
   - Utilizzo di `nltk` per il tokenizing, la rimozione delle stopwords e lo stemming.
   - Pulizia del testo rimuovendo spazi bianchi, numeri, simboli, punteggiature ed emoji.

2. **Indexing**:
   - Creazione di un indice utilizzando `PyTerrier`, una libreria Python per esperimenti di IR basata su Terrier IR platform.

3. **Sviluppo del Motore di Ricerca**:
   - Implementazione di modelli di ranking come TF-IDF e BM25 per recuperare i documenti in base alla loro rilevanza rispetto alla query.

4. **Re-ranking con KNRM e BERT**:
   - Utilizzo di modelli pre-addestrati per il re-ranking dei documenti. KNRM (Kernelized Neural Ranking Model) e BERT (Bidirectional Encoder Representations from Transformers) vengono impiegati per migliorare la rilevanza dei documenti restituiti.

5. **Valutazione**:
   - Misurazione delle prestazioni del motore di ricerca e dei modelli di re-ranking attraverso metriche standard di IR come MAP (Mean Average Precision), Precision@k e NDCG (Normalized Discounted Cumulative Gain).

6. **Tecnologie Utilizzate**:
   - `Python` come linguaggio di programmazione.
   - `PyTerrier` per l'indexing, il retrieval e il re-ranking.
   - `nltk` per il pre-processing del testo.
   - Modelli pre-addestrati come `BERT` e `KNRM` per il re-ranking.

Questo progetto dimostra l'efficacia dell'uso di modelli di deep learning nel migliorare la rilevanza dei documenti restituiti da un motore di ricerca, combinando tecniche classiche di IR con approcci basati su modelli pre-addestrati.

