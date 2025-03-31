## Exam Project: Information Retrieval on the ANTIQUE Dataset

This project was developed for the Information Retrieval exam (academic year 2023-2024). The main objective was to implement and evaluate various Information Retrieval techniques using the **ANTIQUE (Answer Trait Identification for Question Answering)** dataset, a collection of non-factoid questions and answers from Yahoo! Webscope L6. This dataset is particularly interesting because it contains open-ended questions requiring complex answers, such as descriptions, opinions, or explanations, making it a significant testbed for IR systems.

The project followed a structured pipeline:

1.  **Text Preprocessing:**
    * Normalization of the raw text through:
        * Removal of multiple whitespaces.
        * Conversion to lowercase.
        * Removal of numbers, symbols, punctuation, and emojis.
    * Text tokenization.
    * Removal of common English stop words using the NLTK library.

2.  **Indexing:**
    * Creation of an inverted index from the pre-processed corpus using the **PyTerrier** library. This index allows for efficient document retrieval.

3.  **Development of Basic Search Engine (Count-Based):**
    * Implementation of classic term-frequency-based ranking models:
        * **TF-IDF**
        * **BM25**
    * Use of PyTerrier to perform searches and retrieve relevant documents for queries.

4.  **Development of Advanced Search Engine (Neural Re-ranking):**
    * Application of pre-trained neural models to further improve the results obtained by BM25 (considered as the first retrieval stage). The models used include:
        * **KNRM** (Kernel-based Neural Ranking Model) with FastText embeddings ('wordvec_hash').
        * **Vanilla BERT** (a pre-trained model not specifically fine-tuned on the ANTIQUE dataset).
        * **Cross-Encoder** (based on 'cross-encoder/stsb-roberta-base').
        * **Bi-Encoder** (based on 'all-MiniLM-L6-v2').

5.  **Evaluation and Comparison:**
    * Rigorous performance evaluation of the different approaches using standard IR metrics such as **MAP (Mean Average Precision) / AP@100, nDCG (Normalized Discounted Cumulative Gain), P@5 (Precision at 5), and P@10 (Precision at 10)**.
    * Systematic comparison between the baseline models (TF-IDF vs BM25) and between BM25 and its variants with neural re-ranking (BM25 vs KNRM, BM25 vs BERT), including weighted combinations of scores.

### Technologies Used

* **Language:** Python
* **Main Libraries:**
    * **PyTerrier:** Main framework for indexing, retrieval (TF-IDF, BM25), experiment pipelines, and evaluation.
    * **NLTK:** For text preprocessing operations (tokenization, stopword removal).
    * **OpenNIR (onir_pt):** For implementing and applying KNRM and Vanilla BERT neural re-rankers.
    * **Sentence-Transformers:** For using pre-trained Cross-Encoder and Bi-Encoder models.
    * **Pandas & NumPy:** Data manipulation (often used internally by PyTerrier).
    * **FastText:** Pre-trained embeddings used by KNRM.

### Results Obtained

* **BM25 vs TF-IDF:** BM25 showed slightly superior performance compared to TF-IDF on the ANTIQUE dataset, especially in the top-ranked results (P@5).
* **Re-ranking with KNRM:** Directly applying KNRM (with generic embeddings) as a re-ranker worsened performance compared to BM25 alone, suggesting the need for specific fine-tuning or more suitable embeddings. Weighted combinations also showed that a higher weight for KNRM did not yield benefits.
* **Re-ranking with Vanilla BERT (non-tuned):** Using a pre-trained BERT model not fine-tuned on the dataset produced mixed results, improving nDCG but not MAP or P@10 compared to BM25. This highlights the importance of domain-specific tuning.
* **Combination BM25 + BERT-based Models:** Combining the normalized scores of BM25 with those of BERT-based neural models (Vanilla BERT, Bi-Encoder, Cross-Encoder) via a weighted sum (e.g., 50/50) generally led to **significant improvements** over BM25 alone, particularly for MAP, nDCG, and P@10. The 0.5*VBERT + 0.5*BM25 combination was the top performer in the final comparison.
