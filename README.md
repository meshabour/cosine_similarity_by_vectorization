Word similarity engine ,no external NLP libraries, just NumPy.


TF-IDF — term frequency weighted by inverse document frequency, computed manually (no sklearn).
Word Vectors — the TF-IDF matrix is transposed (tf_idf.T), so each row becomes a vector representing one word across all documents, instead of one document across all words.
Cosine Similarity — measures the angle between two word vectors to score how similarly two words behave across the corpus.
recommend(word, top_n) — returns the top_n words most similar to the input word, using a fully vectorized (loop-free) similarity computation via NumPy matrix-vector multiplication.


The stemmer is manual and rule-based, not a real algorithm like Porter or Snowball. It only handles a few suffix patterns (-ing, -ed, -s) with hardcoded length checks. It will mis-stem or under-stem many words (e.g. irregular plurals, words like "was" → "wa" if not filtered as a stop word first).
