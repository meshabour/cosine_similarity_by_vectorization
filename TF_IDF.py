import numpy as np
import string



documents = [
    
    "the cat sat on the mat",
    "the dog sat on the mat",
    "the cat likes the dog",
    "the cat and the dog played on the mat",
    "my cat sleeps on the soft mat",
    "the dog barked at the cat",
    "the small cat chased the dog",
    "the dog and the cat are friends",

    
    "she drank hot coffee in the morning",
    "he likes tea more than coffee",
    "the coffee was too hot to drink",
    "fresh bread and butter for breakfast",
    "she baked warm bread this morning",
    "milk and bread are on the table",
    "he drank cold milk after breakfast",
    "the tea was sweet and warm",

   
    "the sun was hot this morning",
    "it was cold and rainy today",
    "the rain fell all through the night",
    "the weather turned cold and windy",
    "the sun rose over the cold city",
    "heavy rain and strong wind today",

    
    "she read a book about the weather",
    "he studied for the test all night",
    "reading books helps you learn fast",
    "the students studied the book carefully",
    "she likes reading books in the morning",
    "he wrote notes while studying for the test",
]

#data preprocessing 

def lowercase(text):
    return text.lower()

def tokenize(text):
    return text.split()


def remove_punctuation(tokens):
    cleaned = []

    for word in tokens:
        word = word.translate(
            str.maketrans("", "", string.punctuation)
        )

        if word:
            cleaned.append(word)

    return cleaned

stop_words = {
    "the", "is", "are", "a", "an",
    "in", "on", "of", "was", "were",
    "and", "to", "for", "with", "through",
    "this", "that", "at", "all", "more",
    "than", "too", "over", "after", "while"
}


def remove_stop_words(tokens):
    result = []

    for word in tokens:
        if word not in stop_words:
            result.append(word)

    return result



def stem_word(word):

    if word.endswith("ing") and len(word) > 5:
        word = word[:-3]

    elif word.endswith("ed") and len(word) > 4:
        word = word[:-2]

    elif word.endswith("s") and len(word) > 3:
        word = word[:-1]

    return word

def preprocess(text):

    text = lowercase(text)

    tokens = tokenize(text)

    tokens = remove_punctuation(tokens)

    tokens = remove_stop_words(tokens)

    tokens = [stem_word(word) for word in tokens]

    return tokens


processed_documents = [
    preprocess(document)
    for document in documents
]

print(processed_documents)

#TF-IDF

vocabulary = sorted(set(
    word
    for document in processed_documents
    for word in document
))

print("Vocabulary:")
print(vocabulary)




matrix = np.zeros((len(documents), len(vocabulary)))

for i, document in enumerate(documents):

    words = processed_documents[i]

    for j, word in enumerate(vocabulary):
       matrix[i, j] = words.count(word)


print("\nTerm-Document Matrix:")
print(matrix)




def term_frequency(row):

    return row / np.sum(row)


tf = np.zeros_like(matrix, dtype=float)

for i, row in enumerate(matrix):

    tf[i] = term_frequency(row)


print("\nTF:")
print(tf)




def inverse_document_frequency(col):

    corpus_size = len(col)

    doc_count = np.count_nonzero(col > 0)
    if doc_count == 0:
        return 0.0
    return np.log10(corpus_size / doc_count)


idf = np.zeros(len(vocabulary))

for j in range(len(vocabulary)):

    col = matrix[:, j]

    idf[j] = inverse_document_frequency(col)


print("\nIDF:")
print(idf)



tf_idf = tf * idf


print("\nTF-IDF:")
print(tf_idf)

word_vectors = tf_idf.T

print("\nWord Vectors (Transposed TF-IDF):")
print(word_vectors)



def cosine_similarity(v1, v2):
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

def recommend(word, top_n=3):

    word = stem_word(lowercase(word))

    if word not in vocabulary:
        return f"vocabulary does not exist {word}"

    word_index = vocabulary.index(word)
    target_vector = word_vectors[word_index]

    dot_products = word_vectors @ target_vector
    norms = np.linalg.norm(word_vectors, axis=1)
    target_norm = np.linalg.norm(target_vector)

    denom = norms * target_norm
    denom[denom == 0] = 1e-10

    similarities = dot_products / denom
    similarities[word_index] = -1 
    top_indices = np.argsort(similarities)[::-1][:top_n]

    return [(vocabulary[i], similarities[i]) for i in top_indices]


print("\nRecommendations for 'cat':")
print(recommend("cat", top_n=3))

print("\nRecommendations for 'coffee':")
print(recommend("coffee", top_n=3))

print("\nRecommendations for 'rain':")
print(recommend("rain", top_n=3))

print("\nRecommendations for 'book':")
print(recommend("book", top_n=3))