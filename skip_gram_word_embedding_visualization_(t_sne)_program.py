# Skip-Gram Word Embedding Visualization (t-SNE)

# Install and Import Required Libraries
import gensim
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# -------- NEW CORPUS --------
sentences = [
    "Technology is evolving rapidly with advancements in artificial intelligence.",
    "Healthcare systems rely on accurate data for patient diagnosis.",
    "Sports analytics uses data to improve player performance and strategy.",
    "Artificial intelligence helps automate complex decision making.",
    "Doctors analyze medical data to detect early symptoms of diseases.",
    "Football teams use machine learning for performance prediction.",
    "Fitness tracking devices collect real time health data.",
    "Deep learning models are transforming image recognition in healthcare.",
    "Data analytics is becoming essential in modern businesses.",
    "Athletes benefit from personalized training programs based on data insights."
]

# Tokenize
tokenized_sentences = [sentence.lower().split() for sentence in sentences]

# Train Skip-Gram Model
skipgram_model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,
    window=3,
    min_count=1,
    sg=1,
    epochs=100
)

# Inspect Word Vectors
word = "data"
vector = skipgram_model.wv[word]
print(f"Vector size for '{word}':", len(vector))

# Similarity
similarity_score = skipgram_model.wv.similarity("data", "health")
print("Similarity between 'data' and 'health':", similarity_score)

# Most Similar Words
similar_words = skipgram_model.wv.most_similar("data", topn=5)
print("Words similar to 'data':")
for word, score in similar_words:
    print(word, ":", score)

# ---- UPDATED PAIRS ----
pairs = [
    ("data", "health"),
    ("machine", "learning"),
    ("sports", "analytics")
]

scores = [skipgram_model.wv.similarity(w1, w2) for w1, w2 in pairs]
print("Average Similarity Score:", np.mean(scores))

# Visualization using t-SNE
words = list(skipgram_model.wv.index_to_key)
vectors = np.array([skipgram_model.wv[word] for word in words])

tsne = TSNE(n_components=2, random_state=42, perplexity=5)
reduced_vectors = tsne.fit_transform(vectors)

# Plot Word Embeddings
plt.figure(figsize=(8, 6))

for i, word in enumerate(words):
    plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
    plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]))

plt.title("Skip-Gram Word Embedding Visualization (t-SNE)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.show()

