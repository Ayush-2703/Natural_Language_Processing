"""
Topic 1.2 -- Assessing Word Vectors using TF-IDF and t-SNE Dimensionality Reduction
CSE468: Natural Language Processing with Deep Learning

Two assessments of vector quality, both using t-SNE purely for *visualisation*
and a proper distance metric (silhouette score) computed in the original
high-dimensional space for the actual *quantitative* assessment:

  Part A -- Document level: represent each of the 500 Brown corpus documents
            two ways -- sparse TF-IDF, and dense mean-pooled Word2Vec -- and
            compare how cleanly each separates the corpus's 15 genres.

  Part B -- Word level: take the 500 most frequent content words from the
            Word2Vec model trained in Topic 1.1 and check, visually and
            quantitatively, whether the embedding space separates words by
            part of speech -- a property nothing in training explicitly asked
            for, but which falls out of words sharing distributional contexts
            with same-category words.

Run directly:
    python implementation.py
"""

import os
import re
import time
import collections

import matplotlib.pyplot as plt
import nltk
import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import brown, gutenberg, movie_reviews
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(HERE, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# Reuse Topic 1.1's cached Word2Vec model (retrains automatically if absent,
# so this script still runs standalone -- see explanation.md).
W2V_PATH = os.path.join(
    HERE, "..", "1.1-Introduction-to-Word-Vectors-and-Analogies", "artifacts", "word2vec_phase1.model"
)


def ensure_nltk_data():
    for pkg, sub in [
        ("brown", "corpora"), ("gutenberg", "corpora"), ("movie_reviews", "corpora"),
        ("punkt", "tokenizers"), ("punkt_tab", "tokenizers"), ("universal_tagset", "taggers"),
    ]:
        try:
            nltk.data.find(f"{sub}/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def clean(tokens):
    return [w.lower() for w in tokens if re.match(r"^[a-zA-Z]+$", w) and len(w) > 1]


def build_corpus_for_training():
    sentences = []
    for fid in brown.fileids():
        sentences.append(clean(brown.words(fid)))
    for fid in gutenberg.fileids():
        for sent in gutenberg.sents(fid):
            c = clean(sent)
            if len(c) > 3:
                sentences.append(c)
    for fid in movie_reviews.fileids():
        sentences.append(clean(movie_reviews.words(fid)))
    return sentences


def load_or_train_word2vec():
    """Identical recipe to Topic 1.1 -- see that topic's explanation.md for why
    each choice (corpus mix, sg=1, caching) was made."""
    if os.path.exists(W2V_PATH):
        print(f"Loading cached Word2Vec model from {W2V_PATH}")
        return Word2Vec.load(W2V_PATH)
    print("No cached model found -- training from scratch (~3 minutes)...")
    ensure_nltk_data()
    sentences = build_corpus_for_training()
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, sg=1, workers=1, epochs=8)
    os.makedirs(os.path.dirname(W2V_PATH), exist_ok=True)
    model.save(W2V_PATH)
    return model


# --------------------------------------------------------------------------
# Part A -- Document-level: TF-IDF vs. mean-pooled Word2Vec, genre clustering
# --------------------------------------------------------------------------
def load_brown_documents():
    texts, labels = [], []
    for fid in brown.fileids():
        texts.append(" ".join(brown.words(fid)))
        labels.append(brown.categories(fid)[0])
    return texts, labels


def tfidf_document_vectors(texts):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", min_df=2)
    X = vectorizer.fit_transform(texts)  # sparse, (n_docs, <=5000)
    return X


def mean_pooled_w2v_document_vectors(texts, model):
    dim = model.wv.vector_size
    X = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        words = [w for w in clean(text.split()) if w in model.wv]
        if words:
            X[i] = np.mean(model.wv[words], axis=0)
    return X


def assess_document_representation(X, labels, name, color_map, use_svd_first=False):
    """t-SNE for the *picture*; silhouette score in the original space for the
    *number*. These must not be confused -- see theory.md section 4."""
    X_for_tsne = X
    if use_svd_first:
        # Standard practice: knock a very high-dimensional sparse matrix down
        # to ~50 dense dimensions with (truncated) SVD before t-SNE, which is
        # both faster and removes a lot of TF-IDF's noisier long-tail features.
        X_for_tsne = TruncatedSVD(n_components=50, random_state=42).fit_transform(X)

    sil = silhouette_score(X, labels, metric="cosine" if use_svd_first else "euclidean")

    tsne = TSNE(n_components=2, perplexity=25, random_state=42, init="pca")
    X_2d = tsne.fit_transform(X_for_tsne)

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=160)
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab20")
    for i, lab in enumerate(unique_labels):
        idx = [j for j, l in enumerate(labels) if l == lab]
        ax.scatter(X_2d[idx, 0], X_2d[idx, 1], s=22, color=cmap(i / len(unique_labels)), label=lab, alpha=0.85)
    ax.set_title(f"{name}  --  t-SNE of Brown documents by genre\n(silhouette in original space: {sil:.3f})")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, color_map), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return sil


# --------------------------------------------------------------------------
# Part B -- Word-level: do Word2Vec vectors cluster by part of speech?
# --------------------------------------------------------------------------
def build_dominant_pos_lookup():
    counts = collections.defaultdict(collections.Counter)
    for word, tag in brown.tagged_words(tagset="universal"):
        counts[word.lower()][tag] += 1
    return {w: c.most_common(1)[0][0] for w, c in counts.items()}


def assess_word_vectors_by_pos(model, n_words=500, skip_top=50):
    pos_lookup = build_dominant_pos_lookup()
    vocab_by_freq = model.wv.index_to_key  # already ordered most- to least-frequent
    chosen = [w for w in vocab_by_freq[skip_top:] if w in pos_lookup][:n_words]

    vectors = np.array([model.wv[w] for w in chosen])
    pos_tags = [pos_lookup[w] for w in chosen]

    sil = silhouette_score(vectors, pos_tags, metric="cosine")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
    X_2d = tsne.fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=160)
    unique_tags = sorted(set(pos_tags))
    cmap = plt.get_cmap("tab10")
    for i, tag in enumerate(unique_tags):
        idx = [j for j, t in enumerate(pos_tags) if t == tag]
        ax.scatter(X_2d[idx, 0], X_2d[idx, 1], s=22, color=cmap(i / len(unique_tags)), label=tag, alpha=0.85)
    ax.set_title(f"Word2Vec vectors (top {n_words} content words) -- t-SNE coloured by POS\n"
                 f"(silhouette in original 100-d space: {sil:.3f})")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "word2vec_tsne_by_pos.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return sil


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ensure_nltk_data()
    model = load_or_train_word2vec()

    print("Loading Brown documents (500 docs, 15 genres)...")
    texts, labels = load_brown_documents()

    print("Building TF-IDF document vectors...")
    X_tfidf = tfidf_document_vectors(texts)
    sil_tfidf = assess_document_representation(
        X_tfidf, labels, "TF-IDF", "tsne_tfidf_by_genre.png", use_svd_first=True
    )

    print("Building mean-pooled Word2Vec document vectors...")
    X_w2v = mean_pooled_w2v_document_vectors(texts, model)
    sil_w2v = assess_document_representation(
        X_w2v, labels, "Mean-pooled Word2Vec", "tsne_word2vec_by_genre.png", use_svd_first=False
    )

    print(f"\nDocument-level genre separation (higher = cleaner clusters):")
    print(f"  TF-IDF              silhouette = {sil_tfidf:.3f}")
    print(f"  Mean-pooled Word2Vec silhouette = {sil_w2v:.3f}")

    print("\nAssessing word-level vectors by part of speech...")
    sil_pos = assess_word_vectors_by_pos(model)
    print(f"  Word2Vec-by-POS silhouette = {sil_pos:.3f}")

    print(f"\nSaved 3 plots to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
