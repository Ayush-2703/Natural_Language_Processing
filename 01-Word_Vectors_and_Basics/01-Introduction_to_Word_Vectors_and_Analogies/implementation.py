"""
Topic 1.1 -- Introduction to Word Vectors and Word Analogy
CSE468: Natural Language Processing with Deep Learning

Trains (or loads a cached copy of) Word2Vec embeddings on a multi-million-
token corpus assembled from NLTK's Brown, Gutenberg, and Movie Reviews
corpora, then explores the resulting vector space using explicit PyTorch
tensor operations: cosine similarity, nearest-neighbour search, and analogy
solving via vector arithmetic (vec(b) - vec(a) + vec(c)).

Gensim is used purely to *obtain* the vectors quickly (exactly how you would
in production -- nobody reimplements Word2Vec's training loop just to
explore a vector space). Phase 3 of this course dissects Word2Vec's training
mechanics -- CBOW vs. Skip-gram, hierarchical softmax, negative sampling --
from scratch in both NumPy and TensorFlow.

Run directly:
    python implementation.py
"""

import os
import re
import time

import matplotlib.pyplot as plt
import nltk
import numpy as np
import torch
import torch.nn.functional as F
from gensim.models import Word2Vec
from nltk.corpus import brown, gutenberg, movie_reviews

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(HERE, "artifacts")
IMAGE_DIR = os.path.join(HERE, "images")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "word2vec_phase1.model")

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# 1. Corpus assembly
# --------------------------------------------------------------------------
def ensure_nltk_data():
    for pkg in ["brown", "gutenberg", "movie_reviews", "punkt", "punkt_tab"]:
        try:
            nltk.data.find(
                f"corpora/{pkg}" if pkg not in ("punkt", "punkt_tab") else f"tokenizers/{pkg}"
            )
        except LookupError:
            nltk.download(pkg, quiet=True)


def clean(tokens):
    """Lowercase, alphabetic-only tokens; drop single characters."""
    return [w.lower() for w in tokens if re.match(r"^[a-zA-Z]+$", w) and len(w) > 1]


def build_corpus():
    """
    Assemble pseudo-sentences from three NLTK corpora (~4M tokens total).
    Brown and Movie Reviews are chunked per-document; Gutenberg is chunked
    at the sentence level since each of its 18 files is a full book and
    Word2Vec's context window should not cross sentence boundaries.
    """
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


# --------------------------------------------------------------------------
# 2. Word2Vec training, cached to disk so Topics 1.2-1.4 can reuse it
# --------------------------------------------------------------------------
def load_or_train_word2vec():
    if os.path.exists(MODEL_PATH):
        print(f"Loading cached Word2Vec model from {MODEL_PATH}")
        return Word2Vec.load(MODEL_PATH)

    print(
        "No cached model found -- assembling corpus and training Word2Vec "
        "(skip-gram, 100-d, ~3 minutes on a single core)..."
    )
    ensure_nltk_data()

    t0 = time.time()
    sentences = build_corpus()
    n_tokens = sum(len(s) for s in sentences)
    print(f"  corpus: {len(sentences):,} pseudo-sentences, {n_tokens:,} tokens ({time.time()-t0:.1f}s)")

    t1 = time.time()
    model = Word2Vec(
        sentences,
        vector_size=100,
        window=5,
        min_count=5,
        sg=1,  # skip-gram; Phase 3 builds this and CBOW from scratch
        workers=1,
        epochs=8,
    )
    print(f"  trained {len(model.wv):,} word vectors in {time.time()-t1:.1f}s")
    model.save(MODEL_PATH)
    return model


# --------------------------------------------------------------------------
# 3. Tensor-based vector space operations (PyTorch)
# --------------------------------------------------------------------------
class WordVectorSpace:
    """
    Wraps a gensim KeyedVectors object in a single PyTorch tensor so that
    similarity search and analogy solving are explicit tensor operations --
    a matrix-vector product plus a top-k -- rather than a library black box.
    """

    def __init__(self, keyed_vectors):
        self.itos = keyed_vectors.index_to_key  # int index -> string
        self.stoi = {w: i for i, w in enumerate(self.itos)}  # string -> int index
        raw = torch.tensor(keyed_vectors.vectors, dtype=torch.float32)
        # L2-normalise once: every later dot product IS the cosine similarity.
        self.vectors = F.normalize(raw, dim=1)

    def __contains__(self, word):
        return word in self.stoi

    def vec(self, word):
        return self.vectors[self.stoi[word]]

    def nearest(self, query_vec, exclude=(), topn=8):
        """Cosine similarity of a query vector against the whole vocabulary
        via one matrix-vector product, then top-k."""
        q = F.normalize(query_vec.unsqueeze(0), dim=1)
        sims = (self.vectors @ q.T).squeeze(1)  # shape: (vocab_size,)
        for w in exclude:
            if w in self.stoi:
                sims[self.stoi[w]] = -1.0
        top_sims, top_idx = torch.topk(sims, min(topn, sims.shape[0]))
        return [(self.itos[i], top_sims[j].item()) for j, i in enumerate(top_idx.tolist())]

    def similarity(self, w1, w2):
        return torch.dot(self.vec(w1), self.vec(w2)).item()

    def analogy(self, a, b, c, topn=5):
        """Solve a:b :: c:? via vector arithmetic vec(b) - vec(a) + vec(c)."""
        query = self.vec(b) - self.vec(a) + self.vec(c)
        return self.nearest(query, exclude={a, b, c}, topn=topn)


# --------------------------------------------------------------------------
# 4. Evaluation: a small curated analogy test set
# --------------------------------------------------------------------------
# (a, b, c, expected) meaning a:b :: c:expected
ANALOGY_TEST_SET = [
    ("boy", "boys", "girl", "girls"),
    ("good", "better", "bad", "worse"),
    ("day", "days", "night", "nights"),
    ("man", "woman", "king", "queen"),
    ("paris", "france", "london", "england"),
    ("slow", "slower", "fast", "faster"),
    ("write", "writer", "act", "actor"),
]


def evaluate_analogies(space):
    correct_at_1, correct_at_5 = 0, 0
    rows = []
    for a, b, c, expected in ANALOGY_TEST_SET:
        if not all(w in space for w in (a, b, c, expected)):
            rows.append((a, b, c, expected, "OOV (word missing from vocab)", False, False))
            continue
        preds = space.analogy(a, b, c, topn=5)
        pred_words = [w for w, _ in preds]
        hit1 = pred_words[0] == expected
        hit5 = expected in pred_words
        correct_at_1 += hit1
        correct_at_5 += hit5
        rows.append((a, b, c, expected, pred_words, hit1, hit5))

    n = len(ANALOGY_TEST_SET)
    print(f"\nAnalogy accuracy on this {n}-item curated set: top-1 = {correct_at_1}/{n}, top-5 = {correct_at_5}/{n}\n")
    for a, b, c, expected, preds, hit1, hit5 in rows:
        mark = "[hit]" if hit1 else ("[top5]" if hit5 else "[miss]")
        print(f"  {mark:7s} {a}:{b} :: {c}:{expected}  ->  {preds}")
    return correct_at_1 / n, correct_at_5 / n


# --------------------------------------------------------------------------
# 5. Visualisations
# --------------------------------------------------------------------------
def plot_similarity_heatmap(space, words, path):
    words = [w for w in words if w in space]
    n = len(words)
    mat = np.zeros((n, n))
    for i, w1 in enumerate(words):
        for j, w2 in enumerate(words):
            mat[i, j] = space.similarity(w1, w2)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=160)
    im = ax.imshow(mat, cmap="RdYlBu_r", vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_xticklabels(words, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(words)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")
    ax.set_title("Cosine similarity between word vectors")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_analogy_bars(space, a, b, c, expected, path):
    preds = space.analogy(a, b, c, topn=6)
    words_ = [w for w, _ in preds]
    sims = [s for _, s in preds]
    colors = ["#2E7D32" if w == expected else "#4C72B0" for w in words_]

    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=160)
    y = list(range(len(words_)))[::-1]
    ax.barh(y, sims, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(words_)
    ax.set_xlabel(f"cosine similarity to  vec({b}) - vec({a}) + vec({c})")
    ax.set_title(f"Analogy: {a}:{b} :: {c}:?  (expected: {expected})")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------
# 6. Main
# --------------------------------------------------------------------------
def main():
    model = load_or_train_word2vec()
    space = WordVectorSpace(model.wv)

    print("\nNearest neighbours of 'money':", space.nearest(space.vec("money"), exclude={"money"}, topn=6))
    print("Nearest neighbours of 'king': ", space.nearest(space.vec("king"), exclude={"king"}, topn=6))

    evaluate_analogies(space)

    heatmap_words = ["king", "queen", "man", "woman", "money", "cash", "good", "bad", "happy", "sad"]
    plot_similarity_heatmap(space, heatmap_words, os.path.join(IMAGE_DIR, "similarity_heatmap.png"))
    plot_analogy_bars(space, "man", "woman", "king", "queen", os.path.join(IMAGE_DIR, "analogy_king_queen.png"))

    print(f"\nSaved plots to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
