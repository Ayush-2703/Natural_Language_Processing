"""
Topic 3.1 -- Introduction to Word Embedding (CBOW and Skip-Gram)
CSE468: Natural Language Processing with Deep Learning

Builds and caches a Word2Vec-ready corpus and vocabulary (reused by Topics
3.2 and 3.3), then implements both of Mikolov et al. (2013)'s architectures
with a FULL softmax output layer:

  CBOW       -- average several context words' embeddings, predict the
                single centre word they surround.
  Skip-gram  -- take the single centre word's embedding, predict each
                context word independently.

The vocabulary is deliberately capped small (3,000 words) specifically so a
full softmax is computationally tractable here -- this cap, and the cost of
removing it, is the entire motivation for Topic 3.2's hierarchical softmax
and negative sampling.

Run directly:
    python implementation.py
"""

import math
import os
import pickle
import random
import re
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(HERE, "images")
ARTIFACT_DIR = os.path.join(HERE, "artifacts")
DATASET_PATH = os.path.join(ARTIFACT_DIR, "word2vec_dataset.pkl")
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

VOCAB_SIZE = 3000   # deliberately small -- see theory.md section 4
WINDOW = 2
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(1)


# --------------------------------------------------------------------------
# 1. Shared corpus + vocabulary, cached for Topics 3.2 and 3.3
# --------------------------------------------------------------------------
def clean_sentence(words):
    return [w.lower() for w in words if re.match(r"^[a-zA-Z]+$", w) and len(w) > 1]


def build_and_cache_dataset():
    import nltk
    for pkg, sub in [("brown", "corpora"), ("gutenberg", "corpora"), ("movie_reviews", "corpora")]:
        try:
            nltk.data.find(f"{sub}/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    from nltk.corpus import brown, gutenberg, movie_reviews

    raw_sentences = []
    for fid in brown.fileids():
        raw_sentences.append(clean_sentence(brown.words(fid)))
    for fid in gutenberg.fileids():
        for sent in gutenberg.sents(fid):
            c = clean_sentence(sent)
            if len(c) > 3:
                raw_sentences.append(c)
    for fid in movie_reviews.fileids():
        raw_sentences.append(clean_sentence(movie_reviews.words(fid)))

    counts = Counter(w for s in raw_sentences for w in s)
    vocab = [w for w, _ in counts.most_common(VOCAB_SIZE)]
    word2idx = {w: i for i, w in enumerate(vocab)}

    # Standard word2vec preprocessing: DROP out-of-vocabulary words entirely
    # (rather than mapping them to <unk>, as Phase 2's language models did)
    # -- predicting a meaningless catch-all token is not useful here, and
    # word2vec has enough remaining data that simply discarding rare words
    # is the standard, defensible choice. See explanation.md for the
    # consequence this has on how the context window is defined.
    filtered_sentences = []
    for s in raw_sentences:
        f = [word2idx[w] for w in s if w in word2idx]
        if len(f) >= 2:
            filtered_sentences.append(f)

    dataset = {
        "vocab": vocab,
        "word2idx": word2idx,
        "filtered_sentences": filtered_sentences,
        "raw_word_counts": counts,
    }
    with open(DATASET_PATH, "wb") as f:
        pickle.dump(dataset, f)
    return dataset


def load_or_build_dataset():
    if os.path.exists(DATASET_PATH):
        print(f"Loading cached Word2Vec dataset from {DATASET_PATH}")
        with open(DATASET_PATH, "rb") as f:
            return pickle.load(f)
    print("No cached dataset found -- building from Brown + Gutenberg + Movie Reviews...")
    return build_and_cache_dataset()


# --------------------------------------------------------------------------
# 2. Training pair generation
# --------------------------------------------------------------------------
def generate_skipgram_pairs(sentences, window, max_pairs=None, seed=SEED):
    pairs = []
    for s in sentences:
        for i, center in enumerate(s):
            for j in range(max(0, i - window), min(len(s), i + window + 1)):
                if j != i:
                    pairs.append((center, s[j]))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if max_pairs and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
    centers = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    contexts = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    return centers, contexts


def generate_cbow_examples(sentences, window, max_examples=None, seed=SEED):
    """Each example: a fixed-width context window (padded with -1 for
    missing positions at sentence edges) and the centre word to predict."""
    width = 2 * window
    examples = []
    for s in sentences:
        for i, center in enumerate(s):
            ctx = []
            for j in range(i - window, i + window + 1):
                if j == i:
                    continue
                ctx.append(s[j] if 0 <= j < len(s) else -1)
            examples.append((ctx, center))
    rng = random.Random(seed)
    rng.shuffle(examples)
    if max_examples and len(examples) > max_examples:
        examples = examples[:max_examples]
    contexts = torch.tensor([e[0] for e in examples], dtype=torch.long)  # (N, width), -1 = padding
    centers = torch.tensor([e[1] for e in examples], dtype=torch.long)
    return contexts, centers


# --------------------------------------------------------------------------
# 3. Models -- full softmax over the (small) vocabulary
# --------------------------------------------------------------------------
class SkipGramModel(nn.Module):
    """centre word -> predict one context word, full softmax."""

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)   # "input"/centre-word vectors
        self.out_proj = nn.Linear(embed_dim, vocab_size)       # "output"/context-word vectors

    def forward(self, center_idx):
        e = self.in_embed(center_idx)
        return self.out_proj(e)  # logits over the vocabulary


class CBOWModel(nn.Module):
    """averaged context words -> predict the single centre word, full softmax."""

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=vocab_size)  # +1 = padding slot
        self.out_proj = nn.Linear(embed_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, context_idx):
        # context_idx: (batch, width), with -1 marking "no word at this
        # position" (sentence-boundary padding) -- remapped to the
        # dedicated, always-zero padding embedding row before averaging.
        safe_idx = context_idx.clone()
        safe_idx[safe_idx == -1] = self.vocab_size
        e = self.in_embed(safe_idx)                     # (batch, width, embed_dim)
        mask = (context_idx != -1).unsqueeze(-1).float()  # (batch, width, 1)
        summed = (e * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        averaged = summed / counts
        return self.out_proj(averaged)


def train_model(model, inputs, targets, epochs, batch_size, lr=2e-3, label=""):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=True)
    history = []
    for epoch in range(epochs):
        total_loss, n_batches = 0.0, 0
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches
        history.append(avg_loss)
        print(f"  [{label}] epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")
    return history


# --------------------------------------------------------------------------
# 4. Evaluation: nearest neighbours + a small held-out NLL comparison
# --------------------------------------------------------------------------
def nearest_neighbors(embeddings, word2idx, idx2word, word, topn=6):
    vecs = F.normalize(embeddings, dim=1)
    q = vecs[word2idx[word]]
    sims = vecs @ q
    sims[word2idx[word]] = -1.0
    top = torch.topk(sims, topn)
    return [(idx2word[i.item()], round(s.item(), 3)) for s, i in zip(top.values, top.indices)]


# --------------------------------------------------------------------------
# Visualisations
# --------------------------------------------------------------------------
def plot_architecture_comparison(path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), dpi=170)
    fig.patch.set_facecolor("white")

    # CBOW
    ax = axes[0]
    ax.axis("off")
    ctx_words = ["the", "sat", "on", "mat"]
    for i, w in enumerate(ctx_words):
        x = 0.5 + i
        ax.scatter([x], [2.6], s=0)
        ax.text(x, 2.6, w, ha="center", fontsize=10, fontweight="bold",
                 bbox=dict(boxstyle="round", facecolor="#4C72B0", alpha=0.2, edgecolor="#4C72B0"))
        ax.annotate("", xy=(2.5, 1.4), xytext=(x, 2.35),
                     arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.3))
    ax.text(2.5, 1.4, "average", ha="center", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="#C9A227", alpha=0.25, edgecolor="#C9A227"))
    ax.annotate("", xy=(2.5, 0.4), xytext=(2.5, 1.15),
                 arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.5))
    ax.text(2.5, 0.4, '"cat" (predicted)', ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor="#C0504D", alpha=0.2, edgecolor="#C0504D"))
    ax.set_xlim(0, 5); ax.set_ylim(0, 3.2)
    ax.set_title("CBOW: context words -> predict centre word")

    # Skip-gram
    ax = axes[1]
    ax.axis("off")
    ax.text(2.5, 2.6, '"cat" (input)', ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor="#C0504D", alpha=0.2, edgecolor="#C0504D"))
    for i, w in enumerate(ctx_words):
        x = 0.5 + i
        ax.annotate("", xy=(x, 1.15), xytext=(2.5, 2.35),
                     arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.3))
        ax.text(x, 0.8, w, ha="center", fontsize=10, fontweight="bold",
                 bbox=dict(boxstyle="round", facecolor="#4C72B0", alpha=0.2, edgecolor="#4C72B0"))
    ax.set_xlim(0, 5); ax.set_ylim(0, 3.2)
    ax.set_title("Skip-gram: centre word -> predict each context word")

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_loss_curves(cbow_hist, sg_hist, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=160)
    ax.plot(cbow_hist, label="CBOW", color="#4C72B0", marker="o")
    ax.plot(sg_hist, label="Skip-gram", color="#C0504D", marker="o")
    ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy loss")
    ax.set_title("CBOW vs. Skip-gram training loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_rare_vs_frequent(cbow_emb, sg_emb, word2idx, raw_word_counts, vocab, path):
    """Skip-gram is reputed to do better on rare words than CBOW (Mikolov et
    al., 2013) -- check this directly by comparing each model's nearest-
    neighbour self-consistency (mean similarity to its own top-5 neighbours)
    split by word frequency band."""
    freqs = np.array([raw_word_counts[w] for w in vocab])
    order = np.argsort(-freqs)
    frequent_band = order[:300]
    rare_band = order[-300:]

    def mean_topk_sim(embeddings, indices, k=5):
        vecs = F.normalize(embeddings, dim=1)
        sims = vecs[indices] @ vecs.T
        for row, idx in enumerate(indices):
            sims[row, idx] = -1.0
        topk = torch.topk(sims, k, dim=1).values
        return topk.mean().item()

    rows = []
    for name, emb in [("CBOW", cbow_emb), ("Skip-gram", sg_emb)]:
        rows.append((name, "frequent (top 300)", mean_topk_sim(emb, torch.tensor(frequent_band))))
        rows.append((name, "rare (bottom 300)", mean_topk_sim(emb, torch.tensor(rare_band))))

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=160)
    x = np.arange(2)
    width = 0.35
    cbow_vals = [r[2] for r in rows if r[0] == "CBOW"]
    sg_vals = [r[2] for r in rows if r[0] == "Skip-gram"]
    ax.bar(x - width/2, cbow_vals, width, label="CBOW", color="#4C72B0")
    ax.bar(x + width/2, sg_vals, width, label="Skip-gram", color="#C0504D")
    ax.set_xticks(x); ax.set_xticklabels(["frequent words", "rare words"])
    ax.set_ylabel("mean cosine similarity to own top-5 neighbours")
    ax.set_title("Neighbourhood tightness: frequent vs. rare words")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return rows


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    dataset = load_or_build_dataset()
    vocab, word2idx = dataset["vocab"], dataset["word2idx"]
    idx2word = {i: w for w, i in word2idx.items()}
    sentences = dataset["filtered_sentences"]
    V = len(vocab)
    n_tokens = sum(len(s) for s in sentences)
    print(f"Vocabulary size: {V}   Sentences (after filtering): {len(sentences):,}   Tokens: {n_tokens:,}")

    MAX_PAIRS = 320_000
    sg_centers, sg_contexts = generate_skipgram_pairs(sentences, WINDOW, max_pairs=MAX_PAIRS)
    cbow_contexts, cbow_centers = generate_cbow_examples(sentences, WINDOW, max_examples=MAX_PAIRS)
    print(f"Skip-gram training pairs: {len(sg_centers):,}")
    print(f"CBOW training examples:   {len(cbow_centers):,}")

    EMBED_DIM = 50
    EPOCHS = 6
    BATCH = 1024

    print("\n=== Training Skip-gram ===")
    sg_model = SkipGramModel(V, EMBED_DIM)
    t0 = time.time()
    sg_history = train_model(sg_model, sg_centers, sg_contexts, EPOCHS, BATCH, label="skip-gram")
    sg_time = time.time() - t0

    print("\n=== Training CBOW ===")
    cbow_model = CBOWModel(V, EMBED_DIM)
    t0 = time.time()
    cbow_history = train_model(cbow_model, cbow_contexts, cbow_centers, EPOCHS, BATCH, label="cbow")
    cbow_time = time.time() - t0

    print(f"\nWall-clock training time: CBOW = {cbow_time:.1f}s   Skip-gram = {sg_time:.1f}s")

    sg_emb = sg_model.in_embed.weight.data.clone()
    cbow_emb = cbow_model.in_embed.weight.data[:V].clone()  # drop the padding row

    print("\n=== Nearest neighbours ===")
    for w in ["good", "money", "king", "water"]:
        if w in word2idx:
            print(f"  CBOW       nearest to {w!r}: {nearest_neighbors(cbow_emb, word2idx, idx2word, w)}")
            print(f"  Skip-gram  nearest to {w!r}: {nearest_neighbors(sg_emb, word2idx, idx2word, w)}")

    plot_architecture_comparison(os.path.join(IMAGE_DIR, "cbow_vs_skipgram_architecture.png"))
    plot_loss_curves(cbow_history, sg_history, os.path.join(IMAGE_DIR, "loss_curves.png"))
    freq_results = plot_rare_vs_frequent(cbow_emb, sg_emb, word2idx, dataset["raw_word_counts"], vocab,
                                          os.path.join(IMAGE_DIR, "rare_vs_frequent.png"))
    print("\n=== Rare vs. frequent word neighbourhood tightness ===")
    for name, band, val in freq_results:
        print(f"  {name:10s} {band:20s}: {val:.3f}")

    print(f"\nSaved plots to {IMAGE_DIR}")
    print(f"Saved shared dataset to {DATASET_PATH} for Topics 3.2 and 3.3 to reuse")


if __name__ == "__main__":
    main()
