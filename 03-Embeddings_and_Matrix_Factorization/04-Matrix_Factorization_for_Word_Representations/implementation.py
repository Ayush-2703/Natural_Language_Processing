"""
Topic 3.4 -- Matrix Factorization: Introduction, Training, Models, and Regularization
CSE468: Natural Language Processing with Deep Learning

The central idea of this Phase shifts here from "predict the next word
with a neural network" to "decompose a co-occurrence matrix." These two
approaches turn out to produce equivalent vector spaces (Topic 3.5 explains
why formally) -- but they illuminate different aspects of the same problem.

This topic:

  1. Builds a weighted co-occurrence matrix C from the cached corpus, where
     C[w,c] = sum of 1/distance over all (word, context) windows.
  2. Computes PPMI (Positive Pointwise Mutual Information) from C -- the
     standard reweighting that makes the matrix useful for factorization.
  3. Factorizes PPMI via truncated SVD -- fast, exact, non-iterative.
  4. Factorizes C directly via Alternating Least Squares (ALS) -- an
     iterative factorization that reveals the optimization structure shared
     with neural Word2Vec: two matrix factors (word and context embeddings),
     a loss, and a regularization term.
  5. Compares all three sets of embeddings qualitatively and via an analogy-
     accuracy evaluation.

Run directly:
    python implementation.py
"""

import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(HERE, "images")
ARTIFACT_DIR = os.path.join(HERE, "artifacts")
COOC_PATH = os.path.join(ARTIFACT_DIR, "cooc_matrix.npy")
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

DATASET_PATH = os.path.join(
    HERE, "..", "3.1-Word-Embeddings-CBOW-and-SkipGram", "artifacts", "word2vec_dataset.pkl"
)
SEED = 42
np.random.seed(SEED)
WINDOW = 4


def load_or_build_dataset():
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "rb") as f:
            return pickle.load(f)
    sys.path.insert(0, os.path.join(HERE, "..", "3.1-Word-Embeddings-CBOW-and-SkipGram"))
    from implementation import build_and_cache_dataset
    return build_and_cache_dataset()


# --------------------------------------------------------------------------
# 1. Co-occurrence matrix
# --------------------------------------------------------------------------
def build_cooccurrence_matrix(sentences, V, window):
    """Harmonic-weighted co-occurrence: pair (w, c) at distance d contributes
    1/d (not 1), so near contexts count more than distant ones. This is a
    common refinement that often produces slightly better embeddings than
    uniform window weighting."""
    cooc = np.zeros((V, V), dtype=np.float32)
    for s in sentences:
        for i, w in enumerate(s):
            for j in range(max(0, i - window), min(len(s), i + window + 1)):
                if j != i:
                    cooc[w, s[j]] += 1.0 / abs(i - j)
    return cooc


def load_or_build_cooc(sentences, V, window):
    if os.path.exists(COOC_PATH):
        print(f"Loading cached co-occurrence matrix from {COOC_PATH}")
        return np.load(COOC_PATH)
    print("Building co-occurrence matrix (window=4, harmonic weighting)...")
    t0 = time.time()
    cooc = build_cooccurrence_matrix(sentences, V, window)
    print(f"  built in {time.time()-t0:.1f}s  ({np.count_nonzero(cooc):,} non-zero entries)")
    np.save(COOC_PATH, cooc)
    return cooc


# --------------------------------------------------------------------------
# 2. PPMI
# --------------------------------------------------------------------------
def compute_ppmi(cooc):
    """PPMI(w,c) = max(0, log( P(w,c) / (P(w)P(c)) ))"""
    rowsum = cooc.sum(axis=1, keepdims=True)
    colsum = cooc.sum(axis=0, keepdims=True)
    total = cooc.sum()
    pmi = np.log((cooc * total + 1e-10) / (rowsum * colsum + 1e-10))
    return np.maximum(pmi, 0)


# --------------------------------------------------------------------------
# 3. SVD factorisation
# --------------------------------------------------------------------------
def svd_embeddings(ppmi, k):
    """Truncated SVD of the PPMI matrix: PPMI ≈ U Σ Vᵀ.
    Standard heuristic: use U * sqrt(Σ) as the word embeddings (which
    rebalances the singular values between the two factors symmetrically)."""
    U, s, Vt = svds(csr_matrix(ppmi), k=k)
    idx = np.argsort(-s)
    U, s, Vt = U[:, idx], s[idx], Vt[idx, :]
    return (U * np.sqrt(s)).astype(np.float32)


# --------------------------------------------------------------------------
# 4. Alternating Least Squares
# --------------------------------------------------------------------------
def als_factorize(cooc, k, epochs, reg, seed=SEED):
    """
    Objective: min_{W,H}  ‖M - W Hᵀ‖²_F + λ(‖W‖²_F + ‖H‖²_F)
    where M = cooc (the raw, not PPMI, matrix; ALS can handle the raw
    asymmetry naturally by treating rows and columns separately).

    ALS update rules -- derived by setting the partial derivative of the
    loss to zero and solving for one factor while holding the other fixed:

      W_i = (Hᵀ H + λI)^{-1} Hᵀ M[i,:]ᵀ   for each row i
      H_j = (Wᵀ W + λI)^{-1} Wᵀ M[:,j]    for each col j

    In matrix form (updating ALL rows at once):

      W = M  H (HᵀH + λI)^{-1}
      H = Mᵀ W (WᵀW + λI)^{-1}

    which is just two regularised least-squares solves per epoch.
    """
    rng = np.random.RandomState(seed)
    V = cooc.shape[0]
    W = rng.randn(V, k).astype(np.float32) * 0.01
    H = rng.randn(V, k).astype(np.float32) * 0.01
    history = []
    for epoch in range(epochs):
        HtH = H.T @ H + reg * np.eye(k, dtype=np.float32)
        W = np.linalg.solve(HtH.T, (H.T @ cooc.T).T.T).T  # M H (HtH)^{-1}
        WtW = W.T @ W + reg * np.eye(k, dtype=np.float32)
        H = np.linalg.solve(WtW.T, (W.T @ cooc).T.T).T
        # reconstruction loss on a random 5% sample (computing full ‖M-WHᵀ‖ is expensive)
        sample_idx = rng.choice(V, size=max(1, V // 20), replace=False)
        Msub = cooc[sample_idx]
        recon = W[sample_idx] @ H.T
        loss = float(np.mean((Msub - recon) ** 2))
        history.append(loss)
        print(f"  [ALS] epoch {epoch+1}/{epochs}  recon_loss(sample)={loss:.4f}")
    return W.astype(np.float32), history


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
ANALOGY_SET = [
    ("boy", "boys", "girl", "girls"),
    ("good", "better", "bad", "worse"),
    ("day", "days", "night", "nights"),
    ("man", "woman", "king", "queen"),
]


def nearest_neighbors(emb_norm, word2idx, idx2word, word, topn=6, exclude=()):
    q = emb_norm[word2idx[word]]
    sims = emb_norm @ q
    for w in (word,) + tuple(exclude):
        if w in word2idx:
            sims[word2idx[w]] = -1.0
    top = np.argsort(-sims)[:topn]
    return [(idx2word[i], round(float(sims[i]), 3)) for i in top]


def evaluate_analogies(emb_norm, word2idx, idx2word, analogy_set):
    hits, total = 0, 0
    for a, b, c, expected in analogy_set:
        if not all(w in word2idx for w in (a, b, c, expected)):
            continue
        query = emb_norm[word2idx[b]] - emb_norm[word2idx[a]] + emb_norm[word2idx[c]]
        query = query / (np.linalg.norm(query) + 1e-9)
        sims = emb_norm @ query
        for w in (a, b, c):
            sims[word2idx[w]] = -1.0
        pred = idx2word[np.argmax(sims)]
        hits += (pred == expected)
        total += 1
    return hits, total


# --------------------------------------------------------------------------
# Visualisations
# --------------------------------------------------------------------------
def plot_cooc_heatmap(cooc, word2idx, words, path):
    n = len(words)
    mat = np.array([[cooc[word2idx[w1], word2idx[w2]] for w2 in words] for w1 in words])
    mat_log = np.log1p(mat)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=160)
    im = ax.imshow(mat_log, cmap="YlOrRd")
    ax.set_xticks(range(n)); ax.set_xticklabels(words, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(words)
    ax.set_title("log(1 + co-occurrence count) between 10 words")
    fig.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_singular_values(ppmi, path):
    _, s, _ = svds(csr_matrix(ppmi), k=50)
    s = np.sort(s)[::-1]
    cumvar = np.cumsum(s ** 2) / np.sum(s ** 2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
    axes[0].plot(s, color="#4C72B0")
    axes[0].set_xlabel("rank"); axes[0].set_ylabel("singular value")
    axes[0].set_title("Singular value spectrum of PPMI matrix")
    axes[1].plot(cumvar, color="#C0504D")
    axes[1].axhline(0.9, linestyle="--", color="#888")
    axes[1].set_xlabel("rank"); axes[1].set_ylabel("cumulative variance explained")
    axes[1].set_title("How many components to keep?")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_als_convergence(history, path):
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=160)
    ax.plot(history, color="#4F9D69", marker="o")
    ax.set_xlabel("epoch"); ax.set_ylabel("reconstruction MSE (5% sample)")
    ax.set_title("ALS convergence: ‖M - W Hᵀ‖ (sampled)")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    dataset = load_or_build_dataset()
    vocab, word2idx = dataset["vocab"], dataset["word2idx"]
    idx2word = {i: w for w, i in word2idx.items()}
    sentences = dataset["filtered_sentences"]
    V = len(vocab)

    cooc = load_or_build_cooc(sentences, V, WINDOW)
    print(f"Co-occurrence matrix: shape {cooc.shape}, total mass = {cooc.sum():.1f}")

    print("\nComputing PPMI...")
    ppmi = compute_ppmi(cooc)
    print(f"  PPMI non-zeros: {np.count_nonzero(ppmi):,}")

    print("\nSVD factorisation (k=100)...")
    t0 = time.time()
    svd_emb = svd_embeddings(ppmi, k=100)
    print(f"  SVD done in {time.time()-t0:.1f}s")
    svd_norm = normalize(svd_emb)

    print("\nALS factorisation (k=100, 8 epochs, λ=0.1)...")
    als_W, als_history = als_factorize(cooc, k=100, epochs=8, reg=0.1)
    als_norm = normalize(als_W)

    print("\n=== Analogy evaluation ===")
    for name, emb_norm in [("SVD/PPMI", svd_norm), ("ALS", als_norm)]:
        h, t = evaluate_analogies(emb_norm, word2idx, idx2word, ANALOGY_SET)
        print(f"  {name:12s}: {h}/{t} analogies correct")

    print("\n=== Nearest neighbours (SVD/PPMI) ===")
    for w in ["good", "king", "money"]:
        if w in word2idx:
            print(f"  {w!r}: {nearest_neighbors(svd_norm, word2idx, idx2word, w)}")
    print("=== Nearest neighbours (ALS) ===")
    for w in ["good", "king", "money"]:
        if w in word2idx:
            print(f"  {w!r}: {nearest_neighbors(als_norm, word2idx, idx2word, w)}")

    heatmap_words = ["the", "good", "bad", "money", "king", "man", "woman", "water", "day", "night"]
    plot_cooc_heatmap(cooc, word2idx, heatmap_words, os.path.join(IMAGE_DIR, "cooc_heatmap.png"))
    plot_singular_values(ppmi, os.path.join(IMAGE_DIR, "singular_values.png"))
    plot_als_convergence(als_history, os.path.join(IMAGE_DIR, "als_convergence.png"))
    print(f"\nSaved plots to {IMAGE_DIR}")
    print(f"Saved co-occurrence matrix to {COOC_PATH} for Topics 3.5-3.7 to reuse")


if __name__ == "__main__":
    main()
