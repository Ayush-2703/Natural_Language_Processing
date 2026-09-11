"""
Topic 3.7 -- Pointwise Mutual Information (PMI) Implementations

PMI is the bridge between count-based and predictive embeddings. This topic:

  1. Implements the full PMI matrix: PMI, PPMI, SPPMI (shifted PPMI for k
     negatives), and NPMI (normalised PMI, bounded in [-1, +1]).
  2. Empirically checks Topic 3.5's theoretical prediction -- that SGNS dot
     products correlate with PMI -- on the embeddings trained in Topic 3.3.
  3. Directly compares PPMI-based SVD embeddings to SPPMI-based SVD
     embeddings (Topic 3.5 predicts SPPMI, k=5, matches SGNS's implicit
     target; PPMI is the k=1 special case).
"""

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(HERE, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

DATASET_PATH = os.path.join(HERE, "..", "3.1-Word-Embeddings-CBOW-and-SkipGram", "artifacts", "word2vec_dataset.pkl")
COOC_PATH = os.path.join(HERE, "..", "3.4-Matrix-Factorization-for-Word-Representations", "artifacts", "cooc_matrix.npy")
# Topic 3.3's NumPy SGNS vectors (saved to disk if available; recomputed if not)
NUMPY_EMB_PATH = os.path.join(HERE, "..", "3.3-Implementing-Word2Vec-NumPy-and-TensorFlow", "artifacts", "sgns_numpy_embeddings.npy")
SEED = 42


def load_resources():
    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)
    cooc = np.load(COOC_PATH)
    return dataset, cooc


# --------------------------------------------------------------------------
# 1. PMI variants
# --------------------------------------------------------------------------
def compute_pmi_variants(cooc, k_neg=5):
    """
    PMI(w,c) = log( P(w,c) / (P(w)·P(c)) )
    PPMI = max(0, PMI)
    SPPMI_k = max(0, PMI - log k)    [shifted by log(k negatives)]
    NPMI = PMI / (-log P(w,c))        [normalised, in [-1,+1]]
    """
    total = cooc.sum()
    rowsum = cooc.sum(axis=1, keepdims=True)   # (V, 1)
    colsum = cooc.sum(axis=0, keepdims=True)   # (1, V)
    pmi = np.log((cooc * total + 1e-10) / (rowsum * colsum + 1e-10))

    ppmi = np.maximum(pmi, 0.0)
    sppmi = np.maximum(pmi - np.log(k_neg), 0.0)

    log_joint = np.log(cooc / total + 1e-10)
    npmi = np.where(cooc > 0, pmi / (-log_joint + 1e-10), 0.0)
    npmi = np.clip(npmi, -1.0, 1.0)

    return pmi, ppmi, sppmi, npmi


def svd_embeddings(matrix, k):
    U, s, _ = svds(csr_matrix(matrix), k=k)
    idx = np.argsort(-s)
    return (U[:, idx] * np.sqrt(s[idx])).astype(np.float32)


# --------------------------------------------------------------------------
# 2. Empirical check of Levy & Goldberg's prediction
# --------------------------------------------------------------------------
def get_or_retrain_sgns_embeddings(dataset):
    """Load Topic 3.3's trained NumPy SGNS embeddings if available, else
    retrain a quick version on the same corpus."""
    if os.path.exists(NUMPY_EMB_PATH):
        print(f"  Loading cached SGNS embeddings from {NUMPY_EMB_PATH}")
        return np.load(NUMPY_EMB_PATH)

    print("  SGNS embeddings not cached; retraining a compact version (~30s)...")
    import random

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    vocab = dataset["vocab"]
    sentences = dataset["filtered_sentences"]
    V, K, D = len(vocab), 5, 50
    freqs = np.array([dataset["raw_word_counts"][w] for w in vocab], dtype=np.float64)
    noise = np.power(freqs, 0.75); noise /= noise.sum()
    table_size = 1_000_000
    table = np.zeros(table_size, dtype=np.int64)
    cum = np.cumsum(noise); idx = 0
    for w in range(V):
        end = int(cum[w] * table_size); table[idx:end] = w; idx = end
    table[idx:] = V - 1

    pairs = []
    for s in sentences:
        for i, center in enumerate(s):
            for j in range(max(0, i - 2), min(len(s), i + 3)):
                if j != i:
                    pairs.append((center, s[j]))
    random.Random(SEED).shuffle(pairs)
    pairs = pairs[:400_000]
    centers = np.array([p[0] for p in pairs], dtype=np.int64)
    contexts = np.array([p[1] for p in pairs], dtype=np.int64)

    rng = np.random.RandomState(SEED)
    W_in = (rng.rand(V, D) - 0.5) / D
    W_out = np.zeros((V, D))
    for epoch in range(5):
        perm = rng.permutation(len(centers))
        for start in range(0, len(centers), 1024):
            idx2 = perm[start:start + 1024]
            c, o = centers[idx2], contexts[idx2]
            neg = table[rng.randint(0, table_size, size=(len(idx2), K))]
            v_c, v_o, v_neg = W_in[c], W_out[o], W_out[neg]
            s_o = np.sum(v_c * v_o, axis=1)
            s_neg = np.einsum("bd,bkd->bk", v_c, v_neg)
            sig_o, sig_neg = sigmoid(s_o), sigmoid(s_neg)
            g_vc = (sig_o - 1)[:, None] * v_o + np.einsum("bk,bkd->bd", sig_neg, v_neg)
            g_vo = (sig_o - 1)[:, None] * v_c
            g_vn = sig_neg[:, :, None] * v_c[:, None, :]
            np.add.at(W_in, c, -0.02 * g_vc)
            np.add.at(W_out, o, -0.02 * g_vo)
            np.add.at(W_out, neg.reshape(-1), -0.02 * g_vn.reshape(-1, D))

    os.makedirs(os.path.dirname(NUMPY_EMB_PATH), exist_ok=True)
    np.save(NUMPY_EMB_PATH, W_in.astype(np.float32))
    return W_in.astype(np.float32)


def check_sgns_vs_pmi(sgns_emb, pmi_matrix, word2idx, n_sample=500, seed=SEED):
    """
    For a random sample of word pairs, compare:
      - the dot product of their SGNS vectors (v_w · u_c at convergence
        should equal PMI(w,c) - log k, per Levy & Goldberg's proof)
      - the actual PMI value from the count matrix

    Returns Pearson r between the two, computed over observed (non-zero)
    pairs only (PMI is undefined / -inf for unobserved pairs).
    """
    rng = np.random.RandomState(seed)
    vocab_list = list(word2idx.keys())
    # sample from observed pairs only
    nz_w, nz_c = np.nonzero(pmi_matrix)
    idx = rng.choice(len(nz_w), size=min(n_sample, len(nz_w)), replace=False)
    ws, cs = nz_w[idx], nz_c[idx]

    sgns_dots = np.sum(sgns_emb[ws] * sgns_emb[cs], axis=1)
    pmi_vals = pmi_matrix[ws, cs]

    r = float(np.corrcoef(sgns_dots, pmi_vals)[0, 1])
    return r, sgns_dots, pmi_vals


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------
ANALOGY_SET = [
    ("boy", "boys", "girl", "girls"),
    ("good", "better", "bad", "worse"),
    ("day", "days", "night", "nights"),
    ("man", "woman", "king", "queen"),
]


def evaluate_analogies(emb_norm, word2idx, idx2word):
    hits, total = 0, 0
    for a, b, c, expected in ANALOGY_SET:
        if not all(w in word2idx for w in (a, b, c, expected)):
            continue
        q = emb_norm[word2idx[b]] - emb_norm[word2idx[a]] + emb_norm[word2idx[c]]
        q /= np.linalg.norm(q) + 1e-9
        sims = emb_norm @ q
        for w in (a, b, c):
            sims[word2idx[w]] = -1.0
        hits += (idx2word[np.argmax(sims)] == expected)
        total += 1
    return hits, total


# --------------------------------------------------------------------------
# Visualisations
# --------------------------------------------------------------------------
def plot_pmi_comparison(ppmi, sppmi, word2idx, words, path):
    n = len(words)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=160)
    for ax, mat, title in [(axes[0], ppmi, "PPMI"), (axes[1], sppmi, "SPPMI (k=5, shift=log 5)")]:
        sub = np.array([[mat[word2idx[w1], word2idx[w2]] for w2 in words] for w1 in words])
        im = ax.imshow(sub, cmap="YlOrRd", vmin=0)
        ax.set_xticks(range(n)); ax.set_xticklabels(words, rotation=45, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels(words)
        ax.set_title(title); fig.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle("PPMI vs SPPMI: shifting by log k thresholds out low-PMI pairs")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_sgns_vs_pmi_scatter(sgns_dots, pmi_vals, r, path):
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=160)
    ax.scatter(pmi_vals[:500], sgns_dots[:500], s=8, alpha=0.4, color="#4C72B0")
    ax.set_xlabel("PMI(w,c) from co-occurrence matrix")
    ax.set_ylabel("SGNS embedding dot product: v_w · v_c")
    ax.set_title(f"SGNS dot products vs. PMI\n(Pearson r = {r:.3f},  n=500 sampled observed pairs)")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_analogy_all(results, path):
    names, accs = list(results.keys()), [h / t if t > 0 else 0 for h, t in results.values()]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    bars = ax.bar(names, accs, color=["#888888", "#4C72B0", "#4F9D69"])
    for bar, (h, t) in zip(bars, results.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{h}/{t}", ha="center", fontsize=9)
    ax.set_ylabel("analogy accuracy"); ax.set_ylim(0, 1.1)
    ax.set_title("SVD embeddings: PPMI vs SPPMI (k=5)")
    plt.xticks(rotation=15, ha="right"); plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print("Loading cached dataset and co-occurrence matrix...")
    dataset, cooc = load_resources()
    vocab, word2idx = dataset["vocab"], dataset["word2idx"]
    idx2word = {i: w for w, i in word2idx.items()}

    print("\nComputing PMI variants...")
    pmi, ppmi, sppmi, npmi = compute_pmi_variants(cooc, k_neg=5)
    print(f"  PMI  range: [{pmi[cooc>0].min():.2f}, {pmi[cooc>0].max():.2f}] (observed pairs only)")
    print(f"  PPMI non-zeros: {np.count_nonzero(ppmi):,}")
    print(f"  SPPMI(k=5) non-zeros: {np.count_nonzero(sppmi):,}  (fewer -- shift removes low-PMI pairs)")
    print(f"  NPMI range (observed): [{npmi[cooc>0].min():.3f}, {npmi[cooc>0].max():.3f}]")

    print("\nSVD on PPMI and SPPMI (k=100)...")
    ppmi_emb = svd_embeddings(ppmi, k=100)
    sppmi_emb = svd_embeddings(sppmi, k=100)
    ppmi_norm = normalize(ppmi_emb)
    sppmi_norm = normalize(sppmi_emb)

    print("\nEvaluating analogies...")
    results = {
        "SVD/PPMI": evaluate_analogies(ppmi_norm, word2idx, idx2word),
        "SVD/SPPMI (k=5)": evaluate_analogies(sppmi_norm, word2idx, idx2word),
    }
    for name, (h, t) in results.items():
        print(f"  {name:20s}: {h}/{t}")

    print("\nChecking Levy & Goldberg prediction: SGNS dots ≈ PMI - log k ...")
    sgns_emb = get_or_retrain_sgns_embeddings(dataset)
    r, sgns_dots, pmi_vals = check_sgns_vs_pmi(sgns_emb, pmi, word2idx)
    print(f"  Pearson r(SGNS dot products, PMI values) = {r:.3f}  (500 sampled observed pairs)")
    print(f"  (Theory predicts positive correlation; r > 0 confirms it)")

    probe_words = ["good", "bad", "money", "king", "man", "woman", "water", "day"]
    plot_pmi_comparison(ppmi, sppmi, word2idx, probe_words, os.path.join(IMAGE_DIR, "ppmi_vs_sppmi.png"))
    plot_sgns_vs_pmi_scatter(sgns_dots, pmi_vals, r, os.path.join(IMAGE_DIR, "sgns_vs_pmi.png"))
    plot_analogy_all(results, os.path.join(IMAGE_DIR, "ppmi_vs_sppmi_analogy.png"))
    print(f"\nSaved plots to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
