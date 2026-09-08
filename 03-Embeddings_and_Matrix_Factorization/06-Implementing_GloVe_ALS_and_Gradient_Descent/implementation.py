"""
Topic 3.6 -- Implementing GloVe using Alternating Least Squares and Gradient Descent
CSE468: Natural Language Processing with Deep Learning

GloVe (Pennington, Socher & Manning, 2014) is a weighted least-squares
factorisation of the log co-occurrence matrix. This topic implements it two
ways -- ALS and gradient descent -- and compares both to Topic 3.4's SVD/PPMI
embeddings on the same analogy test to close the loop on Phase 3's full
comparison arc: SGNS (3.1-3.3), SVD/PPMI (3.4), GloVe (3.6), PMI (3.7).

Run directly:
    python implementation.py
"""

import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(HERE, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

DATASET_PATH = os.path.join(HERE, "..", "3.1-Word-Embeddings-CBOW-and-SkipGram", "artifacts", "word2vec_dataset.pkl")
COOC_PATH = os.path.join(HERE, "..", "3.4-Matrix-Factorization-for-Word-Representations", "artifacts", "cooc_matrix.npy")
SEED = 42
np.random.seed(SEED)


def load_resources():
    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)
    cooc = np.load(COOC_PATH)
    return dataset, cooc


def glove_weight(x, x_max=100, alpha=0.75):
    """GloVe's f(x): ramps from 0 to 1 for counts below x_max,
    then stays at 1 -- downweights very rare pairs (noisy) but
    does NOT upweight very common ones (already well-estimated)."""
    return np.where(x < x_max, (x / x_max) ** alpha, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# GloVe objective and helpers
# --------------------------------------------------------------------------
def glove_loss(W, b_w, C, b_c, log_cooc, weights):
    diff = (W @ C.T) + b_w[:, None] + b_c[None, :] - log_cooc
    return float(np.sum(weights * diff ** 2))


def glove_als(log_cooc, weights, k, epochs, reg, seed=SEED):
    """
    ALS update for GloVe: at each step, one factor is held fixed while
    the other is updated via weighted ridge regression (a small extension
    of Topic 3.4's plain ALS to incorporate GloVe's f(x) weights).

    For a fixed context matrix C, each word row w_i solves:
        min_{w_i}  Σ_j f(X_ij)(w_i·c_j + b_i + b_j - log X_ij)² + λ‖w_i‖²
    which is a standard weighted least-squares problem per row.
    """
    V = log_cooc.shape[0]
    rng = np.random.RandomState(seed)
    W = rng.randn(V, k).astype(np.float32) * 0.1
    C = rng.randn(V, k).astype(np.float32) * 0.1
    b_w = np.zeros(V, dtype=np.float32)
    b_c = np.zeros(V, dtype=np.float32)
    history = []

    for epoch in range(epochs):
        # Update W and b_w (one row per word)
        for i in range(V):
            wi = weights[i]          # (V,) weight vector for row i
            target_i = log_cooc[i] - b_w[i] - b_c   # (V,) residual ignoring w_i·c_j
            # Weighted LS: (CᵀΩC + λI) w_i = CᵀΩ t_i
            CwC = (C * wi[:, None]).T @ C + reg * np.eye(k)
            Cwt = (C * wi[:, None]).T @ target_i
            W[i] = np.linalg.solve(CwC, Cwt)
            # bias: d/db_w_i [ Σ_j f_ij (w_i·c_j + b_i + b_j - log X_ij)² ] = 0
            b_w[i] = float(np.sum(wi * (log_cooc[i] - W[i] @ C.T - b_c)) / (np.sum(wi) + 1e-8))

        # Update C and b_c (one column per context word)
        for j in range(V):
            wj = weights[:, j]
            target_j = log_cooc[:, j] - b_w - b_c[j]
            WwW = (W * wj[:, None]).T @ W + reg * np.eye(k)
            Wwt = (W * wj[:, None]).T @ target_j
            C[j] = np.linalg.solve(WwW, Wwt)
            b_c[j] = float(np.sum(wj * (log_cooc[:, j] - W @ C[j] - b_w)) / (np.sum(wj) + 1e-8))

        loss = glove_loss(W, b_w, C, b_c, log_cooc, weights)
        history.append(loss)
        print(f"  [GloVe-ALS] epoch {epoch+1}/{epochs}  loss={loss:.2f}")

    return (W + C).astype(np.float32), history   # GloVe convention: sum word+context vectors


def glove_gradient_descent(log_cooc, weights, k, epochs, lr, reg, seed=SEED):
    """
    AdaGrad-based gradient descent -- matches the original GloVe paper's
    optimiser. Using adaptive per-parameter learning rates matters here
    because GloVe's gradients are very unequal: frequently co-occurring
    pairs get many gradient updates while rare pairs get almost none, so
    a single global learning rate either stalls on rare pairs or diverges
    on frequent ones.
    """
    V = log_cooc.shape[0]
    rng = np.random.RandomState(seed)
    W = rng.randn(V, k).astype(np.float64) * 0.1
    C = rng.randn(V, k).astype(np.float64) * 0.1
    b_w = np.zeros(V, dtype=np.float64)
    b_c = np.zeros(V, dtype=np.float64)

    # AdaGrad accumulators
    G_W = np.ones((V, k), dtype=np.float64)
    G_C = np.ones((V, k), dtype=np.float64)
    G_bw = np.ones(V, dtype=np.float64)
    G_bc = np.ones(V, dtype=np.float64)

    # Pre-fetch non-zero indices for efficiency
    nz_i, nz_j = np.nonzero(weights > 0)
    w_vals = weights[nz_i, nz_j].astype(np.float64)
    log_vals = log_cooc[nz_i, nz_j].astype(np.float64)
    n_pairs = len(nz_i)
    history = []

    for epoch in range(epochs):
        perm = rng.permutation(n_pairs)
        total_loss = 0.0
        batch_size = min(4096, n_pairs)
        for start in range(0, n_pairs, batch_size):
            idx = perm[start:start + batch_size]
            ii, jj = nz_i[idx], nz_j[idx]
            wts = w_vals[idx]
            targets = log_vals[idx]

            pred = np.sum(W[ii] * C[jj], axis=1) + b_w[ii] + b_c[jj]
            diff = pred - targets
            wdiff = wts * diff
            total_loss += float(np.sum(wts * diff ** 2))

            g_W = wdiff[:, None] * C[jj]
            g_C = wdiff[:, None] * W[ii]
            g_bw = wdiff
            g_bc = wdiff

            # AdaGrad accumulate + update
            G_W[ii] += g_W ** 2
            W[ii] -= lr * g_W / np.sqrt(G_W[ii])
            G_C[jj] += g_C ** 2
            C[jj] -= lr * g_C / np.sqrt(G_C[jj])
            G_bw[ii] += g_bw ** 2
            b_w[ii] -= lr * g_bw / np.sqrt(G_bw[ii])
            G_bc[jj] += g_bc ** 2
            b_c[jj] -= lr * g_bc / np.sqrt(G_bc[jj])

        avg_loss = total_loss / n_pairs
        history.append(avg_loss)
        print(f"  [GloVe-GD]  epoch {epoch+1}/{epochs}  avg_loss={avg_loss:.6f}")

    return ((W + C) / 2).astype(np.float32), history


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
ANALOGY_SET = [
    ("boy", "boys", "girl", "girls"),
    ("good", "better", "bad", "worse"),
    ("day", "days", "night", "nights"),
    ("man", "woman", "king", "queen"),
]


def nearest_neighbors(emb_norm, word2idx, idx2word, word, topn=6):
    q = emb_norm[word2idx[word]]
    sims = emb_norm @ q
    sims[word2idx[word]] = -1.0
    top = np.argsort(-sims)[:topn]
    return [(idx2word[i], round(float(sims[i]), 3)) for i in top]


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
def plot_loss_curves(als_hist, gd_hist, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=160)
    axes[0].plot(als_hist, color="#4C72B0", marker="o"); axes[0].set_title("GloVe ALS")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("weighted squared loss")
    axes[1].plot(gd_hist, color="#C0504D", marker="o"); axes[1].set_title("GloVe Gradient Descent (AdaGrad)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("avg pair loss")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_analogy_summary(results, path):
    names = list(results.keys())
    hits = [r[0] for r in results.values()]
    totals = [r[1] for r in results.values()]
    accs = [h / t if t > 0 else 0 for h, t in zip(hits, totals)]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
    bars = ax.bar(names, accs, color=["#4C72B0", "#4F9D69", "#C0504D"])
    for bar, h, t in zip(bars, hits, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{h}/{t}", ha="center", fontsize=9)
    ax.set_ylabel("analogy accuracy")
    ax.set_title("Phase 3 embedding comparison — same 4 analogy test pairs")
    ax.set_ylim(0, 1.1)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
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
    V = len(vocab)

    # GloVe's two inputs: weighted log co-occurrence
    log_cooc = np.log(cooc + 1e-8).astype(np.float32)   # log(X_ij), smoothed
    weights = glove_weight(cooc, x_max=100, alpha=0.75)   # f(X_ij)
    n_nonzero = np.count_nonzero(weights > 0)
    print(f"Non-zero weighted pairs: {n_nonzero:,}  (out of {V*V:,} possible)")

    K, ALS_EPOCHS, GD_EPOCHS, REG = 100, 3, 8, 0.1

    print(f"\n=== GloVe ALS (k={K}, {ALS_EPOCHS} epochs, λ={REG}) ===")
    t0 = time.time()
    als_emb, als_hist = glove_als(log_cooc, weights, K, ALS_EPOCHS, REG)
    print(f"  ALS time: {time.time()-t0:.1f}s")

    print(f"\n=== GloVe Gradient Descent / AdaGrad (k={K}, {GD_EPOCHS} epochs) ===")
    t0 = time.time()
    gd_emb, gd_hist = glove_gradient_descent(log_cooc, weights, K, GD_EPOCHS, lr=0.05, reg=REG)
    print(f"  GD time: {time.time()-t0:.1f}s")

    als_norm = normalize(als_emb)
    gd_norm = normalize(gd_emb)

    print("\n=== Analogy evaluation ===")
    results = {}
    results["GloVe ALS"] = evaluate_analogies(als_norm, word2idx, idx2word)
    results["GloVe GD"] = evaluate_analogies(gd_norm, word2idx, idx2word)

    # Also reload SVD/PPMI baseline from Topic 3.4 for comparison
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import svds
        ppmi_cooc = np.maximum(
            np.log((cooc * cooc.sum() + 1e-10) /
                   (cooc.sum(axis=1, keepdims=True) * cooc.sum(axis=0, keepdims=True) + 1e-10)),
            0)
        U, s, _ = svds(csr_matrix(ppmi_cooc), k=K)
        idx = np.argsort(-s)
        svd_emb = (U[:, idx] * np.sqrt(s[idx])).astype(np.float32)
        svd_norm = normalize(svd_emb)
        results["SVD/PPMI (3.4)"] = evaluate_analogies(svd_norm, word2idx, idx2word)
    except Exception:
        pass

    for name, (h, t) in results.items():
        print(f"  {name:20s}: {h}/{t}")

    print("\n=== Nearest neighbours (GloVe GD) ===")
    for w in ["good", "money", "king"]:
        if w in word2idx:
            print(f"  {w!r}: {nearest_neighbors(gd_norm, word2idx, idx2word, w)}")

    plot_loss_curves(als_hist, gd_hist, os.path.join(IMAGE_DIR, "glove_loss_curves.png"))
    plot_analogy_summary(results, os.path.join(IMAGE_DIR, "analogy_comparison.png"))
    print(f"\nSaved plots to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
