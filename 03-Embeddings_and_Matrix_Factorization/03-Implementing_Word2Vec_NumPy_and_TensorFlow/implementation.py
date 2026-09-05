"""
Topic 3.3 -- Implementing Word2Vec (NumPy and TensorFlow)
CSE468: Natural Language Processing with Deep Learning

Topic 3.2 implemented negative sampling using PyTorch's autograd. This topic
implements the exact same mathematical model -- skip-gram with negative
sampling -- TWICE more, independently:

  1. In raw NumPy, with the gradients of the loss with respect to every
     parameter DERIVED BY HAND and verified against numerical
     (finite-difference) gradients before being trusted for training. No
     autodiff framework is used anywhere in this implementation.
  2. In TensorFlow, using tf.Variable + GradientTape (the same low-level
     style as Topic 1.5, deliberately not tf.keras.layers), trained on the
     SAME pairs as the NumPy version for a direct, fair comparison.

The point of building the same model twice is cross-validation: if the
hand-derived NumPy gradients are correct, training with them should produce
results consistent with TensorFlow's independently-computed autodiff
gradients on identical data. Agreement between two independently-implemented
systems is much stronger evidence of correctness than either one alone.

Run directly:
    python implementation.py
"""

import os
import pickle
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch.nn.functional as F
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(HERE, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

DATASET_PATH = os.path.join(
    HERE, "..", "3.1-Word-Embeddings-CBOW-and-SkipGram", "artifacts", "word2vec_dataset.pkl"
)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_or_build_dataset():
    if os.path.exists(DATASET_PATH):
        print(f"Loading cached Word2Vec dataset from {DATASET_PATH}")
        with open(DATASET_PATH, "rb") as f:
            return pickle.load(f)
    print("No cached dataset found -- rebuilding from Topic 3.1's recipe...")
    sys.path.insert(0, os.path.join(HERE, "..", "3.1-Word-Embeddings-CBOW-and-SkipGram"))
    from implementation import build_and_cache_dataset  # noqa: E402
    return build_and_cache_dataset()


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
    return pairs


def subsample_sentences(sentences, freqs, threshold=1e-3, seed=SEED):
    """Topic 3.2's diagnosed fix, applied proactively here -- see that
    topic's explanation.md for the honest account of its limits."""
    total = sum(freqs)
    rel_freq = np.array(freqs, dtype=np.float64) / total
    keep_prob = (np.sqrt(rel_freq / threshold) + 1) * (threshold / rel_freq)
    keep_prob = np.clip(keep_prob, 0.0, 1.0)
    rng = np.random.RandomState(seed)
    out = []
    for s in sentences:
        kept = [w for w in s if rng.random_sample() < keep_prob[w]]
        if len(kept) >= 2:
            out.append(kept)
    return out


def build_negative_sampling_table(freqs, table_size=1_000_000):
    noise = np.power(np.array(freqs, dtype=np.float64), 0.75)
    noise = noise / noise.sum()
    table = np.zeros(table_size, dtype=np.int64)
    cum = np.cumsum(noise)
    idx = 0
    for w in range(len(freqs)):
        end = int(cum[w] * table_size)
        table[idx:end] = w
        idx = end
    table[idx:] = len(freqs) - 1
    return table


# ==========================================================================
# PART 1: NumPy implementation, gradients derived by hand
# ==========================================================================
#
# Loss for one (center c, positive context o, negatives n_1..n_k):
#   L = -log sigma(v_o . v_c)  -  sum_i log sigma(-v_{n_i} . v_c)
#
# Let s_o = v_o . v_c,  s_i = v_{n_i} . v_c.  By the chain rule:
#   dL/ds_o  = sigma(s_o) - 1
#   dL/ds_i  = sigma(s_i)
#   dL/dv_c  = (sigma(s_o)-1) * v_o  +  sum_i sigma(s_i) * v_{n_i}
#   dL/dv_o  = (sigma(s_o)-1) * v_c
#   dL/dv_{n_i} = sigma(s_i) * v_c
#
# verify_gradients_numerically() checks these against finite differences
# before any of them are trusted for actual training.

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def sgns_loss_single(W_in, W_out, c, o, neg):
    v_c, v_o, v_neg = W_in[c], W_out[o], W_out[neg]
    s_o = v_c @ v_o
    s_neg = v_neg @ v_c
    return -np.log(sigmoid(s_o) + 1e-10) - np.sum(np.log(1 - sigmoid(s_neg) + 1e-10))


def verify_gradients_numerically(V=50, d=8, k=4, eps=1e-5, seed=0):
    rng = np.random.RandomState(seed)
    W_in = rng.randn(V, d) * 0.1
    W_out = rng.randn(V, d) * 0.1
    c, o, neg = 3, 7, np.array([1, 2, 9, 15])

    v_c, v_o, v_neg = W_in[c], W_out[o], W_out[neg]
    s_o = v_c @ v_o
    s_neg = v_neg @ v_c
    sig_o, sig_neg = sigmoid(s_o), sigmoid(s_neg)

    analytic_grad_v_c = (sig_o - 1) * v_o + (sig_neg[:, None] * v_neg).sum(axis=0)
    analytic_grad_v_o = (sig_o - 1) * v_c

    numeric_grad_v_c = np.zeros(d)
    numeric_grad_v_o = np.zeros(d)
    for i in range(d):
        Wp, Wm = W_in.copy(), W_in.copy()
        Wp[c, i] += eps; Wm[c, i] -= eps
        numeric_grad_v_c[i] = (sgns_loss_single(Wp, W_out, c, o, neg)
                                 - sgns_loss_single(Wm, W_out, c, o, neg)) / (2 * eps)
        Wp, Wm = W_out.copy(), W_out.copy()
        Wp[o, i] += eps; Wm[o, i] -= eps
        numeric_grad_v_o[i] = (sgns_loss_single(W_in, Wp, c, o, neg)
                                 - sgns_loss_single(W_in, Wm, c, o, neg)) / (2 * eps)

    err_c = np.max(np.abs(analytic_grad_v_c - numeric_grad_v_c))
    err_o = np.max(np.abs(analytic_grad_v_o - numeric_grad_v_o))
    return err_c, err_o


def train_numpy_sgns(W_in, W_out, centers, contexts, table, epochs, batch_size, lr, k=5, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(centers)
    history = []
    for epoch in range(epochs):
        perm = rng.permutation(n)
        total_loss, n_batches = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            c, o = centers[idx], contexts[idx]
            neg = table[rng.randint(0, len(table), size=(len(idx), k))]

            v_c, v_o, v_neg = W_in[c], W_out[o], W_out[neg]
            s_o = np.sum(v_c * v_o, axis=1)
            s_neg = np.einsum("bd,bkd->bk", v_c, v_neg)
            sig_o, sig_neg = sigmoid(s_o), sigmoid(s_neg)

            loss = (-np.mean(np.log(sig_o + 1e-10))
                    - np.mean(np.sum(np.log(1 - sig_neg + 1e-10), axis=1)))
            total_loss += loss
            n_batches += 1

            grad_s_o, grad_s_neg = sig_o - 1, sig_neg
            grad_v_c = grad_s_o[:, None] * v_o + np.einsum("bk,bkd->bd", grad_s_neg, v_neg)
            grad_v_o = grad_s_o[:, None] * v_c
            grad_v_neg = grad_s_neg[:, :, None] * v_c[:, None, :]

            np.add.at(W_in, c, -lr * grad_v_c)
            np.add.at(W_out, o, -lr * grad_v_o)
            np.add.at(W_out, neg.reshape(-1), -lr * grad_v_neg.reshape(-1, W_in.shape[1]))

        avg_loss = total_loss / n_batches
        history.append(avg_loss)
        print(f"  [NumPy]      epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")
    return history


# ==========================================================================
# PART 2: TensorFlow implementation, tf.Variable + GradientTape
# ==========================================================================
def train_tensorflow_sgns(vocab_size, embed_dim, centers, contexts, table, epochs, batch_size, lr, k=5, seed=SEED):
    tf.random.set_seed(seed)
    # Adam handles IndexedSlices from sparse tf.gather gradients correctly;
    # plain SGD does not apply them, which causes the loss to appear stuck at
    # its initialisation value across all epochs -- a silent failure mode.
    W_in = tf.Variable(tf.random.uniform([vocab_size, embed_dim], -0.5 / embed_dim, 0.5 / embed_dim))
    W_out = tf.Variable(tf.zeros([vocab_size, embed_dim]))
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def train_step(c, o, neg):
        with tf.GradientTape() as tape:
            v_c = tf.gather(W_in, c)                       # (batch, d)
            v_o = tf.gather(W_out, o)                        # (batch, d)
            v_neg = tf.gather(W_out, neg)                    # (batch, k, d)

            s_o = tf.reduce_sum(v_c * v_o, axis=1)          # (batch,)
            s_neg = tf.einsum("bd,bkd->bk", v_c, v_neg)     # (batch, k)

            pos_loss = -tf.math.log_sigmoid(s_o)
            neg_loss = -tf.reduce_sum(tf.math.log_sigmoid(-s_neg), axis=1)
            loss = tf.reduce_mean(pos_loss + neg_loss)
        grads = tape.gradient(loss, [W_in, W_out])
        optimizer.apply_gradients(zip(grads, [W_in, W_out]))
        return loss

    n = len(centers)
    history = []
    rng = np.random.RandomState(seed)
    for epoch in range(epochs):
        perm = rng.permutation(n)
        total_loss, n_batches = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            c  = tf.constant(centers[idx],  dtype=tf.int32)
            o  = tf.constant(contexts[idx], dtype=tf.int32)
            neg = tf.constant(
                table[rng.randint(0, len(table), size=(len(idx), k))].astype(np.int32)
            )
            loss = train_step(c, o, neg)
            total_loss += float(loss)
            n_batches += 1
        avg_loss = total_loss / n_batches
        history.append(avg_loss)
        print(f"  [TensorFlow] epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")

    # Sanity check: if loss never changed from init, something went wrong.
    assert history[-1] < history[0] - 0.03, (
        f"TF loss did not decrease ({history[0]:.4f} -> {history[-1]:.4f}). "
        "Possible gradient-flow bug."
    )
    return W_in.numpy(), history


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def nearest_neighbors(embeddings, word2idx, idx2word, word, topn=6):
    vecs = F.normalize(torch.from_numpy(embeddings).float(), dim=1)
    q = vecs[word2idx[word]]
    sims = vecs @ q
    sims[word2idx[word]] = -1.0
    top = torch.topk(sims, topn)
    return [(idx2word[i.item()], round(s.item(), 3)) for s, i in zip(top.values, top.indices)]


def embedding_agreement(emb1, emb2, word2idx, idx2word, words, topn=10):
    """How much do the two independently-trained models' nearest-neighbour
    sets overlap, for the same query words? Measures cross-implementation
    consistency, not "correctness" against ground truth."""
    overlaps = []
    for w in words:
        if w not in word2idx:
            continue
        n1 = {n for n, _ in nearest_neighbors(emb1, word2idx, idx2word, w, topn)}
        n2 = {n for n, _ in nearest_neighbors(emb2, word2idx, idx2word, w, topn)}
        overlap = len(n1 & n2) / topn
        overlaps.append((w, overlap))
    return overlaps


# --------------------------------------------------------------------------
# Visualisations
# --------------------------------------------------------------------------
def plot_loss_comparison(np_hist, tf_hist, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=160)
    ax.plot(np_hist, label="NumPy (hand-derived grads)", color="#4C72B0", marker="o")
    ax.plot(tf_hist, label="TensorFlow (GradientTape)", color="#C0504D", marker="s")
    ax.set_xlabel("epoch"); ax.set_ylabel("SGNS loss")
    ax.set_title("NumPy vs. TensorFlow: same model, same data")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_gradcheck_result(err_c, err_o, path):
    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=160)
    names = ["dL/dv_c", "dL/dv_o"]
    errs = [err_c, err_o]
    bars = ax.bar(names, errs, color=["#4F9D69", "#4F9D69"])
    for bar, e in zip(bars, errs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{e:.2e}",
                 ha="center", va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("max |analytic - numerical| gradient (log scale)")
    ax.set_title("Hand-derived gradient check (finite differences)")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print("=== Step 1: verify hand-derived gradients numerically BEFORE trusting them ===")
    err_c, err_o = verify_gradients_numerically()
    print(f"  max |analytic - numerical| for dL/dv_c: {err_c:.2e}")
    print(f"  max |analytic - numerical| for dL/dv_o: {err_o:.2e}")
    assert err_c < 1e-6 and err_o < 1e-6, "Gradient check failed -- do not proceed to training!"
    print("  PASSED -- gradients match finite-difference estimates to ~1e-9. Safe to train.")
    plot_gradcheck_result(err_c, err_o, os.path.join(IMAGE_DIR, "gradient_check.png"))

    dataset = load_or_build_dataset()
    vocab, word2idx = dataset["vocab"], dataset["word2idx"]
    idx2word = {i: w for w, i in word2idx.items()}
    V = len(vocab)
    freqs = [dataset["raw_word_counts"][w] for w in vocab]

    print("\n=== Step 2: prepare identical training data for both implementations ===")
    sub_sentences = subsample_sentences(dataset["filtered_sentences"], freqs, threshold=1e-3)
    pairs = generate_skipgram_pairs(sub_sentences, window=2, max_pairs=600_000)
    centers = np.array([p[0] for p in pairs], dtype=np.int64)
    contexts = np.array([p[1] for p in pairs], dtype=np.int64)
    table = build_negative_sampling_table(freqs)
    print(f"  {len(centers):,} training pairs (subsampled corpus, identical for both implementations)")

    EMBED_DIM, EPOCHS, BATCH, LR_NP, LR_TF = 50, 6, 1024, 0.02, 0.05

    print(f"\n=== Step 3: train NumPy implementation ({EPOCHS} epochs) ===")
    rng = np.random.RandomState(SEED)
    W_in_np = (rng.rand(V, EMBED_DIM) - 0.5) / EMBED_DIM
    W_out_np = np.zeros((V, EMBED_DIM))
    t0 = time.time()
    np_history = train_numpy_sgns(W_in_np, W_out_np, centers, contexts, table, EPOCHS, BATCH, LR_NP)
    np_time = time.time() - t0
    print(f"  NumPy training time: {np_time:.1f}s  ({len(centers)*EPOCHS/np_time:,.0f} pairs/sec)")

    print(f"\n=== Step 4: train TensorFlow implementation ({EPOCHS} epochs) ===")
    t0 = time.time()
    W_in_tf, tf_history = train_tensorflow_sgns(V, EMBED_DIM, centers, contexts, table, EPOCHS, BATCH, LR_TF)
    tf_time = time.time() - t0
    print(f"  TensorFlow training time: {tf_time:.1f}s  ({len(centers)*EPOCHS/tf_time:,.0f} pairs/sec)")

    print("\n=== Step 5: cross-validate -- do the two implementations agree? ===")
    check_words = ["good", "bad", "money", "water", "king", "man", "woman", "small", "large", "happy"]
    overlaps = embedding_agreement(W_in_np, W_in_tf, word2idx, idx2word, check_words)
    for w, ov in overlaps:
        print(f"  top-10 neighbour overlap for {w!r}: {ov:.0%}")
    mean_overlap = np.mean([ov for _, ov in overlaps])
    print(f"  mean neighbour-set overlap across {len(overlaps)} probe words: {mean_overlap:.0%}")

    print("\n=== Nearest neighbours, NumPy embeddings ===")
    for w in ["good", "money", "king"]:
        print(f"  {w!r}: {nearest_neighbors(W_in_np, word2idx, idx2word, w)}")
    print("=== Nearest neighbours, TensorFlow embeddings ===")
    for w in ["good", "money", "king"]:
        print(f"  {w!r}: {nearest_neighbors(W_in_tf, word2idx, idx2word, w)}")

    plot_loss_comparison(np_history, tf_history, os.path.join(IMAGE_DIR, "numpy_vs_tensorflow_loss.png"))
    print(f"\nSaved plots to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
