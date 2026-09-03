"""
Topic 3.2 -- Word2Vec Training Mechanisms: Hierarchical Softmax and Negative Sampling
CSE468: Natural Language Processing with Deep Learning

Topic 3.1 ended by identifying the bottleneck shared by CBOW and skip-gram:
a full softmax over the entire vocabulary, computed on every training
example. This topic implements the two classical fixes from Mikolov et al.
(2013b), both from scratch, on the exact same cached vocabulary and corpus
as Topic 3.1 so the comparison is apples to apples:

  1. Hierarchical softmax -- build a real Huffman tree over the vocabulary
     by word frequency, and replace the V-way softmax with a sequence of
     O(log V) binary decisions walking root-to-leaf.
  2. Negative sampling -- replace the full softmax with one true positive
     example and k sampled "noise" negatives, each scored with a simple
     binary logistic loss.

Both are benchmarked against full softmax under IDENTICAL conditions (same
pair count, batch size, embedding dimension), then negative sampling -- the
fastest of the three -- is used to train on a substantially larger pair
budget than Topic 3.1's full-softmax models could afford in similar
wall-clock time, to see whether the resulting embeddings are any better.

Run directly:
    python implementation.py
"""

import heapq
import os
import pickle
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(HERE, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

DATASET_PATH = os.path.join(
    HERE, "..", "3.1-Word-Embeddings-CBOW-and-SkipGram", "artifacts", "word2vec_dataset.pkl"
)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(1)


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


# --------------------------------------------------------------------------
# 1. Huffman tree construction for hierarchical softmax
# --------------------------------------------------------------------------
def build_huffman_tree(freqs):
    """Standard Huffman coding via a min-heap: repeatedly merge the two
    lowest-frequency remaining nodes. Leaves are word indices 0..V-1;
    internal nodes are assigned ids V..2V-2 during merging, then re-indexed
    to 0..V-2 in the returned paths so they can index directly into a
    (V-1, d) parameter table. Returns, per word: `paths` (the sequence of
    internal-node ids visited root-to-leaf) and `codes` (the corresponding
    sequence of left/right bits)."""
    V = len(freqs)
    heap = [(freqs[i], i, i) for i in range(V)]  # (freq, tiebreak, node_id)
    heapq.heapify(heap)

    parent = {}
    next_internal_id = V
    tiebreak = V
    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        parent[n1] = (next_internal_id, 0)
        parent[n2] = (next_internal_id, 1)
        heapq.heappush(heap, (f1 + f2, tiebreak, next_internal_id))
        tiebreak += 1
        next_internal_id += 1

    paths, codes = [], []
    for i in range(V):
        path, code = [], []
        node = i
        while node in parent:
            p, bit = parent[node]
            path.append(p - V)  # re-index internal nodes to 0..V-2
            code.append(bit)
            node = p
        paths.append(list(reversed(path)))
        codes.append(list(reversed(code)))
    return paths, codes


def pad_huffman_paths(paths, codes):
    max_depth = max(len(p) for p in paths)
    V = len(paths)
    path_arr = np.zeros((V, max_depth), dtype=np.int64)
    code_arr = np.zeros((V, max_depth), dtype=np.float32)
    mask_arr = np.zeros((V, max_depth), dtype=np.float32)
    for i in range(V):
        L = len(paths[i])
        path_arr[i, :L] = paths[i]
        code_arr[i, :L] = codes[i]
        mask_arr[i, :L] = 1.0
    return path_arr, code_arr, mask_arr, max_depth


# --------------------------------------------------------------------------
# 2. Negative sampling noise table: P(w) proportional to count(w)^0.75
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# 3. Models
# --------------------------------------------------------------------------
class FullSoftmaxSkipGram(nn.Module):
    """Topic 3.1's model, reproduced here as the timing baseline."""

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, center_idx):
        return self.out_proj(self.in_embed(center_idx))


class HierarchicalSoftmaxSkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_internal):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.node_embed = nn.Embedding(n_internal, embed_dim)

    def forward(self, center_idx, path, code, mask):
        e = self.in_embed(center_idx).unsqueeze(1)      # (batch, 1, d)
        nodes = self.node_embed(path)                     # (batch, depth, d)
        scores = (nodes * e).sum(-1)                       # (batch, depth) -- v_node . v_center
        sign = 2 * code - 1                                 # bit 1 -> +1, bit 0 -> -1
        log_probs = F.logsigmoid(sign * scores)
        loss_per_example = -(log_probs * mask).sum(dim=1)
        return loss_per_example.mean()


class NegativeSamplingSkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, center_idx, pos_idx, neg_idx):
        v_c = self.in_embed(center_idx)                    # (batch, d)
        v_pos = self.out_embed(pos_idx)                      # (batch, d)
        pos_loss = -F.logsigmoid((v_c * v_pos).sum(-1))       # (batch,)
        v_neg = self.out_embed(neg_idx)                        # (batch, k, d)
        neg_score = torch.bmm(v_neg, v_c.unsqueeze(2)).squeeze(2)  # (batch, k)
        neg_loss = -F.logsigmoid(-neg_score).sum(dim=1)
        return (pos_loss + neg_loss).mean()


# --------------------------------------------------------------------------
# 4. Fair timing comparison: identical pairs, batch size, embedding dim
# --------------------------------------------------------------------------
def time_full_softmax(centers, contexts, V, embed_dim, batch_size):
    model = FullSoftmaxSkipGram(V, embed_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(centers, contexts), batch_size=batch_size, shuffle=True)
    t0 = time.time()
    for x, y in loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    return time.time() - t0


def time_hierarchical_softmax(centers, contexts_np, path_arr, code_arr, mask_arr, V, embed_dim, batch_size):
    context_path = torch.from_numpy(path_arr[contexts_np])
    context_code = torch.from_numpy(code_arr[contexts_np])
    context_mask = torch.from_numpy(mask_arr[contexts_np])
    model = HierarchicalSoftmaxSkipGram(V, embed_dim, V - 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    loader = DataLoader(TensorDataset(centers, context_path, context_code, context_mask),
                         batch_size=batch_size, shuffle=True)
    t0 = time.time()
    for c, p, cd, m in loader:
        optimizer.zero_grad()
        loss = model(c, p, cd, m)
        loss.backward()
        optimizer.step()
    return time.time() - t0


def time_negative_sampling(centers, contexts, table, V, embed_dim, batch_size, k=5):
    model = NegativeSamplingSkipGram(V, embed_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    loader = DataLoader(TensorDataset(centers, contexts), batch_size=batch_size, shuffle=True)
    table_size = len(table)
    t0 = time.time()
    for c, ctx in loader:
        neg = torch.from_numpy(table[np.random.randint(0, table_size, size=(c.size(0), k))])
        optimizer.zero_grad()
        loss = model(c, ctx, neg)
        loss.backward()
        optimizer.step()
    return time.time() - t0


# --------------------------------------------------------------------------
# Evaluation (same protocol as Topic 3.1, for direct comparability)
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
def plot_huffman_depth_distribution(paths, freqs, path):
    depths = np.array([len(p) for p in paths])
    order = np.argsort(-np.array(freqs))
    ranked_depths = depths[order]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    ax.scatter(np.arange(1, len(ranked_depths) + 1), ranked_depths, s=4, alpha=0.4, color="#4C72B0")
    ax.set_xscale("log")
    ax.set_xlabel("word frequency rank (log scale)")
    ax.set_ylabel("Huffman code length (tree depth)")
    ax.set_title("Hierarchical softmax: frequent words get shorter codes")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_timing_comparison(times, names, path):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    bars = ax.bar(names, times, color=["#888888", "#4F9D69", "#C0504D"])
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(times) * 0.02,
                 f"{t:,.0f} pairs/s", ha="center", fontsize=9)
    ax.set_ylabel("training throughput (pairs/second)")
    ax.set_title("Identical 200,000-pair, single-epoch timing test\n(V=3,000, embed_dim=50, batch=2,048)")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_loss_curve(history, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.3), dpi=160)
    ax.plot(history, color="#C0504D", marker="o", markersize=3)
    ax.set_xlabel("epoch"); ax.set_ylabel("negative sampling loss")
    ax.set_title("Large-scale negative sampling training")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_before_after_subsampling(naive_neighbors, sub_neighbors, path):
    words = list(naive_neighbors.keys())
    fig, axes = plt.subplots(len(words), 2, figsize=(10, 2.1 * len(words)), dpi=160)
    for row, w in enumerate(words):
        for col, (title, neighbors) in enumerate([
            ("naive (no subsampling)", naive_neighbors[w]),
            ("with frequent-word subsampling", sub_neighbors[w]),
        ]):
            ax = axes[row, col]
            ax.axis("off")
            labels = [f"{n}  ({s:.2f})" for n, s in neighbors]
            ax.text(0.02, 0.5, f"\"{w}\"  ->\n" + "\n".join(labels), fontsize=9,
                     va="center", family="monospace")
            if row == 0:
                ax.set_title(title, fontsize=10.5)
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
    freqs = [dataset["raw_word_counts"][w] for w in vocab]
    print(f"Vocabulary size: {V}")

    print("\nBuilding Huffman tree for hierarchical softmax...")
    paths, codes = build_huffman_tree(freqs)
    path_arr, code_arr, mask_arr, max_depth = pad_huffman_paths(paths, codes)
    print(f"  internal nodes: {V-1}   max tree depth: {max_depth}   "
          f"avg depth: {np.mean([len(p) for p in paths]):.1f}   (log2(V) = {np.log2(V):.1f})")

    print("\nBuilding negative sampling noise table (P(w) ~ count(w)^0.75)...")
    neg_table = build_negative_sampling_table(freqs)

    # ---- Fair timing comparison: identical 200K pairs, batch, embed_dim ----
    EMBED_DIM = 50
    BATCH = 2048
    TIMING_PAIRS = 200_000
    timing_pairs = generate_skipgram_pairs(sentences, window=2, max_pairs=TIMING_PAIRS)
    t_centers = torch.tensor([p[0] for p in timing_pairs], dtype=torch.long)
    t_contexts_np = np.array([p[1] for p in timing_pairs], dtype=np.int64)
    t_contexts = torch.from_numpy(t_contexts_np)

    print(f"\n=== Timing comparison on {TIMING_PAIRS:,} identical pairs ===")
    t_full = time_full_softmax(t_centers, t_contexts, V, EMBED_DIM, BATCH)
    print(f"  Full softmax:         {t_full:.2f}s  ({TIMING_PAIRS/t_full:,.0f} pairs/sec)")

    t_hs = time_hierarchical_softmax(t_centers, t_contexts_np, path_arr, code_arr, mask_arr, V, EMBED_DIM, BATCH)
    print(f"  Hierarchical softmax: {t_hs:.2f}s  ({TIMING_PAIRS/t_hs:,.0f} pairs/sec)  "
          f"-- {t_full/t_hs:.2f}x faster than full softmax")

    t_ns = time_negative_sampling(t_centers, t_contexts, neg_table, V, EMBED_DIM, BATCH, k=5)
    print(f"  Negative sampling:    {t_ns:.2f}s  ({TIMING_PAIRS/t_ns:,.0f} pairs/sec)  "
          f"-- {t_full/t_ns:.2f}x faster than full softmax")

    plot_timing_comparison(
        [TIMING_PAIRS / t_full, TIMING_PAIRS / t_hs, TIMING_PAIRS / t_ns],
        ["Full softmax", "Hierarchical\nsoftmax", "Negative\nsampling"],
        os.path.join(IMAGE_DIR, "timing_comparison.png"),
    )
    plot_huffman_depth_distribution(paths, freqs, os.path.join(IMAGE_DIR, "huffman_depth_distribution.png"))

    # ---- Large-scale negative sampling training, using the speed it buys ----
    LARGE_PAIRS = 3_000_000
    EPOCHS = 3
    print(f"\n=== Large-scale negative sampling training: {LARGE_PAIRS:,} pairs x {EPOCHS} epochs ===")
    t0 = time.time()
    large_pairs = generate_skipgram_pairs(sentences, window=2, max_pairs=LARGE_PAIRS)
    print(f"  generated {len(large_pairs):,} pairs in {time.time()-t0:.1f}s")
    l_centers = torch.tensor([p[0] for p in large_pairs], dtype=torch.long)
    l_contexts = torch.tensor([p[1] for p in large_pairs], dtype=torch.long)

    big_model = NegativeSamplingSkipGram(V, EMBED_DIM)
    optimizer = torch.optim.Adam(big_model.parameters(), lr=2e-3)
    loader = DataLoader(TensorDataset(l_centers, l_contexts), batch_size=BATCH, shuffle=True)
    table_size = len(neg_table)
    history = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        total_loss, n_batches = 0.0, 0
        for c, ctx in loader:
            neg = torch.from_numpy(neg_table[np.random.randint(0, table_size, size=(c.size(0), 5))])
            optimizer.zero_grad()
            loss = big_model(c, ctx, neg)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / n_batches
        history.append(avg)
        print(f"  epoch {epoch+1}/{EPOCHS}  loss={avg:.4f}")
    train_time = time.time() - t0
    print(f"  training time: {train_time:.1f}s  ({len(large_pairs)*EPOCHS/train_time:,.0f} pairs/sec)")

    ns_emb = big_model.in_embed.weight.data.clone()
    print("\n=== Nearest neighbours, NAIVE negative-sampling embeddings (3M pairs x 3 epochs, no subsampling) ===")
    naive_neighbors = {}
    for w in ["good", "money", "king", "water"]:
        if w in word2idx:
            naive_neighbors[w] = nearest_neighbors(ns_emb, word2idx, idx2word, w)
            print(f"  {w!r}: {naive_neighbors[w]}")

    # ---- Mikolov et al. (2013b)'s fix: subsample away frequent words ----
    # The naive run above is a well-documented real failure mode: without
    # any check on how often extremely common words (the, a, of, to, ...)
    # participate as training pairs, they dominate the gradient signal and
    # the embedding space ends up organised around raw frequency rather
    # than meaning -- visible directly in the neighbours above. The SAME
    # paper that introduces negative sampling also introduces the standard
    # fix: randomly discard frequent words from the corpus before
    # generating pairs, with a word kept with probability
    #   P_keep(w) = (sqrt(f(w)/t) + 1) * (t/f(w))
    # for relative frequency f(w) and a small threshold t (here 1e-3).
    print("\n=== Applying Mikolov et al.'s frequent-word subsampling ===")
    total_tokens = sum(freqs)
    rel_freq = np.array(freqs, dtype=np.float64) / total_tokens
    threshold = 1e-3
    keep_prob = (np.sqrt(rel_freq / threshold) + 1) * (threshold / rel_freq)
    keep_prob = np.clip(keep_prob, 0.0, 1.0)
    print(f"  keep probability for the MOST frequent word ({vocab[0]!r}): {keep_prob[0]:.3f}")
    print(f"  keep probability for a mid-frequency word ({vocab[1500]!r}): {keep_prob[1500]:.3f}")
    print(f"  keep probability for a rare word ({vocab[-1]!r}): {keep_prob[-1]:.3f}")

    rng = np.random.RandomState(SEED)
    subsampled_sentences = []
    for s in sentences:
        kept = [w for w in s if rng.random_sample() < keep_prob[w]]
        if len(kept) >= 2:
            subsampled_sentences.append(kept)
    n_tok_before = sum(len(s) for s in sentences)
    n_tok_after = sum(len(s) for s in subsampled_sentences)
    print(f"  tokens before subsampling: {n_tok_before:,}   after: {n_tok_after:,}  "
          f"({n_tok_after/n_tok_before:.1%} kept)")

    SUB_PAIRS = 2_000_000
    SUB_EPOCHS = 3
    sub_pairs = generate_skipgram_pairs(subsampled_sentences, window=2, max_pairs=SUB_PAIRS)
    s_centers = torch.tensor([p[0] for p in sub_pairs], dtype=torch.long)
    s_contexts = torch.tensor([p[1] for p in sub_pairs], dtype=torch.long)
    print(f"  generated {len(sub_pairs):,} pairs from the subsampled corpus")

    sub_model = NegativeSamplingSkipGram(V, EMBED_DIM)
    optimizer = torch.optim.Adam(sub_model.parameters(), lr=2e-3)
    loader = DataLoader(TensorDataset(s_centers, s_contexts), batch_size=BATCH, shuffle=True)
    sub_history = []
    t0 = time.time()
    for epoch in range(SUB_EPOCHS):
        total_loss, n_batches = 0.0, 0
        for c, ctx in loader:
            neg = torch.from_numpy(neg_table[np.random.randint(0, table_size, size=(c.size(0), 5))])
            optimizer.zero_grad()
            loss = sub_model(c, ctx, neg)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / n_batches
        sub_history.append(avg)
        print(f"  epoch {epoch+1}/{SUB_EPOCHS}  loss={avg:.4f}")
    print(f"  training time: {time.time()-t0:.1f}s")

    sub_emb = sub_model.in_embed.weight.data.clone()
    print("\n=== Nearest neighbours, WITH subsampling ===")
    sub_neighbors = {}
    for w in ["good", "money", "king", "water"]:
        if w in word2idx:
            sub_neighbors[w] = nearest_neighbors(sub_emb, word2idx, idx2word, w)
            print(f"  {w!r}: {sub_neighbors[w]}")

    plot_loss_curve(history, os.path.join(IMAGE_DIR, "large_scale_loss.png"))
    plot_before_after_subsampling(naive_neighbors, sub_neighbors, os.path.join(IMAGE_DIR, "subsampling_comparison.png"))
    print(f"\nSaved plots to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
