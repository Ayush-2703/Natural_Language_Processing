"""
Topic 5.1 -- Introduction to Recursive Neural Networks
CSE468: Natural Language Processing with Deep Learning

A Recursive Neural Network (RvNN) applies the same neural network at every
node in a parse tree, bottom-up: leaf nodes are initialized with word
embeddings, and each internal node computes its representation by combining
its children's representations through a shared composition function.

This topic:
  1. Demonstrates tree traversal on Penn Treebank parse trees (NLTK).
  2. Implements the simplest RvNN composition function: a single-layer MLP
     applied to concatenated children representations.
  3. Shows how the same parameter-sharing structure lets a fixed-size network
     operate on trees of any shape and depth.
  4. Visualizes tree structures and the computation flow.

Run directly:
    python implementation.py
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
from nltk.corpus import treebank

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(HERE, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)
torch.manual_seed(42)

# --------------------------------------------------------------------------
# Penn Treebank tree utilities
# --------------------------------------------------------------------------
def ensure_data():
    for p, s in [("treebank","corpora"),("punkt","tokenizers"),("punkt_tab","tokenizers")]:
        try: nltk.data.find(f"{s}/{p}")
        except: nltk.download(p, quiet=True)


def simplify_label(label):
    """Strip Penn Treebank function tags (NP-SBJ -> NP)."""
    return label.split("-")[0].split("=")[0]


def tree_depth(tree):
    if isinstance(tree, str):
        return 0
    return 1 + max(tree_depth(c) for c in tree)


def tree_size(tree):
    """Number of nodes (leaves + internal)."""
    if isinstance(tree, str):
        return 1
    return 1 + sum(tree_size(c) for c in tree)


# --------------------------------------------------------------------------
# Toy vocabulary and embeddings
# --------------------------------------------------------------------------
def build_toy_vocab(trees, max_words=200):
    from collections import Counter
    counts = Counter(w.lower() for t in trees for w in t.leaves())
    vocab = ["<unk>"] + [w for w, _ in counts.most_common(max_words - 1)]
    return {w: i for i, w in enumerate(vocab)}


# --------------------------------------------------------------------------
# Simple recursive composition: h = tanh(W [h_left; h_right] + b)
# --------------------------------------------------------------------------
class TreeRNN(nn.Module):
    """
    A simple recursive neural network with a shared composition function.
    For binary trees: h_parent = tanh(W [h_left; h_right] + b)
    For n-ary trees:  applied recursively, left-to-right.
    Leaves: h_leaf = embedding(word)
    """
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.W = nn.Linear(2 * hidden_dim, hidden_dim)

    def compose(self, left, right):
        """Core composition: combines two child vectors into one parent vector."""
        return torch.tanh(self.W(torch.cat([left, right], dim=-1)))

    def forward_tree(self, tree, word2idx):
        """Recursively compute hidden state for any subtree."""
        if isinstance(tree, str):
            # Leaf: look up embedding
            idx = word2idx.get(tree.lower(), 0)
            return self.embedding(torch.tensor([idx])).squeeze(0)
        else:
            # Internal node: compute children first, then compose
            child_vecs = [self.forward_tree(c, word2idx) for c in tree]
            # For n-ary trees, compose left-to-right pairwise
            result = child_vecs[0]
            for cv in child_vecs[1:]:
                result = self.compose(result, cv)
            return result


# --------------------------------------------------------------------------
# Sentiment classification on toy examples (demonstrates end-to-end training)
# --------------------------------------------------------------------------
SENTIMENT_TREES = [
    # (phrase, sentiment: 0=neg, 1=pos)
    ("good film", 1), ("great movie", 1), ("excellent story", 1),
    ("wonderful acting", 1), ("amazing performance", 1),
    ("bad film", 0), ("terrible movie", 0), ("boring story", 0),
    ("poor acting", 0), ("awful performance", 0),
    ("not good", 0), ("not bad", 1), ("very good", 1), ("really terrible", 0),
]


def make_tiny_tree(phrase, word2idx):
    """Construct a simple left-branching binary tree from a phrase."""
    words = phrase.split()
    from nltk import Tree
    if len(words) == 2:
        return Tree("S", [Tree("W", [words[0]]), Tree("W", [words[1]])])
    elif len(words) == 3:
        return Tree("S", [Tree("W", [words[0]]),
                           Tree("S", [Tree("W", [words[1]]), Tree("W", [words[2]])])])
    return Tree("S", [Tree("W", [w]) for w in words])


# --------------------------------------------------------------------------
# Visualisations
# --------------------------------------------------------------------------
def plot_tree_stats(trees, path):
    depths = [tree_depth(t) for t in trees]
    sizes  = [tree_size(t) for t in trees]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
    axes[0].hist(depths, bins=15, color="#4C72B0", alpha=0.8)
    axes[0].set_xlabel("tree depth"); axes[0].set_ylabel("count")
    axes[0].set_title("Penn Treebank parse tree depths")
    axes[1].hist(sizes, bins=20, color="#C0504D", alpha=0.8)
    axes[1].set_xlabel("tree size (nodes)"); axes[1].set_ylabel("count")
    axes[1].set_title("Penn Treebank parse tree sizes")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def plot_sentiment_training(history, path):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    ax.plot(history, color="#4F9D69", marker="o", ms=3)
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss")
    ax.set_title("TreeRNN sentiment classification (toy phrases)")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def plot_hidden_state_similarity(model, word2idx, trees_phrases, labels, path):
    """t-SNE of tree-root hidden states coloured by sentiment."""
    from sklearn.manifold import TSNE
    vecs = []
    for phrase, _ in trees_phrases:
        t = make_tiny_tree(phrase, word2idx)
        with torch.no_grad():
            h = model.forward_tree(t, word2idx)
        vecs.append(h.numpy())
    vecs = np.stack(vecs)
    if len(vecs) < 10:
        coords = vecs[:, :2]
    else:
        coords = TSNE(n_components=2, perplexity=4, random_state=42).fit_transform(vecs)
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=160)
    colors = ["#C0504D" if l==0 else "#4F9D69" for _, l in trees_phrases]
    ax.scatter(coords[:,0], coords[:,1], c=colors, s=60, alpha=0.85)
    for i, (phrase, _) in enumerate(trees_phrases):
        ax.text(coords[i,0], coords[i,1]+0.05, phrase[:12], fontsize=7, ha="center")
    ax.set_title("TreeRNN root vectors after training\n(red=negative, green=positive)")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ensure_data()
    trees = treebank.parsed_sents()
    print(f"Penn Treebank: {len(trees)} parse trees")

    depths = [tree_depth(t) for t in trees]
    sizes  = [tree_size(t) for t in trees]
    print(f"  depth: mean={np.mean(depths):.1f}  max={max(depths)}  min={min(depths)}")
    print(f"  size:  mean={np.mean(sizes):.1f}  max={max(sizes)}  min={min(sizes)}")

    # Show the recursive computation path on one example tree
    t0 = trees[0]
    print(f"\nExample tree leaves: {t0.leaves()}")
    print(f"  depth={tree_depth(t0)}, size={tree_size(t0)}")

    # Build vocabulary from toy sentiment phrases
    all_words = set(w for phrase, _ in SENTIMENT_TREES for w in phrase.split())
    word2idx = {"<unk>": 0, **{w: i+1 for i, w in enumerate(sorted(all_words))}}

    HIDDEN = 32
    model = TreeRNN(vocab_size=len(word2idx), hidden_dim=HIDDEN)
    clf_head = nn.Linear(HIDDEN, 2)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(clf_head.parameters()), lr=5e-3
    )

    print("\nTraining TreeRNN on toy sentiment examples (100 epochs)...")
    history = []
    for epoch in range(100):
        total_loss = 0.0
        for phrase, label in SENTIMENT_TREES:
            tree = make_tiny_tree(phrase, word2idx)
            h = model.forward_tree(tree, word2idx)
            logits = clf_head(h.unsqueeze(0))
            loss = F.cross_entropy(logits, torch.tensor([label]))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        history.append(total_loss / len(SENTIMENT_TREES))
        if (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch+1}/100  loss={history[-1]:.4f}")

    # Evaluate on toy set
    correct = 0
    for phrase, label in SENTIMENT_TREES:
        tree = make_tiny_tree(phrase, word2idx)
        with torch.no_grad():
            h = model.forward_tree(tree, word2idx)
            pred = clf_head(h.unsqueeze(0)).argmax().item()
        correct += (pred == label)
    print(f"\nToy sentiment accuracy: {correct}/{len(SENTIMENT_TREES)}")

    plot_tree_stats(trees, os.path.join(IMAGE_DIR, "tree_stats.png"))
    plot_sentiment_training(history, os.path.join(IMAGE_DIR, "sentiment_training.png"))
    plot_hidden_state_similarity(model, word2idx, SENTIMENT_TREES, None,
                                 os.path.join(IMAGE_DIR, "hidden_states.png"))
    print(f"\nSaved plots to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
