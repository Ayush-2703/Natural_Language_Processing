"""
Topic 5.3 -- Tree Neural Network (TNN) utilizing Recursion
CSE468: Natural Language Processing with Deep Learning

A full TNN trained on Penn Treebank trees with synthetic sentiment labels
(generated from word polarities), demonstrating recursive training with
supervision at every tree node.

Run directly:
    python implementation.py
"""
import os, sys, random
from collections import Counter, defaultdict

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import nltk
from nltk.corpus import treebank

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(HERE, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)
torch.manual_seed(42); random.seed(42); np.random.seed(42)
sys.setrecursionlimit(10000)

# ── Data ────────────────────────────────────────────────────────────────────
POSITIVE_WORDS = {"good","great","excellent","wonderful","amazing","fantastic","brilliant","love","best","perfect"}
NEGATIVE_WORDS = {"bad","terrible","awful","boring","poor","dreadful","hate","worst","horrible","dull"}

def clean_label(l): return l.split("-")[0].split("=")[0]

def sentiment_label(leaves):
    """Synthetic: count positive/negative words to assign a 0/1 label."""
    pos = sum(1 for w in leaves if w.lower() in POSITIVE_WORDS)
    neg = sum(1 for w in leaves if w.lower() in NEGATIVE_WORDS)
    return 1 if pos >= neg else 0


class TNNModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.W = nn.Linear(2 * hidden_dim, hidden_dim)
        self.clf = nn.Linear(hidden_dim, n_classes)
        nn.init.xavier_uniform_(self.W.weight)

    def compose(self, left, right):
        return torch.tanh(self.W(torch.cat([left, right], dim=-1)))

    def forward_tree(self, tree, word2idx):
        """Returns (root_hidden, list_of_(hidden, label) at every node)."""
        if isinstance(tree, str):
            idx = word2idx.get(tree.lower(), 0)
            h = self.embedding(torch.tensor([idx])).squeeze(0)
            label = 1 if tree.lower() in POSITIVE_WORDS else (0 if tree.lower() in NEGATIVE_WORDS else None)
            return h, [(h, label)] if label is not None else [(h, None)]
        child_results = [self.forward_tree(c, word2idx) for c in tree]
        h = child_results[0][0]
        all_node_info = []
        for cr in child_results:
            all_node_info.extend(cr[1])
        for ci in range(1, len(child_results)):
            h = self.compose(h, child_results[ci][0])
        leaves = tree.leaves()
        label = sentiment_label(leaves) if len(leaves) >= 2 else None
        all_node_info.append((h, label))
        return h, all_node_info


def build_vocab(trees, max_words=3000):
    counts = Counter(w.lower() for t in trees for w in t.leaves())
    vocab = ["<pad>", "<unk>"] + [w for w, _ in counts.most_common(max_words - 2)]
    return {w: i for i, w in enumerate(vocab)}


def train_tnn(model, trees, word2idx, epochs=5, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        random.shuffle(trees)
        total_loss, total_nodes = 0.0, 0
        for tree in trees[:300]:  # subsample for speed
            _, node_info = model.forward_tree(tree, word2idx)
            node_losses = []
            for h, label in node_info:
                if label is not None:
                    logits = model.clf(h.unsqueeze(0))
                    node_losses.append(F.cross_entropy(logits, torch.tensor([label])))
            if node_losses:
                loss = torch.stack(node_losses).mean()
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item(); total_nodes += 1
        avg = total_loss / max(total_nodes, 1)
        history.append(avg)
        print(f"  epoch {epoch+1}/{epochs}  loss={avg:.4f}")
    return history


def evaluate_tnn(model, trees, word2idx):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for tree in trees:
            _, node_info = model.forward_tree(tree, word2idx)
            for h, label in node_info:
                if label is not None:
                    pred = model.clf(h.unsqueeze(0)).argmax().item()
                    correct += (pred == label); total += 1
    return correct / max(total, 1)


def main():
    trees = treebank.parsed_sents()
    word2idx = build_vocab(trees)
    random.shuffle(trees)
    train_trees = trees[:int(0.8*len(trees))]
    test_trees  = trees[int(0.8*len(trees)):]

    model = TNNModel(len(word2idx), hidden_dim=64, n_classes=2)
    print(f"Training TNN on {len(train_trees)} trees (5 epochs, 300 per epoch)...")
    history = train_tnn(model, train_trees, word2idx, epochs=5)

    test_acc = evaluate_tnn(model, test_trees[:100], word2idx)
    print(f"Test accuracy (every labelled node): {test_acc:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    ax.plot(history, color="#4C72B0", marker="o")
    ax.set_xlabel("epoch"); ax.set_ylabel("avg node loss")
    ax.set_title(f"TNN training — supervised at every tree node\nTest node acc: {test_acc:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "tnn_training.png"), bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved plots to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
