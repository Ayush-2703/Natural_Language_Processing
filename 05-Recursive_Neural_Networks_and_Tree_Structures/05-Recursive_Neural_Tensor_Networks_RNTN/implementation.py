"""
Topic 5.5 -- Recursive Neural Tensor Networks (RNTN): Concepts and TF Implementation
CSE468: Natural Language Processing with Deep Learning

Implements the RNTN (Socher et al., 2013) in PyTorch — the TNN's composition
function extended with a tensor product term that captures multiplicative
interactions between child vectors. Compares TNN vs RNTN on the same data.

Run directly:
    python implementation.py
"""
import os, sys, random
from collections import Counter
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

POSITIVE_WORDS = {"good","great","excellent","wonderful","amazing","fantastic","brilliant","love","best","perfect"}
NEGATIVE_WORDS = {"bad","terrible","awful","boring","poor","dreadful","hate","worst","horrible","dull"}

def sentence_label(leaves):
    p = sum(1 for w in leaves if w.lower() in POSITIVE_WORDS)
    n = sum(1 for w in leaves if w.lower() in NEGATIVE_WORDS)
    return 1 if p >= n else 0


class TNNModel(nn.Module):
    """Standard TNN for baseline comparison."""
    def __init__(self, vocab_size, d, n_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.W = nn.Linear(2*d, d)
        self.clf = nn.Linear(d, n_classes)
    def compose(self, l, r): return torch.tanh(self.W(torch.cat([l,r],-1)))
    def forward_tree(self, tree, w2i):
        if isinstance(tree, str):
            h = self.emb(torch.tensor([w2i.get(tree.lower(), 0)])).squeeze(0)
            return h, h
        child_hs = [self.forward_tree(c, w2i) for c in tree]
        h = child_hs[0][0]
        for ci in child_hs[1:]: h = self.compose(h, ci[0])
        return h, h


class RNTNModel(nn.Module):
    """
    RNTN: h = tanh( [l;r]^T V[i] [l;r] + W[l;r] + b )
    V ∈ R^{2d×2d×d}: one bilinear form per output dimension.
    """
    def __init__(self, vocab_size, d, n_classes=2):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.V = nn.Parameter(torch.randn(d, 2*d, 2*d) * 0.05)  # tensor
        self.W = nn.Linear(2*d, d)
        self.clf = nn.Linear(d, n_classes)
        nn.init.xavier_uniform_(self.W.weight)

    def compose(self, l, r):
        concat = torch.cat([l, r], dim=-1)        # (2d,)
        # tensor term: for each output dim i, compute concat @ V[i] @ concat
        tensor_term = torch.einsum('i,ijk,k->j', concat, self.V, concat)  # (d,)
        linear_term = self.W(concat)               # (d,)
        return torch.tanh(tensor_term + linear_term)

    def forward_tree(self, tree, w2i):
        if isinstance(tree, str):
            h = self.emb(torch.tensor([w2i.get(tree.lower(), 0)])).squeeze(0)
            return h, h
        child_hs = [self.forward_tree(c, w2i) for c in tree]
        h = child_hs[0][0]
        for ci in child_hs[1:]: h = self.compose(h, ci[0])
        return h, h


def build_vocab(trees, max_words=2000):
    counts = Counter(w.lower() for t in trees for w in t.leaves())
    vocab = ["<pad>", "<unk>"] + [w for w, _ in counts.most_common(max_words-2)]
    return {w: i for i, w in enumerate(vocab)}


def train_model(model, trees, w2i, epochs=4, n_per_epoch=200):
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        random.shuffle(trees)
        total_loss = 0; nb = 0
        for tree in trees[:n_per_epoch]:
            label = sentence_label(tree.leaves())
            try:
                h, _ = model.forward_tree(tree, w2i)
                logits = model.clf(h.unsqueeze(0))
                loss = F.cross_entropy(logits, torch.tensor([label]))
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item(); nb += 1
            except RecursionError:
                pass
        avg = total_loss / max(nb, 1)
        history.append(avg)
        print(f"  epoch {epoch+1}/{epochs}  loss={avg:.4f}")
    return history


def evaluate_model(model, trees, w2i, n=200):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for tree in trees[:n]:
            label = sentence_label(tree.leaves())
            try:
                h, _ = model.forward_tree(tree, w2i)
                pred = model.clf(h.unsqueeze(0)).argmax().item()
                correct += (pred == label); total += 1
            except RecursionError:
                pass
    return correct / max(total, 1)


def plot_comparison(tnn_hist, rntn_hist, tnn_acc, rntn_acc, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=160)
    axes[0].plot(tnn_hist, color="#4C72B0", marker="o", label="TNN")
    axes[0].plot(rntn_hist, color="#C0504D", marker="s", label="RNTN")
    axes[0].legend(); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("TNN vs RNTN training loss")
    bars = axes[1].bar(["TNN", "RNTN"], [tnn_acc, rntn_acc], color=["#4C72B0", "#C0504D"])
    for bar, a in zip(bars, [tnn_acc, rntn_acc]):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f"{a:.4f}", ha="center", fontsize=10)
    axes[1].set_ylim(0.5, 0.85); axes[1].set_ylabel("test accuracy"); axes[1].set_title("TNN vs RNTN: effect of tensor term")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight", facecolor="white"); plt.close()


def main():
    trees = treebank.parsed_sents()
    random.shuffle(trees)
    train_trees = trees[:int(0.8*len(trees))]
    test_trees  = trees[int(0.8*len(trees)):]
    w2i = build_vocab(trees)
    print(f"Vocab: {len(w2i)}  Train: {len(train_trees)}  Test: {len(test_trees)}")

    print("\nTraining TNN (baseline)...")
    tnn = TNNModel(len(w2i), d=32)
    tnn_hist = train_model(tnn, train_trees, w2i, epochs=4, n_per_epoch=250)
    tnn_acc = evaluate_model(tnn, test_trees, w2i)
    print(f"TNN test accuracy: {tnn_acc:.4f}")

    print("\nTraining RNTN (with tensor term)...")
    rntn = RNTNModel(len(w2i), d=32)
    rntn_hist = train_model(rntn, train_trees, w2i, epochs=4, n_per_epoch=250)
    rntn_acc = evaluate_model(rntn, test_trees, w2i)
    print(f"RNTN test accuracy: {rntn_acc:.4f}")

    n_params_tnn  = sum(p.numel() for p in tnn.parameters())
    n_params_rntn = sum(p.numel() for p in rntn.parameters())
    print(f"\nTNN parameters:  {n_params_tnn:,}")
    print(f"RNTN parameters: {n_params_rntn:,}  (extra={n_params_rntn-n_params_tnn:,} from tensor V)")

    plot_comparison(tnn_hist, rntn_hist, tnn_acc, rntn_acc,
                    os.path.join(IMAGE_DIR, "tnn_vs_rntn.png"))
    print(f"\nSaved plots to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
