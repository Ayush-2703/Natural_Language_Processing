"""
Topic 5.4 -- Converting Trees to Sequences
CSE468: Natural Language Processing with Deep Learning

Demonstrates three ways to convert a parse tree into a flat sequence,
then trains a BiLSTM on each representation and compares downstream
performance — showing whether structural information in the sequence
encoding helps the model beyond just reading the original left-to-right words.

Run directly:
    python implementation.py
"""
import os, sys, random
from collections import Counter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch, torch.nn as nn
import nltk
from nltk.corpus import treebank

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(HERE, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)
torch.manual_seed(42); random.seed(42)
sys.setrecursionlimit(10000)

POSITIVE_WORDS = {"good","great","excellent","wonderful","amazing","fantastic","brilliant","love","best","perfect"}
NEGATIVE_WORDS = {"bad","terrible","awful","boring","poor","dreadful","hate","worst","horrible","dull"}

def sentence_label(leaves):
    p = sum(1 for w in leaves if w.lower() in POSITIVE_WORDS)
    n = sum(1 for w in leaves if w.lower() in NEGATIVE_WORDS)
    return 1 if p >= n else 0


# ── Three tree-to-sequence representations ───────────────────────────────────
def tree_to_words(tree):
    """Baseline: original word sequence (no structural info)."""
    return tree.leaves()

def tree_to_linearised(tree):
    """Depth-first linearisation with bracket tokens."""
    if isinstance(tree, str):
        return [tree]
    label = tree.label().split("-")[0]
    tokens = [f"<{label}>"]
    for c in tree:
        tokens.extend(tree_to_linearised(c))
    tokens.append(f"</{label}>")
    return tokens

def tree_to_labels_and_words(tree):
    """Interleave phrase labels as special tokens before each span."""
    if isinstance(tree, str):
        return [tree]
    label = tree.label().split("-")[0]
    tokens = [f"[{label}]"]
    for c in tree:
        tokens.extend(tree_to_labels_and_words(c))
    return tokens


# ── Simple classifier ─────────────────────────────────────────────────────────
class SeqClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*hidden_dim, 2)
        self.drop = nn.Dropout(0.3)
    def forward(self, x):
        e = self.drop(self.emb(x))
        _, (h, _) = self.lstm(e)
        h = torch.cat([h[0], h[1]], dim=-1)
        return self.fc(self.drop(h))


def train_and_eval(sequences, labels, name):
    vocab = Counter(t for s in sequences for t in s)
    w2i = {"<pad>": 0, "<unk>": 1}
    for w, _ in vocab.most_common(5000):
        w2i.setdefault(w, len(w2i))

    idx_seqs = [[w2i.get(t, 1) for t in s] for s in sequences]
    max_len = min(max(len(s) for s in idx_seqs), 120)
    padded = torch.zeros(len(idx_seqs), max_len, dtype=torch.long)
    for i, s in enumerate(idx_seqs):
        l = min(len(s), max_len)
        padded[i, :l] = torch.tensor(s[:l])
    labels_t = torch.tensor(labels, dtype=torch.long)

    perm = torch.randperm(len(labels_t))
    n_train = int(0.8 * len(labels_t))
    tr_X, tr_y = padded[perm[:n_train]], labels_t[perm[:n_train]]
    te_X, te_y = padded[perm[n_train:]], labels_t[perm[n_train:]]

    model = SeqClassifier(len(w2i))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    history = []
    for epoch in range(10):
        model.train()
        perm2 = torch.randperm(len(tr_X))
        total_loss = 0; nb = 0
        for i in range(0, len(tr_X), 64):
            batch_idx = perm2[i:i+64]
            loss = crit(model(tr_X[batch_idx]), tr_y[batch_idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); nb += 1
        history.append(total_loss / nb)
    model.eval()
    with torch.no_grad():
        preds = model(te_X).argmax(1)
        acc = (preds == te_y).float().mean().item()
    print(f"  {name:30s}: test_acc={acc:.4f}  vocab_size={len(w2i)}")
    return acc, history


def main():
    trees = treebank.parsed_sents()
    print(f"Loaded {len(trees)} trees")
    results = {}
    histories = {}
    for name, fn in [
        ("Words only", tree_to_words),
        ("Linearised (brackets)", tree_to_linearised),
        ("Label tokens", tree_to_labels_and_words),
    ]:
        sequences = [fn(t) for t in trees]
        labels = [sentence_label(t.leaves()) for t in trees]
        acc, hist = train_and_eval(sequences, labels, name)
        results[name] = acc; histories[name] = hist

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=160)
    for name, hist in histories.items():
        axes[0].plot(hist, label=name, marker="o", ms=3)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("training loss"); axes[0].legend(fontsize=8); axes[0].set_title("Training loss by sequence representation")
    names, accs = list(results.keys()), list(results.values())
    bars = axes[1].bar(names, accs, color=["#888", "#4C72B0", "#4F9D69"])
    for bar, a in zip(bars, accs):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f"{a:.4f}", ha="center", fontsize=9)
    axes[1].set_ylim(0.45, 0.75); axes[1].set_ylabel("test accuracy"); axes[1].set_title("Tree-to-sequence: does structure help?")
    plt.xticks(rotation=10, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "sequence_comparison.png"), bbox_inches="tight", facecolor="white"); plt.close()
    print(f"Saved plots to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
