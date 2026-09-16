"""
Topic 4.3 -- Neural Networks and RNNs applied to POS Tagging
CSE468: Natural Language Processing with Deep Learning

A bidirectional LSTM POS tagger trained on the same Penn Treebank split as
Topic 4.2's HMM. The model reads each sentence in both directions and
predicts one tag per token directly from the concatenated hidden states --
no Viterbi decoding, no explicit transition model. Results are compared
directly against the HMM baseline on the identical test set.

Run directly:
    python implementation.py
"""

import os
import pickle
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(HERE, "images")
DATASET_PATH = os.path.join(
    HERE, "..", "4.1-Introduction-to-POS-Tagging", "artifacts", "pos_dataset.pkl"
)
os.makedirs(IMAGE_DIR, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(2)


def load_or_build_dataset():
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "rb") as f:
            return pickle.load(f)
    sys.path.insert(0, os.path.join(HERE, "..", "4.1-Introduction-to-POS-Tagging"))
    from implementation import build_and_cache_dataset
    return build_and_cache_dataset()


# --------------------------------------------------------------------------
# Dataset and DataLoader
# --------------------------------------------------------------------------
PAD_IDX = 0
UNK_IDX = 1
PAD_TAG = 0


class POSDataset(Dataset):
    def __init__(self, sents, word2idx, tag2idx):
        self.data = []
        for sent in sents:
            words = torch.tensor(
                [word2idx.get(w.lower(), UNK_IDX) for w, _ in sent], dtype=torch.long
            )
            tags = torch.tensor([tag2idx[t] for _, t in sent], dtype=torch.long)
            self.data.append((words, tags))

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


def collate_fn(batch):
    words, tags = zip(*batch)
    lengths = torch.tensor([len(w) for w in words], dtype=torch.long)
    words_padded = pad_sequence(words, batch_first=True, padding_value=PAD_IDX)
    tags_padded  = pad_sequence(tags,  batch_first=True, padding_value=PAD_TAG)
    return words_padded, tags_padded, lengths


# --------------------------------------------------------------------------
# BiLSTM tagger
# --------------------------------------------------------------------------
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_tags, n_layers=2, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=n_layers, batch_first=True,
            bidirectional=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_dim, n_tags)

    def forward(self, x, lengths):
        e = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.dropout(out)
        return self.fc(out)


# --------------------------------------------------------------------------
# Training and evaluation
# --------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_tokens = 0.0, 0
    for words, tags, lengths in loader:
        words, tags, lengths = words.to(device), tags.to(device), lengths.to(device)
        logits = model(words, lengths)
        B, L, T = logits.shape
        mask = (words != PAD_IDX)
        loss = criterion(logits.view(B * L, T), tags.view(B * L))
        # zero out padding positions in loss
        loss = (loss * mask.view(-1).float()).sum() / mask.sum()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item(); total_tokens += 1
    return total_loss / total_tokens


def evaluate(model, loader, device):
    model.eval()
    correct = total = sent_correct = 0
    with torch.no_grad():
        for words, tags, lengths in loader:
            words, tags, lengths = words.to(device), tags.to(device), lengths.to(device)
            logits = model(words, lengths)
            preds = logits.argmax(dim=-1)
            mask = (words != PAD_IDX)
            correct    += ((preds == tags) & mask).sum().item()
            total      += mask.sum().item()
            sent_correct += ((preds == tags) | ~mask).all(dim=1).sum().item()
    return correct / total, sent_correct / len(loader.dataset)


# --------------------------------------------------------------------------
# Visualisations
# --------------------------------------------------------------------------
def plot_training_curves(train_losses, val_accs, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
    axes[0].plot(train_losses, color="#4C72B0", marker="o", ms=3)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy loss"); axes[0].set_title("Training loss")
    axes[1].plot(val_accs, color="#4F9D69", marker="o", ms=3)
    axes[1].axhline(0.9317, linestyle="--", color="#C0504D", lw=1.5, label="HMM (0.932)")
    axes[1].axhline(0.8726, linestyle="--", color="#888",   lw=1.5, label="MFT (0.873)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation token accuracy")
    axes[1].set_title("Validation accuracy vs. baselines"); axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def plot_comparison(results, path):
    names = list(results.keys()); accs = list(results.values())
    colors = ["#888888", "#4C72B0", "#4F9D69"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=160)
    bars = ax.bar(names, accs, color=colors)
    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{a:.4f}", ha="center", fontsize=9)
    ax.set_ylim(0.84, 0.98); ax.set_ylabel("test token accuracy")
    ax.set_title("POS tagger comparison — identical Penn Treebank test split")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    dataset = load_or_build_dataset()
    train_sents, val_sents, test_sents = dataset["train"], dataset["val"], dataset["test"]
    tagset  = dataset["tagset"]
    tag2idx = dataset["tag2idx"]

    # Build extended vocab with PAD and UNK
    word_counts = Counter(w.lower() for s in train_sents for w, _ in s)
    vocab = ["<PAD>", "<UNK>"] + [w for w, c in word_counts.most_common() if c >= 1]
    word2idx = {w: i for i, w in enumerate(vocab)}

    print(f"Vocabulary: {len(vocab):,}  |  Tags: {len(tagset)}  |  Train sents: {len(train_sents):,}")

    device = torch.device("cpu")
    train_ds = POSDataset(train_sents, word2idx, tag2idx)
    val_ds   = POSDataset(val_sents,   word2idx, tag2idx)
    test_ds  = POSDataset(test_sents,  word2idx, tag2idx)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False, collate_fn=collate_fn)

    model = BiLSTMTagger(len(vocab), embed_dim=100, hidden_dim=128, n_tags=len(tagset),
                         n_layers=2, dropout=0.4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss(reduction="none")

    EPOCHS = 20
    print(f"\nTraining BiLSTM tagger for {EPOCHS} epochs...")
    train_losses, val_accs = [], []
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, _ = evaluate(model, val_loader, device)
        scheduler.step(-val_acc)
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  epoch {epoch+1:2d}/{EPOCHS}  loss={train_loss:.4f}  val_acc={val_acc:.4f}")

    model.load_state_dict(best_state)
    test_tok_acc, test_sent_acc = evaluate(model, test_loader, device)
    print(f"\nTest results (best val checkpoint):")
    print(f"  token accuracy  = {test_tok_acc:.4f}")
    print(f"  sentence accuracy = {test_sent_acc:.4f}")
    print(f"  MFT baseline was 0.8726, HMM baseline was 0.9317")

    results = {"MFT (4.1)": 0.8726, "HMM (4.2)": 0.9317, "BiLSTM (4.3)": test_tok_acc}
    plot_training_curves(train_losses, val_accs, os.path.join(IMAGE_DIR, "training_curves.png"))
    plot_comparison(results, os.path.join(IMAGE_DIR, "tagger_comparison.png"))
    print(f"\nSaved plots to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
