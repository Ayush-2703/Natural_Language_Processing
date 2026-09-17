"""
Topic 4.4 -- NER Basics and NER utilizing RNNs
CSE468: Natural Language Processing with Deep Learning

Named Entity Recognition (NER) labels spans of text as person names (PER),
organisations (ORG), locations (LOC), or miscellaneous named entities (MISC),
using the BIO annotation scheme: B-TYPE marks the Beginning of an entity
span, I-TYPE marks the Interior, O marks a non-entity token.

This topic trains a BiLSTM NER tagger on the CoNLL-2002 Spanish dataset
(NLTK's conll2002 corpus, ~8,300 training sentences) -- the same architecture
as Topic 4.3's POS tagger, with two important differences:
  - Entity-level F1 (not token accuracy) is the correct evaluation metric,
    because a partial match on a multi-token entity (e.g. getting "New" right
    but "York" wrong) is a complete miss in the downstream use case.
  - Class imbalance is severe: ~87% of tokens are O, so a model that always
    predicts O gets 87% token accuracy but 0% F1 -- a degenerate result.

Run directly:
    python implementation.py
"""

import os
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(HERE, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(2)
PAD_IDX = 0; UNK_IDX = 1; PAD_TAG = 0


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_conll2002():
    for pkg in ["conll2002"]:
        try: nltk.data.find(f"corpora/{pkg}")
        except LookupError: nltk.download(pkg, quiet=True)
    from nltk.corpus import conll2002
    train_raw = list(conll2002.iob_sents("esp.train"))
    val_raw   = list(conll2002.iob_sents("esp.testa"))
    test_raw  = list(conll2002.iob_sents("esp.testb"))
    # each sentence is a list of (word, pos, iob) triples
    train = [[(w, t) for w, _, t in s] for s in train_raw]
    val   = [[(w, t) for w, _, t in s] for s in val_raw]
    test  = [[(w, t) for w, _, t in s] for s in test_raw]
    return train, val, test


class NERDataset(Dataset):
    def __init__(self, sents, word2idx, tag2idx):
        self.data = []
        for sent in sents:
            words = torch.tensor([word2idx.get(w.lower(), UNK_IDX) for w, _ in sent], dtype=torch.long)
            tags  = torch.tensor([tag2idx[t] for _, t in sent], dtype=torch.long)
            self.data.append((words, tags))
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


def collate_fn(batch):
    words, tags = zip(*batch)
    lengths = torch.tensor([len(w) for w in words], dtype=torch.long)
    return (pad_sequence(words, batch_first=True, padding_value=PAD_IDX),
            pad_sequence(tags,  batch_first=True, padding_value=PAD_TAG), lengths)


# --------------------------------------------------------------------------
# Model (same BiLSTM architecture as Topic 4.3)
# --------------------------------------------------------------------------
class BiLSTMNER(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_tags, dropout=0.35):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_dim, n_tags)

    def forward(self, x, lengths):
        e = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return self.fc(self.dropout(out))


# --------------------------------------------------------------------------
# Entity-level F1 (seqeval-style, computed from scratch)
# --------------------------------------------------------------------------
def extract_entities(tags_seq, idx2tag):
    """Convert a BIO tag sequence to a set of (type, start, end) spans."""
    entities = set()
    current_type, current_start = None, None
    for i, tag_idx in enumerate(tags_seq):
        tag = idx2tag[tag_idx]
        if tag.startswith("B-"):
            if current_type is not None:
                entities.add((current_type, current_start, i))
            current_type = tag[2:]
            current_start = i
        elif tag.startswith("I-"):
            if current_type is None or tag[2:] != current_type:
                # malformed sequence -- treat as beginning
                if current_type is not None:
                    entities.add((current_type, current_start, i))
                current_type = tag[2:]; current_start = i
        else:  # O
            if current_type is not None:
                entities.add((current_type, current_start, i))
            current_type = None
    if current_type is not None:
        entities.add((current_type, current_start, len(tags_seq)))
    return entities


def entity_f1(model, loader, idx2tag):
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for words, tags, lengths in loader:
            logits = model(words, lengths)
            preds  = logits.argmax(dim=-1)
            for i, length in enumerate(lengths.tolist()):
                gold_ents = extract_entities(tags[i, :length].tolist(), idx2tag)
                pred_ents = extract_entities(preds[i, :length].tolist(), idx2tag)
                tp += len(gold_ents & pred_ents)
                fp += len(pred_ents - gold_ents)
                fn += len(gold_ents - pred_ents)
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1


def token_accuracy(model, loader):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for words, tags, lengths in loader:
            preds = model(words, lengths).argmax(dim=-1)
            mask  = (words != PAD_IDX)
            correct += ((preds == tags) & mask).sum().item()
            total   += mask.sum().item()
    return correct / total


# --------------------------------------------------------------------------
# Visualisations
# --------------------------------------------------------------------------
def plot_training(train_losses, val_f1s, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
    axes[0].plot(train_losses, color="#4C72B0", marker="o", ms=3)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].set_title("Training loss")
    axes[1].plot(val_f1s, color="#C0504D", marker="o", ms=3)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("entity-level F1"); axes[1].set_title("Validation F1")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def plot_entity_distribution(train_sents, tagset, path):
    entity_counts = Counter()
    for sent in train_sents:
        for _, tag in sent:
            if tag.startswith("B-"):
                entity_counts[tag[2:]] += 1
    labels, vals = zip(*entity_counts.most_common())
    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    ax.bar(labels, vals, color=["#4C72B0", "#4F9D69", "#C0504D", "#C9A227"])
    ax.set_ylabel("entity mentions in training data")
    ax.set_title("CoNLL-2002 entity type distribution (training split)")
    for i, v in enumerate(vals):
        ax.text(i, v + 30, str(v), ha="center", fontsize=9)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print("Loading CoNLL-2002 Spanish NER data...")
    train, val, test = load_conll2002()
    print(f"  train: {len(train)} sents  val: {len(val)}  test: {len(test)}")

    tagset = sorted({t for s in train for _, t in s})
    tag2idx = {t: i for i, t in enumerate(tagset)}
    idx2tag = {i: t for t, i in tag2idx.items()}
    print(f"  NER tags: {tagset}")

    wc = Counter(w.lower() for s in train for w, _ in s)
    vocab = ["<PAD>", "<UNK>"] + [w for w, c in wc.most_common() if c >= 2]
    word2idx = {w: i for i, w in enumerate(vocab)}
    print(f"  vocabulary: {len(vocab):,}")

    train_dl = DataLoader(NERDataset(train, word2idx, tag2idx), 64, True, collate_fn=collate_fn)
    val_dl   = DataLoader(NERDataset(val,   word2idx, tag2idx), 128, False, collate_fn=collate_fn)
    test_dl  = DataLoader(NERDataset(test,  word2idx, tag2idx), 128, False, collate_fn=collate_fn)

    model = BiLSTMNER(len(vocab), embed_dim=100, hidden_dim=128, n_tags=len(tagset))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # class-weighted loss: down-weight O tokens so rare entity classes get more gradient
    class_weights = torch.ones(len(tagset))
    o_idx = tag2idx.get("O", 0)
    class_weights[o_idx] = 0.3
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")

    EPOCHS = 12
    print(f"\nTraining BiLSTM NER tagger for {EPOCHS} epochs...")
    train_losses, val_f1s = [], []
    best_val_f1, best_state = 0.0, None

    for epoch in range(EPOCHS):
        model.train(); total_loss, nb = 0.0, 0
        for words, tags, lengths in train_dl:
            logits = model(words, lengths); B, L, T = logits.shape
            mask = (words != PAD_IDX)
            loss = (criterion(logits.view(B*L, T), tags.view(B*L)) * mask.view(-1).float()).sum() / mask.sum()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); nb += 1
        avg_loss = total_loss / nb
        _, _, val_f1 = entity_f1(model, val_dl, idx2tag)
        train_losses.append(avg_loss); val_f1s.append(val_f1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  epoch {epoch+1:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_F1={val_f1:.4f}")

    model.load_state_dict(best_state)
    tok_acc = token_accuracy(model, test_dl)
    p, r, f1 = entity_f1(model, test_dl, idx2tag)
    print(f"\nTest results (best val checkpoint):")
    print(f"  token accuracy = {tok_acc:.4f}  (high -- mostly O tokens)")
    print(f"  entity precision = {p:.4f}  recall = {r:.4f}  F1 = {f1:.4f}")
    print(f"  (entity F1 is the real metric -- token acc is inflated by ~87% O tokens)")

    plot_training(train_losses, val_f1s, os.path.join(IMAGE_DIR, "ner_training.png"))
    plot_entity_distribution(train, tagset, os.path.join(IMAGE_DIR, "entity_distribution.png"))
    print(f"\nSaved plots to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
