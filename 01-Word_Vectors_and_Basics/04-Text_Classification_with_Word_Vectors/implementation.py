"""
Topic 1.4 -- Text Classification utilizing Word Vectors
CSE468: Natural Language Processing with Deep Learning

Sentiment classification on NLTK's Movie Reviews corpus (2,000 documents,
balanced positive/negative), comparing three ways of turning a document into
a feature vector:

  1. TF-IDF + Logistic Regression          (Topic 1.2's representation, as a baseline)
  2. Frozen pretrained Word2Vec  + a small PyTorch classifier head
  3. Fine-tuned    Word2Vec      + the same classifier head

(2) and (3) are implemented with torch.nn.EmbeddingBag, which is exactly the
right tool when "mean-pool a variable number of word vectors per document"
is the operation you need -- it does the lookup-and-average in one fused,
batched op instead of padding every document to the same length.

This is also the most direct illustration in this Phase of Collobert &
Weston's (2008) central argument: a single, generically-trained word
representation can be reused as the input layer for a different downstream
task (here, sentiment) instead of hand-engineering task-specific features.

Run directly:
    python implementation.py
"""

import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(HERE, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

W2V_PATH = os.path.join(
    HERE, "..", "1.1-Introduction-to-Word-Vectors-and-Analogies", "artifacts", "word2vec_phase1.model"
)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def clean(tokens):
    return [w.lower() for w in tokens if re.match(r"^[a-zA-Z]+$", w) and len(w) > 1]


def load_or_train_word2vec():
    """Same recipe and cache file as Topics 1.1-1.3 -- see 1.1's explanation.md."""
    if os.path.exists(W2V_PATH):
        print(f"Loading cached Word2Vec model from {W2V_PATH}")
        return Word2Vec.load(W2V_PATH)
    import nltk
    from nltk.corpus import brown, gutenberg, movie_reviews as mr
    for pkg, sub in [("brown", "corpora"), ("gutenberg", "corpora"), ("movie_reviews", "corpora")]:
        try:
            nltk.data.find(f"{sub}/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    sentences = []
    for fid in brown.fileids():
        sentences.append(clean(brown.words(fid)))
    for fid in gutenberg.fileids():
        for sent in gutenberg.sents(fid):
            c = clean(sent)
            if len(c) > 3:
                sentences.append(c)
    for fid in mr.fileids():
        sentences.append(clean(mr.words(fid)))
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, sg=1, workers=1, epochs=8)
    os.makedirs(os.path.dirname(W2V_PATH), exist_ok=True)
    model.save(W2V_PATH)
    return model


# --------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------
def load_movie_reviews():
    import nltk
    from nltk.corpus import movie_reviews
    try:
        nltk.data.find("corpora/movie_reviews")
    except LookupError:
        nltk.download("movie_reviews", quiet=True)

    texts, labels = [], []
    for label in ["pos", "neg"]:
        for fid in movie_reviews.fileids(label):
            texts.append(clean(movie_reviews.words(fid)))
            labels.append(1 if label == "pos" else 0)
    return texts, labels


# --------------------------------------------------------------------------
# 2. EmbeddingBag classifier
# --------------------------------------------------------------------------
class ReviewDataset(Dataset):
    def __init__(self, token_lists, labels, word2idx):
        self.indexed = [[word2idx.get(w, 0) for w in toks] or [0] for toks in token_lists]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.indexed[i], self.labels[i]


def collate_batch(batch):
    """Standard nn.EmbeddingBag batching: concatenate every example's token
    indices into one long 1-D tensor, and record where each example starts
    in `offsets` -- this is how EmbeddingBag handles variable-length
    documents without ever padding."""
    indices_list, labels = zip(*batch)
    offsets = [0]
    for idx in indices_list[:-1]:
        offsets.append(offsets[-1] + len(idx))
    flat_indices = torch.tensor([i for idx in indices_list for i in idx], dtype=torch.long)
    offsets = torch.tensor(offsets, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    return flat_indices, offsets, labels


class EmbeddingBagClassifier(nn.Module):
    def __init__(self, embedding_matrix, freeze_embeddings):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.embedding.weight.data.copy_(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = not freeze_embeddings
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        )

    def forward(self, flat_indices, offsets):
        pooled = self.embedding(flat_indices, offsets)  # (batch, embed_dim) -- mean-pooled per doc
        return self.classifier(pooled)


def train_pytorch_model(model, train_loader, val_loader, epochs=12, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for flat_idx, offsets, labels in train_loader:
            optimizer.zero_grad()
            logits = model(flat_idx, offsets)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / n_batches

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for flat_idx, offsets, labels in val_loader:
                logits = model(flat_idx, offsets)
                val_loss += criterion(logits, labels).item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_loss /= len(val_loader)
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"  epoch {epoch+1:2d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

    return history


def evaluate_pytorch_model(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for flat_idx, offsets, labels in loader:
            preds = model(flat_idx, offsets).argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    return np.array(all_preds), np.array(all_labels)


# --------------------------------------------------------------------------
# 3. TF-IDF + Logistic Regression baseline
# --------------------------------------------------------------------------
def tfidf_baseline(train_texts, test_texts, train_labels, test_labels):
    vectorizer = TfidfVectorizer(max_features=10000, min_df=2)
    X_train = vectorizer.fit_transform([" ".join(t) for t in train_texts])
    X_test = vectorizer.transform([" ".join(t) for t in test_texts])
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_test)
    return accuracy_score(test_labels, preds)


# --------------------------------------------------------------------------
# Visualisations
# --------------------------------------------------------------------------
def plot_loss_curves(histories, names, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)
    for hist, name in zip(histories, names):
        axes[0].plot(hist["train_loss"], label=f"{name} (train)")
        axes[0].plot(hist["val_loss"], linestyle="--", label=f"{name} (val)")
        axes[1].plot(hist["val_acc"], label=name)
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend(fontsize=8)
    axes[1].set_title("Validation accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_confusion(cm, path, title):
    fig, ax = plt.subplots(figsize=(4.5, 4), dpi=160)
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["neg", "pos"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["neg", "pos"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    w2v = load_or_train_word2vec()
    texts, labels = load_movie_reviews()
    print(f"Loaded {len(texts)} reviews ({sum(labels)} positive, {len(labels)-sum(labels)} negative)")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.15, random_state=SEED, stratify=train_labels
    )
    print(f"train={len(train_texts)}  val={len(val_texts)}  test={len(test_texts)}")

    # --- vocabulary + embedding matrix (index 0 reserved for unknown words) ---
    word2idx = {w: i + 1 for i, w in enumerate(w2v.wv.index_to_key)}
    embedding_matrix = np.vstack([np.zeros((1, w2v.wv.vector_size), dtype=np.float32), w2v.wv.vectors])

    def make_loader(texts_, labels_, shuffle):
        ds = ReviewDataset(texts_, labels_, word2idx)
        return DataLoader(ds, batch_size=32, shuffle=shuffle, collate_fn=collate_batch)

    train_loader = make_loader(train_texts, train_labels, True)
    val_loader = make_loader(val_texts, val_labels, False)
    test_loader = make_loader(test_texts, test_labels, False)

    results = {}

    print("\n=== TF-IDF + Logistic Regression baseline ===")
    results["TF-IDF + LogReg"] = tfidf_baseline(train_texts, test_texts, train_labels, test_labels)
    print(f"  test accuracy = {results['TF-IDF + LogReg']:.3f}")

    print("\n=== Frozen Word2Vec + PyTorch classifier ===")
    frozen_model = EmbeddingBagClassifier(embedding_matrix, freeze_embeddings=True)
    frozen_history = train_pytorch_model(frozen_model, train_loader, val_loader, epochs=12)
    preds, gold = evaluate_pytorch_model(frozen_model, test_loader)
    results["Frozen Word2Vec"] = accuracy_score(gold, preds)
    cm_frozen = confusion_matrix(gold, preds)

    print("\n=== Fine-tuned Word2Vec + PyTorch classifier ===")
    finetuned_model = EmbeddingBagClassifier(embedding_matrix, freeze_embeddings=False)
    finetuned_history = train_pytorch_model(finetuned_model, train_loader, val_loader, epochs=12)
    preds2, gold2 = evaluate_pytorch_model(finetuned_model, test_loader)
    results["Fine-tuned Word2Vec"] = accuracy_score(gold2, preds2)
    cm_finetuned = confusion_matrix(gold2, preds2)

    print("\n=== Test set results ===")
    for name, acc in results.items():
        print(f"  {name:20s}: {acc:.3f}")

    plot_loss_curves(
        [frozen_history, finetuned_history], ["frozen", "fine-tuned"],
        os.path.join(IMAGE_DIR, "loss_curves.png"),
    )
    plot_confusion(cm_frozen, os.path.join(IMAGE_DIR, "confusion_frozen.png"), "Frozen Word2Vec -- confusion matrix")
    plot_confusion(cm_finetuned, os.path.join(IMAGE_DIR, "confusion_finetuned.png"), "Fine-tuned Word2Vec -- confusion matrix")
    print(f"\nSaved plots to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
