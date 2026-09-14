"""
Topic 4.1 -- Introduction to Parts-of-Speech (POS) Tagging

POS tagging assigns a grammatical category (noun, verb, adjective, ...) to
every token in a sentence. This is both a useful NLP sub-task in its own
right and the canonical sequence labelling benchmark for comparing taggers,
which is why Topics 4.2 and 4.3 both evaluate on the data prepared here.

This topic:
  1. Loads the Penn Treebank sample from NLTK (3,913 sentences, 43 POS tags),
     cleans it, and caches the 80/10/10 train/val/test split that Topics 4.2
     and 4.3 reuse.
  2. Implements the simplest possible non-trivial baseline: the Most-Frequent-
     Tag (MFT) tagger -- assign each word the POS tag it most often had in
     training data; unknown words get the globally most common tag.
  3. Visualises the tag distribution and a tag transition heatmap.

Run directly:
    python implementation.py
"""

import os
import pickle
import random
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import nltk
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(HERE, "images")
ARTIFACT_DIR = os.path.join(HERE, "artifacts")
DATASET_PATH = os.path.join(ARTIFACT_DIR, "pos_dataset.pkl")
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

SEED = 42


# --------------------------------------------------------------------------
# 1. Dataset
# --------------------------------------------------------------------------
def ensure_nltk_data():
    for pkg, sub in [("treebank", "corpora"), ("punkt", "tokenizers"),
                     ("punkt_tab", "tokenizers")]:
        try:
            nltk.data.find(f"{sub}/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def clean_sent(tagged_sent):
    """Remove Penn Treebank trace tokens (-NONE-, -LRB-, etc.)."""
    return [(w, t) for w, t in tagged_sent
            if t != "-NONE-" and not t.startswith("-")]


def build_and_cache_dataset():
    ensure_nltk_data()
    from nltk.corpus import treebank

    raw = [clean_sent(s) for s in treebank.tagged_sents()]
    sents = [s for s in raw if len(s) >= 2]

    rng = random.Random(SEED)
    rng.shuffle(sents)
    n = len(sents)
    train = sents[:int(0.80 * n)]
    val   = sents[int(0.80 * n):int(0.90 * n)]
    test  = sents[int(0.90 * n):]

    vocab   = sorted({w for s in train for w, _ in s})
    tagset  = sorted({t for s in train for _, t in s})
    word2idx = {w: i for i, w in enumerate(vocab)}
    tag2idx  = {t: i for i, t in enumerate(tagset)}

    dataset = dict(
        train=train, val=val, test=test,
        vocab=vocab, tagset=tagset,
        word2idx=word2idx, tag2idx=tag2idx,
    )
    with open(DATASET_PATH, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved dataset to {DATASET_PATH}")
    return dataset


def load_or_build_dataset():
    if os.path.exists(DATASET_PATH):
        print(f"Loading cached POS dataset from {DATASET_PATH}")
        with open(DATASET_PATH, "rb") as f:
            return pickle.load(f)
    print("Building POS dataset from Penn Treebank...")
    return build_and_cache_dataset()


# --------------------------------------------------------------------------
# 2. Most-Frequent-Tag baseline
# --------------------------------------------------------------------------
class MostFrequentTagger:
    def __init__(self, train_sents):
        self.word_tag = {}
        word_tag_counts = defaultdict(Counter)
        global_tag_counts = Counter()
        for sent in train_sents:
            for w, t in sent:
                word_tag_counts[w.lower()][t] += 1
                global_tag_counts[t] += 1
        for w, counts in word_tag_counts.items():
            self.word_tag[w] = counts.most_common(1)[0][0]
        self.default_tag = global_tag_counts.most_common(1)[0][0]

    def tag(self, words):
        return [self.word_tag.get(w.lower(), self.default_tag) for w in words]


def evaluate_tagger(tagger, sents):
    correct, total, sent_correct = 0, 0, 0
    for sent in sents:
        words = [w for w, _ in sent]
        gold  = [t for _, t in sent]
        pred  = tagger.tag(words)
        hits  = sum(p == g for p, g in zip(pred, gold))
        correct += hits
        total   += len(gold)
        sent_correct += (hits == len(gold))
    return correct / total, sent_correct / len(sents)


# --------------------------------------------------------------------------
# 3. Visualisations
# --------------------------------------------------------------------------
def plot_tag_distribution(train_sents, path):
    counts = Counter(t for s in train_sents for _, t in s)
    tags, vals = zip(*counts.most_common(25))
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=160)
    colors = plt.cm.tab20(np.linspace(0, 1, len(tags)))
    ax.bar(tags, vals, color=colors)
    ax.set_xlabel("Penn Treebank POS tag"); ax.set_ylabel("frequency in training set")
    ax.set_title("Top-25 POS tag distribution -- Penn Treebank training split")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def plot_transition_heatmap(train_sents, tagset, path):
    """Bigram POS transition matrix -- gives an intuition for the Markov
    structure that the HMM in Topic 4.2 will exploit."""
    T = len(tagset); tag2i = {t: i for i, t in enumerate(tagset)}
    mat = np.zeros((T, T), dtype=np.float32)
    for sent in train_sents:
        tags = [t for _, t in sent]
        for a, b in zip(tags[:-1], tags[1:]):
            mat[tag2i[a], tag2i[b]] += 1
    row_sums = mat.sum(axis=1, keepdims=True)
    mat = mat / (row_sums + 1e-8)

    # Show only top-15 most-frequent tags for readability
    freq = Counter(t for s in train_sents for _, t in s)
    top15 = [t for t, _ in freq.most_common(15)]
    idx   = [tag2i[t] for t in top15]
    sub   = mat[np.ix_(idx, idx)]

    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
    im = ax.imshow(sub, cmap="YlOrRd", vmin=0, vmax=0.6)
    ax.set_xticks(range(15)); ax.set_xticklabels(top15, rotation=45, ha="right")
    ax.set_yticks(range(15)); ax.set_yticklabels(top15)
    ax.set_title("POS tag bigram transition probability P(next tag | current tag)\n(top 15 tags, training data)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def plot_oov_vs_accuracy(tagger, test_sents, train_words, path):
    """Compare accuracy on words seen in training vs. OOV words."""
    seen_c, seen_t = 0, 0
    oov_c,  oov_t  = 0, 0
    for sent in test_sents:
        words = [w for w, _ in sent]
        gold  = [t for _, t in sent]
        pred  = tagger.tag(words)
        for w, g, p in zip(words, gold, pred):
            if w.lower() in train_words:
                seen_t += 1; seen_c += (p == g)
            else:
                oov_t += 1;  oov_c  += (p == g)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
    labels = ["Seen words", "OOV words"]
    values = [seen_c / max(seen_t, 1), oov_c / max(oov_t, 1)]
    counts = [seen_t, oov_t]
    bars = ax.bar(labels, values, color=["#4C72B0", "#C0504D"], alpha=0.8)
    for bar, acc, cnt in zip(bars, values, counts):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01, f"{acc:.3f}\n(n={cnt:,})",
                ha="center", fontsize=9)
    ax.set_ylim(0, 1.1); ax.set_ylabel("token accuracy")
    ax.set_title("MFT tagger: seen-word vs. OOV accuracy")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    dataset = load_or_build_dataset()
    train, val, test = dataset["train"], dataset["val"], dataset["test"]
    tagset  = dataset["tagset"]

    print(f"\nDataset: {len(train)} train / {len(val)} val / {len(test)} test sentences")
    print(f"  train tokens: {sum(len(s) for s in train):,}")
    print(f"  tagset size: {len(tagset)} tags: {tagset[:10]} ...")

    tagger = MostFrequentTagger(train)
    train_tok_acc, train_sent_acc = evaluate_tagger(tagger, train)
    val_tok_acc,   val_sent_acc   = evaluate_tagger(tagger, val)
    test_tok_acc,  test_sent_acc  = evaluate_tagger(tagger, test)

    print(f"\nMost-Frequent-Tag baseline results:")
    print(f"  train  token acc = {train_tok_acc:.4f}  sentence acc = {train_sent_acc:.4f}")
    print(f"  val    token acc = {val_tok_acc:.4f}  sentence acc = {val_sent_acc:.4f}")
    print(f"  test   token acc = {test_tok_acc:.4f}  sentence acc = {test_sent_acc:.4f}")
    print(f"  (lower bound for Topics 4.2 and 4.3 to beat)")

    train_words = {w.lower() for s in train for w, _ in s}
    n_oov = sum(1 for s in test for w, _ in s if w.lower() not in train_words)
    n_test_tok = sum(len(s) for s in test)
    print(f"\n  OOV rate in test set: {n_oov}/{n_test_tok} = {n_oov/n_test_tok:.2%}")

    plot_tag_distribution(train, os.path.join(IMAGE_DIR, "tag_distribution.png"))
    plot_transition_heatmap(train, tagset, os.path.join(IMAGE_DIR, "transition_heatmap.png"))
    plot_oov_vs_accuracy(tagger, test, train_words, os.path.join(IMAGE_DIR, "oov_vs_accuracy.png"))
    print(f"\nSaved plots to {IMAGE_DIR}")
    print(f"Cached dataset to {DATASET_PATH} -- reused by Topics 4.2 and 4.3")


if __name__ == "__main__":
    main()
