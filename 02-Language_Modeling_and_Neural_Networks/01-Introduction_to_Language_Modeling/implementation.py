"""
Topic 2.1 -- Introduction to Language Modeling and Neural Networks
CSE468: Natural Language Processing with Deep Learning

A language model assigns a probability to a sequence of words. This topic:

  1. Builds and caches the train/test split + vocabulary that Topics 2.2 and
     2.3 will both reuse, so all three topics' perplexity numbers are
     directly comparable -- same data, same vocabulary, same <UNK> policy.
  2. Implements the weakest possible baseline -- a UNIGRAM model that
     ignores all context -- and evaluates it with perplexity, the standard
     language-modelling metric used throughout this Phase.
  3. Plots the corpus's word-frequency distribution (Zipf's law), which is
     both the motivation for capping the vocabulary with <UNK> and a preview
     of why n-gram counting runs into trouble (Topic 2.2) in a way neural
     models (Topic 2.3) are built to handle more gracefully.

Run directly:
    python implementation.py
"""

import math
import os
import pickle
import random
import re
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(HERE, "images")
ARTIFACT_DIR = os.path.join(HERE, "artifacts")
DATASET_PATH = os.path.join(ARTIFACT_DIR, "lm_dataset.pkl")
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

VOCAB_SIZE = 5000  # excluding special tokens
SEED = 42

BOS, EOS, UNK = "<bos>", "<eos>", "<unk>"


# --------------------------------------------------------------------------
# 1. Shared dataset: build once here, cached for Topics 2.2 and 2.3
# --------------------------------------------------------------------------
def clean_sentence(words):
    return [w.lower() for w in words if re.match(r"^[a-zA-Z]+$", w) and len(w) > 1]


def build_and_cache_dataset():
    try:
        nltk.data.find("corpora/brown")
    except LookupError:
        nltk.download("brown", quiet=True)
    from nltk.corpus import brown

    sentences = [clean_sentence(s) for s in brown.sents()]
    sentences = [s for s in sentences if len(s) >= 2]  # drop degenerate one-word "sentences"

    random.seed(SEED)
    random.shuffle(sentences)
    n_test = int(0.1 * len(sentences))
    test_sents = sentences[:n_test]
    train_sents = sentences[n_test:]

    # Vocabulary built from TRAINING data only -- the test set must never
    # influence which words are "known" or the perplexity number is biased.
    counts = Counter(w for s in train_sents for w in s)
    most_common = [w for w, _ in counts.most_common(VOCAB_SIZE)]
    vocab = [BOS, EOS, UNK] + most_common
    word2idx = {w: i for i, w in enumerate(vocab)}

    def encode(sents):
        out = []
        for s in sents:
            idx_seq = [word2idx[BOS]] + [word2idx.get(w, word2idx[UNK]) for w in s] + [word2idx[EOS]]
            out.append(idx_seq)
        return out

    dataset = {
        "vocab": vocab,
        "word2idx": word2idx,
        "train_sents": train_sents,      # raw tokens, train, for frequency stats
        "train_encoded": encode(train_sents),
        "test_encoded": encode(test_sents),
        "raw_word_counts": counts,
    }
    with open(DATASET_PATH, "wb") as f:
        pickle.dump(dataset, f)
    return dataset


def load_or_build_dataset():
    if os.path.exists(DATASET_PATH):
        print(f"Loading cached LM dataset from {DATASET_PATH}")
        with open(DATASET_PATH, "rb") as f:
            return pickle.load(f)
    print("No cached dataset found -- building from Brown corpus...")
    return build_and_cache_dataset()


# --------------------------------------------------------------------------
# 2. Unigram language model
# --------------------------------------------------------------------------
class UnigramModel:
    """P(w) = count(w) / total_count, estimated from training data. Excludes
    <bos> from the counts (it's a conditioning device, never a word being
    predicted) but includes <eos> (predicting "the sentence ends now" is
    part of the task)."""

    def __init__(self, dataset):
        word2idx = dataset["word2idx"]
        counts = np.zeros(len(word2idx))
        for seq in dataset["train_encoded"]:
            for idx in seq[1:]:  # skip the leading <bos>
                counts[idx] += 1
        self.probs = counts / counts.sum()
        self.word2idx = word2idx

    def log_prob(self, idx):
        return math.log(self.probs[idx])

    def perplexity(self, encoded_sentences):
        total_log_prob, total_tokens = 0.0, 0
        for seq in encoded_sentences:
            for idx in seq[1:]:  # score every token except the leading <bos>
                total_log_prob += self.log_prob(idx)
                total_tokens += 1
        return math.exp(-total_log_prob / total_tokens)


# --------------------------------------------------------------------------
# Visualisations
# --------------------------------------------------------------------------
def plot_zipf(raw_word_counts, path):
    freqs = sorted(raw_word_counts.values(), reverse=True)
    ranks = np.arange(1, len(freqs) + 1)

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=160)
    ax.loglog(ranks, freqs, color="#4C72B0", lw=1.5)
    ax.axvline(VOCAB_SIZE, color="#C0504D", linestyle="--", lw=1.5,
               label=f"vocabulary cutoff (top {VOCAB_SIZE})")
    ax.set_xlabel("rank (log scale)")
    ax.set_ylabel("frequency (log scale)")
    ax.set_title("Zipf's law: word frequency vs. rank in the Brown corpus")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_chain_rule_diagram(path):
    fig, ax = plt.subplots(figsize=(9, 3), dpi=170)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    words = ["the", "cat", "sat", "down"]
    terms = [
        "P(the)",
        "P(cat | the)",
        "P(sat | the, cat)",
        "P(down | the, cat, sat)",
    ]
    x = 0.5
    for i, (w, t) in enumerate(zip(words, terms)):
        ax.text(x, 0.65, w, fontsize=13, fontweight="bold", ha="center", color="#2E5A88")
        ax.text(x, 0.25, t, fontsize=10.5, ha="center", color="#333")
        if i < len(words) - 1:
            ax.annotate("", xy=(x + 1.55, 0.65), xytext=(x + 0.55, 0.65),
                        arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.6))
        x += 2.1
    ax.text(4.6, 1.0, 'P(the cat sat down)  =  P(the) · P(cat|the) · P(sat|the,cat) · P(down|the,cat,sat)',
            fontsize=10.5, ha="center", style="italic")
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(0, 1.3)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    dataset = load_or_build_dataset()
    print(f"Vocabulary size (incl. special tokens): {len(dataset['vocab'])}")
    print(f"Train sentences: {len(dataset['train_encoded'])}  Test sentences: {len(dataset['test_encoded'])}")
    n_unk_train = sum(1 for seq in dataset["train_encoded"] for idx in seq if idx == dataset["word2idx"][UNK])
    n_tok_train = sum(len(seq) for seq in dataset["train_encoded"])
    print(f"<unk> rate in training data: {n_unk_train/n_tok_train:.2%}")

    model = UnigramModel(dataset)
    train_pp = model.perplexity(dataset["train_encoded"])
    test_pp = model.perplexity(dataset["test_encoded"])
    print(f"\nUnigram model perplexity:  train = {train_pp:.1f}   test = {test_pp:.1f}")
    print(f"(for reference: a model that guessed uniformly at random over "
          f"{len(dataset['vocab'])} words would have perplexity {len(dataset['vocab'])})")

    plot_zipf(dataset["raw_word_counts"], os.path.join(IMAGE_DIR, "zipf_law.png"))
    plot_chain_rule_diagram(os.path.join(IMAGE_DIR, "chain_rule_diagram.png"))
    print(f"\nSaved plots to {IMAGE_DIR}")
    print(f"Saved shared dataset to {DATASET_PATH} for Topics 2.2 and 2.3 to reuse")


if __name__ == "__main__":
    main()
