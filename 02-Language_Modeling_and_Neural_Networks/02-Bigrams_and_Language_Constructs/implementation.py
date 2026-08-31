"""
Topic 2.2 -- Bigrams and Language Constructs
CSE468: Natural Language Processing with Deep Learning

Builds a classical, counting-based bigram language model on the exact
train/test split and vocabulary cached by Topic 2.1, and walks through why
raw maximum-likelihood bigram counting is actually broken in practice:

  1. Raw MLE:        P(w_t|w_{t-1}) = count(w_{t-1},w_t) / count(w_{t-1})
                      -- fails outright the moment a test bigram was never
                      seen in training, which happens constantly.
  2. Laplace (add-1): pretend every possible bigram was seen one extra time.
                      Fixes the zero-probability crash, but over-corrects.
  3. Linear interpolation with the unigram model, with the interpolation
     weight tuned on a held-out validation slice of the training data.

Run directly:
    python implementation.py
"""

import math
import os
import pickle
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(HERE, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

DATASET_PATH = os.path.join(
    HERE, "..", "2.1-Introduction-to-Language-Modeling", "artifacts", "lm_dataset.pkl"
)


def load_or_build_dataset():
    if os.path.exists(DATASET_PATH):
        print(f"Loading cached LM dataset from {DATASET_PATH}")
        with open(DATASET_PATH, "rb") as f:
            return pickle.load(f)
    print("No cached dataset found -- rebuilding from Topic 2.1's recipe...")
    import sys
    sys.path.insert(0, os.path.join(HERE, "..", "2.1-Introduction-to-Language-Modeling"))
    from implementation import build_and_cache_dataset  # noqa: E402
    return build_and_cache_dataset()


# --------------------------------------------------------------------------
# Bigram counting
# --------------------------------------------------------------------------
def count_bigrams(encoded_sentences, vocab_size):
    """unigram_counts here counts each word's occurrences as the FIRST half
    of a bigram (i.e. "how many times did something follow this word") --
    correct as the MLE/Laplace denominator, but NOT a true unigram
    distribution, because <eos> never precedes anything within a sentence
    and would get a count of exactly zero. See true_unigram_distribution()
    for the distribution actually used as the interpolation backoff."""
    unigram_counts = np.zeros(vocab_size)
    bigram_counts = defaultdict(lambda: np.zeros(vocab_size))
    for seq in encoded_sentences:
        for prev, nxt in zip(seq[:-1], seq[1:]):
            unigram_counts[prev] += 1
            bigram_counts[prev][nxt] += 1
    return unigram_counts, bigram_counts


def true_unigram_distribution(encoded_sentences, vocab_size):
    """A real P(w) over every predictable token, INCLUDING <eos> -- matches
    Topic 2.1's UnigramModel exactly (every token in seq[1:], i.e. every
    word except each sequence's leading <bos>). This -- not the bigram-
    context counts above -- is the correct unigram backoff distribution for
    interpolation; using the wrong one would silently give <eos> a backoff
    probability of zero, breaking interpolation for exactly the test
    bigrams it's supposed to rescue."""
    counts = np.zeros(vocab_size)
    for seq in encoded_sentences:
        for idx in seq[1:]:
            counts[idx] += 1
    return counts / counts.sum()


class BigramModel:
    def __init__(self, unigram_counts, bigram_counts, vocab_size, smoothing="none", k=1.0, lam=0.7,
                 unigram_probs=None):
        self.unigram_counts = unigram_counts
        self.bigram_counts = bigram_counts
        self.V = vocab_size
        self.smoothing = smoothing
        self.k = k
        self.lam = lam
        self.unigram_probs = unigram_probs  # only needed for interpolation

    def prob(self, prev, nxt):
        c_prev = self.unigram_counts[prev]
        c_pair = self.bigram_counts[prev][nxt] if prev in self.bigram_counts else 0.0

        if self.smoothing == "none":
            if c_prev == 0:
                return 0.0
            return c_pair / c_prev

        if self.smoothing == "laplace":
            return (c_pair + self.k) / (c_prev + self.k * self.V)

        if self.smoothing == "interpolation":
            p_bigram = c_pair / c_prev if c_prev > 0 else 0.0
            p_unigram = self.unigram_probs[nxt]
            return self.lam * p_bigram + (1 - self.lam) * p_unigram

        raise ValueError(self.smoothing)

    def perplexity(self, encoded_sentences, return_failure_info=False):
        total_log_prob, total_tokens = 0.0, 0
        n_zero = 0
        for seq in encoded_sentences:
            for prev, nxt in zip(seq[:-1], seq[1:]):
                p = self.prob(prev, nxt)
                total_tokens += 1
                if p <= 0:
                    n_zero += 1
                    if not return_failure_info:
                        return float("inf")
                    continue
                total_log_prob += math.log(p)
        if return_failure_info:
            return n_zero, total_tokens
        return math.exp(-total_log_prob / total_tokens)


def tune_interpolation_lambda(train_encoded, unigram_counts, bigram_counts, unigram_probs, vocab_size):
    """Carve a validation slice out of the training sentences and grid-search
    lambda on it -- never touch the test set while choosing a hyperparameter."""
    random.seed(123)
    shuffled = train_encoded[:]
    random.shuffle(shuffled)
    n_val = int(0.1 * len(shuffled))
    val_sents = shuffled[:n_val]

    best_lam, best_pp = None, float("inf")
    results = []
    for lam in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        model = BigramModel(unigram_counts, bigram_counts, vocab_size,
                             smoothing="interpolation", lam=lam, unigram_probs=unigram_probs)
        pp = model.perplexity(val_sents)
        results.append((lam, pp))
        if pp < best_pp:
            best_pp, best_lam = pp, lam
    return best_lam, results


# --------------------------------------------------------------------------
# Visualisations
# --------------------------------------------------------------------------
def plot_bigram_heatmap(bigram_counts, word2idx, idx2word, words, path):
    n = len(words)
    mat = np.zeros((n, n))
    for i, w1 in enumerate(words):
        idx1 = word2idx[w1]
        row = bigram_counts.get(idx1, None)
        total = row.sum() if row is not None else 0
        for j, w2 in enumerate(words):
            idx2 = word2idx[w2]
            mat[i, j] = (row[idx2] / total) if (row is not None and total > 0) else 0.0

    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    im = ax.imshow(mat, cmap="viridis")
    ax.set_xticks(range(n)); ax.set_xticklabels(words, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(words)
    ax.set_xlabel("next word"); ax.set_ylabel("previous word")
    ax.set_title("P(next word | previous word) -- raw MLE bigram estimates")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_perplexity_comparison(results, path):
    names = list(results.keys())
    values = [results[n] if np.isfinite(results[n]) else 0 for n in names]
    is_inf = [not np.isfinite(results[n]) for n in names]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    bars = ax.bar(names, values, color=["#C0504D" if inf else "#4C72B0" for inf in is_inf])
    for bar, name, inf in zip(bars, names, is_inf):
        label = "undefined\n(zero-prob test bigram)" if inf else f"{results[name]:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5 if not inf else 5,
                label, ha="center", fontsize=9)
    ax.set_ylabel("test-set perplexity")
    ax.set_title("Bigram language model: effect of smoothing strategy")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_lambda_tuning(results, path):
    lams, pps = zip(*results)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    ax.plot(lams, pps, marker="o", color="#4C72B0")
    best_lam, best_pp = min(results, key=lambda x: x[1])
    ax.scatter([best_lam], [best_pp], color="#C0504D", zorder=5, s=80, label=f"best: λ={best_lam}")
    ax.set_xlabel("λ  (weight on the bigram estimate)")
    ax.set_ylabel("validation perplexity")
    ax.set_title("Tuning the interpolation weight λ")
    ax.legend()
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
    V = len(vocab)

    unigram_counts, bigram_counts = count_bigrams(dataset["train_encoded"], V)
    unigram_probs = true_unigram_distribution(dataset["train_encoded"], V)
    n_observed_bigrams = sum((row > 0).sum() for row in bigram_counts.values())
    print(f"Vocabulary size: {V}")
    print(f"Distinct bigram contexts observed: {len(bigram_counts)} / possible contexts: {V}")
    print(f"Distinct (prev,next) bigram pairs observed: {n_observed_bigrams:,} "
          f"out of {V*V:,} possible -- {n_observed_bigrams/(V*V):.4%} of the full table")

    results = {}

    print("\n=== Raw MLE bigram model ===")
    raw_model = BigramModel(unigram_counts, bigram_counts, V, smoothing="none")
    n_zero, total = raw_model.perplexity(dataset["test_encoded"], return_failure_info=True)
    print(f"  {n_zero:,} / {total:,} test bigrams ({n_zero/total:.2%}) have ZERO probability under raw MLE")
    print(f"  -> test set log-likelihood is -infinity, so perplexity is undefined")
    results["Raw MLE"] = float("inf")

    print("\n=== Laplace (add-1) smoothing ===")
    laplace_model = BigramModel(unigram_counts, bigram_counts, V, smoothing="laplace", k=1.0)
    pp_laplace = laplace_model.perplexity(dataset["test_encoded"])
    print(f"  test perplexity = {pp_laplace:.1f}")
    results["Laplace (add-1)"] = pp_laplace

    print("\n=== Linear interpolation with unigram (tuning lambda on held-out training data) ===")
    best_lam, lambda_results = tune_interpolation_lambda(
        dataset["train_encoded"], unigram_counts, bigram_counts, unigram_probs, V
    )
    print(f"  best lambda found: {best_lam}")
    interp_model = BigramModel(unigram_counts, bigram_counts, V, smoothing="interpolation",
                                lam=best_lam, unigram_probs=unigram_probs)
    pp_interp = interp_model.perplexity(dataset["test_encoded"])
    print(f"  test perplexity = {pp_interp:.1f}")
    results["Interpolation"] = pp_interp

    print(f"\n=== Summary (lower is better; Topic 2.1's unigram baseline was 336.4) ===")
    for name, pp in results.items():
        print(f"  {name:20s}: {pp}")

    common_words = ["the", "an", "he", "she", "is", "was", "of", "in", "to", "and"]
    plot_bigram_heatmap(bigram_counts, word2idx, idx2word, common_words,
                         os.path.join(IMAGE_DIR, "bigram_heatmap.png"))
    plot_perplexity_comparison(results, os.path.join(IMAGE_DIR, "perplexity_comparison.png"))
    plot_lambda_tuning(lambda_results, os.path.join(IMAGE_DIR, "lambda_tuning.png"))
    print(f"\nSaved plots to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
