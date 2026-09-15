"""
Topic 4.2 -- Hidden Markov Models (HMM) for POS Tagging
CSE468: Natural Language Processing with Deep Learning

A from-scratch HMM POS tagger trained on the Penn Treebank split built in
Topic 4.1, implementing:

  - Maximum-likelihood estimation of emission, transition, and start
    probabilities from training data.
  - Add-k (Laplace) smoothing on all probabilities to handle zero counts.
  - A suffix-based OOV heuristic: unknown words are tagged by matching
    their longest known suffix, which captures that "-ing" words are almost
    always VBG, "-ed" words are often VBD, etc.
  - The Viterbi algorithm for exact MAP decoding in log-space (avoiding
    numeric underflow on long sentences).

Run directly:
    python implementation.py
"""

import os
import pickle
import sys
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(HERE, "images")
DATASET_PATH = os.path.join(
    HERE, "..", "4.1-Introduction-to-POS-Tagging", "artifacts", "pos_dataset.pkl"
)
os.makedirs(IMAGE_DIR, exist_ok=True)


def load_or_build_dataset():
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "rb") as f:
            return pickle.load(f)
    sys.path.insert(0, os.path.join(HERE, "..", "4.1-Introduction-to-POS-Tagging"))
    from implementation import build_and_cache_dataset
    return build_and_cache_dataset()


# --------------------------------------------------------------------------
# HMM
# --------------------------------------------------------------------------
LOG_EPS = -1e9   # stand-in for log(0)


class HMMTagger:
    """
    A first-order HMM: the tag at position t depends only on the tag at t-1,
    and the word at position t depends only on the tag at t.

      P(w_1..w_n, t_1..t_n) = P(t_1) * Π P(t_i|t_{i-1}) * Π P(w_i|t_i)

    Estimated from training data with add-k smoothing throughout.
    """

    def __init__(self, train_sents, tagset, k_trans=1e-3, k_emit=1e-5):
        T = len(tagset)
        self.tagset = tagset
        self.tag2i  = {t: i for i, t in enumerate(tagset)}
        V = T   # vocabulary grows dynamically; we smooth over known words only

        # --- count ---
        start_counts = np.zeros(T, dtype=np.float64)
        trans_counts = np.zeros((T, T), dtype=np.float64)
        emit_counts  = defaultdict(lambda: np.zeros(T, dtype=np.float64))
        word_counts  = Counter()

        for sent in train_sents:
            tags = [self.tag2i[t] for _, t in sent]
            words = [w.lower() for w, _ in sent]
            start_counts[tags[0]] += 1
            for i in range(1, len(tags)):
                trans_counts[tags[i-1], tags[i]] += 1
            for i, w in enumerate(words):
                emit_counts[w][tags[i]] += 1
                word_counts[w] += 1

        # --- smooth ---
        start_counts += k_trans
        self.log_start = np.log(start_counts / start_counts.sum())

        trans_counts += k_trans
        self.log_trans = np.log(trans_counts / trans_counts.sum(axis=1, keepdims=True))

        self.emit_counts = dict(emit_counts)
        self.k_emit = k_emit
        self.T = T

        # --- suffix OOV heuristic ---
        # collect emission distributions for every suffix of length 1..4
        suffix_tag_counts = defaultdict(lambda: np.zeros(T, dtype=np.float64))
        for w, tc in emit_counts.items():
            for suf_len in range(1, 5):
                if len(w) >= suf_len:
                    suffix_tag_counts[w[-suf_len:]] += tc
        self.suffix_log_emit = {}
        for suf, tc in suffix_tag_counts.items():
            tc_smooth = tc + k_emit
            self.suffix_log_emit[suf] = np.log(tc_smooth / tc_smooth.sum())
        self.log_uniform_emit = np.log(np.ones(T) / T)

    def _log_emit(self, word):
        wl = word.lower()
        if wl in self.emit_counts:
            tc = self.emit_counts[wl] + self.k_emit
            return np.log(tc / tc.sum())
        # unknown word: try suffixes from longest to shortest
        for suf_len in range(min(4, len(wl)), 0, -1):
            suf = wl[-suf_len:]
            if suf in self.suffix_log_emit:
                return self.suffix_log_emit[suf]
        return self.log_uniform_emit

    def viterbi(self, words):
        n = len(words)
        T = self.T
        dp   = np.full((T, n), LOG_EPS)
        back = np.zeros((T, n), dtype=np.int32)

        emit0 = self._log_emit(words[0])
        dp[:, 0] = self.log_start + emit0

        for i in range(1, n):
            emit_i = self._log_emit(words[i])
            for t in range(T):
                scores = dp[:, i-1] + self.log_trans[:, t]
                best = np.argmax(scores)
                dp[t, i] = scores[best] + emit_i[t]
                back[t, i] = best

        # backtrack
        seq = [0] * n
        seq[-1] = int(np.argmax(dp[:, -1]))
        for i in range(n-2, -1, -1):
            seq[i] = int(back[seq[i+1], i+1])
        return [self.tagset[s] for s in seq]


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(tagger, sents):
    correct = total = sent_correct = 0
    for sent in sents:
        words = [w for w, _ in sent]
        gold  = [t for _, t in sent]
        pred  = tagger.viterbi(words)
        hits  = sum(p == g for p, g in zip(pred, gold))
        correct     += hits
        total       += len(gold)
        sent_correct += (hits == len(gold))
    return correct / total, sent_correct / len(sents)


def per_tag_accuracy(tagger, sents, tagset):
    per_tag = defaultdict(lambda: [0, 0])  # [correct, total]
    for sent in sents:
        words = [w for w, _ in sent]
        gold  = [t for _, t in sent]
        pred  = tagger.viterbi(words)
        for g, p in zip(gold, pred):
            per_tag[g][1] += 1
            per_tag[g][0] += (p == g)
    return {t: (per_tag[t][0] / per_tag[t][1] if per_tag[t][1] > 0 else 0)
            for t in tagset if per_tag[t][1] > 0}


# --------------------------------------------------------------------------
# Visualisations
# --------------------------------------------------------------------------
def plot_confusion_matrix(tagger, sents, tagset, path, top_n=15):
    from collections import Counter
    freq = Counter(t for s in sents for _, t in s)
    top_tags = [t for t, _ in freq.most_common(top_n)]
    tag2i = {t: i for i, t in enumerate(top_tags)}
    mat = np.zeros((top_n, top_n), dtype=np.int32)
    for sent in sents:
        words = [w for w, _ in sent]
        gold  = [t for _, t in sent]
        pred  = tagger.viterbi(words)
        for g, p in zip(gold, pred):
            if g in tag2i and p in tag2i:
                mat[tag2i[g], tag2i[p]] += 1
    norm = mat / (mat.sum(axis=1, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=(9, 8), dpi=160)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(top_n)); ax.set_xticklabels(top_tags, rotation=45, ha="right")
    ax.set_yticks(range(top_n)); ax.set_yticklabels(top_tags)
    ax.set_xlabel("predicted"); ax.set_ylabel("gold")
    ax.set_title("HMM tagger confusion matrix (top-15 tags, test set)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def plot_per_tag_accuracy(per_tag_acc, path):
    items = sorted(per_tag_acc.items(), key=lambda x: -x[1])
    tags, accs = zip(*items)
    colors = ["#4F9D69" if a >= 0.95 else "#4C72B0" if a >= 0.85 else "#C0504D" for a in accs]
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=160)
    ax.bar(tags, accs, color=colors)
    ax.axhline(0.95, linestyle="--", color="#4F9D69", lw=1.5, label="0.95 threshold")
    ax.axhline(0.85, linestyle="--", color="#C0504D", lw=1.5, label="0.85 threshold")
    ax.set_xlabel("POS tag"); ax.set_ylabel("per-tag accuracy")
    ax.set_title("HMM per-tag accuracy on test set")
    ax.legend(); plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    dataset = load_or_build_dataset()
    train, val, test = dataset["train"], dataset["val"], dataset["test"]
    tagset = dataset["tagset"]

    print("Training HMM tagger on Penn Treebank training split...")
    tagger = HMMTagger(train, tagset, k_trans=1e-3, k_emit=1e-5)

    mft_test_acc = 0.8726   # from Topic 4.1
    print(f"  MFT baseline (Topic 4.1 reference): {mft_test_acc:.4f}")

    print("\nEvaluating...")
    train_tok, train_sent = evaluate(tagger, train[:200])
    val_tok,   val_sent   = evaluate(tagger, val)
    test_tok,  test_sent  = evaluate(tagger, test)

    print(f"  train token acc  = {train_tok:.4f}  (first 200 sents)")
    print(f"  val   token acc  = {val_tok:.4f}")
    print(f"  test  token acc  = {test_tok:.4f}  (MFT was {mft_test_acc:.4f})")
    print(f"  test  sentence acc = {test_sent:.4f}")
    improvement = test_tok - mft_test_acc
    print(f"  improvement over MFT: {improvement:+.4f}")

    per_tag = per_tag_accuracy(tagger, test, tagset)
    worst = sorted(per_tag.items(), key=lambda x: x[1])[:5]
    best  = sorted(per_tag.items(), key=lambda x: -x[1])[:5]
    print(f"\n  5 hardest tags: {worst}")
    print(f"  5 easiest tags: {best}")

    # Example sentence
    example = ["The", "Fed", "raised", "interest", "rates", "last", "week", "."]
    pred = tagger.viterbi(example)
    print(f"\n  Example: {list(zip(example, pred))}")

    plot_confusion_matrix(tagger, test, tagset,
                          os.path.join(IMAGE_DIR, "confusion_matrix.png"))
    plot_per_tag_accuracy(per_tag, os.path.join(IMAGE_DIR, "per_tag_accuracy.png"))
    print(f"\nSaved plots to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
