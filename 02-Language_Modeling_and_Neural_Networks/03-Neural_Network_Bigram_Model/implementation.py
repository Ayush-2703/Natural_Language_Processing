"""
Topic 2.3 -- Implementation of the Neural Network Bigram Model
CSE468: Natural Language Processing with Deep Learning

A neural language model restricted to bigram context -- Bengio et al.
(2003)'s architecture (embedding -> hidden layer with a nonlinearity ->
softmax over the vocabulary) with the context window shrunk to a single
word, so it is the direct neural counterpart of Topic 2.2's counting-based
bigram model and can be benchmarked against it, and against Topic 2.1's
unigram baseline, on identical data with an identical evaluation metric.

Run directly:
    python implementation.py
"""

import math
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(HERE, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

DATASET_PATH = os.path.join(
    HERE, "..", "2.1-Introduction-to-Language-Modeling", "artifacts", "lm_dataset.pkl"
)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(1)


def load_or_build_dataset():
    if os.path.exists(DATASET_PATH):
        print(f"Loading cached LM dataset from {DATASET_PATH}")
        with open(DATASET_PATH, "rb") as f:
            return pickle.load(f)
    print("No cached dataset found -- rebuilding from Topic 2.1's recipe...")
    sys.path.insert(0, os.path.join(HERE, "..", "2.1-Introduction-to-Language-Modeling"))
    from implementation import build_and_cache_dataset  # noqa: E402
    return build_and_cache_dataset()


# --------------------------------------------------------------------------
# 1. Recompute Topics 2.1/2.2's baselines fresh, for a guaranteed-consistent comparison
# --------------------------------------------------------------------------
def compute_baselines(dataset):
    """Topics 2.1 and 2.2 each have a file literally named implementation.py,
    so importing both by package name would have the second import silently
    return the first cached module. UnigramModel (2.1) is imported normally
    since it's the only "implementation" on sys.path at that point; Topic
    2.2's pieces are then loaded explicitly by file path under a distinct
    module name to avoid the clash."""
    sys.path.insert(0, os.path.join(HERE, "..", "2.1-Introduction-to-Language-Modeling"))
    from implementation import UnigramModel  # Topic 2.1

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "topic_2_2_impl", os.path.join(HERE, "..", "2.2-Bigrams-and-Language-Constructs", "implementation.py")
    )
    topic_2_2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(topic_2_2)

    V = len(dataset["vocab"])
    unigram_model = UnigramModel(dataset)
    pp_unigram = unigram_model.perplexity(dataset["test_encoded"])

    unigram_counts, bigram_counts = topic_2_2.count_bigrams(dataset["train_encoded"], V)
    true_unigram_probs = topic_2_2.true_unigram_distribution(dataset["train_encoded"], V)

    laplace_model = topic_2_2.BigramModel(unigram_counts, bigram_counts, V, smoothing="laplace", k=1.0)
    pp_laplace = laplace_model.perplexity(dataset["test_encoded"])

    # Topic 2.2 already ran the lambda grid search and found 0.99 optimal on
    # validation data from this same corpus -- re-running an 8-point grid
    # search here would just burn CPU time to re-derive the same answer, so
    # the interpolation model is rebuilt directly with that value instead.
    best_lam = 0.99
    interp_model = topic_2_2.BigramModel(unigram_counts, bigram_counts, V, smoothing="interpolation",
                                          lam=best_lam, unigram_probs=true_unigram_probs)
    pp_interp = interp_model.perplexity(dataset["test_encoded"])

    return {"Unigram (2.1)": pp_unigram, "Bigram + Laplace (2.2)": pp_laplace,
            "Bigram + Interpolation (2.2)": pp_interp}


# --------------------------------------------------------------------------
# 2. Neural bigram model -- Bengio et al. (2003), context window = 1
# --------------------------------------------------------------------------
class NeuralBigramLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.hidden = nn.Linear(embed_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.activation = nn.Tanh()  # Bengio et al.'s original choice of nonlinearity

    def forward(self, prev_word_idx):
        e = self.embedding(prev_word_idx)        # (batch, embed_dim)
        h = self.activation(self.hidden(e))       # (batch, hidden_dim)
        logits = self.output(h)                   # (batch, vocab_size)
        return logits


def build_pairs(encoded_sentences):
    prevs, nexts = [], []
    for seq in encoded_sentences:
        for p, n in zip(seq[:-1], seq[1:]):
            prevs.append(p)
            nexts.append(n)
    return torch.tensor(prevs, dtype=torch.long), torch.tensor(nexts, dtype=torch.long)


def train_neural_bigram(model, train_loader, val_loader, epochs=10, lr=2e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "val_ppl": []}

    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for prev, nxt in train_loader:
            optimizer.zero_grad()
            logits = model(prev)
            loss = criterion(logits, nxt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / n_batches

        model.eval()
        val_loss, n_val_batches = 0.0, 0
        with torch.no_grad():
            for prev, nxt in val_loader:
                logits = model(prev)
                val_loss += criterion(logits, nxt).item()
                n_val_batches += 1
        val_loss /= n_val_batches
        val_ppl = math.exp(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)
        print(f"  epoch {epoch+1:2d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_ppl={val_ppl:.1f}")

    return history


def evaluate_perplexity(model, loader):
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for prev, nxt in loader:
            logits = model(prev)
            total_loss += criterion(logits, nxt).item()
            total_tokens += nxt.size(0)
    return math.exp(total_loss / total_tokens)


# --------------------------------------------------------------------------
# Visualisations
# --------------------------------------------------------------------------
def plot_training_curves(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy loss"); axes[0].legend()
    axes[0].set_title("Loss")
    axes[1].plot(history["val_ppl"], color="#C0504D")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation perplexity")
    axes[1].set_title("Validation perplexity")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_final_comparison(results, path):
    names = list(results.keys())
    values = list(results.values())
    colors = ["#888888"] * (len(names) - 1) + ["#4F9D69"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    bars = ax.bar(names, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("test-set perplexity (lower is better)")
    ax.set_title("Every language model built in Phase 2, same data, same metric")
    plt.xticks(rotation=20, ha="right")
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

    print("Recomputing Topic 2.1/2.2 baselines for a guaranteed fair comparison...")
    baselines = compute_baselines(dataset)
    for name, pp in baselines.items():
        print(f"  {name:32s}: {pp:.1f}")

    # held-out validation split, carved from the training sentences only
    rng = np.random.RandomState(SEED)
    train_sents = dataset["train_encoded"]
    perm = rng.permutation(len(train_sents))
    n_val = int(0.1 * len(train_sents))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr_sents = [train_sents[i] for i in tr_idx]
    val_sents = [train_sents[i] for i in val_idx]

    train_prev_full, train_next_full = build_pairs(tr_sents)
    val_prev, val_next = build_pairs(val_sents)
    test_prev, test_next = build_pairs(dataset["test_encoded"])

    # Subsample the training pairs: classical counting (2.1/2.2) is a single
    # pass over the data, but gradient-descent training needs many epochs
    # over whatever it's given, and this sandbox has a single CPU core. 200k
    # pairs (roughly a quarter of the ~815k available here) keeps total
    # training time on the order of a couple of minutes rather than tens of
    # minutes, while still being a real, substantial sample -- not a toy. On
    # a normal multi-core machine or GPU, MAX_TRAIN_PAIRS can simply be raised.
    MAX_TRAIN_PAIRS = 200_000
    if len(train_prev_full) > MAX_TRAIN_PAIRS:
        g = torch.Generator().manual_seed(SEED)
        keep = torch.randperm(len(train_prev_full), generator=g)[:MAX_TRAIN_PAIRS]
        train_prev, train_next = train_prev_full[keep], train_next_full[keep]
    else:
        train_prev, train_next = train_prev_full, train_next_full
    print(f"\nTraining pairs: {len(train_prev):,} (subsampled from {len(train_prev_full):,})  "
          f"Validation pairs: {len(val_prev):,}  Test pairs: {len(test_prev):,}")

    train_loader = DataLoader(TensorDataset(train_prev, train_next), batch_size=2048, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_prev, val_next), batch_size=2048, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_prev, test_next), batch_size=2048, shuffle=False)

    print("\nTraining the neural bigram model...")
    model = NeuralBigramLM(vocab_size=V, embed_dim=32, hidden_dim=32)
    history = train_neural_bigram(model, train_loader, val_loader, epochs=5)

    test_ppl = evaluate_perplexity(model, test_loader)
    print(f"\nNeural bigram model test perplexity: {test_ppl:.1f}")

    print("\n=== Qualitative check: top-5 predicted next words ===")
    model.eval()
    for w in ["he", "the", "of", "is"]:
        idx = torch.tensor([word2idx[w]])
        with torch.no_grad():
            probs = torch.softmax(model(idx), dim=-1).squeeze(0)
        top5 = torch.topk(probs, 5)
        preds = [(idx2word[i.item()], round(p.item(), 3)) for p, i in zip(top5.values, top5.indices)]
        print(f"  P(next | {w!r}) top-5: {preds}")

    all_results = dict(baselines)
    all_results["Neural bigram (2.3)"] = test_ppl
    print("\n=== Final comparison (test-set perplexity, lower is better) ===")
    for name, pp in all_results.items():
        print(f"  {name:32s}: {pp:.1f}")

    plot_training_curves(history, os.path.join(IMAGE_DIR, "training_curves.png"))
    plot_final_comparison(all_results, os.path.join(IMAGE_DIR, "final_comparison.png"))
    print(f"\nSaved plots to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
