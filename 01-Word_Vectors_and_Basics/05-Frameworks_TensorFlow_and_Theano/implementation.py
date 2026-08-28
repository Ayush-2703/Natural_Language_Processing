"""
Topic 1.5 -- Basics of Computational Frameworks: TensorFlow and Theano
CSE468: Natural Language Processing with Deep Learning

This topic is about the computational-graph machinery underneath every deep
learning framework, not about a new NLP technique -- so the "model" here is
deliberately the simplest possible one (logistic regression), built without
any high-level tf.keras.layers/model.fit() convenience, using only:

    tf.Variable        -- trainable parameters living in the graph
    tf.GradientTape    -- records operations for reverse-mode autodiff
    @tf.function        -- traces a Python function into a static graph
                            (TensorFlow's modern descendant of exactly what
                            Theano did explicitly and eagerly required you
                            to declare up front -- see theory.md)

The dataset is a small, hand-written, self-contained toy sentiment set (no
external corpus download needed -- this topic is about the framework, not
about NLP data engineering, which is exhausted enough elsewhere in this
Phase) represented with simple bag-of-words counts.

Run directly:
    python implementation.py
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(HERE, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

tf.random.set_seed(42)
np.random.seed(42)

# --------------------------------------------------------------------------
# 1. A tiny, hand-written, self-contained toy sentiment dataset
# --------------------------------------------------------------------------
POSITIVE = [
    "this movie was great and i loved it",
    "an amazing wonderful film with great acting",
    "i really enjoyed this brilliant story",
    "the best film i have seen this year",
    "wonderful acting and a great plot",
    "loved every minute of this amazing movie",
    "a brilliant and wonderful experience",
    "great film with an amazing cast",
    "i loved the story and the acting was great",
    "an enjoyable and wonderful movie experience",
    "fantastic film with brilliant performances",
    "this was a truly great and enjoyable story",
    "amazing visuals and a wonderful plot",
    "the acting was brilliant and the film was great",
    "loved this fantastic and amazing film",
]
NEGATIVE = [
    "this movie was terrible and i hated it",
    "an awful boring film with terrible acting",
    "i really disliked this dreadful story",
    "the worst film i have seen this year",
    "terrible acting and a boring plot",
    "hated every minute of this awful movie",
    "a dreadful and boring experience",
    "awful film with a terrible cast",
    "i hated the story and the acting was terrible",
    "a disappointing and boring movie experience",
    "dreadful film with terrible performances",
    "this was a truly awful and boring story",
    "terrible visuals and a dreadful plot",
    "the acting was awful and the film was terrible",
    "hated this dreadful and awful film",
]


def build_bow_features(positive, negative):
    """Plain-Python bag-of-words counts -- deliberately not using
    sklearn/gensim here, since this topic is about what happens to features
    *after* you have them, not about feature engineering."""
    all_sentences = positive + negative
    vocab = sorted({w for s in all_sentences for w in s.split()})
    word2idx = {w: i for i, w in enumerate(vocab)}

    X = np.zeros((len(all_sentences), len(vocab)), dtype=np.float32)
    for row, sent in enumerate(all_sentences):
        for w in sent.split():
            X[row, word2idx[w]] += 1.0
    y = np.array([1.0] * len(positive) + [0.0] * len(negative), dtype=np.float32)
    return X, y, vocab


# --------------------------------------------------------------------------
# 2. Logistic regression, built directly from tf.Variable + tf.GradientTape
# --------------------------------------------------------------------------
class GraphLogisticRegression:
    """No tf.keras.layers.Dense, no model.fit() -- every node of the
    computational graph (the matmul, the bias add, the sigmoid, the loss)
    is written out explicitly, exactly as theory.md's diagram shows it."""

    def __init__(self, n_features):
        self.W = tf.Variable(tf.random.normal([n_features, 1], stddev=0.05), name="W")
        self.b = tf.Variable(tf.zeros([1]), name="b")

    def forward(self, X):
        z = tf.matmul(X, self.W) + self.b           # the affine node
        return tf.squeeze(tf.sigmoid(z), axis=1)     # the nonlinearity node

    def loss(self, X, y):
        p = self.forward(X)
        eps = 1e-7  # avoid log(0)
        return -tf.reduce_mean(y * tf.math.log(p + eps) + (1 - y) * tf.math.log(1 - p + eps))

    def trainable_variables(self):
        return [self.W, self.b]


def train_step_eager(model, X, y, optimizer):
    """Eager mode: every line below executes immediately, in plain Python
    control flow, the moment it's reached -- easy to debug, but TensorFlow
    re-dispatches each individual op every single call."""
    with tf.GradientTape() as tape:
        loss = model.loss(X, y)
    grads = tape.gradient(loss, model.trainable_variables())
    optimizer.apply_gradients(zip(grads, model.trainable_variables()))
    return loss


@tf.function
def train_step_graph(model, X, y, optimizer):
    """Identical math, but @tf.function traces this Python function ONCE
    into a static tf.Graph the first time it's called, then replays that
    graph directly on every subsequent call -- this is TensorFlow's modern
    answer to exactly what Theano required you to do explicitly and always:
    build the graph, then run it. See theory.md section 3."""
    with tf.GradientTape() as tape:
        loss = model.loss(X, y)
    grads = tape.gradient(loss, model.trainable_variables())
    optimizer.apply_gradients(zip(grads, model.trainable_variables()))
    return loss


def train(model, X, y, steps, use_graph_mode):
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)
    step_fn = train_step_graph if use_graph_mode else train_step_eager
    losses = []
    for _ in range(steps):
        loss = step_fn(model, X, y, optimizer)
        losses.append(float(loss))
    return losses


def accuracy(model, X, y):
    preds = (model.forward(X).numpy() > 0.5).astype(np.float32)
    return float(np.mean(preds == y))


# --------------------------------------------------------------------------
# 3. Eager vs. graph mode: a real timing comparison
# --------------------------------------------------------------------------
def time_execution_modes(X, y, n_features, steps=300, repeats=3):
    eager_times, graph_times = [], []
    for _ in range(repeats):
        m1 = GraphLogisticRegression(n_features)
        t0 = time.perf_counter()
        train(m1, X, y, steps, use_graph_mode=False)
        eager_times.append(time.perf_counter() - t0)

        m2 = GraphLogisticRegression(n_features)
        t0 = time.perf_counter()
        train(m2, X, y, steps, use_graph_mode=True)  # first call pays tracing cost
        graph_times.append(time.perf_counter() - t0)
    return eager_times, graph_times


# --------------------------------------------------------------------------
# Visualisations
# --------------------------------------------------------------------------
def plot_loss_curve(losses, path):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    ax.plot(losses, color="#4C72B0")
    ax.set_xlabel("training step"); ax.set_ylabel("binary cross-entropy loss")
    ax.set_title("Logistic regression loss -- raw tf.Variable + GradientTape")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_timing_comparison(eager_times, graph_times, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=160)
    data = [eager_times, graph_times]
    ax.boxplot(data, tick_labels=["eager", "@tf.function\n(graph mode)"])
    ax.set_ylabel("wall-clock seconds for 300 training steps")
    ax.set_title("Eager vs. graph-mode execution time")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    X_np, y_np, vocab = build_bow_features(POSITIVE, NEGATIVE)
    print(f"Toy dataset: {X_np.shape[0]} sentences, vocabulary size {len(vocab)}")
    X = tf.constant(X_np)
    y = tf.constant(y_np)

    print("\nTraining (graph mode) for 300 steps...")
    model = GraphLogisticRegression(n_features=X_np.shape[1])
    losses = train(model, X, y, steps=300, use_graph_mode=True)
    print(f"  final loss = {losses[-1]:.4f}")
    print(f"  training accuracy = {accuracy(model, X, y):.3f}")

    test_sentences = ["a wonderful and great experience", "a terrible and boring film"]
    word2idx = {w: i for i, w in enumerate(vocab)}
    X_test = np.zeros((len(test_sentences), len(vocab)), dtype=np.float32)
    for row, sent in enumerate(test_sentences):
        for w in sent.split():
            if w in word2idx:
                X_test[row, word2idx[w]] += 1.0
    preds = model.forward(tf.constant(X_test)).numpy()
    for sent, p in zip(test_sentences, preds):
        print(f"  P(positive) = {p:.3f}   <-  \"{sent}\"")

    print("\nTiming eager vs. graph-mode execution (3 repeats of 300 steps each)...")
    eager_times, graph_times = time_execution_modes(X, y, n_features=X_np.shape[1])
    print(f"  eager:  {np.mean(eager_times):.4f}s avg  {eager_times}")
    print(f"  graph:  {np.mean(graph_times):.4f}s avg  {graph_times}")
    speedup = np.mean(eager_times) / np.mean(graph_times)
    print(f"  graph mode was {speedup:.2f}x the speed of eager mode on this workload")

    plot_loss_curve(losses, os.path.join(IMAGE_DIR, "loss_curve.png"))
    plot_timing_comparison(eager_times, graph_times, os.path.join(IMAGE_DIR, "eager_vs_graph_timing.png"))
    print(f"\nSaved plots to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
