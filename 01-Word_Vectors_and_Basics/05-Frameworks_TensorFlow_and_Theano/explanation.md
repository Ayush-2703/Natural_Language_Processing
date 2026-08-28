# 1.5 — Code Explanation

## The toy dataset and feature extraction

```python
def build_bow_features(positive, negative):
    vocab = sorted({w for s in all_sentences for w in s.split()})
    ...
```

This topic deliberately uses plain Python string splitting and a hand-rolled vocabulary instead of `sklearn`'s `CountVectorizer` or anything from `gensim`/NLTK. The point being made in this topic is entirely about what happens to features *after* you have them — the graph, the gradient, the eager/graph distinction — so the feature extraction is kept as transparent and dependency-free as possible on purpose, and the 30-sentence dataset is small enough that the whole script runs in well under a second (apart from the deliberately-timed sections).

## The model, with no `tf.keras.layers`

```python
class GraphLogisticRegression:
    def __init__(self, n_features):
        self.W = tf.Variable(tf.random.normal([n_features, 1], stddev=0.05), name="W")
        self.b = tf.Variable(tf.zeros([1]), name="b")

    def forward(self, X):
        z = tf.matmul(X, self.W) + self.b
        return tf.squeeze(tf.sigmoid(z), axis=1)
```

`tf.Variable` (as opposed to `tf.constant`) is specifically a *trainable, mutable* tensor — it's the TensorFlow equivalent of a PyTorch tensor with `requires_grad=True`, and it's the only kind of tensor `GradientTape` will track gradients for by default. `stddev=0.05` for the initial random weights is a small-but-nonzero initialisation — exactly zero would make every output identical regardless of input on step one, giving the optimiser no useful gradient signal to break the symmetry. The `forward` method is, line for line, the five boxes in `theory.md`'s diagram: `matmul`, `+ b`, `sigmoid` — `tf.squeeze(..., axis=1)` just drops a redundant size-1 dimension so the output is a flat vector of probabilities rather than a column matrix.

```python
def loss(self, X, y):
    p = self.forward(X)
    eps = 1e-7
    return -tf.reduce_mean(y * tf.math.log(p + eps) + (1 - y) * tf.math.log(1 - p + eps))
```

This is binary cross-entropy written out by hand rather than called from `tf.keras.losses` — the same reasoning as everywhere else in this topic: showing the actual computation rather than a library call. `eps = 1e-7` exists because if the model ever becomes extremely confident and wrong (`p` rounds to exactly `0.0` or `1.0` in floating point), `log(0)` is `-inf`, which would poison the gradient; adding a tiny epsilon keeps the loss numerically finite no matter how confident the model gets.

## Eager vs. graph mode, as actual code

```python
def train_step_eager(model, X, y, optimizer):
    with tf.GradientTape() as tape:
        loss = model.loss(X, y)
    grads = tape.gradient(loss, model.trainable_variables())
    optimizer.apply_gradients(zip(grads, model.trainable_variables()))
    return loss

@tf.function
def train_step_graph(model, X, y, optimizer):
    # ... identical body ...
```

These two functions have **identical bodies**. The only difference is the `@tf.function` decorator, which is the entire point: the same Python code can be executed either by the interpreter directly running each line (eager) or by tracing it once into a compiled graph and replaying that graph on every subsequent call. `optimizer.apply_gradients(zip(grads, variables))` is plain gradient descent written via the Keras optimiser API: `apply_gradients` takes `(gradient, variable)` pairs and performs `variable -= learning_rate * gradient` (with whatever extra bookkeeping the chosen optimiser — here, plain `SGD` — adds; `tf.keras.optimizers` is used here purely as a gradient-descent-step utility, not as part of a `model.fit()` pipeline).

## Actual results

```
final loss = 0.0355
training accuracy = 1.000
P(positive) = 0.980   <-  "a wonderful and great experience"
P(positive) = 0.013   <-  "a terrible and boring film"
```

The model fits this tiny, easily-separable toy dataset essentially perfectly (`loss_curve.png` shows smooth, textbook convergence) and generalises correctly to two held-out test sentences built from words it saw during training in new combinations — exactly what you'd hope a from-scratch, five-node computational graph can do on a problem this simple.

```
eager:  1.4469s avg for 300 steps
graph:  0.2441s avg for 300 steps
graph mode was 5.93x the speed of eager mode on this workload
```

This is the number `theory.md` section 3 promises rather than asserts. Across three repeated runs of 300 training steps each, the `@tf.function`-compiled version is consistently about **6x faster** than the identical code running eagerly — visible directly in `eager_vs_graph_timing.png`'s two clearly separated boxes. On a dataset this tiny, every step is dominated by Python-level dispatch overhead rather than actual floating-point work, which is exactly the regime where graph-mode's "trace once, replay many times" strategy pays off most dramatically; on a much larger model where each step does substantial computation, the *relative* speedup from skipping dispatch overhead would shrink, but it would not disappear. This single timing comparison is, in miniature, the entire practical argument for why TensorFlow 1.x and Theano were built the way they were, and why TensorFlow 2.x kept graph compilation available as an opt-in rather than discarding it once eager execution became the friendlier default.
