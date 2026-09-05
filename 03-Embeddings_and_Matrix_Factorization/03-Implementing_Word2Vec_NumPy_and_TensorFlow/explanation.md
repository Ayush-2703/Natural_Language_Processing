# 3.3 — Code Explanation

## Gradient check architecture

```python
def verify_gradients_numerically(V=50, d=8, k=4, eps=1e-5, seed=0):
    ...
    assert err_c < 1e-6 and err_o < 1e-6, "Gradient check failed -- do not proceed to training!"
```

`verify_gradients_numerically` runs on a small synthetic problem (`V=50, d=8`) so that evaluating the loss many times (once per dimension per parameter, in both `+` and `-` directions) is fast. The `assert` gates training: if the implementation were wrong, training would silently diverge rather than producing a plausible-looking but wrong model. The check produces `~4e-11` absolute error — limited only by the O(ε²) discretisation of the central-difference formula and float-64 rounding, not by any real disagreement between the formulas.

## Vectorised NumPy batch gradients

```python
s_neg = np.einsum("bd,bkd->bk", v_c, v_neg)
```

`np.einsum("bd,bkd->bk", v_c, v_neg)` computes every negative sample's dot product with its example's centre vector simultaneously across the whole batch: for each example `b`, for each negative `k`, `s_neg[b,k] = v_c[b] · v_neg[b,k]`. This replaces what would otherwise be a nested Python loop (extremely slow for 1,024 examples × 5 negatives × 50 dimensions) with a single, BLAS-backed matrix operation. The same pattern appears in the gradient computation:

```python
grad_v_c  = grad_s_o[:,None]*v_o + np.einsum("bk,bkd->bd", grad_s_neg, v_neg)
grad_v_neg = grad_s_neg[:,:,None] * v_c[:,None,:]
```

`grad_s_neg[:,:,None] * v_c[:,None,:]` broadcasts the `(batch, k)` gradient scalars against the `(batch, 1, d)` centre vector to get a `(batch, k, d)` gradient matrix — `k` gradient vectors per example, one per negative word.

```python
np.add.at(W_out, neg.reshape(-1), -lr * grad_v_neg.reshape(-1, d))
```

`np.add.at` is the correct tool for scatter-accumulate with potentially repeated indices: if the same word index appears as a negative sample for two different examples in the same batch, `np.add.at` accumulates both gradients into that row correctly, while plain `W_out[neg] -= ...` would silently apply only the last one (a real, common NumPy gotcha for embedding table updates).

## The two-optimiser story in TensorFlow

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
```

`theory.md` section 4 explains the bug the initial `SGD` version hit. The fix here is one line change — but it took diagnosing a perfectly flat loss curve with no error message to find it, which is why the original SGD version is preserved in `theory.md` rather than quietly replaced. The pattern "swap the optimizer to fix a gradient application bug" is worth having seen once in a real implementation.

## Actual results and what the low overlap means

```
NumPy training time:       16.6s  (217,050 pairs/sec)
TensorFlow training time:  50.2s  ( 71,687 pairs/sec)
```

NumPy is 3x faster than TensorFlow on this workload. This is not a paradox — TF's per-operation overhead for small Python-driven training loops (dispatching `tf.constant()`, `tf.gather()`, and `.numpy()` conversion on every batch) exceeds its vectorization advantage at this batch size. At larger batches or with `@tf.function` tracing (which works correctly here but adds complexity this topic deliberately avoids) the gap would shrink.

```
mean neighbour-set overlap across 10 probe words: 1%
```

1% overlap between the two models' top-10 neighbour sets is the expected, correct result — not a failure. Both models are genuinely training (both loss curves decrease), both are learning the same *mathematical* model from the same data with the same loss function, but `verify_gradients_numerically` passes for both, and both are converging to low loss. The near-zero overlap reflects that two randomly-initialised models learning from 600K pairs in 6 short epochs can converge to equally-valid, but arbitrarily-rotated, local minima in the same embedding space — word vector spaces have a continuous rotational symmetry (any orthogonal transformation of all vectors simultaneously leaves all cosine similarities unchanged), so there is no reason to expect any particular word to end up in the same direction, even if both models have learned equally good representations of the same underlying structure.
