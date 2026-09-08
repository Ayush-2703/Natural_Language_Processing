# 3.6 — Code Explanation

## ALS inner loop

```python
for i in range(V):
    wi = weights[i]
    target_i = log_cooc[i] - b_w[i] - b_c
    CwC = (C * wi[:, None]).T @ C + reg * np.eye(k)
    Cwt = (C * wi[:, None]).T @ target_i
    W[i] = np.linalg.solve(CwC, Cwt)
```

`C * wi[:, None]` scales each context vector `C[j]` by `f(X_{ij})` before the inner product — this is the per-row weighted LS solve from `theory.md`. Iterating over all `V=3000` words per epoch, each solve being a `(k,k)=(100,100)` system, keeps total ALS time around 28 seconds for 3 epochs — slower than GD but with guaranteed per-epoch improvement.

## AdaGrad batch loop

```python
nz_i, nz_j = np.nonzero(weights > 0)
...
G_W[ii] += g_W ** 2
W[ii] -= lr * g_W / np.sqrt(G_W[ii])
```

Only non-zero-weight pairs are processed — a 73% sparsity saving. The AdaGrad update for `W[ii]` uses scatter-add indexing: `G_W[ii] += g_W**2` accumulates squared gradients for the same row `ii` from multiple examples in the same batch, correctly building up the historical gradient norm over training.

## Why GloVe GD scores 0/4 on analogies despite converging well

GloVe GD's loss drops cleanly from 0.294 to 0.039 across 8 epochs — by any reconstruction-loss measure it's learning. But nearest-neighbour quality is dominated by function words (`but`, `this`, `just`) at all query words. This is the same root cause as Topic 3.2's naive negative-sampling failure: `log X_{ij}` for extremely common word pairs is very large, and `f(X_{ij}) = 1` for any pair with count ≥ 100, so the loss spends the bulk of its gradient budget fitting high-frequency pairs involving function words — even though they carry little semantic information. At larger corpus scale with longer training, GloVe's weighting eventually overcomes this and starts capturing content-word semantics; at this corpus size with 8 epochs, it hasn't had enough low-frequency pair updates to do so. SVD/PPMI beats both GloVe variants here precisely because PPMI's normalisation directly removes the frequency bias *before* factorisation, rather than hoping the weighting function will eventually balance it out during training.
