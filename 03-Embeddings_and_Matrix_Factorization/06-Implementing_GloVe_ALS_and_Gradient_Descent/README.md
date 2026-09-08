# 3.6 — Implementing GloVe using ALS and Gradient Descent

## 1. The GloVe objective

GloVe minimises a weighted sum of squared reconstruction errors over all observed word-context pairs:

```
L = Σ_{i,j}  f(X_{ij}) · ( w_i · w̃_j + b_i + b̃_j − log X_{ij} )²
```

where `X_{ij}` is the raw co-occurrence count, `w_i` and `w̃_j` are the word and context vectors being learned, `b_i` and `b̃_j` are per-word bias terms, and `f(x)` is a weighting function:

```
f(x) = (x/x_max)^α   if x < x_max,   else   1
```

with `x_max = 100` and `α = 0.75` as in the original paper. `f(x)` ramps from 0 to 1 for rare co-occurrences (discounting noisy, infrequent pairs) and stays at 1 for common ones (not over-weighting already well-estimated entries). This objective is exactly a **weighted matrix factorisation** of the `log X` matrix, with the bias terms absorbing per-word frequency effects.

## 2. ALS for GloVe

Holding `C` (context vectors) and `b_c` fixed, minimising over each word vector `w_i` independently gives a **weighted ridge regression** per word:

```
(Cᵀ Ω_i C + λI) w_i  =  Cᵀ Ω_i t_i
```

where `Ω_i = diag(f(X_{i,1}), ..., f(X_{i,V}))` and `t_i = log X_{i,:} − b_i − b_c`. This is solved exactly (closed-form) with `np.linalg.solve`. ALS alternates between this update for all word vectors and the symmetric update for all context vectors, with bias updates computed analytically between them.

## 3. AdaGrad for GloVe

The original GloVe paper uses **AdaGrad**, which maintains a per-parameter sum of squared gradients `G_θ` and adapts each parameter's effective learning rate:

```
θ ← θ − (lr / sqrt(G_θ)) · ∂L/∂θ
```

Parameters that receive large gradients frequently (common words' vectors, updated on most training pairs) automatically get a smaller effective step than rare words' vectors — exactly the property needed for a sparse co-occurrence matrix where some word pairs have orders of magnitude more observations than others.

## References

1. Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation.* EMNLP.
2. Duchi, J., Hazan, E., & Singer, Y. (2011). *Adaptive Subgradient Methods for Online Learning.* JMLR. (AdaGrad.)
