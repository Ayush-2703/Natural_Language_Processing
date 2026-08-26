# 1.3 — Visualizing Data and Analogies (t-SNE, Embedding Projectors)

Topic 1.2 used t-SNE to ask "do these vectors cluster by category?" This topic asks a different question — "can I *see* the analogy structure from Topic 1.1?" — and the honest answer requires understanding **why t-SNE is the wrong tool for that specific question**, even though it was the right tool for clustering.

## 1. PCA: the mathematics of a linear projection

**Principal Component Analysis** finds the directions along which a dataset varies the most. Given a centred data matrix `X` (n points × d dimensions, with the mean of each column subtracted), its covariance matrix is `C = XᵀX / n`. PCA's principal components are `C`'s eigenvectors `w_1, w_2, ..., w_d`, ordered by eigenvalue `λ_1 ≥ λ_2 ≥ ... ≥ λ_d` — `λ_i` is exactly the variance of the data along direction `w_i`. Projecting onto the top `k` components (`k=2` or `3` here) means computing:

```
Y = X W_k          where W_k = [w_1, w_2, ..., w_k]  (d × k)
```

This single property — **`Y` is a fixed linear function of `X`** — is the whole reason this topic reaches for PCA instead of t-SNE for one specific job. Linear maps distribute over addition and subtraction:

```
if      vec(b) - vec(a) + vec(c)  ≈  vec(d)            (an analogy holds in R^d)
then    (vec(b) - vec(a) + vec(c)) W_k  =  vec(b)W_k - vec(a)W_k + vec(c)W_k  ≈  vec(d) W_k
```

In other words: **whatever additive structure exists in the original 100-dimensional space is, by simple linear algebra, still present in the PCA-projected 2D coordinates** (up to whatever is lost by discarding the `d - k` lowest-variance directions — a real loss, but a quantifiable and "fair" one, not a distortion). t-SNE offers no equivalent guarantee, because t-SNE's mapping from `X` to `Y` is not a fixed function at all — it is the *output of an iterative, nonlinear optimisation* (Topic 1.2, section 2) with no closed form. There is no algebraic sense in which "the y-coordinate of `b - a + c`" relates to "`y_b - y_a + y_c`," because t-SNE was never asked to make that true, and nothing in its KL-divergence objective rewards it for doing so.

## 2. Embedding projectors

TensorFlow's actual **Embedding Projector** (the tool this topic's name references) is an interactive 3D viewer: load a set of vectors, project them (it offers PCA, t-SNE, and UMAP as interchangeable choices), and rotate/zoom/hover to explore. The interactivity matters for exactly the reason a single static picture doesn't fully capture a high-dimensional space: a 3D PCA view at least preserves more of the variance than a 2D one, and being able to rotate it lets you check whether an apparent pattern survives from multiple angles rather than trusting one fixed snapshot. This topic's code builds the same kind of artifact — a real, standalone, rotatable HTML page (`embedding_projector.html`) — using Plotly instead of TensorBoard's particular implementation.

## 3. What this topic's code does, and the result worth paying attention to

Two things: (1) a general-purpose 3D PCA projector over the 400 most frequent words, coloured by frequency, saved as an interactive HTML page; and (2) a focused experiment that takes six singular→plural word pairs and checks whether their "+plural" offset vectors point the same way — visually, under both PCA and t-SNE side by side, and numerically, via cosine similarity between offset vectors computed in the real 100-dimensional space (the only place that number actually means anything, per Topic 1.2's central rule).

The result is more interesting than a clean confirmation would have been. The numeric evidence in `explanation.md` shows the six offset vectors split into two loosely-aligned sub-groups — human nouns (`boy/boys`, `girl/girls`) align with each other strongly, time-unit nouns (`day`, `night`, `week`, `month`) align with each other moderately, and the two groups barely correlate with each other at all. "+plural" is not one universal direction in this model; it's closer to several related but distinct directions, one per broad noun category. And in the t-SNE picture specifically, the time-unit group's arrows happen to *look* fairly consistent in direction — which is precisely the trap: that visual consistency is not backed by any guarantee, and the only way to know whether it reflects something real is to go check the cosine-similarity numbers in the original space, exactly as this topic's code does.

## References

1. Smilkov, D., Thorat, N., Nicholson, C., Reif, E., Viégas, F. B., & Wattenberg, M. (2016). *Embedding Projector: Interactive Visualization and Interpretation of Embeddings.* NeurIPS Workshop on Interpretable ML. (The TensorFlow tool this topic's name and design are inspired by.)
2. Pearson, K. (1901). *On Lines and Planes of Closest Fit to Systems of Points in Space.* Philosophical Magazine. (The original statement of what became PCA.)
3. Mikolov, T., Yih, W., & Zweig, G. (2013). *Linguistic Regularities in Continuous Space Word Representations.* NAACL. (The paper that first showed offset vectors like "+plural" or "+capital-of" are approximately consistent across many word pairs — the property being stress-tested here.)
