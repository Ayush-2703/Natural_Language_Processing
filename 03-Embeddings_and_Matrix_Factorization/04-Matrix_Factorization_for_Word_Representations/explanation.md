# 3.4 — Code Explanation

## Co-occurrence with harmonic weighting

```python
cooc[w, s[j]] += 1.0 / abs(i - j)
```

A word at distance 1 contributes 1.0, at distance 2 contributes 0.5, at distance 3 contributes 0.33, etc. This means the matrix does not treat "words that happen to be in the same window" uniformly — it genuinely encodes *how close* they were. The effect on embeddings: near-synonyms and idiomatically-adjacent pairs (which occur directly adjacent to each other often) are pulled tighter than words that merely appear in the same sentence.

## PPMI in three lines

```python
pmi = np.log((cooc * total + 1e-10) / (rowsum * colsum + 1e-10))
return np.maximum(pmi, 0)
```

`rowsum` and `colsum` broadcast cleanly: `rowsum` has shape `(V,1)` and `colsum` has shape `(1,V)`, so their product gives the full `(V,V)` matrix of expected joint counts under independence. The `+ 1e-10` guards against log(0) for zero-count cells. `np.maximum(pmi, 0)` applies PPMI in one vectorised call.

## ALS closed-form solve

```python
HtH = H.T @ H + reg * np.eye(k, dtype=np.float32)
W = np.linalg.solve(HtH.T, (H.T @ cooc.T).T.T).T
```

`np.linalg.solve(A, B)` solves `AX = B` for `X`, which is algebraically equivalent to `X = A^{-1} B` but numerically more stable than computing the inverse directly. The transpose gymnastics (`cooc.T`, then two `.T.T`) are reshaping the matrix multiply to get the right orientation for `np.linalg.solve`'s `(n,n), (n,k)` calling convention — the update rule in `theory.md` is `W = M H (HtH)^{-1}`, which is equivalent to `(HtH) Wᵀ = Hᵀ Mᵀ`, solved for `Wᵀ` and transposed.

## Actual results: SVD beats ALS, and why

```
SVD/PPMI: 3/4 analogies correct
ALS:      2/4 analogies correct
```

SVD/PPMI produces clearly interpretable nearest neighbours (`good`→`bad, excellent, decent, fine`), while ALS's neighbours have very high similarity scores but weak semantic coherence (`good`→`pleasant, highly, generally, often` at similarities all above 0.95). Two structural reasons: SVD operates on **PPMI** (a well-engineered reweighting designed to emphasise meaningful co-occurrence signal); ALS here operates on the **raw count matrix** (which has no such reweighting, so it spends most of its capacity fitting very common word pairs whose counts are large but semantically uninformative). ALS on PPMI (rather than on raw counts) would likely perform comparably to SVD — but that is Topic 3.6's subject (GloVe, which is effectively weighted matrix factorisation of a related target), not this topic's. Topic 3.4's job is to establish the *structure* of matrix factorisation as an approach, and ALS on raw counts is the simplest possible form of it to illustrate that structure clearly.
