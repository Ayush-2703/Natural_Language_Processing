# 3.4 — Matrix Factorization: Introduction, Training, Models, and Regularization

## 1. The co-occurrence matrix

Given a corpus and a context window of radius `m`, the co-occurrence matrix `C` has one row and one column per vocabulary word, with `C[w, c]` counting how often words `w` and `c` appear within distance `m` of each other. This is the most direct possible operationalisation of the distributional hypothesis (Topic 1.1): it asks, literally, "which words appear near each other?" The variant implemented here uses **harmonic weighting** — a co-occurrence at distance `d` contributes `1/d` rather than `1` — so words directly adjacent to each other influence the matrix more than words at the edges of the window.

The resulting matrix is large (`V × V`, here `3000 × 3000 = 9M` entries) but very sparse (~16.5% non-zero at window=4 on this corpus), and it is symmetric by definition: `C[w,c] = C[c,w]`, since every observed context is symmetric.

## 2. PPMI: the standard reweighting

Raw co-occurrence counts have a well-known flaw: extremely frequent words like `the` co-occur at high absolute counts with almost everything, even with words that have no meaningful semantic relationship to them. **PMI (Pointwise Mutual Information)** corrects for this by measuring how much more often `w` and `c` co-occur than *chance* — where "chance" is defined as the product of their individual unigram probabilities:

```
PMI(w, c) = log( P(w, c) / (P(w) · P(c)) )
```

PMI can be negative (the pair co-occurs less than chance). **PPMI** (Positive PMI) replaces all negative values with zero — keeping only evidence of genuine positive association and discarding the absence-of-co-occurrence signal, which is too noisy to use reliably at this scale.

## 3. SVD factorisation

Truncated SVD decomposes the PPMI matrix as `PPMI ≈ U Σ Vᵀ`, where `U` and `V` are orthogonal and `Σ` is diagonal with decreasing positive values (the singular values). Keeping only the top `k` components (`k = 100` here) discards the dimensions that carry the least variance — a principled, non-parametric form of dimensionality reduction, much older than neural embeddings. The standard heuristic for producing word embeddings from the SVD is to use `U · √Σ` (the left singular vectors scaled by the square root of the singular values), which rebalances the signal between the two factors symmetrically rather than putting all the variance in one of them. The singular value spectrum plot in `images/singular_values.png` shows why truncation at 100 is reasonable: the top 50 singular values account for the vast majority of the total variance.

## 4. Alternating Least Squares (ALS)

ALS minimises the Frobenius norm of the reconstruction error, with L2 regularisation on both factors:

```
min_{W, H}  ‖M - W Hᵀ‖²_F  +  λ(‖W‖²_F + ‖H‖²_F)
```

The key structural insight: **if you hold `H` fixed and minimise over `W`, the problem is a standard ridge regression** with a closed-form solution:

```
W = M H (HᵀH + λI)^{-1}
```

Similarly, holding `W` fixed gives a closed-form solution for `H`. ALS *alternates* between these two closed-form solves — one pass updating all rows of `W`, one pass updating all rows of `H` — which guarantees the objective never increases and (for this class of problem) typically converges in a small number of iterations. `images/als_convergence.png` shows this directly: the sampled reconstruction MSE drops sharply in the first epoch and fluctuates slightly thereafter (the fluctuation is noise from sampling 5% of rows, not instability in the algorithm itself). The regularisation term `λ‖W‖²_F + λ‖H‖²_F` penalises large factors, which keeps the solution numerically stable even when `M` is sparse and many rows have very few observations.

## References

1. Church, K. W., & Hanks, P. (1990). *Word Association Norms, Mutual Information, and Lexicography.* Computational Linguistics. (The origin of PMI as a word-association measure.)
2. Bullinaria, J. A., & Levy, J. P. (2007). *Extracting Semantic Representations from Word Co-occurrence Statistics: A Computational Study.* Behavior Research Methods. (A systematic comparison of PPMI and related reweightings.)
3. Turney, P. D., & Pantel, P. (2010). *From Frequency to Meaning: Vector Space Models of Semantics.* JMLR. (Comprehensive survey of the field this topic opens.)
4. Koren, Y., Bell, R., & Volinsky, C. (2009). *Matrix Factorization Techniques for Recommender Systems.* Computer. (ALS in its most widely-known deployment, with the derivation and regularisation discussion followed here.)
